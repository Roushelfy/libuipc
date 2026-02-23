#include <affine_body/inter_affine_body_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constraint.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>
#include <uipc/common/enumerate.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
static constexpr U64          ConstitutionUID = 18;
class AffineBodyRevoluteJoint final : public InterAffineBodyConstitution
{
  public:
    using InterAffineBodyConstitution::InterAffineBodyConstitution;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    vector<Vector2i> h_body_ids;
    // [    body0   |   body1    ]
    // [    x0, x1  |   x2, x3   ]
    vector<Vector12> h_rest_positions;
    vector<Float>    h_strength_ratio;

    muda::DeviceBuffer<Vector2i> body_ids;
    muda::DeviceBuffer<Vector12> rest_positions;
    muda::DeviceBuffer<Float>    strength_ratio;


    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        list<Vector2i> body_ids_list;
        list<Vector12> rest_positions_list;
        list<Float>    strength_ratio_list;

        info.for_each(
            geo_slots,
            [&](const InterAffineBodyConstitutionManager::ForEachInfo& I, geometry::Geometry& geo)
            {
                auto uid = geo.meta().find<U64>(builtin::constitution_uid);
                U64  uid_value = uid->view()[0];
                UIPC_ASSERT(uid_value == ConstitutionUID,
                            "AffineBodyRevoluteJoint: Geometry constitution UID mismatch");

                auto joint_geo_id = I.geo_info().geo_id;

                auto sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc, "AffineBodyRevoluteJoint: Geometry must be a simplicial complex");

                auto geo_ids = sc->edges().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids, "AffineBodyRevoluteJoint: Geometry must have 'geo_ids' attribute on `edges`");
                auto geo_ids_view = geo_ids->view();

                auto inst_ids = sc->edges().find<Vector2i>("inst_ids");
                UIPC_ASSERT(inst_ids, "AffineBodyRevoluteJoint: Geometry must have 'inst_ids' attribute on `edges`");
                auto inst_ids_view = inst_ids->view();

                auto strength_ratio = sc->edges().find<Float>("strength_ratio");
                UIPC_ASSERT(strength_ratio, "AffineBodyRevoluteJoint: Geometry must have 'strength_ratio' attribute on `edges`");
                auto strength_ratio_view = strength_ratio->view();

                auto Es = sc->edges().topo().view();
                auto Ps = sc->positions().view();
                for(auto&& [i, e] : enumerate(Es))
                {
                    Vector2i geo_id  = geo_ids_view[i];
                    Vector2i inst_id = inst_ids_view[i];

                    Vector3 P0  = Ps[e[0]];
                    Vector3 P1  = Ps[e[1]];
                    Vector3 mid = (P0 + P1) / 2;
                    Vector3 Dir = (P1 - P0);

                    UIPC_ASSERT(Dir.norm() > 1e-12,
                                R"(AffineBodyRevoluteJoint: Edge with zero length detected,
Joint GeometryID = {},
LinkGeoIDs       = ({}, {}),
LinkInstIDs      = ({}, {}),
Edge             = ({}, {}))",
                                joint_geo_id,
                                geo_id(0),
                                geo_id(1),
                                inst_id(0),
                                inst_id(1),
                                e(0),
                                e(1));

                    Vector3 HalfAxis = Dir.normalized() / 2;

                    // Re-define P0 and P1 to be symmetric around the mid-point
                    P0 = mid - HalfAxis;
                    P1 = mid + HalfAxis;

                    Vector2i body_ids = {info.body_id(geo_id(0), inst_id(0)),
                                         info.body_id(geo_id(1), inst_id(1))};
                    body_ids_list.push_back(body_ids);

                    auto left_sc  = info.body_geo(geo_slots, geo_id(0));
                    auto right_sc = info.body_geo(geo_slots, geo_id(1));

                    UIPC_ASSERT(inst_id(0) >= 0
                                    && inst_id(0) < static_cast<IndexT>(
                                           left_sc->instances().size()),
                                "AffineBodyRevoluteJoint: Left instance ID {} is out of range [0, {})",
                                inst_id(0),
                                left_sc->instances().size());
                    UIPC_ASSERT(inst_id(1) >= 0
                                    && inst_id(1) < static_cast<IndexT>(
                                           right_sc->instances().size()),
                                "AffineBodyRevoluteJoint: Right instance ID {} is out of range [0, {})",
                                inst_id(1),
                                right_sc->instances().size());

                    Transform LT{left_sc->transforms().view()[inst_id(0)]};
                    Transform RT{right_sc->transforms().view()[inst_id(1)]};

                    Vector12 rest_pos;
                    rest_pos.segment<3>(0) = LT.inverse() * P0;  // x0_bar
                    rest_pos.segment<3>(3) = LT.inverse() * P1;  // x1_bar

                    rest_pos.segment<3>(6) = RT.inverse() * P0;  // x2_bar
                    rest_pos.segment<3>(9) = RT.inverse() * P1;  // x3_bar
                    rest_positions_list.push_back(rest_pos);
                }

                std::ranges::copy(strength_ratio_view,
                                  std::back_inserter(strength_ratio_list));
            });

        h_body_ids.resize(body_ids_list.size());
        std::ranges::move(body_ids_list, h_body_ids.begin());

        h_rest_positions.resize(rest_positions_list.size());
        std::ranges::move(rest_positions_list, h_rest_positions.begin());

        h_strength_ratio.resize(strength_ratio_list.size());
        std::ranges::move(strength_ratio_list, h_strength_ratio.begin());

        body_ids.copy_from(h_body_ids);
        rest_positions.copy_from(h_rest_positions);
        strength_ratio.copy_from(h_strength_ratio);
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());  // one energy per joint
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace RJ = sym::affine_body_revolute_joint;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(body_ids.size(),
                   [body_ids = body_ids.cviewer().name("body_ids"),
                    rest_positions = rest_positions.cviewer().name("rest_positions"),
                    strength_ratio = strength_ratio.cviewer().name("strength_ratio"),
                    body_masses = info.body_masses().viewer().name("body_masses"),
                    qs = info.qs().viewer().name("qs"),
                    Es = info.energies().viewer().name("Es")] __device__(int I)
                   {
                       using Alu = ActivePolicy::AluScalar;
                       Vector2i bids = body_ids(I);

                       Alu kappa = safe_cast<Alu>(
                           strength_ratio(I)
                           * (body_masses(bids(0)).mass() + body_masses(bids(1)).mass()));

                       const Vector12& X_bar = rest_positions(I);

                       Eigen::Matrix<Alu, 12, 1> q_i = qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_j = qs(bids(1)).template cast<Alu>();

                       ABDJacobi Js[4] = {ABDJacobi{X_bar.segment<3>(0)},
                                          ABDJacobi{X_bar.segment<3>(3)},
                                          ABDJacobi{X_bar.segment<3>(6)},
                                          ABDJacobi{X_bar.segment<3>(9)}};

                       Eigen::Matrix<Alu, 3, 12> J0 = Js[0].to_mat().template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 12> J1 = Js[1].to_mat().template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 12> J2 = Js[2].to_mat().template cast<Alu>();
                       Eigen::Matrix<Alu, 3, 12> J3 = Js[3].to_mat().template cast<Alu>();

                       Eigen::Matrix<Alu, 3, 1> x0 = (J0 * q_i).eval();
                       Eigen::Matrix<Alu, 3, 1> x1 = (J1 * q_i).eval();
                       Eigen::Matrix<Alu, 3, 1> x2 = (J2 * q_j).eval();
                       Eigen::Matrix<Alu, 3, 1> x3 = (J3 * q_j).eval();

                       Alu E0 = (x0 - x2).squaredNorm();
                       Alu E1 = (x1 - x3).squaredNorm();

                       // energy = 1/2 * kappa * (||x0 - x2||^2 + ||x1 - x3||^2)

                       Es(I) = safe_cast<Float>(kappa * safe_cast<Alu>(0.5) * (E0 + E1));
                   });
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(2 * body_ids.size());  // each joint has 2 * Vector12 gradients
        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * body_ids.size());  // each joint has HalfHessianSize * Matrix12x12 hessians
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace RJ = sym::affine_body_revolute_joint;
        auto gradient_only = info.gradient_only();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                body_ids.size(),
                [body_ids = body_ids.cviewer().name("body_ids"),
                 rest_positions = rest_positions.cviewer().name("rest_positions"),
                 strength_ratio = strength_ratio.cviewer().name("strength_ratio"),
                 body_masses = info.body_masses().viewer().name("body_masses"),
                 qs          = info.qs().viewer().name("qs"),
                 G12s        = info.gradients().viewer().name("G12s"),
                 H12x12s = info.hessians().viewer().name("H12x12s"),
                 gradient_only] __device__(int I)
                {
                    using Alu = ActivePolicy::AluScalar;
                    Vector2i        bids  = body_ids(I);
                    const Vector12& X_bar = rest_positions(I);

                    Eigen::Matrix<Alu, 12, 1> q_i = qs(bids(0)).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_j = qs(bids(1)).template cast<Alu>();

                    ABDJacobi Js[4] = {ABDJacobi{X_bar.segment<3>(0)},
                                       ABDJacobi{X_bar.segment<3>(3)},
                                       ABDJacobi{X_bar.segment<3>(6)},
                                       ABDJacobi{X_bar.segment<3>(9)}};

                    Eigen::Matrix<Alu, 3, 12> J0 = Js[0].to_mat().template cast<Alu>();
                    Eigen::Matrix<Alu, 3, 12> J1 = Js[1].to_mat().template cast<Alu>();
                    Eigen::Matrix<Alu, 3, 12> J2 = Js[2].to_mat().template cast<Alu>();
                    Eigen::Matrix<Alu, 3, 12> J3 = Js[3].to_mat().template cast<Alu>();

                    Eigen::Matrix<Alu, 3, 1> D02 = (J0 * q_i - J2 * q_j).eval();
                    Eigen::Matrix<Alu, 3, 1> D13 = (J1 * q_i - J3 * q_j).eval();
                    Alu K = safe_cast<Alu>(
                        strength_ratio(I)
                        * (body_masses(bids(0)).mass() + body_masses(bids(1)).mass()));

                    // Fill Body Gradient:
                    {
                        // G = 0.5 * kappa * (J0^T * (x0 - x2) + J1^T * (x1 - x3))
                        Eigen::Matrix<Alu, 12, 1> G_i =
                            (K * (J0.transpose() * D02 + J1.transpose() * D13)).eval();
                        G12s(2 * I + 0).write(bids(0), downcast_gradient<Float>(G_i));
                    }
                    {
                        // G = 0.5 * kappa * (J2^T * (x2 - x0) + J3^T * (x3 - x1))
                        Eigen::Matrix<Alu, 12, 1> G_j =
                            (K * (J2.transpose() * (-D02) + J3.transpose() * (-D13))).eval();
                        G12s(2 * I + 1).write(bids(1), downcast_gradient<Float>(G_j));
                    }

                    // Fill Body Hessian:
                    if(!gradient_only)
                    {
                        {
                            Eigen::Matrix<Alu, 12, 12> H_ii;
                            RJ::Hess<Alu>(H_ii,
                                          K,
                                          Js[0].x_bar().template cast<Alu>(),
                                          Js[0].x_bar().template cast<Alu>(),
                                          Js[1].x_bar().template cast<Alu>(),
                                          Js[1].x_bar().template cast<Alu>());
                            H12x12s(HalfHessianSize * I + 0)
                                .write(bids(0), bids(0), downcast_hessian<Float>(H_ii));
                        }
                        {
                            Eigen::Matrix<Alu, 12, 12> H;
                            Vector2i    lr = bids;
                            RJ::Hess<Alu>(H,
                                          -K,
                                          Js[0].x_bar().template cast<Alu>(),
                                          Js[2].x_bar().template cast<Alu>(),
                                          Js[1].x_bar().template cast<Alu>(),
                                          Js[3].x_bar().template cast<Alu>());
                            if(bids(0) > bids(1))
                            {
                                H.transposeInPlace();
                                lr = Vector2i{bids(1), bids(0)};
                            }
                            H12x12s(HalfHessianSize * I + 1)
                                .write(lr(0), lr(1), downcast_hessian<Float>(H));
                        }
                        {
                            Eigen::Matrix<Alu, 12, 12> H_jj;
                            RJ::Hess<Alu>(H_jj,
                                          K,
                                          Js[2].x_bar().template cast<Alu>(),
                                          Js[2].x_bar().template cast<Alu>(),
                                          Js[3].x_bar().template cast<Alu>(),
                                          Js[3].x_bar().template cast<Alu>());
                            H12x12s(HalfHessianSize * I + 2)
                                .write(bids(1), bids(1), downcast_hessian<Float>(H_jj));
                        }
                    }
                });
    }

    U64 get_uid() const noexcept override { return ConstitutionUID; }
};

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJoint);
}  // namespace uipc::backend::cuda_mixed
