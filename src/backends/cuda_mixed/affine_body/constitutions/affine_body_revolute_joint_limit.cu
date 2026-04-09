#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/utils.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <utils/matrix_assembler.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed::sym::affine_body_revolute_joint_limit
{
#include <affine_body/constitutions/sym/affine_body_revolute_joint_limit.inl>
}

namespace uipc::backend::cuda_mixed
{
class AffineBodyRevoluteJointLimit final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 670;
    static constexpr U64   JointUID        = 18;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;

    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    using Vector24    = Vector<Float, 24>;
    using Matrix24x24 = Matrix<Float, 24, 24>;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_l_basis;
    vector<Vector6>  h_r_basis;
    vector<Vector24> h_ref_qs;
    vector<Float>    h_lowers;
    vector<Float>    h_uppers;
    vector<Float>    h_strengths;

    muda::DeviceBuffer<Vector2i> body_ids;
    muda::DeviceBuffer<Vector6>  l_basis;
    muda::DeviceBuffer<Vector6>  r_basis;
    muda::DeviceBuffer<Vector24> ref_qs;
    muda::DeviceBuffer<Float>    lowers;
    muda::DeviceBuffer<Float>    uppers;
    muda::DeviceBuffer<Float>    strengths;

    static auto get_revolute_basis(const geometry::SimplicialComplex* L,
                                   IndexT                             L_inst_id,
                                   const geometry::SimplicialComplex* R,
                                   IndexT                             R_inst_id,
                                   const geometry::SimplicialComplex* joint_mesh,
                                   IndexT                             joint_index)
    {
        auto topo_view = joint_mesh->edges().topo().view();
        auto pos_view  = joint_mesh->positions().view();

        Vector2i e = topo_view[joint_index];
        Vector3  t = pos_view[e[1]] - pos_view[e[0]];
        UIPC_ASSERT(t.squaredNorm() > 0.0,
                    "AffineBodyRevoluteJointLimit: joint edge {} has zero length; cannot compute revolute basis",
                    joint_index);
        Vector3  n;
        Vector3  b;
        orthonormal_basis(t, n, b);

        auto compute_bn_bar = [&](const geometry::SimplicialComplex* geo, IndexT inst_id) -> Vector6
        {
            UIPC_ASSERT(geo, "AffineBodyRevoluteJointLimit: geometry is null when computing basis");

            const Matrix4x4& trans = geo->transforms().view()[inst_id];
            Transform        T{trans};
            Matrix3x3        inv_rot = T.rotation().inverse();
            Vector6          bn_bar;
            bn_bar.segment<3>(0) = inv_rot * b;
            bn_bar.segment<3>(3) = inv_rot * n;
            return bn_bar;
        };

        Vector6 L_bn_bar = compute_bn_bar(L, L_inst_id);
        Vector6 R_bn_bar = compute_bn_bar(R, R_inst_id);
        return std::tuple{L_bn_bar, R_bn_bar};
    }

    void do_build(BuildInfo& info) override {}

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        h_body_ids.clear();
        h_l_basis.clear();
        h_r_basis.clear();
        h_ref_qs.clear();
        h_lowers.clear();
        h_uppers.clear();
        h_strengths.clear();

        info.for_each(
            geo_slots,
            [&](geometry::Geometry& geo)
            {
                auto uid = geo.meta().find<U64>(builtin::constitution_uid);
                UIPC_ASSERT(uid && uid->view()[0] == JointUID,
                            "AffineBodyRevoluteJointLimit must be attached on base revolute joint geometry (UID={})",
                            JointUID);

                auto sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc, "AffineBodyRevoluteJointLimit geometry must be SimplicialComplex");

                auto geo_ids_attr = sc->edges().find<Vector2i>("geo_ids");
                UIPC_ASSERT(geo_ids_attr,
                            "AffineBodyRevoluteJointLimit requires `geo_ids` attribute on edges");
                auto geo_ids = geo_ids_attr->view();

                auto inst_ids_attr = sc->edges().find<Vector2i>("inst_ids");
                UIPC_ASSERT(inst_ids_attr,
                            "AffineBodyRevoluteJointLimit requires `inst_ids` attribute on edges");
                auto inst_ids = inst_ids_attr->view();

                auto lower_attr = sc->edges().find<Float>("limit/lower");
                UIPC_ASSERT(lower_attr,
                            "AffineBodyRevoluteJointLimit requires `limit/lower` attribute on edges");
                auto lower_view = lower_attr->view();

                auto upper_attr = sc->edges().find<Float>("limit/upper");
                UIPC_ASSERT(upper_attr,
                            "AffineBodyRevoluteJointLimit requires `limit/upper` attribute on edges");
                auto upper_view = upper_attr->view();

                auto strength_attr = sc->edges().find<Float>("limit/strength");
                UIPC_ASSERT(strength_attr,
                            "AffineBodyRevoluteJointLimit requires `limit/strength` attribute on edges");
                auto strength_view = strength_attr->view();

                auto init_angle_attr = sc->edges().find<Float>("init_angle");
                UIPC_ASSERT(init_angle_attr,
                            "AffineBodyRevoluteJointLimit requires `init_angle` attribute on edges");
                auto init_angle_view = init_angle_attr->view();

                auto edges = sc->edges().topo().view();
                for(auto&& [i, e] : enumerate(edges))
                {
                    Vector2i geo_id  = geo_ids[i];
                    Vector2i inst_id = inst_ids[i];

                    auto* left_sc  = info.body_geo(geo_slots, geo_id[0]);
                    auto* right_sc = info.body_geo(geo_slots, geo_id[1]);

                    UIPC_ASSERT(inst_id[0] >= 0
                                    && inst_id[0] < static_cast<IndexT>(left_sc->instances().size()),
                                "AffineBodyRevoluteJointLimit: left instance ID {} out of range [0, {})",
                                inst_id[0],
                                left_sc->instances().size());
                    UIPC_ASSERT(inst_id[1] >= 0
                                    && inst_id[1] < static_cast<IndexT>(right_sc->instances().size()),
                                "AffineBodyRevoluteJointLimit: right instance ID {} out of range [0, {})",
                                inst_id[1],
                                right_sc->instances().size());

                    Vector2i bid = {
                        info.body_id(geo_id[0], inst_id[0]),
                        info.body_id(geo_id[1], inst_id[1]),
                    };

                    auto [lb, rb] = get_revolute_basis(
                        left_sc, inst_id[0], right_sc, inst_id[1], sc, i);

                    Vector24 ref;
                    ref.segment<12>(0) =
                        transform_to_q(left_sc->transforms().view()[inst_id[0]]);
                    ref.segment<12>(12) =
                        transform_to_q(right_sc->transforms().view()[inst_id[1]]);

                    h_body_ids.push_back(bid);
                    h_l_basis.push_back(lb);
                    h_r_basis.push_back(rb);
                    h_ref_qs.push_back(ref);
                    Float init_angle   = init_angle_view[i];
                    Float actual_lower = lower_view[i] + init_angle;
                    Float actual_upper = upper_view[i] + init_angle;
                    UIPC_ASSERT(actual_lower <= actual_upper,
                                "AffineBodyRevoluteJointLimit: requires `limit/lower + init_angle <= limit/upper + init_angle` on edge {}, but got lower={} upper={} init_angle={}",
                                i,
                                lower_view[i],
                                upper_view[i],
                                init_angle);
                    h_lowers.push_back(actual_lower);
                    h_uppers.push_back(actual_upper);
                    h_strengths.push_back(strength_view[i]);
                }
            });

        body_ids.copy_from(h_body_ids);
        l_basis.copy_from(h_l_basis);
        r_basis.copy_from(h_r_basis);
        ref_qs.copy_from(h_ref_qs);
        lowers.copy_from(h_lowers);
        uppers.copy_from(h_uppers);
        strengths.copy_from(h_strengths);
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace ERJ = sym::affine_body_revolute_joint_limit;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                body_ids.size(),
                [body_ids = body_ids.cviewer().name("body_ids"),
                 l_basis = l_basis.cviewer().name("l_basis"),
                 r_basis = r_basis.cviewer().name("r_basis"),
                 ref_qs  = ref_qs.cviewer().name("ref_qs"),
                 lowers = lowers.cviewer().name("lowers"),
                 uppers = uppers.cviewer().name("uppers"),
                 strengths = strengths.cviewer().name("strengths"),
                 qs      = info.qs().cviewer().name("qs"),
                 q_prevs = info.q_prevs().cviewer().name("q_prevs"),
                 Es      = info.energies().viewer().name("Es")] __device__(int I)
                {
                    using Alu = ActivePolicy::AluScalar;
                    using Store = InterAffineBodyConstitutionManager::StoreScalar;
                    Vector2i bid = body_ids(I);

                    Eigen::Matrix<Alu, 6, 1> lb = l_basis(I).template cast<Alu>();
                    Eigen::Matrix<Alu, 6, 1> rb = r_basis(I).template cast<Alu>();
                    Vector24 ref_q = ref_qs(I);

                    Eigen::Matrix<Alu, 12, 1> qk = qs(bid[0]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> ql = qs(bid[1]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_prevk =
                        q_prevs(bid[0]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_prevl =
                        q_prevs(bid[1]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_refk =
                        ref_q.segment<12>(0).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_refl =
                        ref_q.segment<12>(12).template cast<Alu>();

                    Alu theta0 = 0;
                    ERJ::DeltaTheta<Alu>(theta0, lb, q_prevk, q_refk, rb, q_prevl, q_refl);

                    Alu delta = 0;
                    ERJ::DeltaTheta<Alu>(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                    Alu x        = theta0 + delta;
                    Alu lower    = safe_cast<Alu>(lowers(I));
                    Alu upper    = safe_cast<Alu>(uppers(I));
                    Alu strength = safe_cast<Alu>(strengths(I));

                    Alu E = 0;
                    if(x > upper)
                    {
                        Alu d = x - upper;
                        E       = strength * d * d * d;
                    }
                    else if(x < lower)
                    {
                        Alu d = lower - x;
                        E       = strength * d * d * d;
                    }

                    Es(I) = safe_cast<Float>(E);
                });
    }

    void do_report_gradient_hessian_extent(GradientHessianExtentInfo& info) override
    {
        info.gradient_count(2 * body_ids.size());
        if(info.gradient_only())
            return;

        info.hessian_count(HalfHessianSize * body_ids.size());
    }

    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace ERJ = sym::affine_body_revolute_joint_limit;
        auto gradient_only = info.gradient_only();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                body_ids.size(),
                [body_ids = body_ids.cviewer().name("body_ids"),
                 l_basis = l_basis.cviewer().name("l_basis"),
                 r_basis = r_basis.cviewer().name("r_basis"),
                 ref_qs  = ref_qs.cviewer().name("ref_qs"),
                 lowers = lowers.cviewer().name("lowers"),
                 uppers = uppers.cviewer().name("uppers"),
                 strengths = strengths.cviewer().name("strengths"),
                 qs      = info.qs().cviewer().name("qs"),
                 q_prevs = info.q_prevs().cviewer().name("q_prevs"),
                 G12s    = info.gradients().viewer().name("G12s"),
                 H12x12s = info.hessians().viewer().name("H12x12s"),
                 gradient_only] __device__(int I) mutable
                {
                    using Alu = ActivePolicy::AluScalar;
                    using Store = InterAffineBodyConstitutionManager::StoreScalar;
                    Vector2i bid = body_ids(I);

                    Eigen::Matrix<Alu, 6, 1> lb = l_basis(I).template cast<Alu>();
                    Eigen::Matrix<Alu, 6, 1> rb = r_basis(I).template cast<Alu>();
                    Vector24 ref_q = ref_qs(I);

                    Eigen::Matrix<Alu, 12, 1> qk = qs(bid[0]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> ql = qs(bid[1]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_prevk =
                        q_prevs(bid[0]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_prevl =
                        q_prevs(bid[1]).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_refk =
                        ref_q.segment<12>(0).template cast<Alu>();
                    Eigen::Matrix<Alu, 12, 1> q_refl =
                        ref_q.segment<12>(12).template cast<Alu>();

                    Alu theta0 = 0;
                    ERJ::DeltaTheta<Alu>(theta0, lb, q_prevk, q_refk, rb, q_prevl, q_refl);

                    Alu delta = 0;
                    ERJ::DeltaTheta<Alu>(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                    Alu x        = theta0 + delta;
                    Alu lower    = safe_cast<Alu>(lowers(I));
                    Alu upper    = safe_cast<Alu>(uppers(I));
                    Alu strength = safe_cast<Alu>(strengths(I));

                    Alu dE_dx   = 0;
                    Alu d2E_dx2 = 0;

                    if(x > upper)
                    {
                        Alu d = x - upper;
                        dE_dx   = safe_cast<Alu>(3.0) * strength * d * d;
                        d2E_dx2 = safe_cast<Alu>(6.0) * strength * d;
                    }
                    else if(x < lower)
                    {
                        Alu d = lower - x;
                        dE_dx   = safe_cast<Alu>(-3.0) * strength * d * d;
                        d2E_dx2 = safe_cast<Alu>(6.0) * strength * d;
                    }

                    Eigen::Matrix<Alu, 24, 1> dx_dq;
                    ERJ::dDeltaTheta_dQ<Alu>(dx_dq, lb, qk, q_prevk, rb, ql, q_prevl);

                    Eigen::Matrix<Alu, 24, 1> G = dE_dx * dx_dq;
                    DoubletVectorAssembler DVA{G12s};
                    DVA.segment<2>(2 * I).write(bid, downcast_gradient<Store>(G));

                    if(gradient_only)
                        return;

                    Eigen::Matrix<Alu, 24, 24> H = d2E_dx2 * (dx_dq * dx_dq.transpose());

                    if(dE_dx != safe_cast<Alu>(0.0))
                    {
                        Eigen::Matrix<Alu, 12, 1> F;
                        Eigen::Matrix<Alu, 12, 1> F_prev;
                        ERJ::F<Alu>(F, lb, qk, rb, ql);
                        ERJ::F<Alu>(F_prev, lb, q_prevk, rb, q_prevl);

                        Eigen::Matrix<Alu, 12, 12> ddx_ddF;
                        ERJ::ddDeltaTheta_ddF(ddx_ddF, F, F_prev);

                        Eigen::Matrix<Alu, 12, 12> H_F = dE_dx * ddx_ddF;

                        Eigen::Matrix<Alu, 24, 24> JT_H_J;
                        ERJ::JT_H_J<Alu>(JT_H_J, H_F, lb, rb, lb, rb);
                        H += JT_H_J;
                    }

                    TripletMatrixAssembler TMA{H12x12s};
                    TMA.half_block<2>(HalfHessianSize * I)
                        .write(bid, downcast_hessian<Store>(H));
                });
    }

    U64 get_uid() const noexcept override { return ConstitutionUID; }
};

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointLimit);
}  // namespace uipc::backend::cuda_mixed
