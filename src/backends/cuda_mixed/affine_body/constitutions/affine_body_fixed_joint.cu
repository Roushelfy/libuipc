#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/constitutions/affine_body_fixed_joint_function.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <uipc/common/enumerate.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class AffineBodyFixedJoint final : public InterAffineBodyConstitution
{
  public:
    static constexpr U64   ConstitutionUID = 25;
    static constexpr SizeT HalfHessianSize = 2 * (2 + 1) / 2;
    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    muda::DeviceBuffer<Vector2i> body_ids;
    muda::DeviceBuffer<Vector6> rest_cs;
    muda::DeviceBuffer<Vector6> rest_ts;
    muda::DeviceBuffer<Vector6> rest_ns;
    muda::DeviceBuffer<Vector6> rest_bs;
    muda::DeviceBuffer<Float>   strength_ratios;

    vector<Vector2i> h_body_ids;
    vector<Vector6>  h_rest_cs;
    vector<Vector6>  h_rest_ts;
    vector<Vector6>  h_rest_ns;
    vector<Vector6>  h_rest_bs;
    vector<Float>    h_strength_ratios;

    using Store       = ActivePolicy::StoreScalar;
    using Vector24    = Vector<Store, 24>;
    using Matrix24x24 = Matrix<Store, 24, 24>;

    void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_init(FilteredInfo& info) override
    {
        auto geo_slots = world().scene().geometries();

        list<Vector2i> body_ids_list;
        list<Vector6>  rest_c_list;
        list<Vector6>  rest_t_list;
        list<Vector6>  rest_n_list;
        list<Vector6>  rest_b_list;
        list<Float>    strength_ratio_list;

        info.for_each(
            geo_slots,
            [&](geometry::Geometry& geo)
            {
                auto sc = geo.as<geometry::SimplicialComplex>();

                auto geo_ids_view = sc->edges().find<Vector2i>("geo_ids")->view();
                auto inst_ids_view = sc->edges().find<Vector2i>("inst_ids")->view();
                auto strength_ratio_view =
                    sc->edges().find<Float>("strength_ratio")->view();

                auto Es = sc->edges().topo().view();
                auto Ps = sc->positions().view();
                for(auto&& [i, e] : enumerate(Es))
                {
                    Vector2i geo_id  = geo_ids_view[i];
                    Vector2i inst_id = inst_ids_view[i];

                    body_ids_list.push_back({info.body_id(geo_id(0), inst_id(0)),
                                             info.body_id(geo_id(1), inst_id(1))});

                    Transform LT{
                        info.body_geo(geo_slots, geo_id(0))->transforms().view()[inst_id(0)]};
                    Transform RT{
                        info.body_geo(geo_slots, geo_id(1))->transforms().view()[inst_id(1)]};

                    Vector3 mid_point = (Ps[e[0]] + Ps[e[1]]) / 2.0;

                    Vector6 rest_c;
                    rest_c.segment<3>(0) = LT.inverse() * mid_point;
                    rest_c.segment<3>(3) = RT.inverse() * mid_point;
                    rest_c_list.push_back(rest_c);

                    Matrix3x3 LR = LT.rotation();
                    Matrix3x3 RR = RT.rotation();

                    Vector6 rest_t;
                    rest_t.segment<3>(0) = LR.inverse() * Vector3(1, 0, 0);
                    rest_t.segment<3>(3) = RR.inverse() * Vector3(1, 0, 0);
                    rest_t_list.push_back(rest_t);

                    Vector6 rest_n;
                    rest_n.segment<3>(0) = LR.inverse() * Vector3(0, 1, 0);
                    rest_n.segment<3>(3) = RR.inverse() * Vector3(0, 1, 0);
                    rest_n_list.push_back(rest_n);

                    Vector6 rest_b;
                    rest_b.segment<3>(0) = LR.inverse() * Vector3(0, 0, 1);
                    rest_b.segment<3>(3) = RR.inverse() * Vector3(0, 0, 1);
                    rest_b_list.push_back(rest_b);
                }

                std::ranges::copy(strength_ratio_view,
                                  std::back_inserter(strength_ratio_list));
            });

        auto move_to = [](auto& dst, auto& src)
        {
            dst.resize(src.size());
            std::ranges::move(src, dst.begin());
        };

        move_to(h_body_ids, body_ids_list);
        move_to(h_rest_cs, rest_c_list);
        move_to(h_rest_ts, rest_t_list);
        move_to(h_rest_ns, rest_n_list);
        move_to(h_rest_bs, rest_b_list);
        move_to(h_strength_ratios, strength_ratio_list);

        body_ids.copy_from(h_body_ids);
        rest_cs.copy_from(h_rest_cs);
        rest_ts.copy_from(h_rest_ts);
        rest_ns.copy_from(h_rest_ns);
        rest_bs.copy_from(h_rest_bs);
        strength_ratios.copy_from(h_strength_ratios);
    }

    void do_report_energy_extent(EnergyExtentInfo& info) override
    {
        info.energy_count(body_ids.size());
    }

    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace FJ = sym::affine_body_fixed_joint;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(body_ids.size(),
                   [body_ids = body_ids.cviewer().name("body_ids"),
                    rest_cs  = rest_cs.cviewer().name("rest_cs"),
                    rest_ts  = rest_ts.cviewer().name("rest_ts"),
                    rest_ns  = rest_ns.cviewer().name("rest_ns"),
                    rest_bs  = rest_bs.cviewer().name("rest_bs"),
                    strength_ratio = strength_ratios.cviewer().name("strength_ratio"),
                    body_masses = info.body_masses().cviewer().name("body_masses"),
                    qs = info.qs().viewer().name("qs"),
                    Es = info.energies().viewer().name("Es")] __device__(int I)
                   {
                       using Alu = ActivePolicy::AluScalar;

                       Vector2i bids      = body_ids(I);
                       Alu      kappa_alu = safe_cast<Alu>(strength_ratio(I))
                                      * safe_cast<Alu>(body_masses(bids(0)).mass()
                                                       + body_masses(bids(1)).mass());

                       Eigen::Matrix<Alu, 12, 1> qi_alu = qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> qj_alu = qs(bids(1)).template cast<Alu>();

                       auto& rest_c = rest_cs(I);
                       auto& rest_t = rest_ts(I);
                       auto& rest_n = rest_ns(I);
                       auto& rest_b = rest_bs(I);

                       Eigen::Vector<Alu, 9> Fr_val;
                       FJ::Fr<Alu>(Fr_val,
                                   rest_t.template segment<3>(0).template cast<Alu>(),
                                   rest_n.template segment<3>(0).template cast<Alu>(),
                                   rest_b.template segment<3>(0).template cast<Alu>(),
                                   qi_alu,
                                   rest_t.template segment<3>(3).template cast<Alu>(),
                                   rest_n.template segment<3>(3).template cast<Alu>(),
                                   rest_b.template segment<3>(3).template cast<Alu>(),
                                   qj_alu);
                       Alu Er_val;
                       FJ::Er<Alu>(Er_val, kappa_alu, Fr_val);

                       Eigen::Vector<Alu, 3> Ft_val;
                       FJ::Ft<Alu>(Ft_val,
                                   rest_c.template segment<3>(0).template cast<Alu>(),
                                   qi_alu,
                                   rest_c.template segment<3>(3).template cast<Alu>(),
                                   qj_alu);
                       Alu Et_val;
                       FJ::Et<Alu>(Et_val, kappa_alu, Ft_val);

                       Es(I) = safe_cast<ActivePolicy::EnergyScalar>(Er_val + Et_val);
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
        namespace FJ       = sym::affine_body_fixed_joint;
        auto gradient_only = info.gradient_only();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(body_ids.size(),
                   [body_ids = body_ids.cviewer().name("body_ids"),
                    rest_cs  = rest_cs.cviewer().name("rest_cs"),
                    rest_ts  = rest_ts.cviewer().name("rest_ts"),
                    rest_ns  = rest_ns.cviewer().name("rest_ns"),
                    rest_bs  = rest_bs.cviewer().name("rest_bs"),
                    strength_ratio = strength_ratios.cviewer().name("strength_ratio"),
                    body_masses = info.body_masses().cviewer().name("body_masses"),
                    qs      = info.qs().viewer().name("qs"),
                    sink    = info.sink(),
                    gradient_only] __device__(int I) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;

                       Vector2i bids      = body_ids(I);
                       Alu      kappa_alu = safe_cast<Alu>(strength_ratio(I))
                                      * safe_cast<Alu>(body_masses(bids(0)).mass()
                                                       + body_masses(bids(1)).mass());

                       Eigen::Matrix<Alu, 12, 1> qi_alu = qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> qj_alu = qs(bids(1)).template cast<Alu>();

                       auto& rest_c = rest_cs(I);
                       auto& rest_t = rest_ts(I);
                       auto& rest_n = rest_ns(I);
                       auto& rest_b = rest_bs(I);

                       Eigen::Vector<Alu, 3> ti0 = rest_t.template segment<3>(0).template cast<Alu>();
                       Eigen::Vector<Alu, 3> ni0 = rest_n.template segment<3>(0).template cast<Alu>();
                       Eigen::Vector<Alu, 3> bi0 = rest_b.template segment<3>(0).template cast<Alu>();
                       Eigen::Vector<Alu, 3> tj0 = rest_t.template segment<3>(3).template cast<Alu>();
                       Eigen::Vector<Alu, 3> nj0 = rest_n.template segment<3>(3).template cast<Alu>();
                       Eigen::Vector<Alu, 3> bj0 = rest_b.template segment<3>(3).template cast<Alu>();
                       Eigen::Vector<Alu, 3> ci0 = rest_c.template segment<3>(0).template cast<Alu>();
                       Eigen::Vector<Alu, 3> cj0 = rest_c.template segment<3>(3).template cast<Alu>();

                       // Rotation gradient
                       Eigen::Vector<Alu, 9> Fr_val;
                       FJ::Fr<Alu>(Fr_val, ti0, ni0, bi0, qi_alu, tj0, nj0, bj0, qj_alu);

                       Eigen::Vector<Alu, 9> dEdFr_val;
                       FJ::dEdFr<Alu>(dEdFr_val, kappa_alu, Fr_val);

                       Eigen::Vector<Alu, 24> JrT_Gr_val;
                       FJ::JrT_Gr<Alu>(JrT_Gr_val, dEdFr_val, ti0, ni0, bi0, qi_alu, tj0, nj0, bj0, qj_alu);

                       // Translation gradient
                       Eigen::Vector<Alu, 3> Ft_val;
                       FJ::Ft<Alu>(Ft_val, ci0, qi_alu, cj0, qj_alu);

                       Eigen::Vector<Alu, 3> dEdFt_val;
                       FJ::dEdFt<Alu>(dEdFt_val, kappa_alu, Ft_val);

                       Eigen::Vector<Alu, 24> JtT_Gt_val;
                       FJ::JtT_Gt<Alu>(JtT_Gt_val, dEdFt_val, ci0, cj0);

                       Vector24               G = downcast_gradient<Store>(JrT_Gr_val + JtT_Gt_val);
                       Vector2i               indices = {bids(0), bids(1)};
                       sink.template write_gradient<2>(2 * I, indices, G);

                       if(!gradient_only)
                       {
                           Eigen::Matrix<Alu, 9, 9> ddEdFr_val;
                           FJ::ddEdFr<Alu>(ddEdFr_val, kappa_alu, Fr_val);
                           make_spd(ddEdFr_val);

                           Eigen::Matrix<Alu, 24, 24> JrT_Hr_Jr_val;
                           FJ::JrT_Hr_Jr<Alu>(JrT_Hr_Jr_val, ddEdFr_val,
                                              ti0, ni0, bi0, qi_alu, tj0, nj0, bj0, qj_alu);

                           Eigen::Matrix<Alu, 3, 3> ddEdFt_val;
                           FJ::ddEdFt<Alu>(ddEdFt_val, kappa_alu, Ft_val);
                           make_spd(ddEdFt_val);

                           Eigen::Matrix<Alu, 24, 24> JtT_Ht_Jt_val;
                           FJ::JtT_Ht_Jt<Alu>(JtT_Ht_Jt_val, ddEdFt_val, ci0, cj0);

                           Matrix24x24            H = downcast_hessian<Store>(JrT_Hr_Jr_val + JtT_Ht_Jt_val);
                           sink.template write_hessian_half<2>(
                               HalfHessianSize * I,
                               indices,
                               H);
                       }
                   });
    };

    U64 get_uid() const noexcept override { return ConstitutionUID; }
};
REGISTER_SIM_SYSTEM(AffineBodyFixedJoint);
}  // namespace uipc::backend::cuda_mixed
