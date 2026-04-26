#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_normal_contact_structured(
    SimplexNormalContact::ContactInfo& info)
{
    using namespace muda;
    using namespace sym::codim_ipc_simplex_contact;
    using Alu    = ActivePolicy::AluScalar;
    using Vec3A  = Eigen::Matrix<Alu, 3, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;
    using Vec9A  = Eigen::Matrix<Alu, 9, 1>;
    using Mat9A  = Eigen::Matrix<Alu, 9, 9>;
    using Vec6A  = Eigen::Matrix<Alu, 6, 1>;
    using Mat6A  = Eigen::Matrix<Alu, 6, 6>;

    const auto structured_sink = info.structured_hessian_sink();

    if(info.PTs().size())
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.PTs().size(),
                   [structured_sink,
                    table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs = info.PTs().viewer().name("PTs"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hats = info.d_hats().viewer().name("d_hats"),
                    dt     = info.dt()] __device__(int i) mutable
                   {
                       Vector4i PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};
                       Alu kt2 = safe_cast<Alu>(PT_kappa(table, cids) * dt * dt);

                       const Vector3& P_f  = Ps(PT[0]);
                       const Vector3& T0_f = Ps(PT[1]);
                       const Vector3& T1_f = Ps(PT[2]);
                       const Vector3& T2_f = Ps(PT[3]);
                       Vec3A         P     = P_f.template cast<Alu>();
                       Vec3A         T0    = T0_f.template cast<Alu>();
                       Vec3A         T1    = T1_f.template cast<Alu>();
                       Vec3A         T2    = T2_f.template cast<Alu>();

                       Alu thickness = safe_cast<Alu>(PT_thickness(thicknesses(PT(0)),
                                                                   thicknesses(PT(1)),
                                                                   thicknesses(PT(2)),
                                                                   thicknesses(PT(3))));
                       Alu d_hat = safe_cast<Alu>(PT_d_hat(
                           d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3))));

                       Vector4i flag =
                           distance::point_triangle_distance_flag(P_f, T0_f, T1_f, T2_f);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_triangle_distance2(flag, P_f, T0_f, T1_f, T2_f, D);
                           Vector2 range =
                               D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                           MUDA_ASSERT(is_active_D(range, D),
                                       "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PT(0),
                                       PT(1),
                                       PT(2),
                                       PT(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Vec12A G;
                       Mat12A H;
                       PT_barrier_gradient_hessian(
                           G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                       make_spd(H);
                       structured_sink.template write_hessian_half<4>(PT, H);
                   });
    }

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.EEs().size(),
               [structured_sink,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                EEs         = info.EEs().viewer().name("EEs"),
                Ps          = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                rest_Ps     = info.rest_positions().viewer().name("rest_Ps"),
                d_hats      = info.d_hats().viewer().name("d_hats"),
                dt          = info.dt()] __device__(int i) mutable
               {
                   Vector4i EE = EEs(i);

                   Vector4i cids = {contact_ids(EE[0]),
                                    contact_ids(EE[1]),
                                    contact_ids(EE[2]),
                                    contact_ids(EE[3])};
                   Alu kt2 = safe_cast<Alu>(EE_kappa(table, cids) * dt * dt);

                   const Vector3& E0_f = Ps(EE[0]);
                   const Vector3& E1_f = Ps(EE[1]);
                   const Vector3& E2_f = Ps(EE[2]);
                   const Vector3& E3_f = Ps(EE[3]);
                   Vec3A         E0    = E0_f.template cast<Alu>();
                   Vec3A         E1    = E1_f.template cast<Alu>();
                   Vec3A         E2    = E2_f.template cast<Alu>();
                   Vec3A         E3    = E3_f.template cast<Alu>();

                   const Vector3& t0_Ea0_f = rest_Ps(EE[0]);
                   const Vector3& t0_Ea1_f = rest_Ps(EE[1]);
                   const Vector3& t0_Eb0_f = rest_Ps(EE[2]);
                   const Vector3& t0_Eb1_f = rest_Ps(EE[3]);
                   Vec3A         t0_Ea0    = t0_Ea0_f.template cast<Alu>();
                   Vec3A         t0_Ea1    = t0_Ea1_f.template cast<Alu>();
                   Vec3A         t0_Eb0    = t0_Eb0_f.template cast<Alu>();
                   Vec3A         t0_Eb1    = t0_Eb1_f.template cast<Alu>();

                   Alu thickness = safe_cast<Alu>(EE_thickness(thicknesses(EE(0)),
                                                               thicknesses(EE(1)),
                                                               thicknesses(EE(2)),
                                                               thicknesses(EE(3))));
                   Alu d_hat = safe_cast<Alu>(EE_d_hat(
                       d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3))));

                   Vector4i flag = distance::edge_edge_distance_flag(E0_f, E1_f, E2_f, E3_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::edge_edge_distance2(flag, E0_f, E1_f, E2_f, E3_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                   EE(0),
                                   EE(1),
                                   EE(2),
                                   EE(3),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec12A G;
                   Mat12A H;
                   mollified_EE_barrier_gradient_hessian(G,
                                                         H,
                                                         flag,
                                                         kt2,
                                                         d_hat,
                                                         thickness,
                                                         t0_Ea0,
                                                         t0_Ea1,
                                                         t0_Eb0,
                                                         t0_Eb1,
                                                         E0,
                                                         E1,
                                                         E2,
                                                         E3);
                   make_spd(H);
                   structured_sink.template write_hessian_half<4>(EE, H);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PEs().size(),
               [structured_sink,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PEs     = info.PEs().viewer().name("PEs"),
                Ps      = info.positions().viewer().name("Ps"),
                rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                dt     = info.dt()] __device__(int i) mutable
               {
                   Vector3i PE = PEs(i);

                   Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
                   Alu kt2 = safe_cast<Alu>(PE_kappa(table, cids) * dt * dt);

                   const Vector3& P_f  = Ps(PE[0]);
                   const Vector3& E0_f = Ps(PE[1]);
                   const Vector3& E1_f = Ps(PE[2]);
                   Vec3A         P     = P_f.template cast<Alu>();
                   Vec3A         E0    = E0_f.template cast<Alu>();
                   Vec3A         E1    = E1_f.template cast<Alu>();

                   Alu thickness = safe_cast<Alu>(PE_thickness(
                       thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2))));
                   Alu d_hat = safe_cast<Alu>(
                       PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2))));

                   Vector3i flag = distance::point_edge_distance_flag(P_f, E0_f, E1_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::point_edge_distance2(flag, P_f, E0_f, E1_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                   PE(0),
                                   PE(1),
                                   PE(2),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec9A G;
                   Mat9A H;
                   PE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                   make_spd(H);
                   structured_sink.template write_hessian_half<3>(PE, H);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.PPs().size(),
               [structured_sink,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PPs = info.PPs().viewer().name("PPs"),
                Ps  = info.positions().viewer().name("Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const auto& PP = PPs(i);

                   Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                   Alu      kt2 = safe_cast<Alu>(PP_kappa(table, cids) * dt * dt);

                   const Vector3& P0_f = Ps(PP[0]);
                   const Vector3& P1_f = Ps(PP[1]);
                   Vec3A         P0    = P0_f.template cast<Alu>();
                   Vec3A         P1    = P1_f.template cast<Alu>();

                   Alu thickness =
                       safe_cast<Alu>(PP_thickness(thicknesses(PP(0)), thicknesses(PP(1))));
                   Alu d_hat = safe_cast<Alu>(PP_d_hat(d_hats(PP(0)), d_hats(PP(1))));

                   Vector2i flag = distance::point_point_distance_flag(P0_f, P1_f);

                   if constexpr(RUNTIME_CHECK)
                   {
                       Float D;
                       distance::point_point_distance2(flag, P0_f, P1_f, D);
                       Vector2 range =
                           D_range(safe_cast<Float>(thickness), safe_cast<Float>(d_hat));
                       MUDA_ASSERT(is_active_D(range, D),
                                   "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                   PP(0),
                                   PP(1),
                                   D,
                                   range(0),
                                   range(1));
                   }

                   Vec6A G;
                   Mat6A H;
                   PP_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P0, P1);
                   make_spd(H);
                   structured_sink.template write_hessian_half<2>(PP, H);
               });
}
}  // namespace uipc::backend::cuda_mixed
