#include <contact_system/contact_models/ipc_simplex_frictional_contact_native.h>
#include <linear_system/socu_native_contact_writer.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>

#include <cmath>

namespace uipc::backend::cuda_mixed
{
namespace
{
template <typename T>
__device__ __forceinline__ T dot3(const Eigen::Matrix<T, 3, 1>& a,
                                  const Eigen::Matrix<T, 3, 1>& b)
{
    return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1> cross3(
    const Eigen::Matrix<T, 3, 1>& a,
    const Eigen::Matrix<T, 3, 1>& b)
{
    Eigen::Matrix<T, 3, 1> out;
    out(0) = a(1) * b(2) - a(2) * b(1);
    out(1) = a(2) * b(0) - a(0) * b(2);
    out(2) = a(0) * b(1) - a(1) * b(0);
    return out;
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1> normalized3(
    const Eigen::Matrix<T, 3, 1>& v)
{
    const T inv_norm = T{1} / std::sqrt(dot3(v, v));
    Eigen::Matrix<T, 3, 1> out;
    out(0) = v(0) * inv_norm;
    out(1) = v(1) * inv_norm;
    out(2) = v(2) * inv_norm;
    return out;
}

template <typename T>
__device__ __forceinline__ T pp_normal_force(T kappa,
                                             T d_hat,
                                             T thickness,
                                             T D)
{
    const T xi = thickness;
    const T x0 = xi * xi;
    const T x1 = D - x0;
    const T x2 = d_hat * d_hat;
    const T x3 = d_hat * xi;
    const T x4 = x2 + T{2} * x3;
    const T dBdD =
        -kappa * (T{2} * D - T{2} * x0 - T{2} * x2 - T{4} * x3)
            * std::log(x1 / x4)
        - kappa * (D - x0 - x4) * (D - x0 - x4) / x1;
    return -dBdD * T{2} * std::sqrt(D);
}

template <typename T>
__device__ __forceinline__ void project_spd_2x2(T& a, T& b, T& d)
{
    const T trace = a + d;
    const T diff  = a - d;
    const T disc  = std::sqrt(diff * diff + T{4} * b * b);
    const T l_max = T{0.5} * (trace + disc);
    const T l_min = T{0.5} * (trace - disc);
    if(l_min >= T{0})
        return;
    if(l_max <= T{0})
    {
        a = T{0};
        b = T{0};
        d = T{0};
        return;
    }

    T vx = b;
    T vy = l_max - a;
    T n2 = vx * vx + vy * vy;
    if(n2 == T{0})
    {
        vx = l_max - d;
        vy = b;
        n2 = vx * vx + vy * vy;
    }
    const T inv_n = T{1} / std::sqrt(n2);
    vx *= inv_n;
    vy *= inv_n;
    a = l_max * vx * vx;
    b = l_max * vx * vy;
    d = l_max * vy * vy;
}

template <typename T>
__device__ __noinline__ void compute_PP_friction_hessian_block_scalar(
    Eigen::Matrix<T, 3, 3>&       K,
    T                             kappa,
    T                             d_hat,
    T                             thickness,
    T                             mu,
    T                             eps_vh,
    const Eigen::Matrix<T, 3, 1>& prev_P0,
    const Eigen::Matrix<T, 3, 1>& prev_P1,
    const Eigen::Matrix<T, 3, 1>& P0,
    const Eigen::Matrix<T, 3, 1>& P1)
{
    Eigen::Matrix<T, 3, 1> v01 = prev_P1 - prev_P0;
    Eigen::Matrix<T, 3, 1> unit_x;
    Eigen::Matrix<T, 3, 1> unit_y;
    unit_x << T{1}, T{0}, T{0};
    unit_y << T{0}, T{1}, T{0};

    const auto x_cross = cross3(unit_x, v01);
    const auto y_cross = cross3(unit_y, v01);

    Eigen::Matrix<T, 3, 1> b0;
    Eigen::Matrix<T, 3, 1> b1;
    if(dot3(x_cross, x_cross) > dot3(y_cross, y_cross))
    {
        b0 = normalized3(x_cross);
        b1 = normalized3(cross3(v01, x_cross));
    }
    else
    {
        b0 = normalized3(y_cross);
        b1 = normalized3(cross3(v01, y_cross));
    }

    const Eigen::Matrix<T, 3, 1> rel_dx = (P0 - prev_P0) - (P1 - prev_P1);
    const T tan0 = dot3(b0, rel_dx);
    const T tan1 = dot3(b1, rel_dx);

    const Eigen::Matrix<T, 3, 1> prev_diff = prev_P0 - prev_P1;
    const T D = dot3(prev_diff, prev_diff);
    const T f = pp_normal_force(kappa, d_hat, thickness, D);

    const T sq_norm = tan0 * tan0 + tan1 * tan1;
    const T eps2    = eps_vh * eps_vh;
    const T root    = std::sqrt(sq_norm);
    const T f1 = sq_norm >= eps2 ? T{1} / root : (-root + T{2} * eps_vh) / eps2;

    T h00;
    T h01;
    T h11;
    if(sq_norm >= eps2)
    {
        const T coef = mu * f * f1 / sq_norm;
        h00 = coef * tan1 * tan1;
        h01 = -coef * tan0 * tan1;
        h11 = coef * tan0 * tan0;
    }
    else if(sq_norm == T{0})
    {
        h00 = mu * f * f1;
        h01 = T{0};
        h11 = h00;
    }
    else
    {
        const T f2_over_norm = -T{1} / (eps2 * root);
        h00 = f2_over_norm * tan0 * tan0 + f1;
        h01 = f2_over_norm * tan0 * tan1;
        h11 = f2_over_norm * tan1 * tan1 + f1;
        project_spd_2x2(h00, h01, h11);
        h00 *= mu * f;
        h01 *= mu * f;
        h11 *= mu * f;
    }

    for(IndexT r = 0; r < 3; ++r)
    {
        for(IndexT c = 0; c < 3; ++c)
        {
            const T v = b0(r) * (h00 * b0(c) + h01 * b1(c))
                        + b1(r) * (h01 * b0(c) + h11 * b1(c));
            K(r, c) = v;
        }
    }
}

}  // namespace

void assemble_ipc_simplex_frictional_contact_native_exact_PP(
    SimplexFrictionalContact::ContactInfo& info)
{
    using namespace muda;
    using Alu   = ActivePolicy::AluScalar;
    using Store = ActivePolicy::StoreScalar;
    using Solve = ActivePolicy::SolveScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Mat3A = Eigen::Matrix<Alu, 3, 3>;
    using Mat3S = Eigen::Matrix<Store, 3, 3>;

    if(info.friction_PPs().size() == 0)
        return;

    const auto structured_sink = info.structured_hessian_sink();
    const auto targets         = info.friction_PP_native_contact_targets();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(info.friction_PPs().size(),
               [structured_sink,
                targets,
                table = info.contact_tabular().viewer().name("contact_tabular"),
                contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                PPs     = info.friction_PPs().viewer().name("friction_PPs"),
                Ps      = info.positions().viewer().name("Ps"),
                prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                thicknesses = info.thicknesses().viewer().name("thicknesses"),
                d_hats = info.d_hats().viewer().name("d_hats"),
                eps_v  = info.eps_velocity(),
                dt     = info.dt()] __device__(int i) mutable
               {
                   const Vector2i PP = PPs(i);

                   Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                   const auto coeff = table(cids[0], cids[1]);
                   const Alu  kt2   = safe_cast<Alu>(coeff.kappa * dt * dt);
                   const Alu  mu    = safe_cast<Alu>(coeff.mu);

                   const Vec3A prev_P0 = prev_Ps(PP[0]).template cast<Alu>();
                   const Vec3A prev_P1 = prev_Ps(PP[1]).template cast<Alu>();
                   const Vec3A P0      = Ps(PP[0]).template cast<Alu>();
                   const Vec3A P1      = Ps(PP[1]).template cast<Alu>();

                   const Alu thickness = safe_cast<Alu>(
                       thicknesses(PP[0]) + thicknesses(PP[1]));
                   const Alu d_hat = safe_cast<Alu>(
                       (d_hats(PP[0]) + d_hats(PP[1])) * Float{0.5});

                   Mat3A K;
                   compute_PP_friction_hessian_block_scalar(
                       K,
                       kt2,
                       d_hat,
                       thickness,
                       mu,
                       safe_cast<Alu>(eps_v * dt),
                       prev_P0,
                       prev_P1,
                       P0,
                       P1);

                   constexpr SizeT HalfBlockCount = 3;
                   const SizeT base = static_cast<SizeT>(i) * HalfBlockCount;
                   if(targets.data() == nullptr
                      || base + HalfBlockCount > targets.size())
                       return;

                   SocuNativeContactExactWriter<Store, Solve> writer{
                       structured_sink.sink,
                       structured_sink.abd_vertex_to_J,
                       structured_sink.counters};

                   for(IndexT target_offset = 0;
                       target_offset < static_cast<IndexT>(HalfBlockCount);
                       ++target_offset)
                   {
                       const auto target =
                           targets.data()[base + static_cast<SizeT>(target_offset)];
                       if(target.write_mode == SocuNativeContactWriteMode::Skipped)
                           continue;
                       if(!target.exact_in_band())
                           continue;

                       const Store sign =
                           target.local_row_vertex == target.local_col_vertex
                               ? Store{1}
                               : Store{-1};
                       Mat3S H3;
                       for(IndexT r = 0; r < 3; ++r)
                       {
                           for(IndexT c = 0; c < 3; ++c)
                           {
                               H3(r, c) = sign * safe_cast<Store>(K(r, c));
                           }
                       }
                       writer.write_half_block(target, H3);
                   }
               });
}
}  // namespace uipc::backend::cuda_mixed
