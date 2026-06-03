#include <muda/ext/eigen/eigen_core_cxx20.h>

#include <app/app.h>
#include <muda/buffer/device_buffer.h>
#include <contact_system/contact_models/codim_ipc_simplex_rcc_adhesive_function.h>

#include <array>
#include <functional>

// Finite-difference E/G/H oracle for the per-feature RCC adhesion device
// functions (Phase 6 Step 1 formulas). Proves the PE (Vector9/9x9) and PP
// (Vector6/6x6) normal + tangential adhesion gradients/Hessians match the
// numerical derivative of the energy, with PT (Vector12/12x12) as a reference
// column. The functions are __device__-only, so each evaluation launches a
// 1-thread kernel and copies the result back.
//
// Notes:
//  - We FD the RAW (un-projected) Hessian (the assembly call site SPD-projects
//    the normal block separately); this matches the analytic coeff*HessD.
//  - Tangential adhesion uses a lagged basis/closest-foot from prev positions;
//    FD perturbs only the CURRENT positions, holding prev fixed, so the basis
//    is constant and the energy is exactly quadratic in the current DOFs.
//  - beta is passed >0 directly (the assembler early-outs on beta<=0).

namespace
{
using namespace uipc;

template <int N>
Float max_rel_err(const Eigen::Vector<Float, N>& a, const Eigen::Vector<Float, N>& b)
{
    Float e = 0.0;
    for(int i = 0; i < N; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max<Float>(1.0, std::abs(b[i])));
    return e;
}

template <int N>
Float max_rel_err(const Eigen::Matrix<Float, N, N>& a, const Eigen::Matrix<Float, N, N>& b)
{
    Float e = 0.0;
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            e = std::max(e, std::abs(a(i, j) - b(i, j)) / std::max<Float>(1.0, std::abs(b(i, j))));
    return e;
}

// Central-difference check of an analytic gradient/Hessian against a scalar
// energy(q) and vector gradient(q), where q is the flattened current DOFs.
template <int N>
void fd_check(const Eigen::Vector<Float, N>&                            q0,
              const std::function<Float(const Eigen::Vector<Float, N>&)>& energy,
              const std::function<Eigen::Vector<Float, N>(const Eigen::Vector<Float, N>&)>& gradient,
              const Eigen::Vector<Float, N>&    analytic_G,
              const Eigen::Matrix<Float, N, N>& analytic_H)
{
    constexpr Float        eps = 1e-6;
    Eigen::Vector<Float, N>    fd_g;
    Eigen::Matrix<Float, N, N> fd_h;
    for(int i = 0; i < N; ++i)
    {
        Eigen::Vector<Float, N> qp = q0, qm = q0;
        qp[i] += eps;
        qm[i] -= eps;
        fd_g[i]      = (energy(qp) - energy(qm)) / (2.0 * eps);
        fd_h.col(i)  = (gradient(qp) - gradient(qm)) / (2.0 * eps);
    }
    CHECK(max_rel_err<N>(analytic_G, fd_g) < 1e-6);
    CHECK(max_rel_err<N>(analytic_H, fd_h) < 1e-5);
}

// ---- GPU launch wrappers (one 1-thread kernel per evaluation) ----

namespace sym_rcc = uipc::backend::cuda::sym::codim_ipc_rcc_adhesive;

// PE normal: stencil (P, E0, E1) -> Vector9 / Matrix9x9.
Float launch_PE_normal_E(const std::array<Vector3, 3>& X, Float Cn, Float beta, Float d_hat, Float dt)
{
    using namespace muda;
    DeviceBuffer<Vector3> dX(3);
    dX.view().copy_from(X.data());
    DeviceBuffer<Float> out(1);
    ParallelFor().kernel_name("PE_normal_E").apply(1, [X = dX.cviewer(), out = out.viewer(), Cn, beta, d_hat, dt] __device__(int) mutable
                                                   { out(0) = sym_rcc::PE_normal_adhesion_energy(Cn, beta, d_hat, dt, X(0), X(1), X(2)); });
    Float h;
    out.view().copy_to(&h);
    return h;
}
void launch_PE_normal_GH(const std::array<Vector3, 3>& X, Float Cn, Float beta, Float d_hat, Float dt, Vector9& G, Matrix9x9& H)
{
    using namespace muda;
    DeviceBuffer<Vector3>   dX(3);
    dX.view().copy_from(X.data());
    DeviceBuffer<Vector9>   dG(1);
    DeviceBuffer<Matrix9x9> dH(1);
    ParallelFor().kernel_name("PE_normal_GH").apply(1, [X = dX.cviewer(), dG = dG.viewer(), dH = dH.viewer(), Cn, beta, d_hat, dt] __device__(int) mutable
                                                    {
                                                        Vector9   g;
                                                        Matrix9x9 h;
                                                        sym_rcc::PE_normal_adhesion_gradient_hessian(g, h, Cn, beta, d_hat, dt, X(0), X(1), X(2));
                                                        dG(0) = g;
                                                        dH(0) = h; });
    dG.view().copy_to(&G);
    dH.view().copy_to(&H);
}

// PE tangential: lagged prev stencil + current stencil -> Vector9 / Matrix9x9.
Float launch_PE_tan_E(const std::array<Vector3, 3>& prev, const std::array<Vector3, 3>& X, Float Ct, Float beta, Float d_hat, Float dt)
{
    using namespace muda;
    DeviceBuffer<Vector3> dP(3), dX(3);
    dP.view().copy_from(prev.data());
    dX.view().copy_from(X.data());
    DeviceBuffer<Float> out(1);
    ParallelFor().kernel_name("PE_tan_E").apply(1, [P = dP.cviewer(), X = dX.cviewer(), out = out.viewer(), Ct, beta, d_hat, dt] __device__(int) mutable
                                                { out(0) = sym_rcc::PE_tangential_adhesion_energy(Ct, beta, d_hat, dt, P(0), P(1), P(2), X(0), X(1), X(2)); });
    Float h;
    out.view().copy_to(&h);
    return h;
}
void launch_PE_tan_GH(const std::array<Vector3, 3>& prev, const std::array<Vector3, 3>& X, Float Ct, Float beta, Float d_hat, Float dt, Vector9& G, Matrix9x9& H)
{
    using namespace muda;
    DeviceBuffer<Vector3>   dP(3), dX(3);
    dP.view().copy_from(prev.data());
    dX.view().copy_from(X.data());
    DeviceBuffer<Vector9>   dG(1);
    DeviceBuffer<Matrix9x9> dH(1);
    ParallelFor().kernel_name("PE_tan_GH").apply(1, [P = dP.cviewer(), X = dX.cviewer(), dG = dG.viewer(), dH = dH.viewer(), Ct, beta, d_hat, dt] __device__(int) mutable
                                                 {
                                                     Vector9   g;
                                                     Matrix9x9 h;
                                                     sym_rcc::PE_tangential_adhesion_gradient_hessian(g, h, Ct, beta, d_hat, dt, P(0), P(1), P(2), X(0), X(1), X(2));
                                                     dG(0) = g;
                                                     dH(0) = h; });
    dG.view().copy_to(&G);
    dH.view().copy_to(&H);
}

// PP normal: stencil (P, Q) -> Vector6 / Matrix6x6.
Float launch_PP_normal_E(const std::array<Vector3, 2>& X, Float Cn, Float beta, Float d_hat, Float dt)
{
    using namespace muda;
    DeviceBuffer<Vector3> dX(2);
    dX.view().copy_from(X.data());
    DeviceBuffer<Float> out(1);
    ParallelFor().kernel_name("PP_normal_E").apply(1, [X = dX.cviewer(), out = out.viewer(), Cn, beta, d_hat, dt] __device__(int) mutable
                                                   { out(0) = sym_rcc::PP_normal_adhesion_energy(Cn, beta, d_hat, dt, X(0), X(1)); });
    Float h;
    out.view().copy_to(&h);
    return h;
}
void launch_PP_normal_GH(const std::array<Vector3, 2>& X, Float Cn, Float beta, Float d_hat, Float dt, Vector6& G, Matrix6x6& H)
{
    using namespace muda;
    DeviceBuffer<Vector3>   dX(2);
    dX.view().copy_from(X.data());
    DeviceBuffer<Vector6>   dG(1);
    DeviceBuffer<Matrix6x6> dH(1);
    ParallelFor().kernel_name("PP_normal_GH").apply(1, [X = dX.cviewer(), dG = dG.viewer(), dH = dH.viewer(), Cn, beta, d_hat, dt] __device__(int) mutable
                                                    {
                                                        Vector6   g;
                                                        Matrix6x6 h;
                                                        sym_rcc::PP_normal_adhesion_gradient_hessian(g, h, Cn, beta, d_hat, dt, X(0), X(1));
                                                        dG(0) = g;
                                                        dH(0) = h; });
    dG.view().copy_to(&G);
    dH.view().copy_to(&H);
}

// PP tangential: lagged prev (P0,P1) + current (P0,P1) -> Vector6 / Matrix6x6.
Float launch_PP_tan_E(const std::array<Vector3, 2>& prev, const std::array<Vector3, 2>& X, Float Ct, Float beta, Float d_hat, Float dt)
{
    using namespace muda;
    DeviceBuffer<Vector3> dP(2), dX(2);
    dP.view().copy_from(prev.data());
    dX.view().copy_from(X.data());
    DeviceBuffer<Float> out(1);
    ParallelFor().kernel_name("PP_tan_E").apply(1, [P = dP.cviewer(), X = dX.cviewer(), out = out.viewer(), Ct, beta, d_hat, dt] __device__(int) mutable
                                                { out(0) = sym_rcc::PP_tangential_adhesion_energy(Ct, beta, d_hat, dt, P(0), P(1), X(0), X(1)); });
    Float h;
    out.view().copy_to(&h);
    return h;
}
void launch_PP_tan_GH(const std::array<Vector3, 2>& prev, const std::array<Vector3, 2>& X, Float Ct, Float beta, Float d_hat, Float dt, Vector6& G, Matrix6x6& H)
{
    using namespace muda;
    DeviceBuffer<Vector3>   dP(2), dX(2);
    dP.view().copy_from(prev.data());
    dX.view().copy_from(X.data());
    DeviceBuffer<Vector6>   dG(1);
    DeviceBuffer<Matrix6x6> dH(1);
    ParallelFor().kernel_name("PP_tan_GH").apply(1, [P = dP.cviewer(), X = dX.cviewer(), dG = dG.viewer(), dH = dH.viewer(), Ct, beta, d_hat, dt] __device__(int) mutable
                                                 {
                                                     Vector6   g;
                                                     Matrix6x6 h;
                                                     sym_rcc::PP_tangential_adhesion_gradient_hessian(g, h, Ct, beta, d_hat, dt, P(0), P(1), X(0), X(1));
                                                     dG(0) = g;
                                                     dH(0) = h; });
    dG.view().copy_to(&G);
    dH.view().copy_to(&H);
}

// helpers to pack/unpack flattened DOF vectors <-> stencil arrays
template <int K>
std::array<Vector3, K> unpack(const Eigen::Vector<Float, 3 * K>& q)
{
    std::array<Vector3, K> X;
    for(int k = 0; k < K; ++k)
        X[k] = q.template segment<3>(3 * k);
    return X;
}
template <int K>
Eigen::Vector<Float, 3 * K> pack(const std::array<Vector3, K>& X)
{
    Eigen::Vector<Float, 3 * K> q;
    for(int k = 0; k < K; ++k)
        q.template segment<3>(3 * k) = X[k];
    return q;
}
}  // namespace

TEST_CASE("rcc_adhesion_PE_normal_egh_matches_finite_difference",
          "[rcc_adhesion][oracle][feature_adhesion][cuda]")
{
    using namespace uipc;
    // point genuinely closest to the edge interior (foot at x=0.4 in (0,1))
    std::array<Vector3, 3> X = {Vector3{0.4, 0.12, 0.06}, Vector3{0.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0}};
    const Float Cn = 1.0e3, beta = 0.7, d_hat = 1.0e-2, dt = 1.0e-2;

    Vector9   G;
    Matrix9x9 H;
    launch_PE_normal_GH(X, Cn, beta, d_hat, dt, G, H);
    CHECK(launch_PE_normal_E(X, Cn, beta, d_hat, dt) > 0.0);

    fd_check<9>(
        pack<3>(X),
        [&](const Eigen::Vector<Float, 9>& q) { return launch_PE_normal_E(unpack<3>(q), Cn, beta, d_hat, dt); },
        [&](const Eigen::Vector<Float, 9>& q) { Vector9 g; Matrix9x9 h; launch_PE_normal_GH(unpack<3>(q), Cn, beta, d_hat, dt, g, h); return g; },
        G, H);
}

TEST_CASE("rcc_adhesion_PE_tangential_egh_matches_finite_difference",
          "[rcc_adhesion][oracle][feature_adhesion][cuda]")
{
    using namespace uipc;
    std::array<Vector3, 3> prev = {Vector3{0.4, 0.12, 0.06}, Vector3{0.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0}};
    // current = prev + small tangential motion (so |u| != 0)
    std::array<Vector3, 3> X = {Vector3{0.43, 0.125, 0.061}, Vector3{0.004, -0.002, 0.0}, Vector3{1.003, 0.001, -0.002}};
    const Float Ct = 1.0e3, beta = 0.7, d_hat = 1.0e-2, dt = 1.0e-2;

    Vector9   G;
    Matrix9x9 H;
    launch_PE_tan_GH(prev, X, Ct, beta, d_hat, dt, G, H);
    CHECK(launch_PE_tan_E(prev, X, Ct, beta, d_hat, dt) > 0.0);

    fd_check<9>(
        pack<3>(X),
        [&](const Eigen::Vector<Float, 9>& q) { return launch_PE_tan_E(prev, unpack<3>(q), Ct, beta, d_hat, dt); },
        [&](const Eigen::Vector<Float, 9>& q) { Vector9 g; Matrix9x9 h; launch_PE_tan_GH(prev, unpack<3>(q), Ct, beta, d_hat, dt, g, h); return g; },
        G, H);
}

TEST_CASE("rcc_adhesion_PP_normal_egh_matches_finite_difference",
          "[rcc_adhesion][oracle][feature_adhesion][cuda]")
{
    using namespace uipc;
    std::array<Vector3, 2> X = {Vector3{-0.1, 0.05, 0.08}, Vector3{0.0, 0.0, 0.0}};
    const Float Cn = 1.0e3, beta = 0.7, d_hat = 1.0e-2, dt = 1.0e-2;

    Vector6   G;
    Matrix6x6 H;
    launch_PP_normal_GH(X, Cn, beta, d_hat, dt, G, H);
    CHECK(launch_PP_normal_E(X, Cn, beta, d_hat, dt) > 0.0);

    fd_check<6>(
        pack<2>(X),
        [&](const Eigen::Vector<Float, 6>& q) { return launch_PP_normal_E(unpack<2>(q), Cn, beta, d_hat, dt); },
        [&](const Eigen::Vector<Float, 6>& q) { Vector6 g; Matrix6x6 h; launch_PP_normal_GH(unpack<2>(q), Cn, beta, d_hat, dt, g, h); return g; },
        G, H);
}

TEST_CASE("rcc_adhesion_PP_tangential_egh_matches_finite_difference",
          "[rcc_adhesion][oracle][feature_adhesion][cuda]")
{
    using namespace uipc;
    std::array<Vector3, 2> prev = {Vector3{-0.1, 0.05, 0.08}, Vector3{0.0, 0.0, 0.0}};
    std::array<Vector3, 2> X = {Vector3{-0.094, 0.053, 0.079}, Vector3{0.003, -0.001, 0.002}};
    const Float Ct = 1.0e3, beta = 0.7, d_hat = 1.0e-2, dt = 1.0e-2;

    Vector6   G;
    Matrix6x6 H;
    launch_PP_tan_GH(prev, X, Ct, beta, d_hat, dt, G, H);
    CHECK(launch_PP_tan_E(prev, X, Ct, beta, d_hat, dt) > 0.0);

    fd_check<6>(
        pack<2>(X),
        [&](const Eigen::Vector<Float, 6>& q) { return launch_PP_tan_E(prev, unpack<2>(q), Ct, beta, d_hat, dt); },
        [&](const Eigen::Vector<Float, 6>& q) { Vector6 g; Matrix6x6 h; launch_PP_tan_GH(prev, unpack<2>(q), Ct, beta, d_hat, dt, g, h); return g; },
        G, H);
}
