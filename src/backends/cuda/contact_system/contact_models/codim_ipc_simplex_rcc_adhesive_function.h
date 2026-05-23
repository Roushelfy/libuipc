#pragma once
#include <type_define.h>
#include <contact_system/rcc_adhesive_coeff.h>
#include <utils/distance/distance_flagged.h>
#include <utils/friction_utils.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>

namespace uipc::backend::cuda
{
namespace sym::codim_ipc_rcc_adhesive
{
    // ----------------------------------------------------------------------
    // β evolution helpers (v2). Faithful port of XBow's RCCAdhesionEnergy3D
    // constructor lines 723-770 and 820-841, including all adaptive scaling
    // factors (Cn/dHat, r_scale, W_scale, η·W_scale/10). See
    // docs/specification/contact_models/rcc_adhesion.md "Backend implementation
    // notes" for the deviation table.
    //
    // The barrier-gradient `db_dd2` here is ∂(κ·b)/∂D, returned by libuipc's
    // dKappaBarrierdD(R, κ, D, dHat, ξ) — same as XBow's
    // Math::barrier_gradient(d²−ξ², activeGap², κ).
    //
    // We use ξ = 0 throughout (libuipc's IPC barrier has no thickness offset).
    // ----------------------------------------------------------------------

    inline __device__ Float _pk_and_scales(/*out*/ Float& r_scale,
                                           /*out*/ Float& W_scale,
                                           /*out*/ Float& db_dd2_out,
                                           Float          beta,
                                           Float          kappa,
                                           Float          Cn,
                                           Float          dHat,
                                           Float          D)
    {
        constexpr Float xi  = 0.0;
        constexpr Float xi2 = 0.0;
        Float activeGap2 = dHat * dHat + 2.0 * xi * dHat;

        Float db_dd2 = 0.0;
        codim_ipc_contact::dKappaBarrierdD<Float>(db_dd2, kappa, D, dHat, xi);
        db_dd2_out = db_dd2;

        Float d = (D > 0) ? sqrt(D) : Float{0};
        Float p_k = (-Cn / dHat * beta * beta - dHat * db_dd2 * 2.0) * d;

        // r_scale: 100 / pb_base where pb_base is the barrier pressure at d_base = ξ + dHat/2.
        Float d_base   = xi + 0.5 * dHat;
        Float D_base   = d_base * d_base;
        Float db_base  = 0.0;
        codim_ipc_contact::dKappaBarrierdD<Float>(db_base, kappa, D_base, dHat, xi);
        Float pb_base  = -dHat * db_base * 2.0 * d_base;
        // pb_base is negative (barrier gradient is negative for D in active band);
        // r_scale should be positive. XBow writes 100/pb_base; we keep that
        // literally — sign comes out right because p_k_new also carries a
        // negative db_dd2 factor.
        r_scale = 100.0 / pb_base;
        W_scale = Cn / dHat * activeGap2 / 4.0;

        return p_k;
    }

    // Existing-pair β evolution. β_in is the previous step's β; returns the
    // clamped new β.
    inline __device__ Float PT_beta_evolve_existing(Float beta_in,
                                                    Float kappa,
                                                    Float Cn,
                                                    Float Ct,
                                                    Float W,
                                                    Float eta,
                                                    Float r,
                                                    Float p0,
                                                    Float dHat,
                                                    Float dt,
                                                    Float D,
                                                    Float u_sq)
    {
        Float r_scale, W_scale, db_dd2;
        Float p_k = _pk_and_scales(r_scale, W_scale, db_dd2, beta_in, kappa, Cn, dHat, D);

        Float v_beta = 0.0;
        Float denom  = eta * W_scale / 10.0;
        if(p_k < 0.0)
        {
            // adhesion-dominated → debonding
            Float energy_term =
                W * W_scale - Cn / dHat * beta_in * D - Ct / dHat * beta_in * u_sq;
            v_beta = fmin(energy_term, Float{0}) / denom;
        }
        else if(p_k > 0.0)
        {
            // contact-dominated → bonding (with tangential debonding)
            Float bonding_term = r * r_scale * fmax(p_k - beta_in * p0, Float{0});
            Float tan_term = fmin(W * W_scale - Ct / dHat * beta_in * u_sq, Float{0}) / denom;
            v_beta = bonding_term + tan_term;
        }
        Float beta_new = beta_in + dt * v_beta;
        return fmin(Float{1}, fmax(Float{0}, beta_new));
    }

    // New-pair bonding kick (β implicit 0 → single bonding step). Returns the
    // max with `initial_beta` so users can force "start fully bonded" via
    // initial_beta = 1 (matches all v1 demos).
    inline __device__ Float PT_beta_init_new(Float initial_beta,
                                             Float kappa,
                                             Float Cn,
                                             Float r,
                                             Float dHat,
                                             Float dt,
                                             Float D)
    {
        Float r_scale, W_scale, db_dd2;
        // beta=0 ⇒ Cn/dHat·β² = 0 ⇒ p_k = -dHat·db_dd2·2·d
        Float p_k = _pk_and_scales(r_scale, W_scale, db_dd2, Float{0}, kappa, Cn, dHat, D);
        Float v_beta = r * r_scale * p_k;       // p_k > 0 inside the active band → positive bonding
        Float beta_kick = fmin(Float{1}, fmax(Float{0}, dt * v_beta));
        return fmax(beta_kick, initial_beta);
    }

    // -------- PT-pair sorted-vertex U64 hash key --------
    // (p, t0, t1, t2): sort (t0,t1,t2) into ascending order, then mix.
    // Collision probability across ~10⁶ pairs ≈ N²/2^64 ≈ 1e-8.
    inline __device__ __host__ U64 _mix64(U64 x)
    {
        constexpr U64 P2 = 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 30)) * P2;
        x = (x ^ (x >> 27)) * P2;
        return x ^ (x >> 31);
    }

    inline __device__ __host__ U64 PT_pair_key(IndexT p, IndexT t0, IndexT t1, IndexT t2)
    {
        // sort (t0, t1, t2)
        if(t0 > t1) { IndexT tmp = t0; t0 = t1; t1 = tmp; }
        if(t1 > t2) { IndexT tmp = t1; t1 = t2; t2 = tmp; }
        if(t0 > t1) { IndexT tmp = t0; t0 = t1; t1 = tmp; }

        constexpr U64 P1 = 0x9E3779B97F4A7C15ull;
        U64 a = _mix64(U64(static_cast<U32>(p))  * P1);
        U64 b = _mix64(U64(static_cast<U32>(t0)) * P1 + U64(static_cast<U32>(t1)));
        U64 c = _mix64(U64(static_cast<U32>(t2)) * P1);
        return _mix64(a ^ b ^ c);
    }

    // ---- end β evolution helpers ----

    // ---------------------------------------------------------------------
    // Per-(i,j) RCCAdhesiveCoeff aggregation (averages over stencil vertices).
    // Returns the "effective" RCCAdhesiveCoeff for a stencil: Cn, Ct, etc.
    // are averaged across the cross-element pairs the same way ContactCoeff
    // averages kappa/mu in codim_ipc_simplex_*contact_function.h.
    //
    // `enabled` is treated as a boolean AND across all sampled cells.
    // ---------------------------------------------------------------------

    inline __device__ RCCAdhesiveCoeff
    _avg2(const RCCAdhesiveCoeff& a, const RCCAdhesiveCoeff& b)
    {
        RCCAdhesiveCoeff r;
        r.Cn           = 0.5 * (a.Cn + b.Cn);
        r.Ct           = 0.5 * (a.Ct + b.Ct);
        r.W            = 0.5 * (a.W + b.W);
        r.eta          = 0.5 * (a.eta + b.eta);
        r.bonding_rate = 0.5 * (a.bonding_rate + b.bonding_rate);
        r.p0           = 0.5 * (a.p0 + b.p0);
        r.initial_beta = 0.5 * (a.initial_beta + b.initial_beta);
        r.enabled      = (a.enabled && b.enabled) ? IndexT{1} : IndexT{0};
        return r;
    }

    inline __device__ RCCAdhesiveCoeff
    PT_rcc_coeff(const muda::CDense2D<RCCAdhesiveCoeff>& table, const Vector4i& cids)
    {
        RCCAdhesiveCoeff sum;
        IndexT           enabled_all = 1;
        for(int j = 1; j < 4; ++j)
        {
            RCCAdhesiveCoeff c = table(cids[0], cids[j]);
            sum.Cn += c.Cn;
            sum.Ct += c.Ct;
            sum.W += c.W;
            sum.eta += c.eta;
            sum.bonding_rate += c.bonding_rate;
            sum.p0 += c.p0;
            sum.initial_beta += c.initial_beta;
            enabled_all = enabled_all && c.enabled;
        }
        RCCAdhesiveCoeff r;
        r.Cn           = sum.Cn / 3.0;
        r.Ct           = sum.Ct / 3.0;
        r.W            = sum.W / 3.0;
        r.eta          = sum.eta / 3.0;
        r.bonding_rate = sum.bonding_rate / 3.0;
        r.p0           = sum.p0 / 3.0;
        r.initial_beta = sum.initial_beta / 3.0;
        r.enabled      = enabled_all;
        return r;
    }

    inline __device__ RCCAdhesiveCoeff
    EE_rcc_coeff(const muda::CDense2D<RCCAdhesiveCoeff>& table, const Vector4i& cids)
    {
        RCCAdhesiveCoeff sum;
        IndexT           enabled_all = 1;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                RCCAdhesiveCoeff c = table(cids[j], cids[k]);
                sum.Cn += c.Cn;
                sum.Ct += c.Ct;
                sum.W += c.W;
                sum.eta += c.eta;
                sum.bonding_rate += c.bonding_rate;
                sum.p0 += c.p0;
                sum.initial_beta += c.initial_beta;
                enabled_all = enabled_all && c.enabled;
            }
        }
        RCCAdhesiveCoeff r;
        r.Cn           = sum.Cn / 4.0;
        r.Ct           = sum.Ct / 4.0;
        r.W            = sum.W / 4.0;
        r.eta          = sum.eta / 4.0;
        r.bonding_rate = sum.bonding_rate / 4.0;
        r.p0           = sum.p0 / 4.0;
        r.initial_beta = sum.initial_beta / 4.0;
        r.enabled      = enabled_all;
        return r;
    }

    inline __device__ RCCAdhesiveCoeff
    PE_rcc_coeff(const muda::CDense2D<RCCAdhesiveCoeff>& table, const Vector3i& cids)
    {
        RCCAdhesiveCoeff sum;
        IndexT           enabled_all = 1;
        for(int j = 1; j < 3; ++j)
        {
            RCCAdhesiveCoeff c = table(cids[0], cids[j]);
            sum.Cn += c.Cn;
            sum.Ct += c.Ct;
            sum.W += c.W;
            sum.eta += c.eta;
            sum.bonding_rate += c.bonding_rate;
            sum.p0 += c.p0;
            sum.initial_beta += c.initial_beta;
            enabled_all = enabled_all && c.enabled;
        }
        RCCAdhesiveCoeff r;
        r.Cn           = sum.Cn / 2.0;
        r.Ct           = sum.Ct / 2.0;
        r.W            = sum.W / 2.0;
        r.eta          = sum.eta / 2.0;
        r.bonding_rate = sum.bonding_rate / 2.0;
        r.p0           = sum.p0 / 2.0;
        r.initial_beta = sum.initial_beta / 2.0;
        r.enabled      = enabled_all;
        return r;
    }

    inline __device__ RCCAdhesiveCoeff
    PP_rcc_coeff(const muda::CDense2D<RCCAdhesiveCoeff>& table, const Vector2i& cids)
    {
        return table(cids[0], cids[1]);
    }

    // ---------------------------------------------------------------------
    // Normal adhesion (XBow-port):
    //   E       = energy_scale * Cn / (2 * dHat) * beta^2 * D
    //   ∂E/∂D   = energy_scale * Cn / (2 * dHat) * beta^2     (constant in D)
    //   ∇E      = (∂E/∂D) * ∇D
    //   ∇²E     = (∂E/∂D) * ∇²D                                (∂²E/∂D² = 0)
    // We multiply by `dt*dt` as energy_scale (matches kt2 = kappa*dt^2 convention).
    // ---------------------------------------------------------------------

    // PT  (4 verts -> Vector12, Matrix12x12)
    // V1 IMPORTANT: we deliberately use the UNFLAGGED `g_PT` formula here, i.e.
    // the gradient/hessian for "the point projects onto the triangle's interior
    // plane". The flagged dispatch (which routes to point-edge or point-point
    // for points whose perpendicular foot falls outside the triangle) is
    // correct for IPC barrier (repulsion away from the closest sub-feature is
    // direction-consistent) but WRONG for RCC adhesion: it makes nearby points
    // get attracted toward edges/vertices, producing a visible diagonal pull
    // on a faceted cube top face. Using the plane-projection PT gradient keeps
    // the adhesion force perpendicular to the triangle's plane regardless of
    // where the point projects, which matches the spec intent and removes the
    // diagonal artifact.
    inline __device__ Float PT_normal_adhesion_energy(Float          Cn,
                                                      Float          beta,
                                                      Float          d_hat,
                                                      Float          dt,
                                                      const Vector3& P,
                                                      const Vector3& T0,
                                                      const Vector3& T1,
                                                      const Vector3& T2)
    {
        using namespace distance;
        Float    D;
        point_triangle_distance2(P, T0, T1, T2, D);  // unflagged: plane projection
        return (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta * D;
    }

    inline __device__ void
    PT_normal_adhesion_gradient_hessian(Vector12&      G,
                                        Matrix12x12&   H,
                                        Float          Cn,
                                        Float          beta,
                                        Float          d_hat,
                                        Float          dt,
                                        const Vector3& P,
                                        const Vector3& T0,
                                        const Vector3& T1,
                                        const Vector3& T2)
    {
        using namespace distance;
        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);  // plane projection
        Matrix12x12 HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
        H           = coeff * HessD;
    }

    inline __device__ void PT_normal_adhesion_gradient(Vector12&      G,
                                                       Float          Cn,
                                                       Float          beta,
                                                       Float          d_hat,
                                                       Float          dt,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace distance;
        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);  // plane projection

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
    }

    // EE
    // ----- V1 NOTE: EE adhesion is DISABLED. -----
    // The libuipc trajectory filter (and XBow's too) emits PE/PP/EE pairs
    // alongside PT pairs for the same geometric contact when a point/edge
    // projects near a triangle edge or vertex. The flagged feature-based
    // gradient then pulls cloth verts toward those edges/vertices, producing
    // a visible diagonal-bias artifact on faceted meshes (cube top face split
    // by a diagonal, for example). Until we add proper deduplication or a
    // smooth-distance formulation, v1 keeps only the PT contribution, which
    // (with the unflagged plane-projection gradient above) pulls perpendicular
    // to each triangle's plane regardless of where the projection foot lands
    // — direction-consistent and bias-free for face-aligned cloth contact.
    inline __device__ Float EE_normal_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void EE_normal_adhesion_gradient_hessian(
        Vector12&      G,
        Matrix12x12&   H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector12::Zero();
        H = Matrix12x12::Zero();
    }

    inline __device__ void EE_normal_adhesion_gradient(
        Vector12&      G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector12::Zero();
    }

    // PE  (V1 DISABLED — see EE note above)
    inline __device__ Float PE_normal_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void PE_normal_adhesion_gradient_hessian(
        Vector9&       G,
        Matrix9x9&     H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector9::Zero();
        H = Matrix9x9::Zero();
    }

    inline __device__ void PE_normal_adhesion_gradient(
        Vector9&       G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector9::Zero();
    }

    // PP  (V1 DISABLED — see EE note above)
    inline __device__ Float PP_normal_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void PP_normal_adhesion_gradient_hessian(
        Vector6&       G,
        Matrix6x6&     H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&)
    {
        G = Vector6::Zero();
        H = Matrix6x6::Zero();
    }

    inline __device__ void PP_normal_adhesion_gradient(
        Vector6&       G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&)
    {
        G = Vector6::Zero();
    }

    // ---------------------------------------------------------------------
    // Tangential adhesion (XBow-port):
    //   E_tan   = energy_scale * (Ct/(2*dHat)) * beta^2 * |u|^2
    //   u       = basisᵀ · (rel_dx in mesh space, from x - x_t through stencil rel-dx)
    //   ∇E_tan  = J^T · (coeff_tan · u)                  with coeff_tan = Ct/dHat * beta^2 * dt^2
    //   ∇²E_tan = J^T · (coeff_tan · I_2) · J            (always PSD; no SPD projection)
    //
    // The lagged basis & Jacobian J come from the SAME helpers IPC friction uses
    // (see codim_ipc_simplex_frictional_contact_function.h::PT_friction_basis etc.).
    // The functions below DO NOT compute the basis themselves — they consume a
    // pre-computed lagged basis (and barycentric `beta_param` for PT, or `eta_param`
    // for PE) so the same basis can be reused between energy/grad/hess passes
    // within a single Newton iteration.
    //
    // To match friction's invocation pattern and minimize code, we compute the
    // basis fresh per kernel call (same as friction) since the prev positions are
    // already in registers/shared.
    // ---------------------------------------------------------------------

    // PT tangential
    inline __device__ Float PT_tangential_adhesion_energy(Float          Ct,
                                                          Float          beta,
                                                          Float          d_hat,
                                                          Float          dt,
                                                          const Vector3& prev_P,
                                                          const Vector3& prev_T0,
                                                          const Vector3& prev_T1,
                                                          const Vector3& prev_T2,
                                                          const Vector3& P,
                                                          const Vector3& T0,
                                                          const Vector3& T1,
                                                          const Vector3& T2)
    {
        using namespace distance;
        using namespace friction;
        Vector2             beta_param;
        Matrix<Float, 3, 2> basis;
        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta_param);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dT0 = T0 - prev_T0;
        Vector3 dT1 = T1 - prev_T1;
        Vector3 dT2 = T2 - prev_T2;
        Vector2 u;
        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta_param, u);

        Float coeff = (dt * dt) * (Ct / (2.0 * d_hat)) * beta * beta;
        return coeff * u.squaredNorm();
    }

    inline __device__ void
    PT_tangential_adhesion_gradient_hessian(Vector12&      G,
                                            Matrix12x12&   H,
                                            Float          Ct,
                                            Float          beta,
                                            Float          d_hat,
                                            Float          dt,
                                            const Vector3& prev_P,
                                            const Vector3& prev_T0,
                                            const Vector3& prev_T1,
                                            const Vector3& prev_T2,
                                            const Vector3& P,
                                            const Vector3& T0,
                                            const Vector3& T1,
                                            const Vector3& T2)
    {
        using namespace distance;
        using namespace friction;
        Vector2             beta_param;
        Matrix<Float, 3, 2> basis;
        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta_param);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dT0 = T0 - prev_T0;
        Vector3 dT1 = T1 - prev_T1;
        Vector3 dT2 = T2 - prev_T2;
        Vector2 u;
        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta_param, u);

        Matrix<Float, 2, 12> J;
        point_triangle_jacobi(basis, beta_param, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;  // 2·(1/2) cancels in grad
        G           = J.transpose() * (coeff * u);
        H           = (coeff)*J.transpose() * J;
    }

    inline __device__ void
    PT_tangential_adhesion_gradient(Vector12&      G,
                                    Float          Ct,
                                    Float          beta,
                                    Float          d_hat,
                                    Float          dt,
                                    const Vector3& prev_P,
                                    const Vector3& prev_T0,
                                    const Vector3& prev_T1,
                                    const Vector3& prev_T2,
                                    const Vector3& P,
                                    const Vector3& T0,
                                    const Vector3& T1,
                                    const Vector3& T2)
    {
        using namespace distance;
        using namespace friction;
        Vector2             beta_param;
        Matrix<Float, 3, 2> basis;
        point_triangle_closest_point(prev_P, prev_T0, prev_T1, prev_T2, beta_param);
        point_triangle_tangent_basis(prev_P, prev_T0, prev_T1, prev_T2, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dT0 = T0 - prev_T0;
        Vector3 dT1 = T1 - prev_T1;
        Vector3 dT2 = T2 - prev_T2;
        Vector2 u;
        point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, beta_param, u);

        Matrix<Float, 2, 12> J;
        point_triangle_jacobi(basis, beta_param, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;
        G           = J.transpose() * (coeff * u);
    }

    // ----- V1 NOTE: EE/PE/PP tangential adhesion is DISABLED too. -----
    // Same rationale as the EE/PE/PP normal-adhesion disable above: the
    // pair-list redundancy (multiple feature-classified pairs per geometric
    // contact) plus the feature-direction inconsistency would create
    // spurious tangential pulls toward edges/vertices.

    inline __device__ Float EE_tangential_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void EE_tangential_adhesion_gradient_hessian(
        Vector12&      G,
        Matrix12x12&   H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector12::Zero();
        H = Matrix12x12::Zero();
    }

    inline __device__ void EE_tangential_adhesion_gradient(
        Vector12&      G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector12::Zero();
    }

    inline __device__ Float PE_tangential_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void PE_tangential_adhesion_gradient_hessian(
        Vector9&       G,
        Matrix9x9&     H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector9::Zero();
        H = Matrix9x9::Zero();
    }

    inline __device__ void PE_tangential_adhesion_gradient(
        Vector9&       G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&, const Vector3&,
        const Vector3&, const Vector3&, const Vector3&)
    {
        G = Vector9::Zero();
    }

    inline __device__ Float PP_tangential_adhesion_energy(
        Float, Float, Float, Float,
        const Vector3&, const Vector3&,
        const Vector3&, const Vector3&)
    {
        return Float{0};
    }

    inline __device__ void PP_tangential_adhesion_gradient_hessian(
        Vector6&       G,
        Matrix6x6&     H,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&,
        const Vector3&, const Vector3&)
    {
        G = Vector6::Zero();
        H = Matrix6x6::Zero();
    }

    inline __device__ void PP_tangential_adhesion_gradient(
        Vector6&       G,
        Float, Float, Float, Float,
        const Vector3&, const Vector3&,
        const Vector3&, const Vector3&)
    {
        G = Vector6::Zero();
    }
}  // namespace sym::codim_ipc_rcc_adhesive
}  // namespace uipc::backend::cuda
