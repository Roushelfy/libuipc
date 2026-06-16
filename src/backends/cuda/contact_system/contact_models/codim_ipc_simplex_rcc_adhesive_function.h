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
    //
    // `blocked` is the cross-layer occlusion flag (an intervening triangle
    // sits between P and T → no physical contact possible). When set, we
    // short-circuit to β = 0. β=0 is NOT absorbing in the evolution rule
    // (the bonding_term at p_k > 0 can re-ignite β from zero), so a separate
    // persistent flag is needed.
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
                                                    Float u_sq,
                                                    bool  blocked)
    {
        if(blocked)
            return Float{0};

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

    // -------- PT single-sided adhesion gate (v3) --------
    //
    // Returns true ⇔ this PT pair's adhesion contribution is active given the
    // sticky-side preferences of P and T. For both signs == 0 (double-sided —
    // the default when set_sticky_side has never been called), always returns
    // true so v2 behaviour is bit-for-bit preserved.
    //
    // Geometry: the gate fires when EITHER endpoint's "sticky outward
    // direction" points toward its contact partner — i.e. that endpoint's
    // sticky face is the side touching the partner.
    //
    //   P-side:   sticky_P · n̂_P  points toward T  ⇔  (P − closest)·(sticky_P · n̂_P) < 0
    //   T-side:   sticky_T · n̂_T  points toward P  ⇔  (P − closest)·(sticky_T · n̂_T) > 0
    //
    // (Note the opposite inequalities — `(P − closest)` is "from T toward P",
    //  so for P's sticky-outward pointing TO T it's anti-aligned, dot < 0;
    //  for T's sticky-outward pointing TO P it's aligned, dot > 0.)
    //
    // Adhesion fires when either side passes. This handles:
    //   • shell-vs-rigid (only the shell has a side): the shell-side check
    //     gates, the rigid-side has sticky=0 and contributes nothing.
    //   • rolled-up shell self-contact: the inside-of-roll turn has its
    //     sticky face engaged on one of the two PT pair directions; the
    //     extended gate guarantees that whichever pair direction has the
    //     sticky face on EITHER end fires correctly.
    //
    // A sticky_sign of 0 on one side means "no preference"; that side does
    // not contribute to the OR, the other side decides. If both are 0 the
    // gate falls through to the v2 always-on path.
    //
    // n̂_T is approximated by the vertex normal at T0 (passed in from the
    // lagged per-vertex normal buffer). For a shell with consistent winding
    // this matches the face normal up to area-weighting smoothing; for a
    // closed body where T0 is not a shell vert (n̂_T = 0) the T-side check
    // automatically yields 0 and contributes nothing.
    inline __device__ bool PT_sticky_gate(IndexT         sticky_P,
                                          IndexT         sticky_T,
                                          const Vector3& n_P,
                                          const Vector3& n_T,
                                          const Vector3& P,
                                          const Vector3& T0,
                                          const Vector3& T1,
                                          const Vector3& T2)
    {
        if(sticky_P == 0 && sticky_T == 0)
            return true;

        using namespace friction;
        Vector2 bary;
        point_triangle_closest_point(P, T0, T1, T2, bary);
        Vector3 closest = T0 + bary[0] * (T1 - T0) + bary[1] * (T2 - T0);
        Vector3 v_TP   = P - closest;  // direction from T toward P

        if(sticky_P != 0)
        {
            Float d_P = v_TP.dot(Float(sticky_P) * n_P);
            if(d_P < Float{0})
                return true;  // P's sticky face is the contact side.
        }
        if(sticky_T != 0)
        {
            Float d_T = v_TP.dot(Float(sticky_T) * n_T);
            if(d_T > Float{0})
                return true;  // T's sticky face is the contact side.
        }
        return false;
    }

    // -------- Möller-Trumbore segment-triangle intersection --------
    //
    // Tests whether the open segment (origin, origin + dir) hits the triangle
    // (t0, t1, t2) at parameter tt ∈ (tmin, tmax). `dir` is NOT normalized; it
    // is the full segment vector (so the canonical TMIN=1e-5 and TMAX=1-1e-5
    // bounds exclude tiny grazes at the endpoints).
    //
    // Used by the PT-pair cross-layer occlusion gate (Phase B): for a freshly
    // proposed adhesion pair (P, T0..T2), cast a segment from centroid(T) to
    // P at frame-open positions; if some other shell triangle intervenes the
    // pair is geometrically unreachable and must not bond.
    inline __device__ bool segment_triangle_hit(const Vector3& origin,
                                                const Vector3& dir,
                                                const Vector3& t0,
                                                const Vector3& t1,
                                                const Vector3& t2,
                                                Float          tmin,
                                                Float          tmax)
    {
        constexpr Float EPS = Float{1e-12};
        Vector3 e1 = t1 - t0;
        Vector3 e2 = t2 - t0;
        Vector3 h  = dir.cross(e2);
        Float   a  = e1.dot(h);
        if(a > -EPS && a < EPS)
            return false;  // segment parallel to triangle plane
        Float   f  = Float{1} / a;
        Vector3 s  = origin - t0;
        Float   u  = f * s.dot(h);
        if(u < Float{0} || u > Float{1})
            return false;
        Vector3 q = s.cross(e1);
        Float   v = f * dir.dot(q);
        if(v < Float{0} || u + v > Float{1})
            return false;
        Float tt = f * e2.dot(q);
        return (tt > tmin) && (tt < tmax);
    }

    // Cross-layer occlusion gate shared by the Phase B beta-init kernels and
    // the distance-mode lock-eligibility kernel: cast a segment from
    // centroid(T) toward P; if any other shell triangle blocks it, the pair
    // is separated by an intervening layer and must not bond. The engage test
    // fires for BOTH orientations in which adhesion can engage (mirroring
    // PT_sticky_gate): T's sticky face toward P, OR P's sticky face toward T.
    // Triangles sharing any of the pair's 4 vertices are excluded from the
    // cast. PT = (P, T0, T1, T2) global vertex ids; positions/triangles come
    // in as device viewers so each caller supplies its own position snapshot.
    template <typename PosViewer, typename TriViewer>
    inline __device__ bool VT_occlusion_blocked(const Vector4i&  PT,
                                                const Vector3&   P,
                                                const Vector3&   T0,
                                                const Vector3&   T1,
                                                const Vector3&   T2,
                                                IndexT           sticky_P,
                                                IndexT           sticky_T,
                                                const Vector3&   normal_P,
                                                const PosViewer& Ps,
                                                const TriViewer& shell_tris,
                                                IndexT           n_tris)
    {
        Vector3    N    = (T1 - T0).cross(T2 - T0);
        Vector3    Cen  = (T0 + T1 + T2) * Float{1.0 / 3.0};
        Vector3    sdir = P - Cen;
        const bool engage =
            (sticky_P == 0 && sticky_T == 0)
            || (sticky_T != 0 && Float(sticky_T) * N.dot(sdir) > Float{0})
            || (sticky_P != 0 && Float(sticky_P) * normal_P.dot(sdir) < Float{0});
        if(!engage)
            return false;
        constexpr Float TMIN = Float{1e-5};
        constexpr Float TMAX = Float{1} - Float{1e-5};
        for(IndexT j = 0; j < n_tris; ++j)
        {
            const Vector3i& tri = shell_tris(j);
            if(tri[0] == PT[0] || tri[1] == PT[0] || tri[2] == PT[0])
                continue;
            if(tri[0] == PT[1] || tri[1] == PT[1] || tri[2] == PT[1])
                continue;
            if(tri[0] == PT[2] || tri[1] == PT[2] || tri[2] == PT[2])
                continue;
            if(tri[0] == PT[3] || tri[1] == PT[3] || tri[2] == PT[3])
                continue;
            Vector3 A = Ps(tri[0]);
            Vector3 B = Ps(tri[1]);
            Vector3 C = Ps(tri[2]);
            if(segment_triangle_hit(Cen, sdir, A, B, C, TMIN, TMAX))
                return true;
        }
        return false;
    }

    // -------- distance-locked bonding (Phase 7) lock-gate helpers --------
    // MUDA_GENERIC so the CPU oracle ([rcc_bonded_pt][oracle][distance_lock])
    // can call them on host while the Phase A lock-eligibility kernel calls
    // them on device.

    // Face-interior foot test shared by the beta-mode lock mask and the
    // distance-mode lock gate: true iff P's perpendicular foot lies inside
    // the triangle, or on / within `margin` (barycentric units) of its
    // boundary. Degenerate (zero-area) triangles count as face-exterior —
    // the rest-shape builder would reject them downstream anyway.
    inline MUDA_GENERIC bool VT_lock_face_interior_pass(const Vector3& P,
                                                        const Vector3& A,
                                                        const Vector3& B,
                                                        const Vector3& C,
                                                        Float          margin)
    {
        const Vector3 e0 = B - A, e1 = C - A, ep = P - A;
        const Float   d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1);
        const Float   d20 = ep.dot(e0), d21 = ep.dot(e1);
        const Float   den = d00 * d11 - d01 * d01;
        if(den <= Float{0})
            return false;
        const Float v        = (d11 * d20 - d01 * d21) / den;
        const Float w        = (d00 * d21 - d01 * d20) / den;
        const Float u        = Float{1} - v - w;
        const Float min_bary = fmin(u, fmin(v, w));
        return min_bary >= -margin;
    }

    // Distance-band lock predicate: true iff the VT primitive's true
    // closest-feature distance satisfies d < xi + c*d_hat. The closest-feature
    // flag is recomputed from the given (end-of-step) positions — the lock is
    // a fresh decision about end-of-step geometry, never the lagged
    // begin-of-step ActiveVT.flag. c = 0 can never pass: the band collapses to
    // d < xi, which the IPC barrier never allows.
    inline MUDA_GENERIC bool VT_distance_lock_band_pass(const Vector3& P,
                                                        const Vector3& A,
                                                        const Vector3& B,
                                                        const Vector3& C,
                                                        Float          xi,
                                                        Float          d_hat,
                                                        Float          ratio)
    {
        // ratio <= 0 disables distance locking STRUCTURALLY (the documented
        // "c = 0 never locks" contract) — with xi > 0 the band would otherwise
        // degenerate to a thickness-violation detector d < xi, which only the
        // IPC d > xi invariant keeps empty.
        if(ratio <= Float{0})
            return false;
        const Float band = xi + ratio * d_hat;
        const Vector4i flag = distance::point_triangle_distance_flag(P, A, B, C);
        Float          D;
        distance::point_triangle_distance2(flag, P, A, B, C, D);
        return D < band * band;
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
        IndexT           dlock_all   = 1;
        Float            dlock_ratio = Float{0};
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
            // The triangle's three vertices normally share one contact element,
            // so these agree; AND the mode (lock only if every pair is
            // distance-lock) and average the band ratio.
            dlock_all = dlock_all && c.distance_lock;
            dlock_ratio += c.distance_lock_ratio;
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
        r.distance_lock       = dlock_all;
        r.distance_lock_ratio = dlock_ratio / 3.0;
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
    // d_star = xi + c*d_hat offsets the energy minimum to gap d=d_star; d_star<=0
    // recovers the legacy E=k*D (min at d=0). See VT_normal_adhesion_* for the math.
    inline __device__ Float PT_normal_adhesion_energy(Float          Cn,
                                                      Float          beta,
                                                      Float          d_hat,
                                                      Float          dt,
                                                      Float          d_star,
                                                      const Vector3& P,
                                                      const Vector3& T0,
                                                      const Vector3& T1,
                                                      const Vector3& T2)
    {
        using namespace distance;
        Float    D;
        point_triangle_distance2(P, T0, T1, T2, D);  // unflagged: plane projection
        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
            return k * D;
        Float dm = sqrt(D) - d_star;
        return k * dm * dm;
    }

    inline __device__ void
    PT_normal_adhesion_gradient_hessian(Vector12&      G,
                                        Matrix12x12&   H,
                                        Float          Cn,
                                        Float          beta,
                                        Float          d_hat,
                                        Float          dt,
                                        Float          d_star,
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

        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
        {
            G = k * GradD;
            H = k * HessD;
            return;
        }
        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        Float d   = sqrt(D);
        Float eps = 1e-12 * d_hat;
        Float s   = d > eps ? d : eps;
        Float fp  = k * (1.0 - d_star / s);
        Float fpp = k * d_star / (2.0 * s * s * s);
        G         = fp * GradD;
        H         = fp * HessD + fpp * (GradD * GradD.transpose());
    }

    inline __device__ void PT_normal_adhesion_gradient(Vector12&      G,
                                                       Float          Cn,
                                                       Float          beta,
                                                       Float          d_hat,
                                                       Float          dt,
                                                       Float          d_star,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace distance;
        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);  // plane projection

        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
        {
            G = k * GradD;
            return;
        }
        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        Float d   = sqrt(D);
        Float eps = 1e-12 * d_hat;
        Float s   = d > eps ? d : eps;
        Float fp  = k * (1.0 - d_star / s);
        G         = fp * GradD;
    }

    // EE
    // ----- NOTE: EE adhesion is still DISABLED (Step 1 enables PE/PP only). -----
    // PE/PP adhesion is now implemented above with the TRUE point-edge /
    // point-point feature distance and per-primitive beta — the full-feature
    // model (see docs/architecture.md "Full-Feature Adhesion And Per-Primitive
    // Beta"). EE stays disabled until its edge-edge true-feature distance and
    // per-primitive beta land. Behaviour is unchanged until m_beta_{EE,PE,PP}
    // evolve per primitive: every assembly call site early-outs on beta <= 0,
    // and EE beta stays zero-filled (PE/PP beta is wired in a later step).
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

    // PE normal adhesion (per VT/EE-primitive full-feature adhesion, Step 1).
    // Mirrors PT_normal_adhesion_* but uses the TRUE point-edge feature distance
    // (point_edge_distance2) instead of the plane projection, matching the XBow
    // RCCAdhesionEnergy classified-feature model. Behaviour-neutral until
    // m_beta_PE is evolved per primitive: the assembly call sites early-out on
    // beta <= 0 and m_beta_PE is currently zero-filled. P = point, E0/E1 = edge.
    inline __device__ Float PE_normal_adhesion_energy(Float          Cn,
                                                      Float          beta,
                                                      Float          d_hat,
                                                      Float          dt,
                                                      const Vector3& P,
                                                      const Vector3& E0,
                                                      const Vector3& E1)
    {
        using namespace distance;
        Float D;
        point_edge_distance2(P, E0, E1, D);  // true point-edge feature distance
        return (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta * D;
    }

    inline __device__ void
    PE_normal_adhesion_gradient_hessian(Vector9&       G,
                                        Matrix9x9&     H,
                                        Float          Cn,
                                        Float          beta,
                                        Float          d_hat,
                                        Float          dt,
                                        const Vector3& P,
                                        const Vector3& E0,
                                        const Vector3& E1)
    {
        using namespace distance;
        Vector9 GradD;
        point_edge_distance2_gradient(P, E0, E1, GradD);
        Matrix9x9 HessD;
        point_edge_distance2_hessian(P, E0, E1, HessD);

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
        H           = coeff * HessD;  // caller SPD-projects the normal block
    }

    inline __device__ void PE_normal_adhesion_gradient(Vector9&       G,
                                                       Float          Cn,
                                                       Float          beta,
                                                       Float          d_hat,
                                                       Float          dt,
                                                       const Vector3& P,
                                                       const Vector3& E0,
                                                       const Vector3& E1)
    {
        using namespace distance;
        Vector9 GradD;
        point_edge_distance2_gradient(P, E0, E1, GradD);

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
    }

    // PP normal adhesion (Step 1). Mirrors PT/PE but uses the TRUE point-point
    // feature distance (point_point_distance2). P = point, Q = closest triangle
    // vertex. Behaviour-neutral until m_beta_PP is evolved per primitive.
    inline __device__ Float PP_normal_adhesion_energy(Float          Cn,
                                                      Float          beta,
                                                      Float          d_hat,
                                                      Float          dt,
                                                      const Vector3& P,
                                                      const Vector3& Q)
    {
        using namespace distance;
        Float D;
        point_point_distance2(P, Q, D);  // true point-point feature distance
        return (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta * D;
    }

    inline __device__ void
    PP_normal_adhesion_gradient_hessian(Vector6&       G,
                                        Matrix6x6&     H,
                                        Float          Cn,
                                        Float          beta,
                                        Float          d_hat,
                                        Float          dt,
                                        const Vector3& P,
                                        const Vector3& Q)
    {
        using namespace distance;
        Vector6 GradD;
        point_point_distance2_gradient(P, Q, GradD);
        Matrix6x6 HessD;
        point_point_distance2_hessian(P, Q, HessD);

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
        H           = coeff * HessD;  // point-point distance Hessian is PSD
    }

    inline __device__ void PP_normal_adhesion_gradient(Vector6&       G,
                                                       Float          Cn,
                                                       Float          beta,
                                                       Float          d_hat,
                                                       Float          dt,
                                                       const Vector3& P,
                                                       const Vector3& Q)
    {
        using namespace distance;
        Vector6 GradD;
        point_point_distance2_gradient(P, Q, GradD);

        Float coeff = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        G           = coeff * GradD;
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

    // ----- NOTE: EE tangential adhesion is still DISABLED (PE/PP enabled below). -----
    // PE/PP tangential adhesion is implemented below, mirroring PT with the
    // point-edge / point-point friction basis (basis/closest-foot/jacobi). EE
    // tangential stays disabled until the edge-edge feature path and
    // per-primitive beta are added.

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

    // PE tangential adhesion (Step 1): coeff * |u|^2, where u is the lagged
    // point-edge tangential relative displacement. Mirrors PT_tangential_* with
    // the point-edge basis/closest-foot/jacobi helpers. (prev_*) lagged,
    // (P,E0,E1) current; J^T J is PSD so no SPD projection is needed.
    inline __device__ Float PE_tangential_adhesion_energy(Float          Ct,
                                                          Float          beta,
                                                          Float          d_hat,
                                                          Float          dt,
                                                          const Vector3& prev_P,
                                                          const Vector3& prev_E0,
                                                          const Vector3& prev_E1,
                                                          const Vector3& P,
                                                          const Vector3& E0,
                                                          const Vector3& E1)
    {
        using namespace distance;
        using namespace friction;
        Float               eta;
        Matrix<Float, 3, 2> basis;
        point_edge_closest_point(prev_P, prev_E0, prev_E1, eta);
        point_edge_tangent_basis(prev_P, prev_E0, prev_E1, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dE0 = E0 - prev_E0;
        Vector3 dE1 = E1 - prev_E1;
        Vector2 u;
        point_edge_tan_rel_dx(dP, dE0, dE1, basis, eta, u);

        Float coeff = (dt * dt) * (Ct / (2.0 * d_hat)) * beta * beta;
        return coeff * u.squaredNorm();
    }

    inline __device__ void
    PE_tangential_adhesion_gradient_hessian(Vector9&       G,
                                            Matrix9x9&     H,
                                            Float          Ct,
                                            Float          beta,
                                            Float          d_hat,
                                            Float          dt,
                                            const Vector3& prev_P,
                                            const Vector3& prev_E0,
                                            const Vector3& prev_E1,
                                            const Vector3& P,
                                            const Vector3& E0,
                                            const Vector3& E1)
    {
        using namespace distance;
        using namespace friction;
        Float               eta;
        Matrix<Float, 3, 2> basis;
        point_edge_closest_point(prev_P, prev_E0, prev_E1, eta);
        point_edge_tangent_basis(prev_P, prev_E0, prev_E1, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dE0 = E0 - prev_E0;
        Vector3 dE1 = E1 - prev_E1;
        Vector2 u;
        point_edge_tan_rel_dx(dP, dE0, dE1, basis, eta, u);

        Matrix<Float, 2, 9> J;
        point_edge_jacobi(basis, eta, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;  // 2*(1/2) cancels
        G           = J.transpose() * (coeff * u);
        H           = (coeff)*J.transpose() * J;
    }

    inline __device__ void
    PE_tangential_adhesion_gradient(Vector9&       G,
                                    Float          Ct,
                                    Float          beta,
                                    Float          d_hat,
                                    Float          dt,
                                    const Vector3& prev_P,
                                    const Vector3& prev_E0,
                                    const Vector3& prev_E1,
                                    const Vector3& P,
                                    const Vector3& E0,
                                    const Vector3& E1)
    {
        using namespace distance;
        using namespace friction;
        Float               eta;
        Matrix<Float, 3, 2> basis;
        point_edge_closest_point(prev_P, prev_E0, prev_E1, eta);
        point_edge_tangent_basis(prev_P, prev_E0, prev_E1, basis);

        Vector3 dP  = P - prev_P;
        Vector3 dE0 = E0 - prev_E0;
        Vector3 dE1 = E1 - prev_E1;
        Vector2 u;
        point_edge_tan_rel_dx(dP, dE0, dE1, basis, eta, u);

        Matrix<Float, 2, 9> J;
        point_edge_jacobi(basis, eta, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;
        G           = J.transpose() * (coeff * u);
    }

    // PP tangential adhesion (Step 1). Mirrors PT/PE with the point-point
    // basis; no closest-foot parameter (the two points are the stencil).
    // (prev_P0, prev_P1) lagged, (P0, P1) current; J^T J is PSD.
    inline __device__ Float PP_tangential_adhesion_energy(Float          Ct,
                                                          Float          beta,
                                                          Float          d_hat,
                                                          Float          dt,
                                                          const Vector3& prev_P0,
                                                          const Vector3& prev_P1,
                                                          const Vector3& P0,
                                                          const Vector3& P1)
    {
        using namespace distance;
        using namespace friction;
        Matrix<Float, 3, 2> basis;
        point_point_tangent_basis(prev_P0, prev_P1, basis);

        Vector3 dP0 = P0 - prev_P0;
        Vector3 dP1 = P1 - prev_P1;
        Vector2 u;
        point_point_tan_rel_dx(dP0, dP1, basis, u);

        Float coeff = (dt * dt) * (Ct / (2.0 * d_hat)) * beta * beta;
        return coeff * u.squaredNorm();
    }

    inline __device__ void
    PP_tangential_adhesion_gradient_hessian(Vector6&       G,
                                            Matrix6x6&     H,
                                            Float          Ct,
                                            Float          beta,
                                            Float          d_hat,
                                            Float          dt,
                                            const Vector3& prev_P0,
                                            const Vector3& prev_P1,
                                            const Vector3& P0,
                                            const Vector3& P1)
    {
        using namespace distance;
        using namespace friction;
        Matrix<Float, 3, 2> basis;
        point_point_tangent_basis(prev_P0, prev_P1, basis);

        Vector3 dP0 = P0 - prev_P0;
        Vector3 dP1 = P1 - prev_P1;
        Vector2 u;
        point_point_tan_rel_dx(dP0, dP1, basis, u);

        Matrix<Float, 2, 6> J;
        point_point_jacobi(basis, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;
        G           = J.transpose() * (coeff * u);
        H           = (coeff)*J.transpose() * J;
    }

    inline __device__ void
    PP_tangential_adhesion_gradient(Vector6&       G,
                                    Float          Ct,
                                    Float          beta,
                                    Float          d_hat,
                                    Float          dt,
                                    const Vector3& prev_P0,
                                    const Vector3& prev_P1,
                                    const Vector3& P0,
                                    const Vector3& P1)
    {
        using namespace distance;
        using namespace friction;
        Matrix<Float, 3, 2> basis;
        point_point_tangent_basis(prev_P0, prev_P1, basis);

        Vector3 dP0 = P0 - prev_P0;
        Vector3 dP1 = P1 - prev_P1;
        Vector2 u;
        point_point_tan_rel_dx(dP0, dP1, basis, u);

        Matrix<Float, 2, 6> J;
        point_point_jacobi(basis, J);

        Float coeff = (dt * dt) * (Ct / d_hat) * beta * beta;
        G           = J.transpose() * (coeff * u);
    }
    // =====================================================================
    // VT-primitive wrappers (Phase 6 Step 2/3): a single per-VT-primitive
    // entry point that switches on the lagged closest-feature `flag` (the
    // Vector4i from point_triangle_distance_flag) and produces full 12-DOF
    // (4-vertex) gradient/Hessian blocks against the (P,T0,T1,T2) stencil.
    //
    //  - Normal: uses the FLAGGED point_triangle_distance2 dispatch, which
    //    routes PT->plane, PE->point-edge, PP->point-point and already
    //    scatters the reduced feature derivative into the 12-DOF block with
    //    zero-padding for the inactive vertex (same machinery the IPC barrier
    //    uses). So one path covers all three closest-feature cases.
    //  - Tangential: there is no flagged friction basis, so we switch on the
    //    degenerate dim, call the reduced PT/PE/PP tangential helper, and
    //    scatter its (Vector12/9/6) result into the 12-DOF block via the
    //    degenerate offsets.
    //
    // Coeff/d_hat are aggregated over the full VT primitive by the caller
    // (PT_rcc_coeff / PT_d_hat / PT_contact_coeff) so evolution and assembly
    // stay consistent regardless of which sub-feature is closest.
    // =====================================================================

    // d_star = xi + c*d_hat is the target gap (energy minimum). d_star <= 0
    // recovers the legacy energy (min at d=0) bitwise: E = k*D, G = k*GradD,
    // H = k*HessD. Otherwise E = k*(sqrt(D) - d_star)^2 with
    //   f'(D)  = k*(1 - d_star/sqrt(D)),  f''(D) = k*d_star/(2*D^{3/2}),
    //   G = f'(D)*GradD,  H = f'(D)*HessD + f''(D)*GradD*GradD^T.
    inline __device__ Float VT_normal_adhesion_energy(Float           Cn,
                                                      Float           beta,
                                                      Float           d_hat,
                                                      Float           dt,
                                                      Float           d_star,
                                                      const Vector4i& flag,
                                                      const Vector3&  P,
                                                      const Vector3&  T0,
                                                      const Vector3&  T1,
                                                      const Vector3&  T2)
    {
        using namespace distance;
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);  // true closest-feature distance
        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
            return k * D;  // legacy: min at d=0 (no sqrt, bitwise identical)
        Float dm = sqrt(D) - d_star;
        return k * dm * dm;  // E = k (sqrt(D) - d*)^2, min at d = d*
    }

    inline __device__ void
    VT_normal_adhesion_gradient_hessian(Vector12&       G,
                                        Matrix12x12&    H,
                                        Float           Cn,
                                        Float           beta,
                                        Float           d_hat,
                                        Float           dt,
                                        Float           d_star,
                                        const Vector4i& flag,
                                        const Vector3&  P,
                                        const Vector3&  T0,
                                        const Vector3&  T1,
                                        const Vector3&  T2)
    {
        using namespace distance;
        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);
        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);
        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
        {
            G = k * GradD;
            H = k * HessD;  // caller SPD-projects the normal block
            return;
        }
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        Float d   = sqrt(D);
        Float eps = 1e-12 * d_hat;              // relative gap floor for the ratios
        Float s   = d > eps ? d : eps;
        Float fp  = k * (1.0 - d_star / s);             // f'(D)
        Float fpp = k * d_star / (2.0 * s * s * s);     // f''(D) = k d*/(2 D^{3/2})
        G         = fp * GradD;
        H         = fp * HessD + fpp * (GradD * GradD.transpose());
        // caller SPD-projects the normal block (handles fp<0 when d<d*)
    }

    inline __device__ void VT_normal_adhesion_gradient(Vector12&       G,
                                                       Float           Cn,
                                                       Float           beta,
                                                       Float           d_hat,
                                                       Float           dt,
                                                       Float           d_star,
                                                       const Vector4i& flag,
                                                       const Vector3&  P,
                                                       const Vector3&  T0,
                                                       const Vector3&  T1,
                                                       const Vector3&  T2)
    {
        using namespace distance;
        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);
        Float k = (dt * dt) * (Cn / (2.0 * d_hat)) * beta * beta;
        if(d_star <= 0)
        {
            G = k * GradD;
            return;
        }
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        Float d   = sqrt(D);
        Float eps = 1e-12 * d_hat;
        Float s   = d > eps ? d : eps;
        Float fp  = k * (1.0 - d_star / s);
        G         = fp * GradD;
    }

    // Lagged tangential relative-displacement magnitude squared for the VT
    // primitive (feature-classified). Used by the beta-evolution law.
    inline __device__ Float VT_tangential_rel_dx_sq(const Vector4i& flag,
                                                    const Vector3&  pP,
                                                    const Vector3&  pT0,
                                                    const Vector3&  pT1,
                                                    const Vector3&  pT2,
                                                    const Vector3&  P,
                                                    const Vector3&  T0,
                                                    const Vector3&  T1,
                                                    const Vector3&  T2)
    {
        using namespace distance;
        using namespace friction;
        Vector4i offsets;
        IndexT   dim = degenerate_point_triangle(flag, offsets);
        Vector3  prev[4] = {pP, pT0, pT1, pT2};
        Vector3  cur[4]  = {P, T0, T1, T2};
        Vector2  u;
        if(dim == 4)
        {
            Vector2             bary;
            Matrix<Float, 3, 2> basis;
            point_triangle_closest_point(pP, pT0, pT1, pT2, bary);
            point_triangle_tangent_basis(pP, pT0, pT1, pT2, basis);
            Vector3 dP = P - pP, dT0 = T0 - pT0, dT1 = T1 - pT1, dT2 = T2 - pT2;
            point_triangle_tan_rel_dx(dP, dT0, dT1, dT2, basis, bary, u);
        }
        else if(dim == 3)
        {
            IndexT              a = offsets(0), b = offsets(1), c = offsets(2);
            Float               eta;
            Matrix<Float, 3, 2> basis;
            point_edge_closest_point(prev[a], prev[b], prev[c], eta);
            point_edge_tangent_basis(prev[a], prev[b], prev[c], basis);
            Vector3 dPa = cur[a] - prev[a], dPb = cur[b] - prev[b], dPc = cur[c] - prev[c];
            point_edge_tan_rel_dx(dPa, dPb, dPc, basis, eta, u);
        }
        else  // dim == 2
        {
            IndexT              a = offsets(0), b = offsets(1);
            Matrix<Float, 3, 2> basis;
            point_point_tangent_basis(prev[a], prev[b], basis);
            Vector3 dPa = cur[a] - prev[a], dPb = cur[b] - prev[b];
            point_point_tan_rel_dx(dPa, dPb, basis, u);
        }
        return u.squaredNorm();
    }

    inline __device__ Float VT_tangential_adhesion_energy(Float           Ct,
                                                          Float           beta,
                                                          Float           d_hat,
                                                          Float           dt,
                                                          const Vector4i& flag,
                                                          const Vector3&  pP,
                                                          const Vector3&  pT0,
                                                          const Vector3&  pT1,
                                                          const Vector3&  pT2,
                                                          const Vector3&  P,
                                                          const Vector3&  T0,
                                                          const Vector3&  T1,
                                                          const Vector3&  T2)
    {
        using namespace distance;
        Vector4i offsets;
        IndexT   dim     = degenerate_point_triangle(flag, offsets);
        Vector3  prev[4] = {pP, pT0, pT1, pT2};
        Vector3  cur[4]  = {P, T0, T1, T2};
        if(dim == 4)
            return PT_tangential_adhesion_energy(Ct, beta, d_hat, dt, pP, pT0, pT1, pT2, P, T0, T1, T2);
        if(dim == 3)
        {
            IndexT a = offsets(0), b = offsets(1), c = offsets(2);
            return PE_tangential_adhesion_energy(Ct, beta, d_hat, dt,
                                                 prev[a], prev[b], prev[c],
                                                 cur[a], cur[b], cur[c]);
        }
        IndexT a = offsets(0), b = offsets(1);
        return PP_tangential_adhesion_energy(Ct, beta, d_hat, dt, prev[a], prev[b], cur[a], cur[b]);
    }

    inline __device__ void
    VT_tangential_adhesion_gradient_hessian(Vector12&       G,
                                            Matrix12x12&    H,
                                            Float           Ct,
                                            Float           beta,
                                            Float           d_hat,
                                            Float           dt,
                                            const Vector4i& flag,
                                            const Vector3&  pP,
                                            const Vector3&  pT0,
                                            const Vector3&  pT1,
                                            const Vector3&  pT2,
                                            const Vector3&  P,
                                            const Vector3&  T0,
                                            const Vector3&  T1,
                                            const Vector3&  T2)
    {
        using namespace distance;
        G = Vector12::Zero();
        H = Matrix12x12::Zero();
        Vector4i offsets;
        IndexT   dim     = degenerate_point_triangle(flag, offsets);
        Vector3  prev[4] = {pP, pT0, pT1, pT2};
        Vector3  cur[4]  = {P, T0, T1, T2};
        if(dim == 4)
        {
            PT_tangential_adhesion_gradient_hessian(G, H, Ct, beta, d_hat, dt,
                                                    pP, pT0, pT1, pT2, P, T0, T1, T2);
        }
        else if(dim == 3)
        {
            Vector9   g9;
            Matrix9x9 h9;
            IndexT    idx[3] = {offsets(0), offsets(1), offsets(2)};
            PE_tangential_adhesion_gradient_hessian(g9, h9, Ct, beta, d_hat, dt,
                                                    prev[idx[0]], prev[idx[1]], prev[idx[2]],
                                                    cur[idx[0]], cur[idx[1]], cur[idx[2]]);
            for(int r = 0; r < 3; ++r)
            {
                G.template segment<3>(3 * idx[r]) = g9.template segment<3>(3 * r);
                for(int s = 0; s < 3; ++s)
                    H.template block<3, 3>(3 * idx[r], 3 * idx[s]) =
                        h9.template block<3, 3>(3 * r, 3 * s);
            }
        }
        else  // dim == 2
        {
            Vector6   g6;
            Matrix6x6 h6;
            IndexT    idx[2] = {offsets(0), offsets(1)};
            PP_tangential_adhesion_gradient_hessian(g6, h6, Ct, beta, d_hat, dt,
                                                    prev[idx[0]], prev[idx[1]],
                                                    cur[idx[0]], cur[idx[1]]);
            for(int r = 0; r < 2; ++r)
            {
                G.template segment<3>(3 * idx[r]) = g6.template segment<3>(3 * r);
                for(int s = 0; s < 2; ++s)
                    H.template block<3, 3>(3 * idx[r], 3 * idx[s]) =
                        h6.template block<3, 3>(3 * r, 3 * s);
            }
        }
    }

    inline __device__ void
    VT_tangential_adhesion_gradient(Vector12&       G,
                                    Float           Ct,
                                    Float           beta,
                                    Float           d_hat,
                                    Float           dt,
                                    const Vector4i& flag,
                                    const Vector3&  pP,
                                    const Vector3&  pT0,
                                    const Vector3&  pT1,
                                    const Vector3&  pT2,
                                    const Vector3&  P,
                                    const Vector3&  T0,
                                    const Vector3&  T1,
                                    const Vector3&  T2)
    {
        using namespace distance;
        G = Vector12::Zero();
        Vector4i offsets;
        IndexT   dim     = degenerate_point_triangle(flag, offsets);
        Vector3  prev[4] = {pP, pT0, pT1, pT2};
        Vector3  cur[4]  = {P, T0, T1, T2};
        if(dim == 4)
        {
            PT_tangential_adhesion_gradient(G, Ct, beta, d_hat, dt,
                                            pP, pT0, pT1, pT2, P, T0, T1, T2);
        }
        else if(dim == 3)
        {
            Vector9 g9;
            IndexT  idx[3] = {offsets(0), offsets(1), offsets(2)};
            PE_tangential_adhesion_gradient(g9, Ct, beta, d_hat, dt,
                                            prev[idx[0]], prev[idx[1]], prev[idx[2]],
                                            cur[idx[0]], cur[idx[1]], cur[idx[2]]);
            for(int r = 0; r < 3; ++r)
                G.template segment<3>(3 * idx[r]) = g9.template segment<3>(3 * r);
        }
        else  // dim == 2
        {
            Vector6 g6;
            IndexT  idx[2] = {offsets(0), offsets(1)};
            PP_tangential_adhesion_gradient(g6, Ct, beta, d_hat, dt,
                                            prev[idx[0]], prev[idx[1]],
                                            cur[idx[0]], cur[idx[1]]);
            for(int r = 0; r < 2; ++r)
                G.template segment<3>(3 * idx[r]) = g6.template segment<3>(3 * r);
        }
    }

}  // namespace sym::codim_ipc_rcc_adhesive
}  // namespace uipc::backend::cuda
