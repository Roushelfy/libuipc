"""
Helper utilities for the wound-tape asset pipeline.

Creators:
    make_ring_hub        — closed-surface ring (donut) trimesh for ABD use
    make_rod             — closed-surface cylinder rod (axis along +z), ABD-ready
    make_flat_tape       — long thin shell strip (sticky-up by default winding)
    make_wound_tape      — tape strip with vertex positions laid along an
                           Archimedean spiral (the procedural "pre-rolled"
                           configuration)

Plus a thin save/load layer:
    save_tape_asset / load_tape_asset — np.savez round-trip of the
                                        non-topology asset state.

ABD on a closed trimesh:
    `AffineBodyConstitution.apply_to(sc, kappa, mass_density)` internally
    calls `compute_mesh_volume(sc)` (divergence theorem on the surface;
    asserts the surface is closed) + `compute_dyadic_mass(sc, rho)` to
    populate the abd_mass / abd_mass_x_bar / abd_mass_x_bar_x_bar
    attributes automatically. So the hub only needs to be a valid closed
    surface — no tetrahedral interior required.
    See src/constitution/affine_body_constitution.cpp:136-144.
"""

from __future__ import annotations

import os
from typing import Tuple
import numpy as np

from uipc.geometry import (
    SimplicialComplex,
    trimesh,
    label_surface,
    mesh_partition,
)


# ----------------------------------------------------------------------
# Ring hub (ABD-ready closed trimesh)
# ----------------------------------------------------------------------
def make_ring_hub(R_outer: float,
                  R_inner: float,
                  height: float,
                  n_radial: int = 48,
                  center: tuple = (0.0, 0.0, 0.0)) -> SimplicialComplex:
    """Build a closed trimesh ring (donut prism) with its axis along +y.

    Geometry:
        Top + bottom annular faces (each n_radial wedges, 2 tris/wedge)
      + outer cylindrical side + inner cylindrical side
        (each n_radial faces, 2 tris/face)
        → 8 · n_radial triangles, 4 · n_radial vertices.

    Triangle windings are chosen so face normals point OUT of the body
    on every face. `is_trimesh_closed` will pass, so ABD's automatic
    mass computation works.
    """
    assert R_outer > R_inner > 0.0, "need 0 < R_inner < R_outer"
    assert height > 0.0
    cx, cy, cz = center
    y_top = cy + 0.5 * height
    y_bot = cy - 0.5 * height

    theta = np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # vertex layout:
    #   [0 .. n)              outer top   (y = y_top)
    #   [n .. 2n)             inner top   (y = y_top)
    #   [2n .. 3n)            outer bot   (y = y_bot)
    #   [3n .. 4n)            inner bot   (y = y_bot)
    n = n_radial
    verts = np.empty((4 * n, 3), dtype=np.float64)
    verts[0:n, 0] = cx + R_outer * cos_t
    verts[0:n, 1] = y_top
    verts[0:n, 2] = cz + R_outer * sin_t

    verts[n:2*n, 0] = cx + R_inner * cos_t
    verts[n:2*n, 1] = y_top
    verts[n:2*n, 2] = cz + R_inner * sin_t

    verts[2*n:3*n, 0] = cx + R_outer * cos_t
    verts[2*n:3*n, 1] = y_bot
    verts[2*n:3*n, 2] = cz + R_outer * sin_t

    verts[3*n:4*n, 0] = cx + R_inner * cos_t
    verts[3*n:4*n, 1] = y_bot
    verts[3*n:4*n, 2] = cz + R_inner * sin_t

    def OT(i): return i % n               # outer top
    def IT(i): return n + (i % n)         # inner top
    def OB(i): return 2*n + (i % n)       # outer bot
    def IB(i): return 3*n + (i % n)       # inner bot

    tris = []
    for i in range(n):
        j = (i + 1) % n
        # top annulus — normal +y. wedge (OT_i, IT_i, IT_j, OT_j)
        tris.append([OT(i), IT(i), IT(j)])
        tris.append([OT(i), IT(j), OT(j)])
        # bottom annulus — normal -y. opposite winding.
        tris.append([OB(i), OB(j), IB(j)])
        tris.append([OB(i), IB(j), IB(i)])
        # outer cylinder — normal +radial.  wedge (OT_i, OT_j, OB_j, OB_i)
        tris.append([OT(i), OT(j), OB(i)])
        tris.append([OT(j), OB(j), OB(i)])
        # inner cylinder — normal -radial (points into the hole, OUT of body).
        tris.append([IT(i), IB(i), IT(j)])
        tris.append([IT(j), IB(i), IB(j)])

    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    return sc


# ----------------------------------------------------------------------
# Rod (closed-surface cylinder, axis along +z)
# ----------------------------------------------------------------------
def make_rod(R: float,
             length: float,
             n_radial: int = 24,
             center: tuple = (0.0, 0.0, 0.0)) -> SimplicialComplex:
    """Closed-surface cylinder with its axis along +z.

    Geometry: two endcap fans + side cylinder, with outward-pointing
    normals on every face. Suitable for ABD via the divergence-theorem
    mass path.

    Designed to be threaded through a ring hub (whose axis has been
    rotated to +z). `R` should be smaller than the hub's inner radius
    so the rod fits with radial clearance.
    """
    assert R > 0.0 and length > 0.0
    cx, cy, cz = center
    z_hi = cz + 0.5 * length
    z_lo = cz - 0.5 * length

    theta = np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    n = n_radial
    # vertex layout:
    #   [0 .. n)     ring at z_hi
    #   [n .. 2n)    ring at z_lo
    #   [2n]         apex at z_hi (top cap fan center)
    #   [2n + 1]     apex at z_lo (bottom cap fan center)
    verts = np.empty((2 * n + 2, 3), dtype=np.float64)
    verts[0:n, 0] = cx + R * cos_t
    verts[0:n, 1] = cy + R * sin_t
    verts[0:n, 2] = z_hi
    verts[n:2*n, 0] = cx + R * cos_t
    verts[n:2*n, 1] = cy + R * sin_t
    verts[n:2*n, 2] = z_lo
    verts[2*n]     = (cx, cy, z_hi)
    verts[2*n + 1] = (cx, cy, z_lo)

    HI = 2 * n
    LO = 2 * n + 1

    tris = []
    for i in range(n):
        j = (i + 1) % n
        # top cap (z = z_hi): outward = +z. fan (HI, i, j) wraps CCW
        # in xy as viewed from +z, so cross(i-HI, j-HI) is +z.
        tris.append([HI, i, j])
        # bottom cap (z = z_lo): outward = -z. flip winding.
        tris.append([LO, n + j, n + i])
        # side: outward = +radial. quad (i_hi, j_hi, j_lo, i_lo) split.
        tris.append([i,     n + i, j])
        tris.append([j,     n + i, n + j])

    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    return sc


# ----------------------------------------------------------------------
# Flat tape strip (sticky face UP by default winding)
# ----------------------------------------------------------------------
def make_flat_tape(length: float,
                   width: float,
                   NX: int,
                   NZ: int,
                   base_y: float = 0.0,
                   center_xz: tuple = (0.0, 0.0)) -> SimplicialComplex:
    """Long thin shell in the xz plane at y=base_y. Triangle winding gives
    the upper face (+y) the +n̂ direction, so `set_sticky_side(tape, +1)`
    marks the upper face as sticky."""
    cx, cz = center_xz
    dx = length / NX
    dz = width  / NZ
    verts = np.empty(((NX + 1) * (NZ + 1), 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(NZ + 1):
            x = cx + i * dx - 0.5 * length
            z = cz + j * dz - 0.5 * width
            verts[i * (NZ + 1) + j] = (x, base_y, z)

    def vid(i, j): return i * (NZ + 1) + j

    tris = []
    # winding chosen so (B-A) × (C-A) has positive y component.
    for i in range(NX):
        for j in range(NZ):
            v00 = vid(i,     j)
            v10 = vid(i + 1, j)
            v01 = vid(i,     j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v01, v11])
            tris.append([v00, v11, v10])
    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc


# ----------------------------------------------------------------------
# Procedurally-wound tape (Archimedean spiral)
# ----------------------------------------------------------------------
def _archimedean_sample(R0: float, b: float, theta_end: float, ds_target: float):
    """Sample theta values along an Archimedean spiral r(θ) = R0 + b·θ
    at approximately equal arc length `ds_target`. Uses the closed-form
    arclength s(θ) = ½·[θ√(θ²+a²) + a²·sinh⁻¹(θ/a)] where a = R0/b for
    the simple Archimedean spiral; here we use a robust numerical
    cumtrapz on |r'(θ)| since the closed form mixes the (R0, b) constants
    nontrivially. Returns (theta_samples, total_length).
    """
    n_fine = max(2000, int(theta_end * 200))
    th_fine = np.linspace(0.0, theta_end, n_fine)
    r_fine = R0 + b * th_fine
    drdth = b
    # |dp/dθ| = sqrt(r² + (dr/dθ)²)
    speed = np.sqrt(r_fine ** 2 + drdth ** 2)
    s_fine = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                                              * np.diff(th_fine))])
    total_length = float(s_fine[-1])
    n_samples = max(2, int(np.ceil(total_length / ds_target)))
    s_target = np.linspace(0.0, total_length, n_samples)
    th_samples = np.interp(s_target, s_fine, th_fine)
    return th_samples, total_length


def make_wound_tape(hub_R_outer: float,
                    n_turns: float,
                    tape_width: float,
                    layer_thickness: float,
                    NZ: int = 6,
                    ds: float | None = None,
                    hub_center: tuple = (0.0, 0.0, 0.0),
                    start_gap: float | None = None,
                    tape_thickness: float = 0.0,
                    d_hat: float | None = None) -> Tuple[SimplicialComplex, float]:
    """Construct a tape strip whose vertex positions lie along an
    Archimedean spiral wrapping around `hub_center` in the xz plane.

    Parameters
    ----------
    hub_R_outer
        Outer radius of the hub (= starting radius of the innermost turn).
    n_turns
        Number of turns of tape (e.g. 5 means 5 full 2π wraps).
    tape_width
        Width of the tape along the y axis (perpendicular to the spiral).
    layer_thickness
        Radial spacing between adjacent turns. Roughly equals d_hat of the
        contact; the IPC barrier's active band sets the effective gap.
    NZ
        Number of cells across the tape width.
    ds
        Approximate arc length between consecutive vertices along the
        spiral direction. Defaults to layer_thickness (so cells are
        roughly square).
    hub_center
        Center of the spiral (xz only; the y component is the tape's
        midplane).
    start_gap
        Radial clearance between the hub surface (r = hub_R_outer) and
        the innermost turn (r = hub_R_outer + start_gap). For IPC's
        thickness-augmented barrier `D_range(xi, d_hat) = [xi², (xi+d_hat)²]`
        with `xi = tape_thickness + hub_thickness` (hub thickness = 0 for
        ABD), the safe band for tape-hub is
            tape_thickness  <  start_gap  <  tape_thickness + d_hat
        Default (when both tape_thickness and d_hat are passed) is the
        midpoint of that band; otherwise `0.5 * layer_thickness`.
    tape_thickness
        Per-vertex `builtin::thickness` set by `NeoHookeanShell.apply_to`
        on the tape. Used to pick safe defaults for `start_gap` and to
        sanity-check `layer_thickness`. Leave 0 if you don't apply
        thickness (rare for shells in practice).
    d_hat
        IPC contact active band. Same purpose as `tape_thickness` —
        feeds the auto-pick of `start_gap` and the layer-thickness
        sanity check.

    Returns
    -------
    sc, total_length
        SimplicialComplex with the spiral-laid positions. Triangle
        winding is chosen so the face normal points radially OUTWARD
        (away from the hub center). Caller should
        `RCCAdhesive.set_sticky_side(sc, -1)` to make the inner (sticky)
        face point toward the hub — matching how a real roll is wound.

    Geometry
    --------
    Tape thickness = 0 (shell). The "layer thickness" only spaces
    successive turns radially; collision/adhesion between turns is via
    the IPC barrier active band (d_hat).
    """
    cx, cy, cz = hub_center
    b = layer_thickness / (2.0 * np.pi)
    theta_end = 2.0 * np.pi * n_turns

    # ---- sanity checks against IPC's thickness-augmented barrier band ----
    # (only fire when the caller supplies both d_hat and tape_thickness;
    # silently fall back to half-layer otherwise to preserve old callers).
    if tape_thickness > 0.0 and d_hat is not None:
        xi_TT = 2.0 * tape_thickness     # tape-tape: xi = t + t
        if layer_thickness <= xi_TT:
            raise ValueError(
                f"layer_thickness={layer_thickness} <= 2*tape_thickness={xi_TT}: "
                f"adjacent turns will be in geometric penetration at init.")
        if layer_thickness >= xi_TT + d_hat:
            # not a hard error — adhesion just won't fire between adjacent
            # turns (they sit outside the active band).
            import warnings
            warnings.warn(
                f"layer_thickness={layer_thickness} >= 2*tape_thickness + d_hat="
                f"{xi_TT + d_hat}: adjacent turns are outside the IPC active band, "
                f"so layer-vs-layer barrier+adhesion will NOT fire.")
        if start_gap is None:
            # midpoint of the tape-hub safe band [t, t + d_hat]
            start_gap = tape_thickness + 0.5 * d_hat

    if start_gap is None:
        start_gap = 0.5 * layer_thickness
    R0 = hub_R_outer + start_gap   # radius of the innermost turn at θ = 0

    if ds is None:
        ds = layer_thickness
    th_samples, total_length = _archimedean_sample(R0, b, theta_end, ds)
    NX = len(th_samples) - 1  # number of cells along the spiral direction

    n_strip = NX + 1
    n_width = NZ + 1
    verts = np.empty((n_strip * n_width, 3), dtype=np.float64)

    for i, th in enumerate(th_samples):
        r = R0 + b * th
        # tangent direction in xz at this θ:
        #   p(θ) = (r cos θ, _, r sin θ),  p'(θ) = (-r sin θ + b cos θ, _,
        #                                          r cos θ + b sin θ)
        # the spiral-outward normal in xz (perp to tangent, pointing
        # toward larger r): (cos θ, _, sin θ) is the radial direction.
        cos_t = np.cos(th)
        sin_t = np.sin(th)
        for j in range(n_width):
            y = cy + (j / NZ - 0.5) * tape_width
            x = cx + r * cos_t
            z = cz + r * sin_t
            verts[i * n_width + j] = (x, y, z)

    def vid(i, j): return i * n_width + j

    tris = []
    # We want each tape "face" triangle to have its normal pointing
    # radially OUTWARD (away from hub center). At any spiral point, the
    # local tangent is along p'(θ); the radial outward direction is
    # (cos θ, 0, sin θ); the y axis is the across-the-width direction.
    # cross(tangent, y) = ±radial. Choosing the right winding requires
    # walking θ in the positive direction, which is what we do below.
    # Empirically for an Archimedean spiral wound CCW (θ increasing) the
    # winding (i,j) → (i+1,j) → (i+1,j+1) yields normals pointing
    # radially outward; we use that order.
    for i in range(NX):
        for j in range(NZ):
            v00 = vid(i,     j)
            v10 = vid(i + 1, j)
            v01 = vid(i,     j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc, total_length


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------
def save_tape_asset(npz_path: str,
                    hub_transform: np.ndarray,
                    tape_positions: np.ndarray,
                    params: dict,
                    pair_state=None) -> None:
    """Save a wound tape asset.

    Stores:
        hub_transform   : 4×4 float64
        tape_positions  : (N, 3) float64  — the wound positions
        params          : a dict of build parameters (hub R_outer, n_turns,
                          tape_width, layer_thickness, NZ, ds, etc.) so the
                          loader can reconstruct the SC topology with the
                          same generator call.
        pair_state      : optional (keys, betas) tuple from
                          RCCAdhesionStateAccessorFeature.dump_pt_state().
                          `keys` is uint64 (sorted vertex-tuple hashes),
                          `betas` is float64 (per-pair adhesion intensity).
                          When provided, the unwind/drop demos can restore
                          the wound bond state on load. When omitted, the
                          .npz is backward-compatible with old loaders.
    """
    payload = dict(
        hub_transform=np.asarray(hub_transform, dtype=np.float64),
        tape_positions=np.asarray(tape_positions, dtype=np.float64),
        params=np.array([params], dtype=object),
    )
    if pair_state is not None:
        keys, betas = pair_state
        keys  = np.asarray(keys,  dtype=np.uint64)
        betas = np.asarray(betas, dtype=np.float64)
        if keys.shape != betas.shape:
            raise ValueError(
                f"save_tape_asset: pair_state keys/betas shape mismatch "
                f"({keys.shape} vs {betas.shape})")
        payload["pair_state_pt_keys"]  = keys
        payload["pair_state_pt_betas"] = betas
    np.savez(npz_path, **payload)


def load_tape_asset(npz_path: str):
    """Inverse of save_tape_asset.

    Returns a 4-tuple:
        hub_transform, tape_positions, params, pair_state

    `pair_state` is either None (legacy asset without saved β) or a tuple
    (keys_uint64, betas_float64) ready to feed into
    RCCAdhesionStateAccessorFeature.load_pt_state().
    """
    data = np.load(npz_path, allow_pickle=True)
    pair_state = None
    if "pair_state_pt_keys" in data.files:
        pair_state = (np.asarray(data["pair_state_pt_keys"],  dtype=np.uint64),
                      np.asarray(data["pair_state_pt_betas"], dtype=np.float64))
    return (data["hub_transform"],
            data["tape_positions"],
            data["params"][0],
            pair_state)


# ----------------------------------------------------------------------
# Parameter presets — split by demo
# ----------------------------------------------------------------------
# The two demos sweep DIFFERENT variables and therefore have separate
# preset namespaces:
#
#   WIND_PRESETS   — geometry / dimensions / IPC numerics. Adhesion is
#                    OFF during winding, so material values are only
#                    needed for the NeoHookean shell (kept at sensible
#                    defaults; rarely the variable you tune).
#
#   UNWIND_PRESETS — material (Young's, ν, ρ) + adhesion (Cn, Ct, W, η)
#                    + pull SPC. Adhesion is ON during the peel. Geometry
#                    + TAPE_THICKNESS + D_HAT come from the loaded asset
#                    (saved by the wind demo), not from the unwind preset.
#                    NOTE: D_HAT/TAPE_THICKNESS may still appear in unwind
#                    presets as defaults used only when the asset doesn't
#                    record them (legacy .npz files).
#
# CLI examples:
#   python rcc_adhesive_tape_winding_demo.py --preset temflex175
#   python rcc_adhesive_tape_unwind_demo.py  --preset temflex175-medium \
#                                            --asset temflex175
#
# Asset coupling:
#   wind   saves to ASSET_DIR/{preset_name}.npz by default
#   unwind loads ASSET_DIR/{--asset NAME}.npz; if --asset not given,
#   defaults to ASSET_DIR/{preset_name}.npz (often wrong — use --asset).
WIND_PRESETS = {
    "default": {
        # Wind-demo defaults — the working config from before the
        # Temflex sweep, with ONLY the hub radii corrected to actual
        # 3M Temflex 175 measurements:
        #   R_inner = 1.5" / 19.05 mm (cardboard core hole)
        #   R_outer ≈ 21.1 mm        (cardboard outer surface)
        # HUB_HEIGHT stays generous (5 cm) so the 4 cm tape doesn't
        # slip off either end; tape width and the rest are unchanged
        # from the working config. For a fully Temflex-accurate
        # geometry (narrow tape, low-profile hub), use temflex175.
        "HUB_R_OUTER":       0.0211,
        "HUB_R_INNER":       0.01905,
        "HUB_HEIGHT":        0.02,
        "TAPE_WIDTH":        0.019,
        "TAPE_LENGTH":       0.80,
        "N_TURNS":           5,
        "TAPE_NZ":          10,        # mesh cells across tape width;
                                        # TAPE_NX auto-derived for square
                                        # cells unless explicitly --set.
        "TAPE_YOUNGS":       1.0e8,
        "TAPE_POISSON":      0.4,
        "TAPE_MASS_DENSITY": 2.0e2,
        "TAPE_THICKNESS":    1.0e-4,
        # D_HAT_RATIO: d_hat = D_HAT_RATIO * TAPE_THICKNESS, so reducing
        # TAPE_THICKNESS auto-shrinks the IPC active band. Avoids stale
        # d_hat triggering spurious self-contact on fine meshes. Explicit
        # `--set D_HAT=...` still overrides if needed.
        "D_HAT_RATIO":       10.0,     # → D_HAT = 1.0e-3
        "LAYER_THICKNESS":   7.0e-4,
        "BUFFER_LENGTH":     0.04,     # wind demo: free-bend window length
        # RCC adhesion applied during wind to tape↔tape and tape↔hub pairs.
        # initial_beta=0 makes new contacts start with NO adhesion force, so
        # the β=0→1 jump that fires when trajectory-filter detects a fresh
        # pair (and which used to cause visible jitter / line-search blips)
        # is replaced by smooth growth via the bonding-rate rule. With
        # bonding_rate=5 a layer under sustained SPC compression reaches
        # β≈1 within a few frames — by the time it's saved into the asset,
        # the wound region is fully bonded.
        "ADH_CN":            1.0e4,
        "ADH_CT":            1.0e5,
        "ADH_W":             1.0,
        "ADH_ETA":           2.0,
        "ADH_BONDING_RATE":  5.0,
        "ADH_INITIAL_BETA":  0.0,
    },
    # ===== 3M Temflex 175 vinyl electrical tape =====
    # Geometry from 3M's official datasheet:
    #   - 1.5" core diameter → R_outer = 19.05 mm
    #   - 3/4" tape width → 19 mm
    #   - 0.178 mm physical total thickness (PVC backing + rubber PSA)
    # Material from PVC backing estimates (3M doesn't publish E/ν/ρ):
    #   - E ≈ 50 MPa secant (flexible plasticised PVC, tape-scale)
    #   - ν ≈ 0.45 (PVC near-incompressible)
    #   - ρ ≈ 1300 kg/m³ (flexible PVC density)
    # IPC numerics: TAPE_THICKNESS = half the physical thickness, so
    # IPC's xi = 2·t matches the real per-layer spacing. To compensate
    # for the lost bending/membrane stiffness from the smaller sim
    # thickness, TAPE_YOUNGS is bumped (linear in membrane, cubic in
    # bending). The advisor's "membrane > 1e6 Pa" recommendation is
    # easily met by all temflex175-* presets below.
    "temflex175": {
        "HUB_R_OUTER":       0.0211,    # cardboard outer (measured ≈ 21.1 mm)
        "HUB_R_INNER":       0.01905,   # 1.5" core hole (= 19.05 mm)
        "HUB_HEIGHT":        0.020,     # ≈ tape width (measured 19 mm) + 1 mm sim margin
        "TAPE_WIDTH":        0.019,
        "TAPE_LENGTH":       0.73,    # ≈ 5cm slack after 5 turns
        "N_TURNS":           5,
        "TAPE_NZ":           10,
        "TAPE_YOUNGS":       1.0e8,    # 50 MPa × 2 (sim-thickness comp)
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,   # = ½ physical 0.178 mm
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "LAYER_THICKNESS":   2.5e-4,   # > 2·t=0.18mm, < 2·t+D_HAT=0.58mm
        "BUFFER_LENGTH":     0.04,
        # RCC adhesion applied during wind to tape↔tape and tape↔hub pairs.
        # See `default` preset for the rationale on initial_beta=0.
        "ADH_CN":            1.0e4,
        "ADH_CT":            1.0e5,
        "ADH_W":             1.0,
        "ADH_ETA":           2.0,
        "ADH_BONDING_RATE":  5.0,
        "ADH_INITIAL_BETA":  0.0,
    },
    "temflex175-3turn": {
        # Quick test: 3 turns instead of 5. Faster wind sim for iteration.
        "HUB_R_OUTER":       0.0211,
        "HUB_R_INNER":       0.01905,
        "HUB_HEIGHT":        0.020,
        "TAPE_WIDTH":        0.019,
        "TAPE_LENGTH":       0.46,    # ≈ 5cm slack after 3 turns
        "N_TURNS":           3,
        "TAPE_NZ":           10,
        "TAPE_YOUNGS":       1.0e8,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "LAYER_THICKNESS":   2.5e-4,
        "BUFFER_LENGTH":     0.04,
        # RCC adhesion applied during wind to tape↔tape and tape↔hub pairs.
        # See `default` preset for the rationale on initial_beta=0.
        "ADH_CN":            1.0e4,
        "ADH_CT":            1.0e5,
        "ADH_W":             1.0,
        "ADH_ETA":           2.0,
        "ADH_BONDING_RATE":  5.0,
        "ADH_INITIAL_BETA":  0.0,
    },
    "temflex175-thick": {
        # Sim TAPE_THICKNESS = full physical 0.178 mm. Larger IPC band
        # required. Bending stiffness ~8× higher than the half-thickness
        # presets above (cubic in t). Use this if the half-thickness
        # presets feel too floppy on bend.
        "HUB_R_OUTER":       0.0211,    # cardboard outer (measured ≈ 21.1 mm)
        "HUB_R_INNER":       0.01905,   # 1.5" core hole (= 19.05 mm)
        "HUB_HEIGHT":        0.020,     # ≈ tape width (measured 19 mm) + 1 mm sim margin
        "TAPE_WIDTH":        0.019,
        "TAPE_LENGTH":       0.75,    # ≈ 5cm slack after 5 turns
        "N_TURNS":           5,
        "TAPE_NZ":           10,
        "TAPE_YOUNGS":       5.0e7,    # physical 50 MPa, no comp needed
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    1.78e-4,
        "D_HAT_RATIO":       6.0 / 1.78,  # ≈ 3.371 → D_HAT = 6.0e-4
        "LAYER_THICKNESS":   4.5e-4,
        "BUFFER_LENGTH":     0.04,
        # RCC adhesion applied during wind to tape↔tape and tape↔hub pairs.
        # See `default` preset for the rationale on initial_beta=0.
        "ADH_CN":            1.0e4,
        "ADH_CT":            1.0e5,
        "ADH_W":             1.0,
        "ADH_ETA":           2.0,
        "ADH_BONDING_RATE":  5.0,
        "ADH_INITIAL_BETA":  0.0,
    },
    "temflex175-10turn": {
        # 10 turns instead of 5 — more impressive spool, slower to sim.
        "HUB_R_OUTER":       0.0211,    # cardboard outer (measured ≈ 21.1 mm)
        "HUB_R_INNER":       0.01905,   # 1.5" core hole (= 19.05 mm)
        "HUB_HEIGHT":        0.020,     # ≈ tape width (measured 19 mm) + 1 mm sim margin
        "TAPE_WIDTH":        0.019,
        "TAPE_LENGTH":       1.44,    # ≈ 5cm slack after 10 turns
        "N_TURNS":           10,
        "TAPE_NZ":           10,
        "TAPE_YOUNGS":       1.0e9,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "LAYER_THICKNESS":   2.5e-4,
        "BUFFER_LENGTH":     0.04,
        # RCC adhesion applied during wind to tape↔tape and tape↔hub pairs.
        # See `default` preset for the rationale on initial_beta=0.
        "ADH_CN":            1.0e4,
        "ADH_CT":            1.0e5,
        "ADH_W":             1.0,
        "ADH_ETA":           2.0,
        "ADH_BONDING_RATE":  5.0,
        "ADH_INITIAL_BETA":  0.0,
    },
}


# ----------------------------------------------------------------------
# Unwind presets — material + adhesion + SPC.
# Geometry / TAPE_THICKNESS / D_HAT come from the loaded asset (the
# values below are only defaults for legacy .npz files that don't
# record them). Use --asset NAME to choose which wind-saved asset to
# peel.
# ----------------------------------------------------------------------
UNWIND_PRESETS = {
    "default": {
        # Original unwind-demo defaults — the working config from before
        # the Temflex sweep. Matches the wind-demo "default" IPC band
        # (TAPE_THICKNESS=0.1mm, D_HAT=1mm), so it pairs naturally with
        # an asset saved by `wind --preset default`.
        "TAPE_YOUNGS":       1.0e9,
        "TAPE_POISSON":      0.4,
        "TAPE_MASS_DENSITY": 1300,
        # IPC dims — should match the wind asset's; left here as
        # fallback for legacy .npz files without saved IPC.
        "TAPE_THICKNESS":    1.0e-4,
        "D_HAT_RATIO":       10.0,        # → D_HAT = 1.0e-3
        # Adhesion (the user's iterated values)
        "ADH_CN":            5.0e1,
        "ADH_CT":            2.0e3,
        "ADH_W":             1.0,
        "ADH_ETA":           100.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        # Pull stiffness
        "SPC_STRENGTH":      1.0e9,
    },
    "soft-bond": {
        # Easy peel: low Cn / low W → β decays fast on minor load.
        # Membrane very stiff so tape doesn't stretch.
        "TAPE_YOUNGS":       1.0e9,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "ADH_CN":            1.0e1,
        "ADH_CT":            1.0e2,
        "ADH_W":             0.5,
        "ADH_ETA":           100.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        "SPC_STRENGTH":      1.0e9,
    },
    "medium-bond": {
        # Moderate peel — bond resists for a bit before failing.
        "TAPE_YOUNGS":       1.0e9,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "ADH_CN":            1.0e2,
        "ADH_CT":            1.0e3,
        "ADH_W":             1.0,
        "ADH_ETA":           100.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        "SPC_STRENGTH":      1.0e9,
    },
    "strong-bond": {
        # Hard peel: high Cn, high W → β survives substantial load.
        # Expect "cascade pull" — bond holds, then fails suddenly.
        "TAPE_YOUNGS":       1.0e9,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "ADH_CN":            1.0e3,
        "ADH_CT":            1.0e4,
        "ADH_W":             2.0,
        "ADH_ETA":           50.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        "SPC_STRENGTH":      1.0e10,
    },
    "stretchy": {
        # Membrane soft enough to actually stretch under pull. Useful
        # to verify the inextensible-recommendation: tape should
        # visibly elongate during peel, layer-2 may detach from layer-3.
        "TAPE_YOUNGS":       5.0e7,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "ADH_CN":            1.0e2,
        "ADH_CT":            1.0e3,
        "ADH_W":             1.0,
        "ADH_ETA":           100.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        "SPC_STRENGTH":      1.0e9,
    },
    "rigid": {
        # Near-inextensible tape (advisor's "membrane > 1e6 Pa" with huge
        # margin) + medium bond. Default recommended for clean visuals.
        "TAPE_YOUNGS":       5.0e9,
        "TAPE_POISSON":      0.45,
        "TAPE_MASS_DENSITY": 1300,
        "TAPE_THICKNESS":    9.0e-5,
        "D_HAT_RATIO":       40.0 / 9.0,  # ≈ 4.444 → D_HAT = 4.0e-4
        "ADH_CN":            1.0e2,
        "ADH_CT":            1.0e3,
        "ADH_W":             1.0,
        "ADH_ETA":           100.0,
        "ADH_BONDING_RATE":  1.0,
        "ADH_INITIAL_BETA":  1.0,
        "SPC_STRENGTH":      1.0e10,
    },
}


def parse_tape_cli(presets: dict, argv=None) -> dict:
    """Parse --preset / --set / --list / --asset / --list-assets.

    `presets` is the namespace to draw from: WIND_PRESETS for the wind
    demo, UNWIND_PRESETS for the unwind demo.

    Returns a resolved config dict (a copy of one presets entry with
    any --set overrides applied), plus internal markers
    `__preset_name__`, `__asset_arg__`, `__list_assets__`.

    `--set KEY=VALUE` accepts any key (preset-defined or not). Unknown
    CLI args are passed through (so polyscope etc. can add flags
    without breaking us).
    """
    import argparse
    import sys as _sys

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--preset", default="default", choices=list(presets),
                   help="named parameter bundle (see --list)")
    p.add_argument("--set", action="append", default=[], dest="overrides",
                   metavar="KEY=VALUE",
                   help="override any preset key (repeatable)")
    p.add_argument("--asset", default=None, metavar="NAME_OR_PATH",
                   help="wind demo: save target; unwind demo: load source. "
                        "Bare name (e.g. 'my_trial') → ASSET_DIR/my_trial.npz; "
                        "path with '/' or '.npz' → used as-is. "
                        "Default: ASSET_DIR/{preset_name}.npz.")
    p.add_argument("--list", action="store_true",
                   help="print all presets and their values, then exit")
    p.add_argument("--list-assets", action="store_true", dest="list_assets",
                   help="print all saved .npz assets in ASSET_DIR, then exit")
    args, _unknown = p.parse_known_args(argv)

    if args.list:
        for name, vals in presets.items():
            print(f"[{name}]")
            for k, v in vals.items():
                if isinstance(v, float):
                    print(f"  {k:<20s} = {v:.4g}")
                else:
                    print(f"  {k:<20s} = {v}")
            print()
        _sys.exit(0)

    cfg = dict(presets[args.preset])
    explicitly_set = set()
    for kv in args.overrides:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got: {kv}")
        key, _, val = kv.partition("=")
        if key in cfg and isinstance(cfg[key], int) and not isinstance(cfg[key], bool):
            try:
                cfg[key] = int(val)
            except ValueError:
                cfg[key] = int(float(val))
        else:
            try:
                cfg[key] = float(val)
            except ValueError:
                cfg[key] = val
        explicitly_set.add(key)

    # Auto-derive D_HAT unless `--set D_HAT=X` is given. Two candidates,
    # take the smaller (conservative against spurious self-contact):
    #
    #   (1) thickness-based:  D_HAT = D_HAT_RATIO × TAPE_THICKNESS
    #         Base value tuned per material/preset. Doesn't know about mesh
    #         resolution — wins when the mesh is coarse enough.
    #
    #   (2) mesh-aware clamp: D_HAT = D_HAT_SAFETY × h/√2 − 2·TAPE_THICKNESS
    #         where h = min(TAPE_WIDTH/TAPE_NZ, TAPE_LENGTH/TAPE_NX) is the
    #         smaller in-plane cell side. Enforces that the IPC active band
    #         [2t, 2t+d_hat] stays narrower than the cell's interior
    #         self-distance ≈ h/√2, so a flat tape at rest doesn't trigger
    #         neighbour-cell self-contact. D_HAT_SAFETY defaults to 0.8.
    #         TAPE_NX falls back to the square-cell rule
    #         (round(TAPE_LENGTH/(TAPE_WIDTH/TAPE_NZ))) if not given —
    #         matches what rcc_adhesive_tape_winding_demo.py uses.
    #
    # So increasing TAPE_NZ (refining the mesh) shrinks the mesh-aware
    # candidate, and once it drops below the thickness-based one it takes
    # over → D_HAT auto-tracks NZ without the user thinking about it.
    if "D_HAT" not in explicitly_set:
        candidates = []
        if "D_HAT_RATIO" in cfg and "TAPE_THICKNESS" in cfg:
            candidates.append(float(cfg["D_HAT_RATIO"]) * float(cfg["TAPE_THICKNESS"]))
        if all(k in cfg for k in ("TAPE_THICKNESS", "TAPE_WIDTH",
                                  "TAPE_LENGTH", "TAPE_NZ")):
            nz = float(cfg["TAPE_NZ"])
            if "TAPE_NX" in cfg:
                nx = float(cfg["TAPE_NX"])
            else:
                # square-cell default (mirrors the demo's derivation)
                nx = max(1.0, round(float(cfg["TAPE_LENGTH"]) * nz
                                    / float(cfg["TAPE_WIDTH"])))
            h = min(float(cfg["TAPE_WIDTH"]) / nz,
                    float(cfg["TAPE_LENGTH"]) / nx)
            safety = float(cfg.get("D_HAT_SAFETY", 0.8))
            cfg["D_HAT_SAFETY"] = safety   # surface in --list
            d_hat_mesh = safety * h / np.sqrt(2.0) - 2.0 * float(cfg["TAPE_THICKNESS"])
            if d_hat_mesh > 0.0:
                candidates.append(d_hat_mesh)
        if candidates:
            cfg["D_HAT"] = min(candidates)

    cfg["__preset_name__"] = args.preset
    cfg["__asset_arg__"] = args.asset
    cfg["__list_assets__"] = args.list_assets
    # Record which keys the user touched on the CLI — used by the unwind
    # demo so asset-saved values (D_HAT, TAPE_THICKNESS, …) override the
    # preset's literals for keys the user didn't explicitly --set.
    cfg["__explicit__"] = explicitly_set
    return cfg


def resolve_asset_path(asset_dir: str, asset_arg: str | None, preset_name: str) -> str:
    """Compute the .npz path for save/load.

    - `asset_arg` None → `{asset_dir}/{preset_name}.npz`
    - bare name (no `/`, no `.npz`) → `{asset_dir}/{asset_arg}.npz`
    - path containing `/` or ending in `.npz` → used as-is
    """
    if not asset_arg:
        return os.path.join(asset_dir, f"{preset_name}.npz")
    if "/" in asset_arg or asset_arg.endswith(".npz"):
        return asset_arg
    return os.path.join(asset_dir, f"{asset_arg}.npz")


def list_assets(asset_dir: str) -> None:
    """Print every .npz file in `asset_dir` with a preset-name hint if the
    asset has one stored under params['__preset_name__']."""
    import os as _os
    import sys as _sys
    if not _os.path.isdir(asset_dir):
        print(f"(no asset directory yet: {asset_dir})")
        _sys.exit(0)
    npzs = sorted(f for f in _os.listdir(asset_dir) if f.endswith(".npz"))
    if not npzs:
        print(f"(no .npz files in {asset_dir})")
        _sys.exit(0)
    print(f"Assets in {asset_dir}:")
    for f in npzs:
        path = _os.path.join(asset_dir, f)
        try:
            _, _, params, pair_state = load_tape_asset(path)
            preset = params.get("__preset_name__", "?")
            extra = []
            if "TAPE_THICKNESS" in params:
                extra.append(f"t={params['TAPE_THICKNESS']*1e3:.3f}mm")
            if "D_HAT" in params:
                extra.append(f"d_hat={params['D_HAT']*1e3:.3f}mm")
            if "TAPE_YOUNGS" in params:
                extra.append(f"E={params['TAPE_YOUNGS']:.1e}")
            if pair_state is not None:
                _keys, _betas = pair_state
                extra.append(f"β:n={len(_betas)},mean={_betas.mean():.2f}")
            extras = f"  ({', '.join(extra)})" if extra else ""
            print(f"  {f}  [preset={preset}]{extras}")
        except Exception as e:
            print(f"  {f}  (could not read: {e})")
    _sys.exit(0)
