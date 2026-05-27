"""
Guided-winding asset demo (Option C).

A straight tape strip is positioned tangent to a FIXED ABD hub:
  - The anchor end (i=0) starts in contact with the hub surface at angle 0.
  - The free end (i=NX) is pinned by `SoftPositionConstraint` and animated
    along a spiral trajectory around the hub axis (+z). As the free end
    orbits + spirals inward, the tape between it and the anchor drapes
    onto the hub and bonds layer-by-layer via v3 single-sided adhesion.

Timeline:
    Phase 0 (preheat) : both ends pinned; adhesion β rises to 1 on anchor.
    Phase 1 (wind)    : anchor SPC released; free-end target follows the
                        winding spiral. Tape physically wraps onto hub.
    Phase 2 (settle)  : free-end stops; system relaxes.

Run:
    python python/examples/rcc_adhesive_tape_winding_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This example requires `polyscope`. Install it with `pip install polyscope`."
    ) from exc

from uipc import (
    Logger,
    Matrix4x4,
    Engine,
    World,
    Scene,
    SceneIO,
    Animation,
    view,
    builtin,
)
from uipc.geometry import trimesh, label_surface, mesh_partition
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
    SoftPositionConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- CLI / preset resolution ----
# Pick a parameter bundle with `--preset`; tweak any single key with
# `--set KEY=VALUE`; list all bundles with `--list`. See
# `tape_asset_lib.TAPE_PRESETS` for what each preset contains.
_CFG = L.parse_tape_cli(L.WIND_PRESETS)
print(f"[wind] preset={_CFG['__preset_name__']}: "
      f"E={_CFG['TAPE_YOUNGS']:.1e} Pa, ν={_CFG['TAPE_POISSON']}, "
      f"ρ={_CFG['TAPE_MASS_DENSITY']} kg/m³, t={_CFG['TAPE_THICKNESS']*1e3:.3f} mm, "
      f"d_hat={_CFG['D_HAT']*1e3:.3f} mm, LAYER={_CFG['LAYER_THICKNESS']*1e3:.3f} mm, "
      f"hub R={_CFG['HUB_R_OUTER']*1e3:.1f} mm × W={_CFG['TAPE_WIDTH']*1e3:.1f} mm, "
      f"N_TURNS={_CFG['N_TURNS']}")

# ---- hub geometry (from preset) ----
HUB_R_OUTER       = _CFG["HUB_R_OUTER"]
HUB_R_INNER       = _CFG["HUB_R_INNER"]
HUB_HEIGHT        = _CFG["HUB_HEIGHT"]
N_RADIAL          = 48

# ---- tape geometry / discretization ----
# TAPE_NZ comes from the preset (override with --set TAPE_NZ=N).
# TAPE_NX defaults to the square-cell rule (TAPE_LENGTH / TAPE_DS); if
# the preset / CLI supplies TAPE_NX explicitly, that wins.
N_TURNS           = int(_CFG["N_TURNS"])
TAPE_LENGTH       = _CFG["TAPE_LENGTH"]
TAPE_WIDTH        = _CFG["TAPE_WIDTH"]
TAPE_NZ           = int(_CFG.get("TAPE_NZ", 10))
TAPE_DS           = TAPE_WIDTH / TAPE_NZ
_nx_override      = _CFG.get("TAPE_NX")
TAPE_NX           = (int(_nx_override) if _nx_override
                     else max(1, int(round(TAPE_LENGTH / TAPE_DS))))

# ---- IPC contact band (from preset; see preset block for the
# 2t < LAYER < 2t+d_hat constraint chain — LAYER is used by procedural
# demo, not here; wind grows layers implicitly via 2·TAPE_THICKNESS) ----
D_HAT             = _CFG["D_HAT"]
TAPE_THICKNESS    = _CFG["TAPE_THICKNESS"]

# ---- tape material (from preset) ----
TAPE_YOUNGS       = _CFG["TAPE_YOUNGS"]
TAPE_POISSON      = _CFG["TAPE_POISSON"]
TAPE_MASS_DENSITY = _CFG["TAPE_MASS_DENSITY"]

# ---- hub material ----
HUB_KAPPA         = 1.0e8
HUB_MASS_DENSITY  = 1000.0

# ---- adhesion ----
ADH_CN            = 1.0e4
ADH_CT            = 1.0e5
ADH_W             = 1.0
ADH_ETA           = 2.0
ADH_BONDING_RATE  = 1.0
ADH_INITIAL_BETA  = 1.0

# ---- SPC ----
SPC_STRENGTH      = 1.0e5        # matches rcc_adhesive_cloth_peel_demo.py
# Free tail strategy: every vertex whose arc-length from the anchor is
# more than (L_wound + BUFFER_LENGTH) is pinned along the tangent line
# from the wrap-off point. The unpinned BUFFER_LENGTH worth of tape
# right at the wrap-off region is where bending onto the hub actually
# happens. Smaller buffer → sharper bend (stress concentration);
# larger buffer → tape doesn't actually contact the hub in time.
BUFFER_LENGTH     = 0.04         # ~ 1 hub circumference's worth of slack
# Anchor strip: the first ANCHOR_ROWS mesh rows (at i=0..ANCHOR_ROWS-1)
# are locked at their rest pose for the ENTIRE simulation — mimics a
# tape with a permanent glue-tab. Specified in *cell count* rather than
# meters so the topology footprint stays consistent across NZ values
# (varying NZ changes dy, but the anchor still pins the same number of
# rows). 3 rows ≈ 5–6 mm at NZ=10; the exact metric depends on
# TAPE_DS = TAPE_WIDTH / TAPE_NZ.
ANCHOR_ROWS       = 3

# ---- timeline (dt=0.01) ----
# Free-end tangential speed = r · ω. At start, r ≈ TAPE_LENGTH, so a
# fast schedule launches the free end at ~5 m/s — way past d_hat per
# step. WIND_FRAMES = 1500 (= 3 s/turn) keeps it under ~2 m/s and
# gives IPC time to engage each new contact pair.
PREHEAT_FRAMES    = 30
WIND_FRAMES       = 1500
SETTLE_FRAMES     = 60
TOTAL_FRAMES      = PREHEAT_FRAMES + WIND_FRAMES + SETTLE_FRAMES
THETA_END         = 2.0 * np.pi * N_TURNS

# Anchor radius: inside the tape-hub IPC active band so adhesion fires.
R_ANCHOR          = HUB_R_OUTER + TAPE_THICKNESS + 0.5 * D_HAT

ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output",
    "rcc_adhesive_tape_winding")
if _CFG.get("__list_assets__"):
    L.list_assets(ASSET_DIR)
ASSET_OUT_PATH = L.resolve_asset_path(
    ASSET_DIR, _CFG.get("__asset_arg__"), _CFG["__preset_name__"])
print(f"[wind] save target: {ASSET_OUT_PATH}")


# ----------------------------------------------------------------------
# Trajectory math
# ----------------------------------------------------------------------
def R_eff(theta: float) -> float:
    """Effective wrap radius — grows by 2·TAPE_THICKNESS per turn so each
    new layer settles on top of the previous one."""
    return HUB_R_OUTER + TAPE_THICKNESS * (1.0 + theta / np.pi)


def L_wound(theta: float) -> float:
    """Total tape arc length wound on hub after angle θ (analytical
    integral of R_eff)."""
    return HUB_R_OUTER * theta + TAPE_THICKNESS * (theta + theta * theta / (2.0 * np.pi))


def free_end_center(theta: float) -> np.ndarray:
    """Free-end center position (xy plane, z=0) at wind angle θ.

    Wrap-off at angle θ on hub: P_wrap = R_eff(θ)·(cosθ, sinθ).
    Tangent (CCW): T = (-sinθ, cosθ).
    Free tail length: L_tape - L_wound(θ), extended along T.
    """
    r = R_eff(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    p_wrap = np.array([r * cos_t, r * sin_t, 0.0])
    tangent = np.array([-sin_t, cos_t, 0.0])
    l_free = max(TAPE_LENGTH - L_wound(theta), 0.0)
    return p_wrap + l_free * tangent


def theta_at_frame(f: int) -> float:
    if f < PREHEAT_FRAMES:
        return 0.0
    if f < PREHEAT_FRAMES + WIND_FRAMES:
        t = (f - PREHEAT_FRAMES) / WIND_FRAMES
        return t * THETA_END
    return THETA_END


def phase_at(f: int) -> str:
    if f < PREHEAT_FRAMES:
        return "preheat"
    if f < PREHEAT_FRAMES + WIND_FRAMES:
        return "wind"
    return "settle"


# ----------------------------------------------------------------------
# Build the tangent-oriented tape mesh directly (avoids modifying
# `make_flat_tape`'s xz-plane output in-place).
# ----------------------------------------------------------------------
def _make_tangent_tape(R_anchor, length, width, NX, NZ):
    """Flat shell with long axis +y, width centered on z=0, normal +x.

    Vertices: (R_anchor, i·dy, j·dz - W/2) for i ∈ [0,NX], j ∈ [0,NZ].
    Triangle winding chosen so face normals = +x (radially outward
    from the hub center). Caller does `set_sticky_side(tape, -1)` to
    make the sticky face point inward (toward hub).
    """
    dy = length / NX
    dz = width / NZ
    n_width = NZ + 1
    verts = np.empty(((NX + 1) * n_width, 3), dtype=np.float64)
    for i in range(NX + 1):
        for j in range(n_width):
            verts[i * n_width + j] = (R_anchor, i * dy, j * dz - 0.5 * width)

    def vid(i, j):
        return i * n_width + j

    tris = []
    # Winding analysis (face in yz plane at constant x):
    #   edges (i,j)→(i+1,j) = +y;  (i,j)→(i,j+1) = +z;  cross +y×+z = +x ✓
    # tri 1: (v00, v10, v11). v10-v00 = +y·dy. v11-v00 = (+y·dy, +z·dz).
    #   cross(+y·dy, +y·dy + +z·dz) = +y×+y + +y×+z·dz = 0 + +x·dz ⇒ +x.
    # tri 2: (v00, v11, v01). v11-v00 = (+y·dy, +z·dz). v01-v00 = +z·dz.
    #   cross((+y·dy, +z·dz), +z·dz) = +y×+z·dy·dz + 0 ⇒ +x.
    for i in range(NX):
        for j in range(NZ):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    sc = trimesh(verts, np.asarray(tris, dtype=np.int32))
    label_surface(sc)
    mesh_partition(sc, 16)
    return sc, verts


def vid(i, j):
    return i * (TAPE_NZ + 1) + j


# ----------------------------------------------------------------------
# Build scene
# ----------------------------------------------------------------------
def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def _hub_axis_to_z() -> Matrix4x4:
    """R_x(π/2) so the hub's natural +y axis maps to +z."""
    Rx = np.eye(4, dtype=np.float64)
    Rx[1, 1] =  0.0;  Rx[1, 2] = -1.0
    Rx[2, 1] =  1.0;  Rx[2, 2] =  0.0
    return _mat4_to_uipc(Rx)


def build_demo(adhesion_on: bool = True):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]   # no gravity — pure winding
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = D_HAT
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    spc = SoftPositionConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact  = tabular.default_element()
    tape_contact = tabular.create("tape")
    tabular.insert(tape_contact, tape_contact, 0.5, 1.0e9)
    tabular.insert(tape_contact, hub_contact,  0.5, 1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )
        adhesive.set(
            tabular, tape_contact, tape_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0,
            initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )
        adhesive.set(
            tabular, tape_contact, hub_contact,
            Cn=ADH_CN, Ct=ADH_CT, W=ADH_W, eta=ADH_ETA,
            bonding_rate=ADH_BONDING_RATE, p0=0.0,
            initial_beta=ADH_INITIAL_BETA,
            enabled=True,
        )

    # ---- hub (fixed in place) ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=N_RADIAL,
        center=(0.0, 0.0, 0.0))
    abd.apply_to(hub_sc, HUB_KAPPA, HUB_MASS_DENSITY)
    hub_contact.apply_to(hub_sc)
    # Rotate the ABD instance so the hub axis aligns with +z (wrap plane = xy).
    view(hub_sc.transforms())[0] = _hub_axis_to_z()
    # Lock the hub via builtin.is_fixed (no animator, no STC needed).
    view(hub_sc.instances().find(builtin.is_fixed))[0] = 1

    hub_obj = scene.objects().create("hub")
    hub_geo, _ = hub_obj.geometries().create(hub_sc)

    # ---- tape (tangent-oriented straight strip) ----
    tape_sc, tape_verts = _make_tangent_tape(
        R_anchor=R_ANCHOR,
        length=TAPE_LENGTH, width=TAPE_WIDTH,
        NX=TAPE_NX, NZ=TAPE_NZ)
    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    nhs.apply_to(tape_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    tape_contact.apply_to(tape_sc)
    spc.apply_to(tape_sc, SPC_STRENGTH)
    if adhesion_on:
        # tape normal = +x (outward). sticky = -1 → sticky face is -x = inward
        # (toward hub center). Matches the wound-tape v3 gate semantics.
        RCCAdhesive.set_sticky_side(tape_sc, -1)

    tape_obj = scene.objects().create("tape")
    tape_geo, _ = tape_obj.geometries().create(tape_sc)

    # Snapshot rest positions for animator z-lookup.
    rest_positions = tape_verts.copy()
    # Per-row arc length from anchor (i=0): row i is at arc length i·dy
    # along the (initially straight) tape.
    dy_tape = TAPE_LENGTH / TAPE_NX
    arc_at_row = np.arange(TAPE_NX + 1) * dy_tape
    # Anchor strip: rows in [0, ANCHOR_ROWS) — always pinned at rest pose.
    # ANCHOR_ROWS is a module-level cell count (NZ-independent topology).

    def animate_tape(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        f = max(info.frame() - 1, 0)
        is_c = view(geo.vertices().find(builtin.is_constrained))
        aim  = view(geo.vertices().find(builtin.aim_position))
        is_c[:] = 0

        # 1) Anchor strip — always locked at rest pose. The very start of
        #    the tape never moves, even during the wind animation.
        for i in range(ANCHOR_ROWS):
            for j in range(TAPE_NZ + 1):
                k = vid(i, j)
                is_c[k] = 1
                aim[k] = rest_positions[k].reshape(3, 1)

        # 2) Tail — pinned along the tangent line beyond (l_wound + BUFFER).
        theta = theta_at_frame(f)
        l_wound = L_wound(theta)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r = R_eff(theta)
        p_wrap = np.array([r * cos_t, r * sin_t])
        tangent = np.array([-sin_t, cos_t])

        # i_pin_start is the first row beyond the active-bend window. Clamp
        # so we don't pin into the anchor strip if l_wound is still tiny.
        i_pin_start = int(np.ceil((l_wound + BUFFER_LENGTH) / dy_tape))
        i_pin_start = max(ANCHOR_ROWS, min(i_pin_start, TAPE_NX + 1))

        if i_pin_start <= TAPE_NX:
            s_offsets = arc_at_row[i_pin_start:] - l_wound       # (N_pinned,)
            xy = p_wrap[None, :] + s_offsets[:, None] * tangent[None, :]
            for ii, i in enumerate(range(i_pin_start, TAPE_NX + 1)):
                x, y = xy[ii]
                for j in range(TAPE_NZ + 1):
                    k = vid(i, j)
                    is_c[k] = 1
                    aim[k] = np.array(
                        [x, y, rest_positions[k, 2]],
                        dtype=np.float64,
                    ).reshape(3, 1)

    scene.animator().insert(tape_obj, animate_tape)

    world.init(scene)
    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "hub_geo": hub_geo,
        "tape_geo": tape_geo,
    }


# ----------------------------------------------------------------------
# Polyscope viewer
# ----------------------------------------------------------------------
def run_demo():
    state = {"adhesion_on": True, "saved": False}
    sim = build_demo(state["adhesion_on"])

    # Trajectory sanity check — print free-end positions at sampled angles.
    print(f"tape mesh: NX={TAPE_NX}, NZ={TAPE_NZ}  "
          f"(dy={TAPE_LENGTH/TAPE_NX*1000:.3f} mm, dz={TAPE_WIDTH/TAPE_NZ*1000:.3f} mm)")
    print(f"L_tape={TAPE_LENGTH:.4f} m, "
          f"L_wound(θ_end)={L_wound(THETA_END):.4f} m, "
          f"slack={TAPE_LENGTH - L_wound(THETA_END):.4f} m")
    for k in range(N_TURNS + 1):
        th = k * 2.0 * np.pi
        p = free_end_center(th)
        print(f"  θ={th:6.3f} ({k} turns): R_eff={R_eff(th):.5f}, "
              f"L_free={max(TAPE_LENGTH - L_wound(th), 0.0):.4f}, "
              f"free_end=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "wound_tape",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(0.3)

    ui = {"run": False}

    def update_visual():
        nonlocal mesh
        merged = fresh_surface()
        v = merged.positions().view().reshape(-1, 3)
        t = merged.triangles().topo().view().reshape(-1, 3)
        if mesh.n_vertices() != v.shape[0]:
            ps.remove_surface_mesh("wound_tape")
            mesh = ps.register_surface_mesh("wound_tape", v, t)
            mesh.set_edge_width(0.3)
        else:
            mesh.update_vertex_positions(v)

    def step_once():
        if sim["world"].frame() >= TOTAL_FRAMES:
            ui["run"] = False
            return
        sim["world"].advance()
        if not sim["world"].is_valid():
            ui["run"] = False
            return
        sim["world"].retrieve()
        update_visual()

    def save_asset():
        os.makedirs(os.path.dirname(ASSET_OUT_PATH), exist_ok=True)
        hub_geo = sim["hub_geo"].geometry()
        tape_geo = sim["tape_geo"].geometry()
        hub_T = np.array(view(hub_geo.transforms()), copy=True).reshape(4, 4)
        tape_pos = np.array(view(tape_geo.positions()), copy=True).reshape(-1, 3)
        params = dict(
            # geometry (needed to rebuild SC topology on load)
            HUB_R_OUTER=HUB_R_OUTER, HUB_R_INNER=HUB_R_INNER,
            HUB_HEIGHT=HUB_HEIGHT,
            N_TURNS=N_TURNS, TAPE_LENGTH=TAPE_LENGTH,
            TAPE_WIDTH=TAPE_WIDTH,
            TAPE_NX=TAPE_NX, TAPE_NZ=TAPE_NZ,
            # IPC numerics — unwind MUST match these to keep the active
            # band aligned. Stored so the unwind demo can warn on mismatch.
            TAPE_THICKNESS=TAPE_THICKNESS, D_HAT=D_HAT,
            # material params (informational; unwind can use different ones)
            TAPE_YOUNGS=TAPE_YOUNGS, TAPE_POISSON=TAPE_POISSON,
            TAPE_MASS_DENSITY=TAPE_MASS_DENSITY,
            # provenance
            __preset_name__=_CFG["__preset_name__"],
        )
        L.save_tape_asset(ASSET_OUT_PATH, hub_T, tape_pos, params)
        print(f"saved wound-tape asset → {ASSET_OUT_PATH}")
        state["saved"] = True

    def reset():
        nonlocal sim
        sim = build_demo(state["adhesion_on"])
        update_visual()
        state["saved"] = False

    def on_update():
        if psim.Button("run / pause"):
            ui["run"] = not ui["run"]
        psim.SameLine()
        if psim.Button("step"):
            step_once()
        psim.SameLine()
        if psim.Button(f"adhesion: {'ON' if state['adhesion_on'] else 'OFF'}"):
            state["adhesion_on"] = not state["adhesion_on"]
        psim.SameLine()
        if psim.Button("reset"):
            reset()
        psim.SameLine()
        if psim.Button("save asset"):
            save_asset()

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        theta = theta_at_frame(f)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}    Phase: {phase_at(f)}")
        psim.Text(f"θ = {theta:.2f} rad   ({theta/(2*np.pi):.2f} turns)")
        psim.Text(f"L_wound = {L_wound(theta):.3f} / {TAPE_LENGTH} m")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        if state["saved"]:
            psim.Text(f"asset @ {ASSET_OUT_PATH}")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
