"""
Sim-roll showcase (Option A, rod-driven).

A flat sticky tape strip lies on the ground. An ABD hub (a flat ring
with its axis along +z) is threaded onto a horizontal ABD rod, and the
rod is animated by SoftTransformConstraint:
    Phase 0 (press) : rod descends; the hub hangs from it under gravity
                      until the hub bottom enters the tape's d_hat band.
    Phase 1 (bond)  : rod still; RCC adhesion β rises to 1 at the first
                      hub-tape contact pairs.
    Phase 2 (roll)  : rod translates along +x. The hub is pushed
                      forward via the inner-cylinder contact; hub-tape
                      friction + adhesion induce rolling. ω = v / r is
                      self-organized — no need to prescribe rotation as
                      the wrapped tape grows the effective radius.
    Phase 3 (settle): rod still; system relaxes.

Run:
    python python/examples/rcc_adhesive_tape_roll_demo.py
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
from uipc.geometry import ground
from uipc.constitution import (
    AffineBodyConstitution,
    NeoHookeanShell,
    SoftTransformConstraint,
    ElasticModuli2D,
    RCCAdhesive,
)

sys.path.insert(0, os.path.dirname(__file__))
import tape_asset_lib as L

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
from asset import AssetDir


# ---- geometry ----
TAPE_LENGTH    = 1.2
TAPE_WIDTH     = 0.04
TAPE_NX        = 300
TAPE_NZ        = 10

HUB_R_OUTER    = 0.02
HUB_R_INNER    = 0.018
HUB_HEIGHT     = 0.044     # slightly wider than the tape
N_RADIAL       = 48

ROD_R          = 0.006     # < HUB_R_INNER → radial clearance = 0.006 m
ROD_LENGTH     = 0.12      # extends well beyond the hub width (0.044)
ROD_N_RADIAL   = 24

# ---- tape material (NeoHookeanShell) ----
# `tape_thickness` is the SHELL's virtual thickness: it scales mass and
# bending stiffness, but the tape stays a geometrically-2D shell — the
# effective inter-layer / tape-vs-hub gap is set by D_HAT below.
TAPE_YOUNGS       = 1.0e7   # Pa
TAPE_POISSON      = 0.4
TAPE_MASS_DENSITY = 2.0e2   # kg/m³ (NeoHookeanShell default)
TAPE_THICKNESS    = 0.0001   # m, virtual shell thickness

# ---- vertical positions ----
GROUND_Y       = 0.0
TAPE_Y         = GROUND_Y + 0.01        # tape sits 1 cm above ground

# When the hub hangs from the rod under gravity, the top of the hub's
# inner cylinder rests on top of the rod, so:
#   hub_y = rod_y - (HUB_R_INNER - ROD_R)
HANG_OFFSET    = HUB_R_INNER - ROD_R     # = 0.006 m

# Pressed-down state: hub bottom should sit 1 mm "below" tape surface so
# the barrier is inside d_hat (5 mm), adhesion fires, and the hub is
# weight-supported by the tape barrier.
HUB_PRESS_Y    = TAPE_Y + HUB_R_OUTER - 0.001
ROD_PRESS_Y    = HUB_PRESS_Y - HANG_OFFSET - 0.003

# Initial (rod high in the air, hub starts concentric so there is NO
# rod-hub overlap at init; gravity will pull the hub down to the natural
# hang position within the first few frames before PRESS_START).
ROD_INITIAL_Y  = ROD_PRESS_Y + 0.10
HUB_INITIAL_Y  = ROD_INITIAL_Y               # concentric → zero contact at frame 0

# Start the hub just in front of the tape (its rightmost contact is one
# R_outer left of the tape's leading edge), so it begins on bare ground
# and rolls into the tape from frame ~BOND_END onward.
HUB_FRONT_GAP  = 0.02
X_START        = -0.5 * TAPE_LENGTH - HUB_R_OUTER - HUB_FRONT_GAP

# ---- contact ----
D_HAT          = 0.001

# ---- timeline (dt=0.01) ----
PRESS_START    = 20       # leave a few frames for the hub to settle onto the rod
PRESS_END      = 70       # 50 frames to descend smoothly
BOND_END       = 200      # frames of hold-still to let β rise
ROLL_FRAMES    = 500      # 5 seconds of rolling
ROLL_END       = BOND_END + ROLL_FRAMES
SETTLE_END     = ROLL_END + 50
TOTAL_FRAMES   = SETTLE_END

# ---- roll kinematics ----
ROLL_DISTANCE  = 1.24     # rod travels +x by this much during the roll phase
# Hub rotation is NOT prescribed; it emerges from rod-hub barrier
# pushing the hub forward + hub-tape friction/adhesion resisting slip
# at the bottom, which torques the hub into rolling.


def smooth01(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def rod_x_at(frame: int) -> float:
    if frame < BOND_END:
        return X_START
    if frame < ROLL_END:
        t = (frame - BOND_END) / ROLL_FRAMES
        return X_START + smooth01(t) * ROLL_DISTANCE
    return X_START + ROLL_DISTANCE


def rod_y_at(frame: int) -> float:
    if frame < PRESS_START:
        return ROD_INITIAL_Y
    if frame < PRESS_END:
        t = (frame - PRESS_START) / (PRESS_END - PRESS_START)
        return ROD_INITIAL_Y + smooth01(t) * (ROD_PRESS_Y - ROD_INITIAL_Y)
    return ROD_PRESS_Y


def phase_at(frame: int) -> str:
    if frame < PRESS_START: return "init"
    if frame < PRESS_END:   return "press"
    if frame < BOND_END:    return "bond"
    if frame < ROLL_END:    return "roll"
    return "settle"


def _mat4_to_uipc(M: np.ndarray) -> Matrix4x4:
    out = Matrix4x4.Identity()
    for i in range(4):
        for j in range(4):
            out[i, j] = M[i, j]
    return out


def make_hub_initial_transform(x: float, y: float) -> Matrix4x4:
    """Place the ring at (x, y, 0) and rotate its natural +y axis to +z
    (so the rolling plane is xy and the hub rolls along x). No rolling
    angle — hub rotation evolves dynamically."""
    Rx = np.eye(4, dtype=np.float64)
    Rx[1, 1] =  0.0;  Rx[1, 2] = -1.0
    Rx[2, 1] =  1.0;  Rx[2, 2] =  0.0
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    return _mat4_to_uipc(T @ Rx)


def make_rod_transform(x: float, y: float) -> Matrix4x4:
    """Rod axis is already +z natively — only translation needed."""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    return _mat4_to_uipc(T)


def build_demo(adhesion_on: bool = True):
    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = D_HAT
    config["extras"]["strict_mode"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    abd = AffineBodyConstitution()
    nhs = NeoHookeanShell()
    stc = SoftTransformConstraint()

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0e9)
    hub_contact    = tabular.default_element()       # the hub uses the default element
    tape_contact   = tabular.create("tape")
    rod_contact    = tabular.create("rod")
    ground_contact = tabular.create("ground")
    tabular.insert(tape_contact, tape_contact,   0.5, 1.0e9)
    tabular.insert(tape_contact, hub_contact,    0.5, 1.0e9)
    tabular.insert(tape_contact, ground_contact, 0.5, 1.0e9)
    # Rod-hub: LOW friction. The rod's job is to push the hub forward via
    # inner-cylinder barrier; the hub should rotate freely about the rod
    # so ground-tape friction can do the rolling.
    tabular.insert(rod_contact,  hub_contact,    0.05, 1.0e9)
    # Rod-tape: should not normally contact, but keep a sane default.
    tabular.insert(rod_contact,  tape_contact,   0.1,  1.0e9)
    tabular.insert(rod_contact,  rod_contact,    0.5,  1.0e9)
    tabular.insert(rod_contact,  ground_contact, 0.3,  1.0e9)
    tabular.insert(hub_contact,  ground_contact, 0.3,  1.0e9)

    if adhesion_on:
        adhesive = RCCAdhesive()
        adhesive.default_model(
            tabular,
            Cn=0.0, Ct=0.0, W=0.0, eta=2.0,
            bonding_rate=0.0, p0=0.0, initial_beta=0.0,
            enabled=False,
        )
        adhesive.set(
            tabular, tape_contact, hub_contact,
            Cn=1.0e4, Ct=1.0e5, W=1.0, eta=2.0,
            bonding_rate=1.0, p0=0.0, initial_beta=1.0,
            enabled=True,
        )
        adhesive.set(
            tabular, tape_contact, tape_contact,
            Cn=1.0e4, Ct=1.0e5, W=1.0, eta=2.0,
            bonding_rate=1.0, p0=0.0, initial_beta=1.0,
            enabled=True,
        )
        # Weak tape-ground adhesion: ~30× weaker than the tape-hub bond.
        # Just enough to hold the leading edge of the tape down so it
        # doesn't curl up in front of the rolling hub.
        adhesive.set(
            tabular, tape_contact, ground_contact,
            Cn=3.0e2, Ct=3.0e3, W=0.1, eta=2.0,
            bonding_rate=1.0, p0=0.0, initial_beta=1.0,
            enabled=True,
        )

    # ---- ground ----
    ground_sc = ground(GROUND_Y)
    ground_contact.apply_to(ground_sc)
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground_sc)

    # ---- hub (free ABD; no transform constraint) ----
    hub_sc = L.make_ring_hub(
        R_outer=HUB_R_OUTER, R_inner=HUB_R_INNER,
        height=HUB_HEIGHT, n_radial=N_RADIAL,
        center=(0.0, 0.0, 0.0))     # rest-pose origin; transformed below
    abd.apply_to(hub_sc, 1.0e8, 1000.0)
    hub_contact.apply_to(hub_sc)
    view(hub_sc.transforms())[0] = make_hub_initial_transform(X_START, HUB_INITIAL_Y)
    hub_obj = scene.objects().create("hub")
    hub_obj.geometries().create(hub_sc)

    # ---- rod (ABD + SoftTransformConstraint, animated) ----
    rod_sc = L.make_rod(
        R=ROD_R, length=ROD_LENGTH, n_radial=ROD_N_RADIAL,
        center=(0.0, 0.0, 0.0))
    abd.apply_to(rod_sc, 1.0e8, 1000.0)
    rod_contact.apply_to(rod_sc)
    stc.apply_to(rod_sc, np.array([1.0e8, 1.0e8], dtype=np.float64))
    view(rod_sc.transforms())[0] = make_rod_transform(X_START, ROD_INITIAL_Y)
    rod_obj = scene.objects().create("rod")
    rod_obj.geometries().create(rod_sc)

    # ---- flat tape ----
    tape_sc = L.make_flat_tape(
        length=TAPE_LENGTH, width=TAPE_WIDTH,
        NX=TAPE_NX, NZ=TAPE_NZ, base_y=TAPE_Y)
    moduli = ElasticModuli2D.youngs_poisson(TAPE_YOUNGS, TAPE_POISSON)
    nhs.apply_to(tape_sc, moduli,
                 mass_density=TAPE_MASS_DENSITY,
                 thickness=TAPE_THICKNESS)
    tape_contact.apply_to(tape_sc)
    if adhesion_on:
        # sticky=0 (omnidirectional) so the bottom face can also adhere
        # to the ground via the weak tape-ground pair. In this demo the
        # tape's two faces are naturally separated by geometry — top
        # only ever sees the hub / other tape layers, bottom only ever
        # sees the ground — so the v3 orientation gate isn't needed
        # here.
        RCCAdhesive.set_sticky_side(tape_sc, 0)
    tape_obj = scene.objects().create("tape")
    tape_obj.geometries().create(tape_sc)

    def animate_rod(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.instances().find(builtin.is_constrained))
        aim_transform  = view(geo.instances().find(builtin.aim_transform))
        is_constrained[0] = 1
        f = max(info.frame() - 1, 0)
        aim_transform[0] = make_rod_transform(rod_x_at(f), rod_y_at(f))

    scene.animator().insert(rod_obj, animate_rod)

    world.init(scene)
    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
    }


def run_demo():
    state = {"adhesion_on": True}
    sim = build_demo(state["adhesion_on"])

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")

    def fresh_surface():
        return sim["scene_io"].simplicial_surface()

    surface = fresh_surface()
    mesh = ps.register_surface_mesh(
        "tape_roll_sim",
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
            ps.remove_surface_mesh("tape_roll_sim")
            mesh = ps.register_surface_mesh("tape_roll_sim", v, t)
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

    def reset():
        nonlocal sim
        sim = build_demo(state["adhesion_on"])
        update_visual()

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

        if ui["run"]:
            step_once()

        f = min(sim["world"].frame(), TOTAL_FRAMES)
        psim.Separator()
        psim.Text(f"Frame: {f} / {TOTAL_FRAMES}")
        psim.Text(f"Phase: {phase_at(f)}")
        psim.Text(f"Rod (x, y): ({rod_x_at(f):+.3f}, {rod_y_at(f):.3f})")
        psim.Text(f"Adhesion: {'ENABLED' if state['adhesion_on'] else 'DISABLED'}")
        if f >= TOTAL_FRAMES:
            psim.Text("Sim done — hit `reset` to rebuild.")

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo()
