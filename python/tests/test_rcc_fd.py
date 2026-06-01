"""Finite-difference derivative tests for RCC adhesive energy functions.

Tests the analytical gradient and Hessian of the two active (v1) PT RCC energies
computed by the CUDA backend via RCCAdhesiveDiagnoserFeature:

  * Normal adhesion:
        E = dt² · Cn/(2·dHat) · β² · D(P,T0,T1,T2)
        D = unflagged plane-projection squared distance
        ∇E = coeff_n · ∇D,   ∇²E = coeff_n · ∇²D

  * Tangential adhesion:
        E = dt² · Ct/(2·dHat) · β² · |u|²
        u = J(prev) · (x − x_prev) ∈ R²
        ∇E = coeff_t · Jᵀu,   ∇²E = coeff_t · JᵀJ

EE, PE, PP adhesion return 0 in v1 — not tested.

All derivative checks use the Schroeder central-difference + central-average
formula (SIGGRAPH '22 course, Section 4.1) with δ=1e-6, threshold 1e-4 —
identical to test_contact_fd.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason
from uipc import Logger, view
from uipc.core import Engine, World, Scene, RCCAdhesiveDiagnoserFeature
from uipc.geometry import Geometry, trimesh

Logger.set_level(Logger.Level.Warn)

# ── tuning constants ──────────────────────────────────────────────────────────
_DELTA       = 1e-6
_GRAD_THRESH = 1e-4
_HESS_THRESH = 1e-4

# Default RCC parameters
_CN    = 1e4
_CT    = 5e3
_BETA  = 0.7
_D_HAT = 0.1
_DT    = 0.01

# Fixed triangle in the y=0 plane, shared by all test cases
_T0 = np.array([-1.0,  0.0, -0.5])
_T1 = np.array([ 1.0,  0.0, -0.5])
_T2 = np.array([ 0.0,  0.0,  1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Session fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rcc_session(tmp_path_factory):
    """CUDA engine + RCCAdhesiveDiagnoserFeature shared across all tests."""
    try:
        workspace = str(tmp_path_factory.mktemp("rcc_fd"))
        engine    = Engine("cuda", workspace)
        world     = World(engine)
        scene     = Scene(Scene.default_config())
        world.init(scene)
        rcc = world.features().find(RCCAdhesiveDiagnoserFeature)
        assert rcc is not None, "RCCAdhesiveDiagnoserFeature not found"
    except Exception as exc:
        pytest.skip(f"CUDA / RCC diagnoser unavailable: {exc}")
    yield engine, world, rcc


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────────

def _pt_sc(P, T0, T1, T2):
    """Single-point + single-triangle scene as a trimesh (for the diagnoser interface)."""
    sc = trimesh(
        np.array([P, T0, T1, T2], dtype=np.float64),
        np.array([[1, 2, 3]], dtype=np.int32),
    )
    return sc


def _set_params(R, beta=_BETA, d_hat=_D_HAT):
    """Write RCC parameters as instance attributes on the result geometry."""
    for name, val in [
        ("rcc/Cn",    _CN),
        ("rcc/Ct",    _CT),
        ("rcc/beta",  beta),
        ("rcc/d_hat", d_hat),
        ("rcc/dt",    _DT),
    ]:
        attr = R.instances().find(name)
        if attr is None:
            R.instances().create(name, float(val))


def _get_scalar(R, name):
    attr = R.instances().find(name)
    return None if attr is None else float(view(attr).flatten()[0])


def _get_vec(R, name):
    attr = R.instances().find(name)
    return None if attr is None else view(attr).flatten()[:12].astype(np.float64)


def _get_mat(R, name):
    attr = R.instances().find(name)
    return None if attr is None else view(attr).flatten()[:144].astype(np.float64).reshape(12, 12)


# ─────────────────────────────────────────────────────────────────────────────
# Schroeder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply(sc, orig_pos, dx12):
    """Write orig_pos + dx12 into the first 4 vertices of sc."""
    pos = view(sc.positions())
    pos[:4] = orig_pos + dx12.reshape(4, 3, 1)


def schroeder_grad(compute_fn, sc, R, energy_name, grad_name,
                   delta=_DELTA, seed=0):
    orig = view(sc.positions()).copy()
    rng  = np.random.default_rng(seed)
    dx   = rng.uniform(-delta, delta, 12)

    _apply(sc, orig, +dx)
    compute_fn()
    fp = _get_scalar(R, energy_name)
    gp = _get_vec(R, grad_name)

    _apply(sc, orig, -dx)
    compute_fn()
    fm = _get_scalar(R, energy_name)
    gm = _get_vec(R, grad_name)

    _apply(sc, orig, np.zeros(12))
    compute_fn()

    if any(v is None for v in [fp, fm, gp, gm]):
        return None

    normed = abs((fp - fm) - float(np.dot(gp + gm, dx))) / delta
    print(f"\n    grad  |f_diff|={abs(fp-fm):.3e}  err/delta={normed:.3e}")
    return normed


def schroeder_hess(compute_fn, sc, R, grad_name, hess_name,
                   delta=_DELTA, seed=1):
    orig = view(sc.positions()).copy()
    rng  = np.random.default_rng(seed)
    dx   = rng.uniform(-delta, delta, 12)

    _apply(sc, orig, +dx)
    compute_fn()
    gp = _get_vec(R, grad_name)
    Hp = _get_mat(R, hess_name)

    _apply(sc, orig, -dx)
    compute_fn()
    gm = _get_vec(R, grad_name)
    Hm = _get_mat(R, hess_name)

    _apply(sc, orig, np.zeros(12))
    compute_fn()

    if any(v is None for v in [gp, gm, Hp, Hm]):
        return None

    residual = (gp - gm) - (Hp + Hm) @ dx
    normed   = np.linalg.norm(residual) / delta
    print(f"\n    hess  |g_diff|={np.linalg.norm(gp-gm):.3e}  err/delta={normed:.3e}")
    return normed


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (P, beta, d_hat, label)
_NORMAL_CASES = [
    # ── d within d_hat, beta=0.7 ─────────────────────────────────────────────
    (np.array([ 0.0,  0.06,  0.0]),  0.7, 0.1, "centroid_b07"),
    (np.array([ 0.3,  0.08, -0.1]),  0.7, 0.1, "off_center_b07"),
    (np.array([-0.5,  0.05,  0.3]),  0.7, 0.1, "near_edge_b07"),
    (np.array([ 0.0,  0.09,  0.8]),  0.7, 0.1, "near_vertex_b07"),
    # ── near-contact (d << d_hat) ─────────────────────────────────────────────
    (np.array([ 0.0,  0.001, 0.0]),  0.7, 0.1, "near_contact_b07"),
    (np.array([ 0.3,  0.001,-0.1]),  0.7, 0.1, "near_contact_offcenter_b07"),
    # ── d > d_hat (outside activation range) ─────────────────────────────────
    (np.array([ 0.0,  0.15,  0.0]),  0.7, 0.1, "beyond_dhat_b07"),
    (np.array([ 0.3,  0.20, -0.1]),  0.7, 0.1, "beyond_dhat_offcenter_b07"),
    # ── beta=1.0 (fully bonded — primary production value) ───────────────────
    (np.array([ 0.0,  0.06,  0.0]),  1.0, 0.1, "centroid_b10"),
    (np.array([ 0.0,  0.001, 0.0]),  1.0, 0.1, "near_contact_b10"),
    (np.array([ 0.0,  0.15,  0.0]),  1.0, 0.1, "beyond_dhat_b10"),
    # ── beta=0.3 ──────────────────────────────────────────────────────────────
    (np.array([ 0.0,  0.06,  0.0]),  0.3, 0.1, "centroid_b03"),
    (np.array([ 0.0,  0.15,  0.0]),  0.3, 0.1, "beyond_dhat_b03"),
]

# Each entry: (prev_P, curr_P, beta, d_hat, label)
_TANGENTIAL_CASES = [
    # ── d within d_hat, beta=0.7 ─────────────────────────────────────────────
    (np.array([0.0, 0.06, 0.0]),  np.array([0.02, 0.06, 0.0]),  0.7, 0.1, "slide_x_b07"),
    (np.array([0.2, 0.07, 0.1]),  np.array([0.18, 0.07, 0.13]), 0.7, 0.1, "slide_diag_b07"),
    (np.array([0.0, 0.05, 0.0]),  np.array([0.001,0.05, 0.0]),  0.7, 0.1, "tiny_slide_b07"),
    (np.array([0.0, 0.08, 0.0]),  np.array([0.05, 0.08, 0.05]), 0.7, 0.1, "large_slide_b07"),
    # ── near-contact ──────────────────────────────────────────────────────────
    (np.array([0.0, 0.001,0.0]),  np.array([0.02, 0.001,0.0]),  0.7, 0.1, "near_contact_b07"),
    # ── d > d_hat ─────────────────────────────────────────────────────────────
    (np.array([0.0, 0.15, 0.0]),  np.array([0.02, 0.15, 0.0]),  0.7, 0.1, "beyond_dhat_b07"),
    # ── beta=1.0 ──────────────────────────────────────────────────────────────
    (np.array([0.0, 0.06, 0.0]),  np.array([0.02, 0.06, 0.0]),  1.0, 0.1, "slide_x_b10"),
    (np.array([0.0, 0.001,0.0]),  np.array([0.02, 0.001,0.0]),  1.0, 0.1, "near_contact_b10"),
    (np.array([0.0, 0.15, 0.0]),  np.array([0.02, 0.15, 0.0]),  1.0, 0.1, "beyond_dhat_b10"),
    # ── beta=0.3 ──────────────────────────────────────────────────────────────
    (np.array([0.0, 0.06, 0.0]),  np.array([0.02, 0.06, 0.0]),  0.3, 0.1, "slide_x_b03"),
    (np.array([0.0, 0.15, 0.0]),  np.array([0.02, 0.15, 0.0]),  0.3, 0.1, "beyond_dhat_b03"),
]


@pytest.mark.basic
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("P,beta,d_hat,label", _NORMAL_CASES,
                         ids=[c[3] for c in _NORMAL_CASES])
def test_pt_normal_grad_fd(rcc_session, P, beta, d_hat, label):
    """PT normal adhesion: Schroeder gradient test via CUDA RCCAdhesiveDiagnoserFeature."""
    _, _, rcc = rcc_session

    sc      = _pt_sc(P, _T0, _T1, _T2)
    prev_sc = _pt_sc(P, _T0, _T1, _T2)
    R       = Geometry()
    _set_params(R, beta=beta, d_hat=d_hat)

    def compute():
        rcc.compute_pt_adhesion(R, sc, sc, prev_sc, prev_sc)

    compute()
    d = abs(P[1])  # plane-projection distance (y-component for our y=0 triangle)
    print(f"\n[normal/{label}]  d={d:.4f} d_hat={d_hat}  beta={beta}  E={_get_scalar(R,'normal/energy'):.4e}")

    err = schroeder_grad(compute, sc, R, "normal/energy", "normal/grad", seed=0)
    assert err is not None and err < _GRAD_THRESH, (
        f"[normal/{label}] grad err/delta={err:.3e} >= {_GRAD_THRESH}"
    )


@pytest.mark.basic
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("P,beta,d_hat,label", _NORMAL_CASES,
                         ids=[c[3] for c in _NORMAL_CASES])
def test_pt_normal_hess_fd(rcc_session, P, beta, d_hat, label):
    """PT normal adhesion: Schroeder Hessian test via CUDA RCCAdhesiveDiagnoserFeature."""
    _, _, rcc = rcc_session

    sc      = _pt_sc(P, _T0, _T1, _T2)
    prev_sc = _pt_sc(P, _T0, _T1, _T2)
    R       = Geometry()
    _set_params(R, beta=beta, d_hat=d_hat)

    def compute():
        rcc.compute_pt_adhesion(R, sc, sc, prev_sc, prev_sc)

    compute()

    err = schroeder_hess(compute, sc, R, "normal/grad", "normal/hess", seed=1)
    assert err is not None and err < _HESS_THRESH, (
        f"[normal/{label}] hess err/delta={err:.3e} >= {_HESS_THRESH}"
    )


@pytest.mark.basic
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("Pp,P,beta,d_hat,label", _TANGENTIAL_CASES,
                         ids=[c[4] for c in _TANGENTIAL_CASES])
def test_pt_tangential_grad_fd(rcc_session, Pp, P, beta, d_hat, label):
    """PT tangential adhesion: Schroeder gradient test (prev positions fixed)."""
    _, _, rcc = rcc_session

    sc      = _pt_sc(P,  _T0, _T1, _T2)
    prev_sc = _pt_sc(Pp, _T0, _T1, _T2)
    R       = Geometry()
    _set_params(R, beta=beta, d_hat=d_hat)

    def compute():
        rcc.compute_pt_adhesion(R, sc, sc, prev_sc, prev_sc)

    compute()
    print(f"\n[tan/{label}]  d={abs(P[1]):.4f} d_hat={d_hat}  beta={beta}  E={_get_scalar(R,'tan/energy'):.4e}")

    err = schroeder_grad(compute, sc, R, "tan/energy", "tan/grad", seed=2)
    assert err is not None and err < _GRAD_THRESH, (
        f"[tangential/{label}] grad err/delta={err:.3e} >= {_GRAD_THRESH}"
    )


@pytest.mark.basic
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("Pp,P,beta,d_hat,label", _TANGENTIAL_CASES,
                         ids=[c[4] for c in _TANGENTIAL_CASES])
def test_pt_tangential_hess_fd(rcc_session, Pp, P, beta, d_hat, label):
    """PT tangential adhesion: Schroeder Hessian test (JᵀJ is constant in x)."""
    _, _, rcc = rcc_session

    sc      = _pt_sc(P,  _T0, _T1, _T2)
    prev_sc = _pt_sc(Pp, _T0, _T1, _T2)
    R       = Geometry()
    _set_params(R, beta=beta, d_hat=d_hat)

    def compute():
        rcc.compute_pt_adhesion(R, sc, sc, prev_sc, prev_sc)

    compute()

    err = schroeder_hess(compute, sc, R, "tan/grad", "tan/hess", seed=3)
    assert err is not None and err < _HESS_THRESH, (
        f"[tangential/{label}] hess err/delta={err:.3e} >= {_HESS_THRESH}"
    )
