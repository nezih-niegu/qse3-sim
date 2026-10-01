"""Tests for the entangled swarm, its chaotic guidance and the Lyapunov machinery."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from typer.testing import CliRunner

from qse3_sim.algebra import generators
from qse3_sim.cli import app
from qse3_sim.dynamics import QSE3Dynamics, RobotState
from qse3_sim.physics import GAUGE_FIELDS, MATTER_PROFILES, FlavorHamiltonian
from qse3_sim.swarm import (
    EntangledSwarm,
    FormationController,
    GuidanceNetwork,
    LorenzReference,
    SwarmParams,
    SwarmState,
    formation_offsets,
    product_state,
    reduced_density,
    von_neumann_bits,
)
from qse3_sim.swarm_plots import hue_pal


@pytest.fixture(scope="module")
def ref() -> LorenzReference:
    return LorenzReference(lorenz_times=1.5)


def _swarm(ref, robots=3, entangle=True, mu_ent=2e-3, aware=True, enabled=True, **kw):
    ctl = FormationController(aware=aware, enabled=enabled)
    return EntangledSwarm(
        FlavorHamiltonian(4.0),
        MATTER_PROFILES["uniform"](),
        GAUGE_FIELDS["helical"](),
        SwarmParams(mu_ent=mu_ent, entangle=entangle),
        ctl,
        ref,
        formation_offsets(robots, 40.0),
        GuidanceNetwork(**kw),
    )


# -- palette -------------------------------------------------------------------


def test_hue_pal_matches_ggplot2() -> None:
    assert hue_pal(2) == ["#F8766D", "#00BFC4"]
    assert hue_pal(3) == ["#F8766D", "#00BA38", "#619CFF"]


# -- algebra behind the entangler -----------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_casimir_pair_operator_is_a_swap(n: int) -> None:
    """sum_a T_a (x) T_a = SWAP/2 - 1/(2N): the pair term is a flavor exchange."""
    t = generators(n)
    lhs = sum(np.kron(m, m) for m in t)
    swap = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            swap[i * n + j, j * n + i] = 1.0
    assert np.allclose(lhs, 0.5 * swap - np.eye(n * n) / (2 * n))


def test_reduced_state_of_product_is_pure() -> None:
    psi = product_state(3, [0, 1, 2])
    for ax in range(3):
        rho = reduced_density(psi, ax)
        assert abs(np.trace(rho) - 1) < 1e-12
        assert von_neumann_bits(rho) < 1e-10


# -- reduction to the single robot ------------------------------------------------


def test_single_robot_swarm_reproduces_qse3_dynamics(ref) -> None:
    """R = 1 with control off is exactly the original integrator."""
    sw = _swarm(ref, robots=1, entangle=False, enabled=False)
    single = QSE3Dynamics(sw.h, sw.matter, sw.gauge, mass_km_inv=sw.p.mass)
    beta = np.array([0.4, 0.2, -0.1])
    gam = 1 / np.sqrt(1 - beta @ beta)
    u = np.array([gam, *(gam * beta)])
    x = np.array([0.0, 30.0, -20.0, 10.0])
    psi = np.array([0.0, 1.0, 0.0], dtype=complex)
    s_sw = SwarmState(x[None].copy(), u[None].copy(), psi[None].copy(), ref.s(0.0)[None])
    s_one = RobotState(x.copy(), u.copy(), psi.copy())
    for k in range(200):
        s_sw = sw.step(s_sw, 5.0 * k, 5.0, 1.0)
        s_one = single.step(s_one, 5.0)
    assert np.allclose(s_sw.x[0], s_one.x, atol=1e-9)
    assert np.allclose(s_sw.u[0], s_one.u, atol=1e-12)
    assert np.allclose(s_sw.psi[0], s_one.psi, atol=1e-12)


# -- entanglement changes the kinematics ------------------------------------------


def test_no_entangler_keeps_robots_pure(ref) -> None:
    sw = _swarm(ref, entangle=False)
    tr = sw.run(sw.initial_state(), 800.0, 80, twins=False)
    assert np.max(tr.entropy) < 1e-9
    assert np.allclose(tr.eta, 1.0, atol=1e-9)


def test_entangler_shrinks_charge_and_conserves(ref) -> None:
    sw = _swarm(ref, mu_ent=5e-3)
    tr = sw.run(sw.initial_state(), 1500.0, 150, twins=False)
    assert np.max(tr.entropy[-1]) > 0.3
    assert np.max(tr.eta[-1]) < 0.95
    assert np.max(np.abs(tr.norm - 1)) < 1e-12
    assert np.max(np.abs(tr.mass_shell - 1)) < 1e-12


def test_naive_controller_loses_bandwidth_when_entangled() -> None:
    offsets = formation_offsets(4, 40.0)
    eta = np.full(4, 0.4)
    naive = FormationController(aware=False)
    aware = FormationController(aware=True)
    assert naive.contraction_rate(offsets, eta) < aware.contraction_rate(offsets, eta)
    assert np.isclose(aware.contraction_rate(offsets, eta), aware.contraction_rate(offsets))


def test_governor_throttles_entangler() -> None:
    ctl = FormationController(eta_floor=0.35, eta_target=0.6)
    assert ctl.entangler_throttle(np.array([0.9, 0.8])) == 1.0
    assert ctl.entangler_throttle(np.array([0.9, 0.30])) == 0.0
    assert FormationController(aware=False).entangler_throttle(np.array([0.1])) == 1.0


# -- chaos ---------------------------------------------------------------------------


def test_reference_lyapunov_exponent_is_lorenz() -> None:
    lr = LorenzReference(lorenz_times=40.0)
    lam = lr.lyapunov_spectrum(dt=2e-3) * lr.time_scale
    assert 0.8 < lam[0] < 1.0  # 0.906 in the long-time limit
    assert abs(lam[1]) < 0.1
    assert abs(lam.sum() + (10 + 1 + 8 / 3)) < 0.1  # trace of the Jacobian


def test_reference_is_subluminal(ref) -> None:
    speeds = [np.linalg.norm(ref(t)[1]) for t in np.linspace(0, ref.duration_km, 400)]
    assert max(speeds) <= ref.v_max + 1e-6


def test_sync_margin_shrinks_with_ring_size() -> None:
    g = GuidanceNetwork(kappa_c=6e-4, kappa_p=1.5e-3)
    mus = [g.mu_min(r) for r in (3, 4, 6, 8)]
    assert all(a > b for a, b in itertools.pairwise(mus))


def test_guidance_exponent_matches_prediction(ref) -> None:
    """Nonlinear twin exponent = free-flow stretching along the path - m_r, per robot."""
    sw = _swarm(ref, robots=4, entangle=False)
    tr = sw.run(sw.initial_state(scatter_km=0.0), ref.duration_km, 300, twins=False)
    predicted = tr.lyap_guidance_pred.mean(axis=0)  # free flow along each path, minus m_r
    # measured by nonlinear driven twins through the coupled equations
    assert np.allclose(tr.guidance_lyapunov, predicted, rtol=0.02)


def test_mechanical_exponent_matches_linear_prediction(ref) -> None:
    sw = _swarm(ref, robots=3, mu_ent=5e-4)
    tr = sw.run(sw.initial_state(), 2500.0, 250, twins=True)
    predicted = -sw.ctl.contraction_rate(sw.offsets)
    assert np.all(tr.mechanical_lyapunov < 0)
    assert np.allclose(tr.mechanical_lyapunov, predicted, rtol=0.1)


def test_scattered_guidance_synchronizes(ref) -> None:
    sw = _swarm(ref, robots=4, entangle=False)
    tr = sw.run(sw.initial_state(scatter_km=10.0), ref.duration_km, 300, twins=False)
    err = np.linalg.norm(tr.sync_error, axis=2).max(axis=1)
    assert err[-1] < 0.05 * err[0]


# -- CLI -----------------------------------------------------------------------------


def test_cli_swarm(tmp_path) -> None:
    res = CliRunner().invoke(
        app,
        ["swarm", "--robots", "3", "--steps", "120", "--lorenz-times", "1",
         "--no-compare", "-o", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "SYNCHRONIZES" in res.output
    for name in ("swarm_paths.png", "swarm_lyapunov_phase3d.png",
                 "swarm_lyapunov_sync3d.png", "swarm_dashboard.png", "swarm.json"):
        assert (tmp_path / name).exists()


# -- identities used in the paper ------------------------------------------------------


def test_eta_is_a_function_of_purity(ref) -> None:
    """eta_r^2 = (N Tr rho_r^2 - 1) / (N - 1): efficiency is purity, exactly."""
    sw = _swarm(ref, mu_ent=5e-3)
    tr = sw.run(sw.initial_state(), 1500.0, 150, twins=False, diagnostics=False)
    n = 3
    assert np.allclose(tr.eta**2, (n * tr.purity - 1) / (n - 1), atol=1e-10)


def test_force_spread_vanishes_without_field(ref) -> None:
    sw = EntangledSwarm(
        FlavorHamiltonian(4.0), MATTER_PROFILES["uniform"](), GAUGE_FIELDS["off"](3),
        SwarmParams(), FormationController(), ref, formation_offsets(3, 40.0),
    )
    tr = sw.run(sw.initial_state(), 200.0, 20, twins=False)
    assert np.max(tr.accel_spread) < 1e-12


def test_slowest_sync_mode_obeys_the_proposition() -> None:
    """Projected desync = exp(-mu_min t) x free tangent flow, in the linear regime."""
    from qse3_sim.swarm import simulate_guidance, slowest_mode_check

    lr = LorenzReference(lorenz_times=3.0)
    for g in (GuidanceNetwork(kappa_c=1e-4, kappa_p=2e-4), GuidanceNetwork()):
        out = simulate_guidance(g, lr, 4, scatter_km=1e-2, steps=1200)
        chk = slowest_mode_check(g, lr, 4, out)
        lin = np.exp(chk["log_measured"]) < 0.5  # linear regime
        lin &= np.exp(chk["log_measured"]) > 1e-4  # above the integrator floor
        gap = np.abs(chk["log_measured"] - chk["log_predicted"])[lin]
        assert lin.sum() > 100
        assert gap.max() < 0.05  # within 5% in amplitude


def test_ablation_flags_are_independent() -> None:
    gain_only = FormationController(governor=False)
    gov_only = FormationController(aware=False, governor=True)
    assert gain_only.uses_schedule and not gain_only.uses_governor
    assert gov_only.uses_governor and not gov_only.uses_schedule
    assert gain_only.entangler_throttle(np.array([0.1])) == 1.0
    assert gov_only.entangler_throttle(np.array([0.1])) == 0.0


def test_product_fast_path_is_exact(ref) -> None:
    """Without the entangler the product-state path equals the full tensor evolution."""
    fast = _swarm(ref, robots=3, entangle=False)
    full = _swarm(ref, robots=3, entangle=False)
    full.product = False  # force the N^R tensor path
    tf = fast.run(fast.initial_state(scatter_km=5.0), 600.0, 60, twins=False)
    tu = full.run(full.initial_state(scatter_km=5.0), 600.0, 60, twins=False)
    assert fast.product and not full.product
    # The two paths renormalize differently after each RK4 step (per robot vs the
    # joint vector); they differ only at the level of the per-step drift (~1e-8).
    assert np.allclose(tf.x, tu.x, atol=1e-6)
    assert np.allclose(tf.charge, tu.charge, atol=1e-6)
    assert np.allclose(tf.accel_phys, tu.accel_phys, atol=1e-9)


def test_record_every_subsamples_without_changing_dynamics(ref) -> None:
    sw = _swarm(ref, robots=3)
    a = sw.run(sw.initial_state(), 400.0, 40, twins=False, diagnostics=False)
    b = sw.run(sw.initial_state(), 400.0, 40, twins=False, diagnostics=False, record_every=5)
    assert len(b.t) == 9
    assert np.allclose(a.x[::5], b.x, atol=1e-12)
