"""Tests for the QSE(3) simulator."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from qse3_sim.algebra import (
    casimir_pure_state,
    color_charge,
    generators,
    is_unitary,
    pmns,
    structure_constants,
)
from qse3_sim.cli import app
from qse3_sim.dynamics import QSE3Dynamics, boosted_state
from qse3_sim.kinematics import KinematicChain
from qse3_sim.physics import (
    GAUGE_FIELDS,
    FlavorHamiltonian,
    matter_potential_km_inv,
    resonance_density,
    uniform_profile,
)
from qse3_sim.singularity import analyze_path, effective_mixing_angle

runner = CliRunner()


# -- algebra ---------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_generator_normalisation(n: int) -> None:
    t = generators(n)
    assert len(t) == n * n - 1
    for a, ta in enumerate(t):
        assert np.allclose(ta, ta.conj().T)  # Hermitian
        assert abs(np.trace(ta)) < 1e-12  # traceless
        for b, tb in enumerate(t):
            expected = 0.5 if a == b else 0.0
            assert abs(np.trace(ta @ tb) - expected) < 1e-12


def test_gell_mann_structure_constants() -> None:
    f = structure_constants(3)
    assert f[0, 1, 2] == pytest.approx(1.0)
    assert f[0, 3, 6] == pytest.approx(0.5)
    assert f[1, 4, 6] == pytest.approx(0.5)
    assert f[3, 4, 7] == pytest.approx(np.sqrt(3) / 2)
    assert np.allclose(f, -np.transpose(f, (1, 0, 2)))  # antisymmetric


def test_pmns_is_unitary() -> None:
    assert is_unitary(pmns())


def test_pure_state_casimir() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        psi = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi /= np.linalg.norm(psi)
        assert np.sum(color_charge(psi) ** 2) == pytest.approx(casimir_pure_state(3))


# -- physics ---------------------------------------------------------------


def test_matter_potential_magnitude() -> None:
    # sqrt(2) G_F N_e for rho=1, Y_e=0.5 is close to the textbook 3.8e-5 km^-1.
    v = matter_potential_km_inv(1.0, 0.5)
    assert 1.8e-4 < v < 2.0e-4


def test_resonance_condition_is_self_consistent() -> None:
    e = 7.0
    rho = resonance_density(e)
    # at the resonance density the two-flavor matter angle passes 45 degrees
    assert effective_mixing_angle(rho, e) == pytest.approx(45.0, abs=0.5)


def test_oscillation_phase_matches_standard_formula() -> None:
    """A two-level gap should reproduce 1.267 * dm2 * L / E."""
    h = FlavorHamiltonian(energy_gev=1.0, dm_sq=(0.0, 0.0, 2.515e-3))
    w = np.linalg.eigvalsh(h.total(0.0))
    gap = w[-1] - w[0]
    assert gap == pytest.approx(2 * 1.2669 * 2.515e-3 / 1.0, rel=1e-3)


# -- kinematics ------------------------------------------------------------


def test_forward_kinematics_unitary() -> None:
    chain = KinematicChain()
    assert is_unitary(chain.forward())
    p = chain.transition_matrix()
    assert np.allclose(p.sum(axis=0), 1.0)


def test_jacobian_rank_bounded_by_state_manifold() -> None:
    chain = KinematicChain()
    diag = chain.manipulability()
    assert diag["max_rank"] == 4
    assert diag["rank"] <= diag["max_rank"]


def test_zero_length_chain_is_singular() -> None:
    """With no propagation and no twist there is nothing left to steer with."""
    from qse3_sim.kinematics import Link

    chain = KinematicChain((Link(0.0, 0.0, 0.0, 0.0),))
    assert chain.manipulability()["rank"] < 4


# -- dynamics --------------------------------------------------------------


def _sim(**kw) -> tuple[QSE3Dynamics, object]:
    h = FlavorHamiltonian(energy_gev=5.0)
    sim = QSE3Dynamics(h, uniform_profile(3.0), GAUGE_FIELDS["helical"](n=3), **kw)
    s0 = boosted_state([0.99, 0.0, 0.0], np.array([1, 0, 0], dtype=complex))
    return sim, s0


def test_conservation_laws() -> None:
    sim, s0 = _sim(mass_km_inv=1e-2, topological_gain=0.0)
    traj = sim.run(s0, 400.0, 1500)
    assert np.max(np.abs(traj.norm - 1)) < 1e-10
    assert np.max(np.abs(traj.mass_shell - 1)) < 1e-8
    assert np.max(np.abs(traj.casimir - casimir_pure_state(3))) < 1e-10


def test_wong_precession_matches_expectation_values() -> None:
    """The doc's dI^a/dt law is exactly d<T_a>/dt when H_flavor is off."""
    sim, s0 = _sim(mass_km_inv=1e-2, topological_gain=0.0, wong_only=True)
    traj = sim.run(s0, 400.0, 1500)
    assert np.max(np.linalg.norm(sim.wong_precession(traj) - traj.charge, axis=1)) < 1e-6


def test_no_internal_motion_without_spatial_velocity() -> None:
    """The internal state cannot torque unless the robot moves through A_mu."""
    h = FlavorHamiltonian(energy_gev=5.0)
    sim = QSE3Dynamics(
        h, uniform_profile(0.0), GAUGE_FIELDS["helical"](n=3), wong_only=True,
        topological_gain=0.0,
    )
    at_rest = boosted_state([0.0, 0.0, 0.0], np.array([1, 1, 0], dtype=complex) / np.sqrt(2))
    traj = sim.run(at_rest, 200.0, 400)
    assert np.max(np.abs(traj.charge - traj.charge[0])) < 1e-9


def test_refractive_energy_conserved_at_frozen_flavor() -> None:
    from qse3_sim.physics import waveguide

    h = FlavorHamiltonian(energy_gev=4.0)
    sim = QSE3Dynamics(
        h, waveguide(), GAUGE_FIELDS["off"](3), mass_km_inv=1e-2,
        topological_gain=5e3, wong_only=True,
    )
    s0 = boosted_state([0.99, 0, 0], np.array([1, 0, 0], dtype=complex),
                       x0=np.array([0.0, 0.0, 40.0, 0.0]))
    traj = sim.run(s0, 3000.0, 3000)
    assert np.max(np.abs(traj.energy / traj.energy[0] - 1)) < 1e-8
    assert traj.x[:, 2].min() < 0.0  # actually focused through the axis


# -- singularity -----------------------------------------------------------


def test_sharp_slab_is_non_adiabatic_and_stellar_is_not() -> None:
    from qse3_sim.physics import gaussian_slab, stellar

    h = FlavorHamiltonian(energy_gev=10.0)

    s = np.linspace(0, 400, 800)
    slab = gaussian_slab()
    rho = np.array([slab.rho(np.array([si, si, 0, 0])) for si in s])
    sharp = analyze_path(s, rho, h).resonant_pair
    assert sharp.has_crossing
    assert sharp.adiabaticity[sharp.crossing_index] < 1.0

    s2 = np.linspace(0, 1.5e7, 800)
    star = stellar()
    rho2 = np.array([star.rho(np.array([si, si, 0, 0])) for si in s2])
    smooth = analyze_path(s2, rho2, h, star.y_e).resonant_pair
    assert smooth.has_crossing
    assert smooth.adiabaticity[smooth.crossing_index] > 10.0


def test_crossing_density_matches_prediction() -> None:
    from qse3_sim.physics import gaussian_slab

    h = FlavorHamiltonian(energy_gev=10.0)
    s = np.linspace(0, 400, 1200)
    slab = gaussian_slab()
    rho = np.array([slab.rho(np.array([si, si, 0, 0])) for si in s])
    rep = analyze_path(s, rho, h)
    found = rep.rho[rep.resonant_pair.crossing_index]
    assert found == pytest.approx(rep.resonance_rho, rel=0.05)


# -- CLI -------------------------------------------------------------------


def test_cli_help() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0


def test_cli_verify() -> None:
    result = runner.invoke(app, ["verify", "--steps", "500"])
    assert result.exit_code == 0
    assert "ALL CHECKS PASSED" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["kinematics"],
        ["dynamics", "--steps", "400", "-L", "200"],
        ["singularities", "--steps", "200"],
    ],
)
def test_cli_commands(tmp_path, args) -> None:
    result = runner.invoke(app, [*args, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert any(tmp_path.iterdir())


# -- control ---------------------------------------------------------------


def test_damped_step_finite_on_rank_deficient_jacobian() -> None:
    from qse3_sim.control import damped_step

    a = np.zeros((6, 8))
    a[0, 0] = 1.0  # rank 1, five structurally empty task directions
    e = np.ones(6)
    for lam in (0.0, 1e-3, 0.1):
        dq = damped_step(a, e, lam)
        assert np.all(np.isfinite(dq))
        assert np.linalg.norm(dq) < 10.0


def test_align_phase_is_a_projection() -> None:
    from qse3_sim.control import align_phase

    psi = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    chi = np.exp(2.3j) * psi
    aligned = align_phase(psi, chi)
    assert np.allclose(aligned, psi)


def test_control_reaches_target_with_mixing_generators() -> None:
    """T_2 and T_5 touch the nu_e sector, so nu_mu -> nu_e is steerable."""
    from qse3_sim import control as ctl
    from qse3_sim.kinematics import DEFAULT_CHAIN

    chain = KinematicChain(
        DEFAULT_CHAIN, FlavorHamiltonian(energy_gev=4.0), phase_index=2, twist_index=5
    )
    psi0 = np.array([0, 1, 0], dtype=complex)
    res = ctl.solve_best(chain, psi0, ctl.named_target(3, "0"), restarts=6, actuation="gauge")
    assert res.converged
    assert res.final_infidelity < 1e-6
    # only the gauge coordinates moved
    assert all(lab.startswith(("theta", "alpha")) for lab, *_ in res.command_table())


def test_reachability_wall_with_diagonal_generators() -> None:
    """T_3 and T_7 leave flavor 1 alone, so no local inversion opens it."""
    from qse3_sim import control as ctl

    chain = KinematicChain(hamiltonian=FlavorHamiltonian(energy_gev=4.0))
    psi0 = np.array([0, 1, 0], dtype=complex)
    res = ctl.solve_best(
        chain, psi0, ctl.named_target(3, "0"), restarts=4, actuation="gauge", max_iter=120
    )
    assert not res.converged
    assert res.final_infidelity > 0.5


def test_control_step_stays_bounded() -> None:
    from qse3_sim import control as ctl

    chain = KinematicChain(hamiltonian=FlavorHamiltonian(energy_gev=4.0))
    psi0 = np.array([0, 1, 0], dtype=complex)
    res = ctl.solve_best(
        chain, psi0, ctl.named_target(3, "uniform"), restarts=3, actuation="all", max_iter=150
    )
    for lab, _start, end, _delta in res.command_table():
        if lab.startswith(("theta", "alpha")):
            assert abs(end) < 100.0  # no runaway phase excursions
        if lab.startswith("d"):
            assert 0.0 <= end <= 2000.0
        if lab.startswith("a"):
            assert 0.0 <= end <= 20.0


def test_pure_gauge_has_zero_field_strength() -> None:
    """Gauge covariance: A = (i/g)(dU)U^-1 is pure gauge for H = +g A xdot T, so F = 0."""
    from scipy.linalg import expm

    from qse3_sim.physics import GaugeField

    t = generators(3)

    def u_of(x):
        return (expm(1j * 0.3 * x[1] * t[0]) @ expm(1j * 0.2 * x[2] * t[4])
                @ expm(1j * 0.1 * x[3] * t[6]))

    def a_pure(x, h=1e-5):
        ui = np.linalg.inv(u_of(x))
        out = np.zeros((4, 8))
        for mu in range(4):
            xp, xm = x.copy(), x.copy()
            xp[mu] += h
            xm[mu] -= h
            am = 1j * (u_of(xp) - u_of(xm)) / (2 * h) @ ui
            out[mu] = [2 * np.real(np.trace(am @ m)) for m in t]
        return out

    f = GaugeField("pure", a_pure, 1.0, 3).field_strength(np.array([0.0, 1.3, -0.7, 0.4]))
    assert np.max(np.abs(f)) < 1e-7


def test_pure_gauge_transports_state_trivially() -> None:
    """Around a closed loop in a pure-gauge field the internal state returns to itself."""
    from scipy.linalg import expm

    t = generators(3)
    g = 1.0

    def u_of(x):
        return expm(1j * 0.3 * x[1] * t[0]) @ expm(1j * 0.2 * x[2] * t[4])

    def a_mat(x, xdot, h=1e-6):
        ui = np.linalg.inv(u_of(x))
        tot = np.zeros((3, 3), dtype=complex)
        for mu in (1, 2):
            xp, xm = x.copy(), x.copy()
            xp[mu] += h
            xm[mu] -= h
            tot += xdot[mu] * 1j / g * (u_of(xp) - u_of(xm)) / (2 * h) @ ui
        return tot

    psi = np.array([1.0, 0.0, 0.0], dtype=complex)
    n = 4000
    for k in range(n):  # unit circle in the x-y plane, H = +g A.xdot
        th, dth = 2 * np.pi * k / n, 2 * np.pi / n
        x = np.array([0.0, np.cos(th + dth / 2), np.sin(th + dth / 2), 0.0])
        xdot = np.array([0.0, -np.sin(th + dth / 2), np.cos(th + dth / 2), 0.0])
        psi = expm(-1j * g * a_mat(x, xdot) * dth) @ psi
    assert abs(abs(np.vdot([1, 0, 0], psi)) - 1) < 1e-5
