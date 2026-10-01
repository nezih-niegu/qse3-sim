"""Command line entry point for the QSE(3) nanorobot simulator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from . import control as ctl
from . import plots
from .algebra import casimir_pure_state, structure_constants
from .dynamics import QSE3Dynamics, boosted_state
from .kinematics import DEFAULT_CHAIN, KinematicChain
from .physics import GAUGE_FIELDS, MATTER_PROFILES, FlavorHamiltonian, resonance_density
from .singularity import analyze_path

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Simulator for a nanoscale robot on C = Min(1,3) x SU(N).",
)

FLAVORS = ("nu_e", "nu_mu", "nu_tau")


def _flavor_names(n: int) -> list[str]:
    return list(FLAVORS[:n]) if n <= 3 else [f"s{i+1}" for i in range(n)]


def _outdir(out: Path | None) -> Path:
    d = out or Path.cwd() / "qse3_output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _initial_state(n: int, flavor: int) -> np.ndarray:
    psi = np.zeros(n, dtype=complex)
    psi[min(flavor, n - 1)] = 1.0
    return psi


# ---------------------------------------------------------------------------


@app.command()
def kinematics(
    energy: float = typer.Option(10.0, "--energy", "-E", help="Wave packet energy in GeV"),
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    flavor: int = typer.Option(0, "--flavor", help="Initial eigenstate index (0-based)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
) -> None:
    """Forward kinematics: path-ordered exponential over the DH table."""
    h = FlavorHamiltonian(energy_gev=energy, n=n)
    chain = KinematicChain(DEFAULT_CHAIN, h)
    q = chain.pack()
    u_total = chain.forward(q)
    p = chain.transition_matrix(q)
    names = _flavor_names(n)

    typer.echo("Quantum-relativistic DH table")
    typer.echo("  link   theta[rad]      d[km]   a[g/cm3]   alpha[rad]")
    for i, link in enumerate(chain.links, start=1):
        typer.echo(
            f"  {i:>4}   {link.theta:>9.3f}   {link.d:>8.1f}   {link.a:>8.2f}   {link.alpha:>9.3f}"
        )

    typer.echo("\nU_total unitarity residual: "
               f"{np.max(np.abs(u_total.conj().T @ u_total - np.eye(n))):.2e}")
    typer.echo("\nTransition probabilities P(row -> column)")
    typer.echo("            " + "".join(f"{c:>10}" for c in names))
    for i, r in enumerate(names):
        typer.echo(f"  {r:>9}   " + "".join(f"{p[j, i]:>10.4f}" for j in range(n)))

    psi0 = _initial_state(n, flavor)
    diag = chain.manipulability(q, psi0)
    typer.echo("\nGeometric Jacobian (Fubini-Study metric on the joint vector)")
    typer.echo(f"  joint coordinates : {len(q)}")
    typer.echo(f"  rank              : {diag['rank']} of {diag['max_rank']} reachable")
    typer.echo(f"  condition number  : {diag['condition']:.3e}")
    typer.echo(f"  manipulability    : {diag['volume']:.3e}")
    typer.echo(f"  Berry curvature   : |F| = {diag['berry_norm']:.3e}")
    typer.echo("  status            : " + ("SINGULAR" if diag["singular"] else "controllable"))

    d = _outdir(out)
    fig = plots.plot_jacobian(diag, d / "jacobian_spectrum.png")
    _write_json(
        d / "kinematics.json",
        {
            "energy_gev": energy,
            "dof": n,
            "transition_matrix": p.tolist(),
            "jacobian": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in diag.items()},
        },
    )
    typer.echo(f"\nWrote {fig}")


@app.command()
def dynamics(
    energy: float = typer.Option(4.0, "--energy", "-E", help="Wave packet energy in GeV"),
    gauge: str = typer.Option("helical", "--gauge", "-g", help=f"one of {list(GAUGE_FIELDS)}"),
    profile: str = typer.Option(
        "gaussian-slab", "--profile", "-p", help=f"one of {list(MATTER_PROFILES)}"
    ),
    distance: float = typer.Option(1600.0, "--distance", "-L", help="Coordinate time in km"),
    steps: int = typer.Option(6000, "--steps", help="Integration steps"),
    beta: float = typer.Option(0.99995, "--beta", help="Initial speed along +x, units of c"),
    mass: float = typer.Option(1.0e-2, "--mass", help="Mass parameter m in km^-1"),
    gain: float = typer.Option(1.0, "--topological-gain", help="Scale of Q_topological"),
    offset_y: float = typer.Option(0.0, "--offset-y", help="Initial transverse offset in km"),
    flavor: int = typer.Option(1, "--flavor", help="Initial eigenstate index (0-based)"),
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
) -> None:
    """Integrate the coupled Wong / MSW / state-precession system."""
    h = FlavorHamiltonian(energy_gev=energy, n=n)
    matter = MATTER_PROFILES[profile]()
    field = GAUGE_FIELDS[gauge](n=n) if gauge != "off" else GAUGE_FIELDS[gauge](n)
    sim = QSE3Dynamics(h, matter, field, mass_km_inv=mass, topological_gain=gain)

    s0 = boosted_state(
        [beta, 0.0, 0.0],
        _initial_state(n, flavor),
        x0=np.array([0.0, 0.0, offset_y, 0.0]),
    )
    traj = sim.run(s0, distance, steps)
    names = _flavor_names(n)

    typer.echo(f"Gauge field   : {field.name}")
    typer.echo(f"Matter model  : {matter.name}")
    typer.echo(f"Energy        : {energy} GeV,  m = {mass} km^-1,  N = {n}")
    typer.echo(f"Path length   : {traj.path_length[-1]:.2f} km over {distance:.0f} km of x^0")

    typer.echo("\nConserved quantities (max deviation)")
    typer.echo(f"  <psi|psi> - 1     : {np.max(np.abs(traj.norm - 1)):.3e}")
    typer.echo(f"  u.u - 1           : {np.max(np.abs(traj.mass_shell - 1)):.3e}")
    cas = casimir_pure_state(n)
    typer.echo(f"  I^a I^a - {cas:.4f}   : {np.max(np.abs(traj.casimir - cas)):.3e}")
    typer.echo(
        f"  M u^0 drift       : {np.max(np.abs(traj.energy / traj.energy[0] - 1)):.3e}"
        "  (exact only at frozen flavor)"
    )

    typer.echo("\nFinal internal state")
    for i, name in enumerate(names):
        typer.echo(f"  P({name:<7}) = {traj.probs[-1, i]:.4f}   (initial {traj.probs[0, i]:.4f})")

    dev = traj.x[-1, 2:]
    typer.echo(f"\nTransverse deflection : dy = {dev[0] - offset_y:+.3f} km, dz = {dev[1]:+.3f} km")
    typer.echo(f"Lorentz factor        : {traj.u[0, 0]:.1f} -> {traj.u[-1, 0]:.1f}")
    fg, ft = float(np.mean(traj.f_gauge)), float(np.mean(traj.f_top))
    typer.echo(f"Force budget          : |Q_gauge| = {fg:.3e}, |Q_top| = {ft:.3e} km^-2")
    typer.echo(f"                        ratio Q_top / Q_gauge = {ft / fg:.3e}" if fg > 0 else "")

    wong = sim.wong_precession(traj)
    torque = np.max(np.linalg.norm(wong - traj.charge, axis=1))
    typer.echo(f"Flavor torque         : max |I_wong - <T>| = {torque:.3e}")

    d = _outdir(out)
    f1 = plots.plot_trajectory(traj, d / "trajectory.png")
    f2 = plots.plot_flavor(traj, d / "flavor_and_charge.png")
    _write_json(
        d / "dynamics.json",
        {
            "meta": traj.meta,
            "final_probabilities": dict(zip(names, traj.probs[-1].tolist())),
            "deflection_km": {"y": float(dev[0]), "z": float(dev[1])},
            "path_length_km": float(traj.path_length[-1]),
            "flavor_torque": float(torque),
            "force_gauge_mean": fg,
            "force_topological_mean": ft,
        },
    )
    typer.echo(f"\nWrote {f1}\nWrote {f2}")


@app.command()
def singularities(
    energy: float = typer.Option(10.0, "--energy", "-E", help="Wave packet energy in GeV"),
    profile: str = typer.Option(
        "gaussian-slab", "--profile", "-p", help=f"one of {list(MATTER_PROFILES)}"
    ),
    distance: float = typer.Option(400.0, "--distance", "-L", help="Path length in km"),
    steps: int = typer.Option(1500, "--steps", help="Samples along the path"),
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
) -> None:
    """Locate MSW resonance zones and quantify the loss of control."""
    h = FlavorHamiltonian(energy_gev=energy, n=n)
    matter = MATTER_PROFILES[profile]()
    s = np.linspace(0.0, distance, steps)
    rho = np.array([matter.rho(np.array([si, si, 0.0, 0.0])) for si in s])
    rep = analyze_path(s, rho, h, matter.y_e)
    w = rep.worst()
    res = w["resonant"]

    typer.echo(f"Matter model            : {matter.name}")
    typer.echo(f"Resonance density at {energy:g} GeV : {rep.resonance_rho:.3f} g/cm^3")
    typer.echo(f"Density range on path   : {rho.min():.3f} .. {rho.max():.3f} g/cm^3")
    typer.echo(f"Crosses resonance       : {rho.min() <= rep.resonance_rho <= rho.max()}")

    typer.echo("\nLevel pairs")
    typer.echo("  pair   crossing      s[km]   rho     gap[km^-1]    L_osc[km]     gamma   P_hop")
    for p in w["all_pairs"]:
        typer.echo(
            f"  {p['levels'][0]}-{p['levels'][1]}    "
            f"{'yes' if p['avoided_crossing'] else ' no':>8}  "
            f"{p['position_km']:>9.1f}  {p['density']:>5.2f}   "
            f"{p['gap_km_inv']:>10.3e}  {p['oscillation_length_km']:>10.1f}  "
            f"{p['gamma']:>9.3g}  {p['hop_probability']:>6.3f}"
        )

    typer.echo(
        f"\nResonant pair {res['levels'][0]}-{res['levels'][1]}"
        f" (gap contrast {res['gap_contrast']:.1f}x)"
    )
    typer.echo(f"  avoided crossing at : {res['position_km']:.2f} km, rho = {res['density']:.2f}")
    typer.echo(f"  gap at crossing     : {res['gap_km_inv']:.4e} km^-1")
    typer.echo(f"  oscillation length  : {res['oscillation_length_km']:.1f} km")
    typer.echo(f"  adiabaticity gamma  : {res['gamma']:.3f}")
    typer.echo(f"  Landau-Zener hop    : {res['hop_probability']:.4f}")
    typer.echo(f"  path with gamma < 1 : {100 * res['danger_fraction']:.1f}%")
    typer.echo(
        "  verdict             : "
        + (
            "eigenbasis commands are unreliable in this zone"
            if res["gamma"] < 1
            else "adiabatic through the crossing, steering stays faithful"
        )
    )

    d = _outdir(out)
    fig = plots.plot_resonance(rep, d / "resonance.png")
    _write_json(d / "singularities.json", {"energy_gev": energy, "profile": profile, **w})
    typer.echo(f"\nWrote {fig}")


@app.command()
def verify(
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    steps: int = typer.Option(2000, "--steps", help="Integration steps"),
) -> None:
    """Self-consistency checks on the algebra, the integrator and Wong's law."""
    ok = True

    f = structure_constants(n)
    anti = max(
        np.max(np.abs(f + np.transpose(f, (1, 0, 2)))),
        np.max(np.abs(f - np.transpose(f, (1, 2, 0)))),
    )
    typer.echo(f"[1] f^abc total antisymmetry residual : {anti:.2e}")
    ok &= anti < 1e-10
    if n == 3:
        typer.echo(f"    f^123 = {f[0,1,2]:.4f} (expected 1.0000)")
        typer.echo(f"    f^458 = {f[3,4,7]:.4f} (expected {np.sqrt(3)/2:.4f})")
        ok &= abs(f[0, 1, 2] - 1.0) < 1e-10 and abs(f[3, 4, 7] - np.sqrt(3) / 2) < 1e-10

    h = FlavorHamiltonian(energy_gev=10.0, n=n)
    matter = MATTER_PROFILES["uniform"]()
    field = GAUGE_FIELDS["helical"](n=n)

    # With H_flavor switched off, <psi|T_a|psi> must obey Wong's precession law
    # exactly, since that is the equation the doc writes down.
    sim = QSE3Dynamics(h, matter, field, wong_only=True, topological_gain=0.0)
    s0 = boosted_state([0.995, 0.0, 0.0], _initial_state(n, 0))
    traj = sim.run(s0, 300.0, steps)
    wong = sim.wong_precession(traj)
    err = float(np.max(np.linalg.norm(wong - traj.charge, axis=1)))
    typer.echo(f"[2] Wong precession vs <psi|T_a|psi>  : {err:.2e}")
    ok &= err < 1e-6

    typer.echo(f"[3] unitarity  max||psi|^2 - 1|       : {np.max(np.abs(traj.norm - 1)):.2e}")
    typer.echo(f"[4] mass shell max|u.u - 1|           : {np.max(np.abs(traj.mass_shell - 1)):.2e}")
    cas = casimir_pure_state(n)
    dev = float(np.max(np.abs(traj.casimir - cas)))
    typer.echo(f"[5] Casimir I^aI^a = {cas:.4f}, dev     : {dev:.2e}")
    ok &= dev < 1e-8

    # Now switch the flavor Hamiltonian back on: the doc's precession law alone
    # no longer accounts for the motion of the internal state.
    sim2 = QSE3Dynamics(h, matter, field, wong_only=False, topological_gain=0.0)
    traj2 = sim2.run(s0, 300.0, steps)
    torque = float(np.max(np.linalg.norm(sim2.wong_precession(traj2) - traj2.charge, axis=1)))
    typer.echo(f"[6] flavor torque with H_flavor on    : {torque:.3e} (expected non-zero)")

    # Refractive energy M u^0 is exactly conserved at frozen flavor weight.
    from .physics import waveguide

    sim3 = QSE3Dynamics(
        h, waveguide(), GAUGE_FIELDS["off"](n), mass_km_inv=1e-2, topological_gain=5.0e3
    )
    frozen = boosted_state([0.99, 0.0, 0.0], _initial_state(n, 0), x0=np.array([0.0, 0.0, 40.0, 0.0]))
    sim3.wong_only = True  # freeze the flavor weight so M depends on x alone
    tr3 = sim3.run(frozen, 4000.0, steps)
    drift = float(np.max(np.abs(tr3.energy / tr3.energy[0] - 1)))
    typer.echo(f"[7] refractive energy M u^0 drift     : {drift:.2e}")
    typer.echo(
        f"    focusing check y: {tr3.x[0, 2]:.1f} -> {tr3.x[:, 2].min():.1f} km"
        " (matter waveguide)"
    )
    ok &= drift < 1e-6

    rho_res = resonance_density(10.0)
    typer.echo(f"[8] MSW resonance density at 10 GeV   : {rho_res:.3f} g/cm^3")

    typer.echo("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def control(
    target: str = typer.Option("0", "--target", "-t", help="Eigenstate index, 'uniform' or 'bell'"),
    flavor: int = typer.Option(1, "--flavor", help="Initial eigenstate index (0-based)"),
    actuation: str = typer.Option(
        "gauge", "--actuation", "-a", help=f"one of {list(ctl.ACTUATION_SETS)}"
    ),
    phase_gen: int = typer.Option(3, "--phase-gen", help="Generator index for theta_i"),
    twist_gen: int = typer.Option(7, "--twist-gen", help="Generator index for alpha_i"),
    energy: float = typer.Option(4.0, "--energy", "-E", help="Wave packet energy in GeV"),
    restarts: int = typer.Option(8, "--restarts", help="Random restarts"),
    max_iter: int = typer.Option(250, "--max-iter", help="Iterations per restart"),
    damping: float = typer.Option(0.08, "--damping", help="lambda_0 for damped least squares"),
    max_link: float = typer.Option(2000.0, "--max-link", help="Upper bound on each d_i, km"),
    undamped: bool = typer.Option(False, "--undamped", help="Use the raw pseudo-inverse"),
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
) -> None:
    """Invert the geometric Jacobian to find a joint schedule reaching a target state."""
    h = FlavorHamiltonian(energy_gev=energy, n=n)
    chain = KinematicChain(
        DEFAULT_CHAIN, h, phase_index=phase_gen, twist_index=twist_gen
    )
    psi0 = _initial_state(n, flavor)
    chi = ctl.named_target(n, target)
    names = _flavor_names(n)

    res = ctl.solve_best(
        chain,
        psi0,
        chi,
        restarts=restarts,
        actuation=actuation,
        max_iter=max_iter,
        lam0=damping,
        damped=not undamped,
        max_link=max_link,
    )

    typer.echo(f"Start state      : {names[min(flavor, n-1)]}")
    typer.echo(f"Target           : {target}")
    typer.echo(f"Actuated set     : {actuation} -> {len(res.actuated)} of {len(res.labels)} joints")
    typer.echo(f"Generators       : phase T_{phase_gen}, twist T_{twist_gen}")
    typer.echo(f"Damping          : {'off (raw pseudo-inverse)' if undamped else f'lambda_0 = {damping}'}")

    typer.echo(
        f"\nResult           : {'CONVERGED' if res.converged else 'STALLED'}"
        f" after {res.iterations} iterations"
    )
    typer.echo(f"  infidelity     : {res.final_infidelity:.3e}")
    typer.echo(f"  fidelity       : {1 - res.final_infidelity:.6f}")
    typer.echo(f"  min sigma seen : {res.sigma_min.min():.3e}")
    typer.echo(f"  damping engaged: {int(np.sum(res.damping > 0))} of {res.iterations} steps")
    total = sum(res.q[j] for j, lab in enumerate(res.labels) if lab.startswith("d_"))
    typer.echo(f"  chain length   : {total:.1f} km (link bound {max_link:.0f} km)")

    psi_f = chain.state(res.q, psi0)
    typer.echo("\nFinal occupation")
    for i, name in enumerate(names):
        typer.echo(f"  P({name:<7}) = {abs(psi_f[i])**2:.4f}")

    typer.echo("\nCommanded schedule")
    typer.echo("  joint        start        end       delta")
    for lab, a0, a1, da in res.command_table():
        typer.echo(f"  {lab:<9} {a0:>10.4f} {a1:>10.4f} {da:>11.4f}")

    if not res.converged:
        typer.echo(
            "\nThe target is out of reach for this actuation set. The commanded"
            "\ngenerators and the link propagators do not generate enough of"
            "\nsu(N) to move the state there; try --actuation all, or pick"
            "\ngenerators that touch the missing sector (e.g. --phase-gen 2"
            "\n--twist-gen 5 to open the nu_e channel), or raise --max-link so"
            "\nthe links accumulate enough oscillation phase on their own."
        )

    d = _outdir(out)
    fig = plots.plot_control(res, d / "control.png")
    _write_json(
        d / "control.json",
        {
            "target": target,
            "actuation": actuation,
            "converged": res.converged,
            "iterations": res.iterations,
            "infidelity": res.final_infidelity,
            "schedule": [
                {"joint": lab, "start": a0, "end": a1, "delta": da}
                for lab, a0, a1, da in res.command_table()
            ],
        },
    )
    typer.echo(f"\nWrote {fig}")


@app.command()
def swarm(
    robots: int = typer.Option(6, "--robots", "-r", help="Number of robots R (state dim N^R)"),
    mu_ent: float = typer.Option(1.0e-4, "--mu-ent", help="Engineered entangler, km^-1"),
    mu_nunu: float = typer.Option(2.0e-3, "--mu-nunu", help="nu-nu forward scattering, km^-1"),
    ent_range: float = typer.Option(150.0, "--ent-range", help="Entangler kernel length, km"),
    no_entangle: bool = typer.Option(False, "--no-entangle", help="Product states only"),
    controller: str = typer.Option("aware", "--controller", help="'aware' or 'naive'"),
    compare: bool = typer.Option(True, "--compare/--no-compare",
                                 help="Also run naive and unentangled baselines"),
    cancel_forces: bool = typer.Option(False, "--cancel-forces",
                                       help="Oracle feed-forward of the physical forces"),
    kappa_c: float = typer.Option(1.0e-3, "--kappa-c", help="Guidance neighbour coupling, km^-1"),
    kappa_p: float = typer.Option(3.0e-3, "--kappa-p", help="Guidance pinning gain, km^-1"),
    topology: str = typer.Option("ring", "--topology", help="Guidance graph: 'ring' or 'all'"),
    omega: float = typer.Option(0.02, "--omega", help="Tracking bandwidth omega, km^-1"),
    zeta: float = typer.Option(1.0, "--zeta", help="Damping ratio of the tracking loop"),
    k_form: float = typer.Option(2.0e-4, "--k-form", help="Distance-rigidity gain, km^-2"),
    k_cons: float = typer.Option(2.0e-3, "--k-cons", help="Velocity-consensus gain, km^-1"),
    a_max: float = typer.Option(0.03, "--a-max", help="Field-limited acceleration, km^-1"),
    eta_floor: float = typer.Option(0.35, "--eta-floor", help="Governor floor on eta_r"),
    radius: float = typer.Option(40.0, "--radius", help="Formation radius, km"),
    scatter: float = typer.Option(10.0, "--scatter", help="Initial guidance desync, km"),
    scale: float = typer.Option(8.0, "--scale", help="km per Lorenz unit"),
    v_max: float = typer.Option(0.55, "--v-max", help="Peak reference speed, c"),
    lorenz_times: float = typer.Option(6.0, "--lorenz-times", help="Duration, Lorenz units"),
    steps: int = typer.Option(3000, "--steps", help="Integration steps"),
    ftle_every: int = typer.Option(10, "--ftle-every", help="Steps per Lyapunov window"),
    no_twins: bool = typer.Option(False, "--no-twins",
                                  help="Skip the mechanical Benettin twins (7x faster)"),
    energy: float = typer.Option(4.0, "--energy", "-E", help="Wave packet energy in GeV"),
    gauge: str = typer.Option("helical", "--gauge", "-g", help=f"one of {list(GAUGE_FIELDS)}"),
    profile: str = typer.Option("uniform", "--profile", "-p",
                                help=f"one of {list(MATTER_PROFILES)}"),
    mass: float = typer.Option(5.0e-2, "--mass", help="Mass parameter m in km^-1"),
    gain: float = typer.Option(1.0, "--topological-gain", help="Scale of Q_topological"),
    n: int = typer.Option(3, "--dof", "-n", help="Internal degrees of freedom, SU(N)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output directory"),
) -> None:
    """Entangled swarm of R robots on a chaotic global trajectory, holding formation."""
    from . import swarm as sw
    from . import swarm_plots as sp

    if controller not in ("aware", "naive"):
        raise typer.BadParameter("--controller must be 'aware' or 'naive'")
    h = FlavorHamiltonian(energy_gev=energy, n=n)
    matter = MATTER_PROFILES[profile]()
    field = GAUGE_FIELDS[gauge](n=n) if gauge != "off" else GAUGE_FIELDS[gauge](n)
    ref = sw.LorenzReference(length_scale=scale, v_max=v_max, lorenz_times=lorenz_times)
    offsets = sw.formation_offsets(robots, radius)
    guide = sw.GuidanceNetwork(kappa_c=kappa_c, kappa_p=kappa_p, topology=topology)

    def build(aware: bool, entangle: bool) -> sw.EntangledSwarm:
        ctl = sw.FormationController(
            omega=omega, zeta=zeta, k_form=k_form, k_cons=k_cons, a_max=a_max,
            aware=aware, cancel_forces=cancel_forces, eta_floor=eta_floor,
            eta_target=max(eta_floor + 0.25, 0.6),
        )
        params = sw.SwarmParams(mass=mass, topological_gain=gain, mu_ent=mu_ent,
                                mu_nunu=mu_nunu, ent_range=ent_range, entangle=entangle)
        return sw.EntangledSwarm(h, matter, field, params, ctl, ref, offsets, guide)

    lam_ref = ref.lyapunov_spectrum()
    lam_max = float(lam_ref[0])
    main_sim = build(controller == "aware", not no_entangle)
    mu_min = guide.mu_min(robots)
    m_self = guide.self_gain(robots)
    mech_rate = main_sim.ctl.contraction_rate(offsets)
    anchors = [int(i) + 1 for i in np.flatnonzero(guide.graph(robots)[1])]

    typer.echo(f"Swarm: {robots} robots, joint SU({n}) state of dimension {n**robots}")
    typer.echo(f"Global trajectory: Lorenz attractor, {scale:g} km/unit, "
               f"T = {ref.time_scale:.0f} km per Lorenz time, {ref.duration_km:.0f} km long")
    typer.echo("  Lyapunov spectrum [km^-1]: " + ", ".join(f"{v:+.3e}" for v in lam_ref)
               + f"   (per Lorenz time: {lam_max * ref.time_scale:+.3f})")
    typer.echo(f"Guidance network: {topology}, anchors {anchors}, "
               f"kappa_c = {kappa_c:.1e}, kappa_p = {kappa_p:.1e} km^-1")
    typer.echo(f"  sync margin mu_min / lambda_max = {mu_min / lam_max:.2f}   "
               f"-> network transverse exponent {lam_max - mu_min:+.2e} km^-1   "
               + ("SYNCHRONIZES" if mu_min > lam_max else "NOT GUARANTEED: desync can grow"))
    typer.echo(f"Mechanical tracking '{controller}': predicted exponent {-mech_rate:+.2e} km^-1")
    typer.echo(f"Integrating {steps} steps"
               + ("" if no_twins else f" with {robots} Benettin twins") + " ...")

    s0 = main_sim.initial_state(scatter_km=scatter)
    traj = main_sim.run(s0, ref.duration_km, steps, twins=not no_twins, ftle_every=ftle_every)

    comparisons: dict[str, sw.SwarmTrajectory] = {}
    if compare:
        typer.echo("Baselines ...")
        if controller == "aware":
            b = build(False, not no_entangle)
            comparisons["naive controller, entangled"] = b.run(
                b.initial_state(scatter_km=scatter), ref.duration_km, steps, twins=False)
        if not no_entangle:
            b = build(controller == "aware", False)
            comparisons["no entanglement"] = b.run(
                b.initial_state(scatter_km=scatter), ref.duration_km, steps, twins=False)

    settle = len(traj.t) // 3  # skip the synchronization transient
    names = _flavor_names(n)

    def stats(tr: sw.SwarmTrajectory) -> dict:
        de = tr.distance_error[settle:]
        return {
            "distance_error_mean_pct": float(100 * de.mean()),
            "distance_error_max_pct": float(100 * de.max()),
            "sync_error_final_km": float(np.linalg.norm(tr.sync_error[-1], axis=1).max()),
            "tracking_error_max_km": float(
                np.linalg.norm(tr.tracking_error[settle:], axis=2).max()),
            "eta_final": tr.eta[-1].tolist(),
            "entropy_final_bits": tr.entropy[-1].tolist(),
            "saturation_fraction": float(tr.saturated.mean()),
        }

    main_stats = stats(traj)
    lg = traj.guidance_lyapunov
    lm = traj.mechanical_lyapunov
    pos_frac = (traj.lyap_guidance > 0).mean(axis=0)
    typer.echo("\nPer robot (main run)          guidance layer (chaotic)           "
               "mechanical layer")
    typer.echo("  robot  flavor0 S_end  eta_end    m_r    <lambda>  predicted  %t>0  "
               "<lambda>  predicted")
    for r in range(robots):
        tag = "*" if r + 1 in anchors else " "
        typer.echo(
            f"  {r + 1:>4}{tag}  {names[r % n]:>6} {traj.entropy[-1, r]:>5.2f}  "
            f"{traj.eta[-1, r]:>6.3f}  {m_self[r]:.1e} {lg[r]:+.2e} {lam_max - m_self[r]:+.2e}"
            f"  {100 * pos_frac[r]:>3.0f}  {lm[r]:+.2e} {-mech_rate:+.2e}"
        )
    typer.echo("  (* anchor: sees the global trajectory directly; units km^-1)")

    typer.echo("\nFormation after the transient: D(t) = worst-pair relative distance error")
    typer.echo(f"  {'run':<30} {'<D> %':>8} {'max D %':>8} {'sync km':>9} {'track km':>9}")
    rows = {f"{controller} (main)": main_stats} | {k: stats(v) for k, v in comparisons.items()}
    for k, v in rows.items():
        typer.echo(f"  {k:<30} {v['distance_error_mean_pct']:>8.3f} "
                   f"{v['distance_error_max_pct']:>8.3f} {v['sync_error_final_km']:>9.1e} "
                   f"{v['tracking_error_max_km']:>9.3f}")
    typer.echo(f"\n  |<Psi|Psi> - 1| max : {np.max(np.abs(traj.norm - 1)):.1e}")
    typer.echo(f"  |u.u - 1| max       : {np.max(np.abs(traj.mass_shell - 1)):.1e}")
    typer.echo(f"  max |v| on any slot : {np.max(np.linalg.norm(np.diff(traj.p_star, axis=0), axis=2)) / traj.meta['dt_km']:.3f} c")

    d = _outdir(out)
    figs = [
        sp.plot_swarm_paths(traj, d / "swarm_paths.png"),
        sp.plot_lyapunov_phase_3d(traj, d / "swarm_lyapunov_phase3d.png", "position"),
        sp.plot_lyapunov_phase_3d(traj, d / "swarm_lyapunov_sync3d.png", "sync"),
        sp.plot_swarm_dashboard(traj, d / "swarm_dashboard.png", lam_max, lam_max - mu_min,
                                eta_floor, comparisons),
        sp.plot_flavor_swarm(traj, d / "swarm_flavor.png", names),
    ]
    _write_json(
        d / "swarm.json",
        {
            "meta": traj.meta,
            "reference": {
                "length_scale_km": scale, "time_scale_km": ref.time_scale,
                "duration_km": ref.duration_km,
                "lyapunov_spectrum_km_inv": lam_ref.tolist(),
            },
            "guidance": {
                "mu_min_km_inv": mu_min, "sync_margin": mu_min / lam_max,
                "network_transverse_exponent_km_inv": lam_max - mu_min,
                "self_gain_km_inv": m_self.tolist(), "anchors": anchors,
                "conditional_lyapunov_km_inv": lg.tolist(),
                "fraction_time_locally_unstable": pos_frac.tolist(),
            },
            "mechanical": {
                "predicted_exponent_km_inv": -mech_rate,
                "conditional_lyapunov_km_inv": lm.tolist(),
            },
            "runs": rows,
        },
    )
    for f in figs:
        typer.echo(f"Wrote {f}")


def main() -> None:
    """Entry point declared in [project.scripts]."""
    app()


if __name__ == "__main__":
    main()
