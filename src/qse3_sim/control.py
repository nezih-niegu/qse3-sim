"""Control topology: inverting the geometric Jacobian to steer the state.

The operator's problem is the inverse-kinematics problem of a conventional
manipulator, transplanted onto CP^(N-1). Given a target internal state |chi>,
find a schedule of joint coordinates q such that U_total(q)|psi_0> = |chi> up
to a global phase.

The loop is resolved-rate control:

    e     = phase-aligned residual  |chi'> - |psi(q)>,  as a real 2N vector
    A_jk  = d psi_k / d q_j, projected orthogonally to |psi>, as a real 2N x m
    dq    = A^T (A A^T + lambda^2 I)^-1 e

`A A^T` is the Fubini-Study metric of the design document restricted to the
actuated coordinates, so this is literally an inversion of the geometric
Jacobian. The damping lambda is what keeps the loop finite at the kinematic
singularities that `singularity.py` locates: near a rank drop the undamped
pseudo-inverse asks for unbounded joint rates in the direction that has stopped
responding, while the damped one trades a little tracking error for a bounded
command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import KinematicChain

# Which DH coordinates the operator can actually command. theta_i is a gauge or
# Berry phase and alpha_i is an SU(N) twist, so both are set by the background
# field; d_i and a_i are properties of the environment the robot flies through.
ACTUATION_SETS: dict[str, tuple[str, ...]] = {
    "gauge": ("theta", "alpha"),
    "phase": ("theta",),
    "twist": ("alpha",),
    "environment": ("d", "a"),
    "all": ("theta", "d", "a", "alpha"),
}


@dataclass
class ControlResult:
    converged: bool
    iterations: int
    q0: np.ndarray
    q: np.ndarray
    actuated: np.ndarray  # indices of the commanded coordinates
    infidelity: np.ndarray  # 1 - |<chi|psi>|^2 per iteration
    sigma_min: np.ndarray  # smallest singular value of A per iteration
    damping: np.ndarray  # lambda actually applied per iteration
    step_norm: np.ndarray  # |dq| per iteration
    labels: list[str] = field(default_factory=list)

    @property
    def final_infidelity(self) -> float:
        return float(self.infidelity[-1])

    def command_table(self) -> list[tuple[str, float, float, float]]:
        """(label, start, end, delta) for each actuated coordinate."""
        rows = []
        for j in self.actuated:
            rows.append((self.labels[j], float(self.q0[j]), float(self.q[j]),
                         float(self.q[j] - self.q0[j])))
        return rows


# Largest sensible single-iteration change, per coordinate kind: radians for
# the phases and twists, km for the intervals, g/cm^3 for matter refraction.
_MAX_LINK_KM = 2000.0
_STEP_CAP = {"theta": 0.5, "alpha": 0.5, "d": 30.0, "a": 1.0}


def _apply_bounds(q: np.ndarray, labels: list[str], max_link: float = _MAX_LINK_KM) -> None:
    """Keep the environment coordinates physical: no negative lengths or densities."""
    for j, lab in enumerate(labels):
        kind = lab.split("_")[0]
        if kind == "d":
            q[j] = min(max(q[j], 10.0), max_link)
        elif kind == "a":
            q[j] = min(max(q[j], 0.0), 20.0)


def align_phase(psi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate the target by the global phase that best matches psi.

    Physical states are rays, so the residual must be measured after removing
    the phase freedom; otherwise the controller chases a coordinate artefact.
    """
    ov = np.vdot(psi, target)
    if abs(ov) < 1e-15:
        return target
    return target * np.conj(ov) / abs(ov)


def task_jacobian(
    chain: KinematicChain,
    q: np.ndarray,
    psi0: np.ndarray,
    actuated: np.ndarray,
    h: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (psi, A) with A the real 2N x m projected state Jacobian."""
    psi = chain.state(q, psi0)
    dpsi = chain.d_state(q, psi0, h)[:, actuated]
    # remove the component along |psi>: global phase carries no physical motion
    dpsi = dpsi - np.outer(psi, psi.conj() @ dpsi)
    return psi, np.vstack([dpsi.real, dpsi.imag])


def damped_step(a: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    """Minimum-norm damped least-squares step, via the SVD.

        dq = sum_k  sigma_k / (sigma_k^2 + lambda^2)  (u_k . e)  v_k

    Written this way it stays finite when A is exactly rank-deficient, which it
    always is here (the state manifold has fewer dimensions than the chain has
    joints), and it degrades smoothly to the plain pseudo-inverse at lambda=0.
    """
    u, sigma, vt = np.linalg.svd(a, full_matrices=False)
    if lam <= 0.0:
        cutoff = 1e-10 * (sigma[0] if sigma.size else 1.0)
        factor = np.where(sigma > cutoff, sigma / np.maximum(sigma**2, 1e-300), 0.0)
    else:
        factor = sigma / (sigma**2 + lam**2)
    return vt.T @ (factor * (u.T @ e))


def adaptive_damping(sigma_min: float, lam0: float, epsilon: float) -> float:
    """Nakamura-Hanafusa damping: zero away from singularities, lam0 at them."""
    if sigma_min >= epsilon:
        return 0.0
    return lam0 * np.sqrt(max(0.0, 1.0 - (sigma_min / epsilon) ** 2))


def solve(
    chain: KinematicChain,
    psi0: np.ndarray,
    target: np.ndarray,
    actuation: str = "gauge",
    max_iter: int = 200,
    tol: float = 1e-8,
    step: float = 1.0,
    lam0: float = 0.08,
    epsilon: float = 0.05,
    damped: bool = True,
    seed: int | None = 0,
    spread: float = 1e-3,
    max_step: float = 0.75,
    max_link: float = _MAX_LINK_KM,
) -> ControlResult:
    """Drive |psi(q)> onto |target> by damped least-squares resolved rates."""
    labels = chain.labels
    wanted = ACTUATION_SETS[actuation]
    actuated = np.array(
        [j for j, lab in enumerate(labels) if lab.split("_")[0] in wanted], dtype=int
    )
    if actuated.size == 0:
        raise ValueError(f"no coordinates selected by actuation set {actuation!r}")

    q = chain.pack().astype(float)
    if seed is not None and spread > 0:
        # Global multi-start: resample each actuated coordinate across its own
        # natural range rather than jittering the nominal pose. The map is
        # periodic in the angles, so a local jitter never leaves the basin it
        # started in and every restart returns the same answer.
        rng = np.random.default_rng(seed)
        for j in actuated:
            kind = labels[j].split("_")[0]
            if kind in ("theta", "alpha"):
                q[j] = rng.uniform(-np.pi, np.pi)
            elif kind == "d":
                q[j] = rng.uniform(20.0, max_link)
            else:
                q[j] = rng.uniform(0.0, 12.0)
        _apply_bounds(q, labels, max_link)
    q0 = q.copy()

    target = np.asarray(target, dtype=complex)
    target = target / np.linalg.norm(target)

    inf_hist, sig_hist, lam_hist, step_hist = [], [], [], []
    converged = False
    # The state manifold is CP^(N-1), so A has at most 2(N-1) non-zero singular
    # values however many joints are actuated. Damping has to key off the
    # smallest *meaningful* one; keying off sigma[-1] would see a structural
    # zero at every configuration and clamp the loop permanently.
    rank_max = 2 * (chain.n - 1)

    def infidelity_at(qq: np.ndarray) -> float:
        return 1.0 - abs(np.vdot(target, chain.state(qq, psi0))) ** 2

    for _ in range(max_iter):
        psi, a = task_jacobian(chain, q, psi0, actuated)
        residual = align_phase(psi, target) - psi
        e = np.concatenate([residual.real, residual.imag])

        infidelity = 1.0 - abs(np.vdot(target, psi)) ** 2
        sigma = np.linalg.svd(a, compute_uv=False)
        sigma_min = float(sigma[min(rank_max, len(sigma)) - 1])
        lam = adaptive_damping(sigma_min, lam0, epsilon) if damped else 0.0

        inf_hist.append(infidelity)
        sig_hist.append(sigma_min)
        lam_hist.append(lam)

        if infidelity < tol:
            step_hist.append(0.0)
            converged = True
            break

        # Column scaling. theta and alpha are radians, d is km, a is g/cm^3;
        # without preconditioning the length coordinates dominate the step and
        # the matter coordinates never move at all.
        col = np.linalg.norm(a, axis=0)
        # Floor the scaling. A coordinate that barely moves the state has a
        # near-zero column, and dividing by it turns a dead joint into the
        # loudest one in the command.
        col = np.maximum(col, 1e-3 * max(col.max(), 1e-12))
        dq = damped_step(a / col, e, lam) / col
        # Trust region. Poorly conditioned directions otherwise ask for joint
        # values of order 1e10 radians: numerically fine, since the map is
        # periodic, but useless as an operator schedule.
        cap = np.array([_STEP_CAP[labels[j].split("_")[0]] for j in actuated])
        excess = np.max(np.abs(dq) / cap)
        if excess > max_step:
            dq *= max_step / excess
        # Backtracking: the map q -> psi(q) is strongly nonlinear, so a full
        # Gauss-Newton stride overshoots and the loop limit-cycles.
        scale = step
        for _ in range(12):
            trial = q.copy()
            trial[actuated] += scale * dq
            _apply_bounds(trial, labels, max_link)
            if infidelity_at(trial) < infidelity:
                break
            scale *= 0.5
        else:
            step_hist.append(0.0)
            break

        step_hist.append(float(np.linalg.norm(scale * dq)))
        q = trial

    return ControlResult(
        converged=converged,
        iterations=len(inf_hist),
        q0=q0,
        q=q,
        actuated=actuated,
        infidelity=np.array(inf_hist),
        sigma_min=np.array(sig_hist),
        damping=np.array(lam_hist),
        step_norm=np.array(step_hist),
        labels=labels,
    )


def named_target(n: int, name: str) -> np.ndarray:
    """Convenience targets: an eigenstate index, 'uniform', or 'bell'."""
    name = name.strip().lower()
    if name.isdigit():
        v = np.zeros(n, dtype=complex)
        v[min(int(name), n - 1)] = 1.0
        return v
    if name == "uniform":
        return np.ones(n, dtype=complex) / np.sqrt(n)
    if name == "bell" and n >= 2:
        v = np.zeros(n, dtype=complex)
        v[0] = 1.0
        v[1] = 1j
        return v / np.sqrt(2)
    raise ValueError(f"unknown target {name!r}")


def solve_best(
    chain: KinematicChain,
    psi0: np.ndarray,
    target: np.ndarray,
    restarts: int = 8,
    **kwargs,
) -> ControlResult:
    """Run `solve` from several seeds and keep the best result.

    The map q -> |psi(q)> is periodic and strongly nonlinear, so a single
    Gauss-Newton descent lands in whichever basin it started in. Restarts are
    the cheapest way to tell a genuine reachability limit apart from a local
    minimum, which matters here because the two have completely different
    fixes: change the background field, or just try again.
    """
    best: ControlResult | None = None
    for k in range(max(1, restarts)):
        kwargs["seed"] = None if k == 0 else 1000 + 37 * k
        scale = 1.0 if k == 0 else 0.9
        res = solve(chain, psi0, target, spread=0.0 if k == 0 else scale, **kwargs)
        if best is None or res.final_infidelity < best.final_infidelity:
            best = res
        if best.converged:
            break
    assert best is not None
    return best
