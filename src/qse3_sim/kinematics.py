"""Forward kinematics on C = Min(1,3) x SU(N).

Each "link" of the chain contributes a unitary built from the four
quantum-relativistic DH parameters:

    theta_i : joint phase        -> exp(-i theta_i T_phase)
    d_i     : spacetime interval -> propagation length of the link
    a_i     : matter refraction  -> MSW potential strength on that link
    alpha_i : topological twist  -> exp(-i alpha_i T_twist)

    A_i = exp(-i alpha_i T_tw) exp(-i (H_vac + a_i V) d_i) exp(-i theta_i T_ph)

and the chain is composed as a path-ordered product U_total = A_n ... A_1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .algebra import generators
from .physics import FlavorHamiltonian, matter_potential_km_inv


@dataclass
class Link:
    """One row of the quantum-relativistic DH table."""

    theta: float  # rad
    d: float  # km
    a: float  # g/cm^3 (matter refraction)
    alpha: float  # rad


DEFAULT_CHAIN: tuple[Link, ...] = (
    Link(theta=0.00, d=120.0, a=0.30, alpha=0.00),
    Link(theta=0.60, d=180.0, a=3.20, alpha=0.35),
    Link(theta=1.20, d=240.0, a=5.00, alpha=-0.50),
    Link(theta=0.25, d=150.0, a=0.80, alpha=0.90),
)


class KinematicChain:
    """Path-ordered forward kinematics plus the geometric Jacobian."""

    def __init__(
        self,
        links: tuple[Link, ...] = DEFAULT_CHAIN,
        hamiltonian: FlavorHamiltonian | None = None,
        phase_index: int = 3,
        twist_index: int = 7,
        y_e: float = 0.5,
    ) -> None:
        self.links = tuple(links)
        self.y_e = y_e
        self.h = hamiltonian or FlavorHamiltonian()
        self.n = self.h.n
        t = generators(self.n)
        self.t_phase = t[min(phase_index, len(t)) - 1]
        self.t_twist = t[min(twist_index, len(t)) - 1]

    # -- parameter packing -------------------------------------------------

    def pack(self) -> np.ndarray:
        """Joint vector q = [theta_1..n, d_1..n, a_1..n, alpha_1..n]."""
        return np.concatenate(
            [
                [ll.theta for ll in self.links],
                [ll.d for ll in self.links],
                [ll.a for ll in self.links],
                [ll.alpha for ll in self.links],
            ]
        )

    def unpack(self, q: np.ndarray) -> tuple[Link, ...]:
        k = len(self.links)
        return tuple(Link(q[i], q[k + i], q[2 * k + i], q[3 * k + i]) for i in range(k))

    @property
    def labels(self) -> list[str]:
        k = len(self.links)
        return (
            [f"theta_{i+1}" for i in range(k)]
            + [f"d_{i+1}" for i in range(k)]
            + [f"a_{i+1}" for i in range(k)]
            + [f"alpha_{i+1}" for i in range(k)]
        )

    # -- forward kinematics -----------------------------------------------

    def link_unitary(self, link: Link) -> np.ndarray:
        v = matter_potential_km_inv(link.a, self.y_e)
        prop = expm(-1j * self.h.total(v) * link.d)
        return expm(-1j * link.alpha * self.t_twist) @ prop @ expm(-1j * link.theta * self.t_phase)

    def forward(self, q: np.ndarray | None = None) -> np.ndarray:
        """U_total = P exp(-i int [H_geom + A_mu dx^mu/ds] ds), link by link."""
        links = self.links if q is None else self.unpack(q)
        u = np.eye(self.n, dtype=complex)
        for link in links:
            u = self.link_unitary(link) @ u
        return u

    def state(self, q: np.ndarray | None = None, psi0: np.ndarray | None = None) -> np.ndarray:
        if psi0 is None:
            psi0 = np.zeros(self.n, dtype=complex)
            psi0[0] = 1.0
        return self.forward(q) @ psi0

    def transition_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        """P(nu_alpha -> nu_beta) = |U_total|^2."""
        return np.abs(self.forward(q)) ** 2

    # -- geometric Jacobian ------------------------------------------------

    def d_state(
        self, q: np.ndarray, psi0: np.ndarray | None = None, h: float = 1e-6
    ) -> np.ndarray:
        """Columns are |d psi / d q_j> by central differences."""
        cols = []
        for j in range(len(q)):
            qp, qm = q.copy(), q.copy()
            step = h * max(1.0, abs(q[j]))
            qp[j] += step
            qm[j] -= step
            cols.append((self.state(qp, psi0) - self.state(qm, psi0)) / (2 * step))
        return np.array(cols).T

    def quantum_geometric_tensor(
        self, q: np.ndarray | None = None, psi0: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Q, g_FS, berry_curvature).

        Q_jk = <d_j psi|(1 - |psi><psi|)|d_k psi>, so that
        g_jk = Re Q_jk is the Fubini-Study metric (the geometric Jacobian's
        Gram matrix) and F_jk = -2 Im Q_jk is the Berry curvature.
        """
        q = self.pack() if q is None else q
        psi = self.state(q, psi0)
        dpsi = self.d_state(q, psi0)
        overlap = psi.conj() @ dpsi  # <psi|d_j psi>
        qgt = dpsi.conj().T @ dpsi - np.outer(overlap.conj(), overlap)
        return qgt, np.real(qgt), -2.0 * np.imag(qgt)

    def manipulability(
        self, q: np.ndarray | None = None, psi0: np.ndarray | None = None, tol: float = 1e-12
    ) -> dict:
        """Rank / conditioning diagnostics of the geometric Jacobian.

        The reachable state manifold is CP^(N-1), of real dimension 2(N-1), so
        the Fubini-Study metric can never have rank above that. A drop below
        that ceiling is a genuine kinematic singularity: some combination of
        joint motions produces no change of internal state at all.
        """
        _, g_fs, berry = self.quantum_geometric_tensor(q, psi0)
        evals = np.linalg.eigvalsh(g_fs)
        evals = np.sort(evals)[::-1]
        max_rank = 2 * (self.n - 1)
        positive = evals[evals > tol * max(1.0, evals[0])]
        rank = len(positive)
        return {
            "eigenvalues": evals,
            "rank": rank,
            "max_rank": max_rank,
            "singular": rank < max_rank,
            "volume": float(np.sqrt(np.prod(positive))) if rank else 0.0,
            "condition": float(evals[0] / positive[-1]) if rank else np.inf,
            "berry_norm": float(np.linalg.norm(berry)),
        }
