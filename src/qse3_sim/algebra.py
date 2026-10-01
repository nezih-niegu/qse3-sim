"""SU(N) Lie algebra machinery for the internal state space.

Implements the generalized Gell-Mann basis T_a (a = 1 .. N^2-1) normalized as

    Tr(T_a T_b) = delta_ab / 2,        [T_a, T_b] = i f^{abc} T_c

together with the totally antisymmetric structure constants f^{abc} and the
PMNS-style mixing matrix used as the "topological twist" reference frame.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def generators(n: int) -> tuple[np.ndarray, ...]:
    """Return the N^2-1 generalized Gell-Mann matrices T_a = lambda_a / 2.

    Ordering for N=3 reproduces the standard Gell-Mann ordering, so T_3 and
    T_8 are the diagonal (Cartan) generators and T_2, T_5, T_7 are the
    antisymmetric 1-2, 1-3 and 2-3 rotation generators.
    """
    if n < 2:
        raise ValueError("SU(N) requires N >= 2")

    mats: list[np.ndarray] = []
    for k in range(n):
        for ll in range(k + 1, n):
            sym = np.zeros((n, n), dtype=complex)
            sym[k, ll] = sym[ll, k] = 0.5
            asym = np.zeros((n, n), dtype=complex)
            asym[k, ll] = -0.5j
            asym[ll, k] = +0.5j
            mats.append(sym)
            mats.append(asym)

    diag: list[np.ndarray] = []
    for m in range(1, n):
        d = np.zeros(n, dtype=complex)
        d[:m] = 1.0
        d[m] = -m
        diag.append(np.diag(d) / np.sqrt(2.0 * m * (m + 1)))

    # Interleave so that N=3 gives exactly (l1..l8)/2 in the canonical order.
    if n == 3:
        ordered = [mats[0], mats[1], diag[0], mats[2], mats[3], mats[4], mats[5], diag[1]]
    else:
        ordered = mats + diag
    return tuple(np.ascontiguousarray(m) for m in ordered)


@lru_cache(maxsize=8)
def structure_constants(n: int) -> np.ndarray:
    """Totally antisymmetric f^{abc} from f^{abc} = -2i Tr([T_a, T_b] T_c)."""
    t = generators(n)
    dim = len(t)
    f = np.zeros((dim, dim, dim))
    for a in range(dim):
        for b in range(dim):
            comm = t[a] @ t[b] - t[b] @ t[a]
            for c in range(dim):
                f[a, b, c] = np.real(-2j * np.trace(comm @ t[c]))
    f[np.abs(f) < 1e-12] = 0.0
    return f


def casimir_pure_state(n: int) -> float:
    """Value of sum_a <T_a>^2 for any pure state in the fundamental rep."""
    return (n - 1) / (2.0 * n)


def color_charge(psi: np.ndarray) -> np.ndarray:
    """Wong color charge I^a = <psi| T_a |psi> (real for Hermitian T_a)."""
    t = generators(psi.shape[0])
    return np.array([np.real(np.vdot(psi, m @ psi)) for m in t])


# ---------------------------------------------------------------------------
# Mixing matrix
# ---------------------------------------------------------------------------

# NuFIT-style normal-ordering best fits (degrees). These set the default
# "topological twist" frame alpha_i between internal eigenstates.
PMNS_DEFAULT = {"theta12": 33.44, "theta13": 8.57, "theta23": 49.2, "delta_cp": 194.0}


def pmns(
    theta12: float = PMNS_DEFAULT["theta12"],
    theta13: float = PMNS_DEFAULT["theta13"],
    theta23: float = PMNS_DEFAULT["theta23"],
    delta_cp: float = PMNS_DEFAULT["delta_cp"],
) -> np.ndarray:
    """Standard 3x3 PMNS matrix; angles in degrees."""
    s12, c12 = np.sin(np.radians(theta12)), np.cos(np.radians(theta12))
    s13, c13 = np.sin(np.radians(theta13)), np.cos(np.radians(theta13))
    s23, c23 = np.sin(np.radians(theta23)), np.cos(np.radians(theta23))
    d = np.exp(1j * np.radians(delta_cp))

    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.conj(d)],
            [
                -s12 * c23 - c12 * s23 * s13 * d,
                c12 * c23 - s12 * s23 * s13 * d,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * d,
                -c12 * s23 - s12 * c23 * s13 * d,
                c23 * c13,
            ],
        ],
        dtype=complex,
    )


def su_n_mixing(n: int) -> np.ndarray:
    """Reference mixing matrix for arbitrary N (PMNS for N=3, DFT otherwise)."""
    if n == 3:
        return pmns()
    j = np.arange(n)
    return np.exp(2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)


def is_unitary(u: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=tol))
