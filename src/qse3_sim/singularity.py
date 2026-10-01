"""Kinematic singularities: MSW resonance zones where control breaks down.

Two distinct failure modes are diagnosed here.

1. Geometric singularity. The Fubini-Study metric built from the geometric
   Jacobian loses rank, so some combination of joint motions no longer moves
   the internal state at all. Handled in `kinematics.manipulability`.

2. Dynamical singularity (MSW resonance). Two instantaneous eigenvalues of
   H(V(s)) approach each other. The effective mixing angle in matter sweeps
   through 45 degrees, the level gap collapses, and the adiabaticity ratio

       gamma_ij = |E_i - E_j| / (2 |<i| d/ds |j>|)

   falls to order unity. Below that, the state no longer follows its
   instantaneous eigenvector and hops with Landau-Zener probability
   P = exp(-pi gamma / 2). Steering commands issued in the eigenbasis stop
   doing what the operator intends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import FlavorHamiltonian, matter_potential_km_inv, resonance_density


@dataclass
class PairDiagnostic:
    """Everything about one pair of instantaneous levels along the path."""

    levels: tuple[int, int]
    gap: np.ndarray  # km^-1
    adiabaticity: np.ndarray  # gamma
    hop: np.ndarray  # Landau-Zener probability
    crossing_index: int  # interior gap minimum, -1 if none
    contrast: float  # boundary gap / interior minimum gap

    @property
    def has_crossing(self) -> bool:
        return self.crossing_index >= 0 and self.contrast > 1.05

    def summary(self, s: np.ndarray, rho: np.ndarray) -> dict:
        i = self.crossing_index if self.has_crossing else int(np.argmin(self.gap))
        return {
            "levels": [int(self.levels[0] + 1), int(self.levels[1] + 1)],
            "avoided_crossing": bool(self.has_crossing),
            "position_km": float(s[i]),
            "density": float(rho[i]),
            "gap_km_inv": float(self.gap[i]),
            "oscillation_length_km": float(2 * np.pi / max(self.gap[i], 1e-30)),
            "gamma": float(self.adiabaticity[i]),
            "hop_probability": float(self.hop[i]),
            "gap_contrast": float(self.contrast),
            "danger_fraction": float(np.mean(self.adiabaticity < 1.0)),
        }


@dataclass
class ResonanceReport:
    s: np.ndarray  # path coordinate, km
    rho: np.ndarray  # density, g/cm^3
    eigenvalues: np.ndarray  # (steps, N), km^-1
    pairs: list[PairDiagnostic]
    effective_theta13: np.ndarray  # degrees, two-flavor matter angle
    resonance_rho: float  # g/cm^3 for the given energy

    @property
    def resonant_pair(self) -> PairDiagnostic:
        """The pair with the sharpest avoided crossing on this path."""
        crossers = [p for p in self.pairs if p.has_crossing]
        pool = crossers or self.pairs
        return max(pool, key=lambda p: p.contrast)

    @property
    def min_gap(self) -> np.ndarray:
        return np.min(np.array([p.gap for p in self.pairs]), axis=0)

    @property
    def adiabaticity(self) -> np.ndarray:
        return self.resonant_pair.adiabaticity

    @property
    def danger_mask(self) -> np.ndarray:
        return self.resonant_pair.adiabaticity < 1.0

    def worst(self) -> dict:
        res = self.resonant_pair
        return {
            "resonant": res.summary(self.s, self.rho),
            "all_pairs": [p.summary(self.s, self.rho) for p in self.pairs],
            "resonance_rho": self.resonance_rho,
        }


def effective_mixing_angle(
    rho: np.ndarray | float,
    energy_gev: float,
    dm_sq: float = 2.515e-3,
    theta_deg: float = 8.57,
    y_e: float = 0.5,
) -> np.ndarray:
    """Two-flavor matter mixing angle: tan(2 theta_m) = sin2t / (cos2t - A/dm2)."""
    from .physics import EV2_PER_GEV_TO_KM_INV

    theta = np.radians(theta_deg)
    v = matter_potential_km_inv(np.asarray(rho, dtype=float), y_e)
    dm_km = EV2_PER_GEV_TO_KM_INV * dm_sq / (2.0 * energy_gev)
    ratio = v / dm_km  # = A / dm^2, with A = 2 E V
    return np.degrees(0.5 * np.arctan2(np.sin(2 * theta), np.cos(2 * theta) - ratio))


def analyze_path(
    s: np.ndarray,
    rho: np.ndarray,
    hamiltonian: FlavorHamiltonian,
    y_e: float = 0.5,
) -> ResonanceReport:
    """Instantaneous eigenstructure of H along a path, plus adiabaticity."""
    s = np.asarray(s, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n = hamiltonian.n
    steps = len(s)

    evals = np.zeros((steps, n))
    evecs = np.zeros((steps, n, n), dtype=complex)
    for i, r in enumerate(rho):
        h = hamiltonian.total(matter_potential_km_inv(r, y_e))
        w, v = np.linalg.eigh(h)
        # fix arbitrary phase / sign against the previous step for smoothness
        if i > 0:
            for k in range(n):
                ov = np.vdot(evecs[i - 1, :, k], v[:, k])
                if abs(ov) > 1e-12:
                    v[:, k] *= np.conj(ov) / abs(ov)
        evals[i] = w
        evecs[i] = v

    ds = np.gradient(s)
    dvec = np.gradient(evecs, axis=0) / ds[:, None, None]

    pairs: list[PairDiagnostic] = []
    for a in range(n):
        for b in range(a + 1, n):
            gap = np.abs(evals[:, a] - evals[:, b])
            # <a| d/ds |b> is the rate at which the eigenbasis itself rotates
            rate = np.abs(np.einsum("si,si->s", evecs[:, :, a].conj(), dvec[:, :, b]))
            gamma = np.where(rate > 1e-30, gap / (2.0 * np.maximum(rate, 1e-30)), np.inf)
            hop = np.exp(-0.5 * np.pi * np.clip(gamma, 0.0, 700.0))

            interior = gap[1:-1]
            idx = int(np.argmin(interior)) + 1 if steps > 2 else 0
            boundary = min(gap[0], gap[-1])
            is_local_min = 0 < idx < steps - 1 and gap[idx] < boundary
            contrast = float(boundary / max(gap[idx], 1e-30)) if is_local_min else 1.0
            pairs.append(
                PairDiagnostic(
                    levels=(a, b),
                    gap=gap,
                    adiabaticity=gamma,
                    hop=hop,
                    crossing_index=idx if is_local_min else -1,
                    contrast=contrast,
                )
            )

    return ResonanceReport(
        s=s,
        rho=rho,
        eigenvalues=evals,
        pairs=pairs,
        effective_theta13=effective_mixing_angle(rho, hamiltonian.energy_gev, y_e=y_e),
        resonance_rho=resonance_density(hamiltonian.energy_gev, y_e=y_e),
    )
