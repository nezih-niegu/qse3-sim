"""Units, Hamiltonians, matter profiles and background gauge fields.

Unit convention (natural units, hbar = c = 1):

    length, time, proper time   ->  km
    energy, Hamiltonian, A_mu   ->  km^-1
    field strength F_munu       ->  km^-2
    mass parameter m            ->  km^-1
    color charge I^a, coupling g -> dimensionless

Metric signature is (+,-,-,-) with x^mu = (t, x, y, z) in km.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .algebra import generators, structure_constants, su_n_mixing

# ---------------------------------------------------------------------------
# Constants and conversions
# ---------------------------------------------------------------------------

HBAR_C_EV_KM = 1.973269804e-10  # hbar*c in eV km
KM_INV_PER_EV = 1.0 / HBAR_C_EV_KM  # 5.0677e9 km^-1 per eV
G_F_GEV2 = 1.1663787e-5  # Fermi constant, GeV^-2
N_AVOGADRO = 6.02214076e23
HBAR_C_GEV_CM = 1.973269804e-14  # hbar*c in GeV cm

# sqrt(2) G_F N_A expressed in eV cm^3 / g  ->  V = SQRT2_GF_NA * rho * Y_e  [eV]
SQRT2_GF_NA_EV = (
    np.sqrt(2.0) * G_F_GEV2 * N_AVOGADRO * HBAR_C_GEV_CM**3 * 1.0e9
)  # ~7.63e-14

# (1 eV^2 / GeV) expressed in km^-1; equals 2 * 1.2669, the usual oscillation factor.
EV2_PER_GEV_TO_KM_INV = 1.0e-9 * KM_INV_PER_EV

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])

# Oscillation parameters (normal ordering), eV^2
DM21_SQ = 7.42e-5
DM31_SQ = 2.515e-3


def matter_potential_km_inv(rho_g_cm3: float | np.ndarray, y_e: float = 0.5):
    """Charged-current matter potential V = sqrt(2) G_F N_e, in km^-1."""
    return SQRT2_GF_NA_EV * rho_g_cm3 * y_e * KM_INV_PER_EV


def resonance_density(
    energy_gev: float, dm_sq: float = DM31_SQ, theta_deg: float = 8.57, y_e: float = 0.5
) -> float:
    """Matter density (g/cm^3) at which the MSW resonance condition is met."""
    v_res_km = (
        EV2_PER_GEV_TO_KM_INV * dm_sq * np.cos(2 * np.radians(theta_deg)) / (2.0 * energy_gev)
    )
    return float(v_res_km / (SQRT2_GF_NA_EV * KM_INV_PER_EV * y_e))


# ---------------------------------------------------------------------------
# Matter profiles: the source of Q_topological
# ---------------------------------------------------------------------------


@dataclass
class MatterProfile:
    """Electron-density environment n_e(x), expressed via mass density."""

    name: str
    density: Callable[[np.ndarray], float]
    y_e: float = 0.5

    def rho(self, x: np.ndarray) -> float:
        """Mass density in g/cm^3 at 4-position x."""
        return float(self.density(x))

    def potential(self, x: np.ndarray) -> float:
        """MSW potential V(x) in km^-1."""
        return float(matter_potential_km_inv(self.rho(x), self.y_e))

    def grad_potential(self, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """Covariant gradient d_mu V, 4-vector with lower index, km^-2."""
        g = np.zeros(4)
        for mu in range(4):
            xp, xm = x.copy(), x.copy()
            xp[mu] += h
            xm[mu] -= h
            g[mu] = (self.potential(xp) - self.potential(xm)) / (2 * h)
        return g


def uniform_profile(rho: float = 3.0, y_e: float = 0.5) -> MatterProfile:
    return MatterProfile("uniform", lambda x: rho, y_e)


def gaussian_slab(
    rho_bg: float = 0.1,
    rho_peak: float = 6.0,
    center_km: float = 200.0,
    width_km: float = 45.0,
    y_e: float = 0.5,
) -> MatterProfile:
    """A dense slab centred on the x-axis: drives the robot through resonance."""

    def dens(x: np.ndarray) -> float:
        return rho_bg + rho_peak * np.exp(-(((x[1] - center_km) / width_km) ** 2))

    return MatterProfile("gaussian-slab", dens, y_e)


def linear_ramp(
    rho0: float = 0.2, slope: float = 0.05, y_e: float = 0.5
) -> MatterProfile:
    return MatterProfile("linear-ramp", lambda x: max(0.0, rho0 + slope * x[1]), y_e)


def waveguide(
    rho_axis: float = 7.87,
    curvature: float = 2.0e-3,
    y_e: float = 0.5,
) -> MatterProfile:
    """Matter channel whose density rises quadratically off the x-axis.

    The transverse gradient produces a genuine restoring force through
    Q_topological, but only on the nu_e component of the internal state, so the
    robot is focused exactly as strongly as it is electron-flavored. Default
    on-axis density sits at the MSW resonance for a 4 GeV wave packet.
    """

    def dens(x: np.ndarray) -> float:
        r2 = x[2] ** 2 + x[3] ** 2
        return rho_axis + curvature * r2

    return MatterProfile("waveguide", dens, y_e)


def stellar(
    rho_core: float = 150.0, scale_height_km: float = 3.0e6, y_e: float = 0.67
) -> MatterProfile:
    """Exponentially stratified stellar interior.

    The density scale height is enormous compared with the oscillation length
    at resonance, so the crossing is adiabatic: the state tracks its
    instantaneous eigenvector and steering commands stay faithful. This is the
    contrast case to the sharp `gaussian-slab`.
    """

    def dens(x: np.ndarray) -> float:
        return rho_core * np.exp(-max(x[1], 0.0) / scale_height_km)

    return MatterProfile("stellar", dens, y_e)


def earth_chord(y_e: float = 0.5) -> MatterProfile:
    """Two-shell PREM-like density as a function of radius from the origin."""
    r_earth, r_core = 6371.0, 3480.0

    def dens(x: np.ndarray) -> float:
        r = float(np.linalg.norm(x[1:]))
        if r > r_earth:
            return 0.0
        if r > r_core:
            return 5.6 - 2.4 * (r / r_earth)  # mantle
        return 12.6 - 3.9 * (r / r_core) ** 2  # core

    return MatterProfile("earth-chord", dens, y_e)


MATTER_PROFILES: dict[str, Callable[[], MatterProfile]] = {
    "uniform": uniform_profile,
    "gaussian-slab": gaussian_slab,
    "linear-ramp": linear_ramp,
    "waveguide": waveguide,
    "stellar": stellar,
    "earth-chord": earth_chord,
}


# ---------------------------------------------------------------------------
# Flavor Hamiltonian: H_geom in the path-ordered exponential
# ---------------------------------------------------------------------------


@dataclass
class FlavorHamiltonian:
    """Vacuum + MSW Hamiltonian in the flavor basis, in km^-1."""

    energy_gev: float = 10.0
    n: int = 3
    dm_sq: tuple[float, ...] = (0.0, DM21_SQ, DM31_SQ)
    mixing: np.ndarray = field(default_factory=lambda: su_n_mixing(3))

    def __post_init__(self) -> None:
        if self.mixing.shape[0] != self.n:
            self.mixing = su_n_mixing(self.n)
        if len(self.dm_sq) != self.n:
            extra = tuple(
                self.dm_sq[-1] * (1.0 + 0.5 * k) for k in range(1, self.n - len(self.dm_sq) + 1)
            )
            self.dm_sq = tuple(self.dm_sq) + extra
        # H = (1/2E) U diag(m_i^2) U^dagger, converted from eV^2/GeV into km^-1.
        self._vac = (
            EV2_PER_GEV_TO_KM_INV
            / (2.0 * self.energy_gev)
            * (self.mixing @ np.diag(np.asarray(self.dm_sq, dtype=complex)) @ self.mixing.conj().T)
        )

    @property
    def vacuum(self) -> np.ndarray:
        return self._vac

    def matter_term(self, potential_km_inv: float) -> np.ndarray:
        """diag(V, 0, ..., 0): only nu_e feels charged-current scattering."""
        m = np.zeros((self.n, self.n), dtype=complex)
        m[0, 0] = potential_km_inv
        return m

    def total(self, potential_km_inv: float) -> np.ndarray:
        return self._vac + self.matter_term(potential_km_inv)


# ---------------------------------------------------------------------------
# Background gauge field A^a_mu(x): the operator's control handle
# ---------------------------------------------------------------------------


@dataclass
class GaugeField:
    """Non-Abelian background potential A^a_mu(x) and its field strength.

    ``potential(x)`` returns an array of shape (4, N^2-1) holding A^a_mu with a
    LOWER spacetime index. Field strengths are formed numerically:

        F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu - g f^{abc} A^b_mu A^c_nu

    The sign of the commutator term is fixed by the internal Hamiltonian
    H_gauge = +g A^a_mu xdot^mu T_a, i.e. the covariant derivative
    D_mu = d_mu + i g A_mu. Gauge covariance requires F = [D_mu, D_nu] / (i g),
    and a pure-gauge potential then has F = 0 exactly (tested). An earlier
    version used +g f A A, which is the convention for D = d - i g A and was
    inconsistent with the Hamiltonian.
    """

    name: str
    potential: Callable[[np.ndarray], np.ndarray]
    coupling: float = 1.0
    n: int = 3

    def a_mu(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.potential(x), dtype=float)

    def field_strength(self, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """F^a_{mu nu} with both spacetime indices lowered, shape (4, 4, dim)."""
        f_abc = structure_constants(self.n)
        a = self.a_mu(x)
        d_a = np.zeros((4, 4, a.shape[1]))  # d_mu A_nu
        for mu in range(4):
            xp, xm = x.copy(), x.copy()
            xp[mu] += h
            xm[mu] -= h
            d_a[mu] = (self.a_mu(xp) - self.a_mu(xm)) / (2 * h)
        f = d_a - np.transpose(d_a, (1, 0, 2))
        f -= self.coupling * np.einsum("abc,mb,nc->mna", f_abc, a, a)
        return f

    def field_strength_upper(self, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
        """F^{a mu nu} with both spacetime indices raised."""
        f_low = self.field_strength(x, h)
        return np.einsum("ma,nb,abc->mnc", METRIC, METRIC, f_low)

    def connection_term(self, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        """H_gauge = g A^a_mu (dx^mu/ds) T_a, an N x N Hermitian matrix."""
        t = generators(self.n)
        a = self.a_mu(x)
        coeff = self.coupling * np.einsum("m,ma->a", xdot, a)
        return sum(c * m for c, m in zip(coeff, t))


def _zero_field(n: int = 3) -> GaugeField:
    dim = n * n - 1
    return GaugeField("off", lambda x: np.zeros((4, dim)), 0.0, n)


def chromomagnetic(b: float = 3.0e-4, n: int = 3, g: float = 1.0) -> GaugeField:
    """Uniform color-3 magnetic field along z: A^3_y = B x  =>  F^3_xy = B."""
    dim = n * n - 1

    def a(x: np.ndarray) -> np.ndarray:
        out = np.zeros((4, dim))
        out[2, 2] = b * x[1]
        return out

    return GaugeField("chromomagnetic", a, g, n)


def chromoelectric(e: float = 1.0e-4, n: int = 3, g: float = 1.0) -> GaugeField:
    """Color-8 electric field along x: A^8_0 = -E x  =>  F^8_{01} = E."""
    dim = n * n - 1

    def a(x: np.ndarray) -> np.ndarray:
        out = np.zeros((4, dim))
        out[0, min(7, dim - 1)] = -e * x[1]
        return out

    return GaugeField("chromoelectric", a, g, n)


def nonabelian_crossed(c: float = 2.0e-2, n: int = 3, g: float = 1.0) -> GaugeField:
    """Constant crossed potentials A^1_x = A^2_y = c.

    Every derivative vanishes, so the entire field strength comes from the
    commutator: F^3_xy = -g f^{123} c^2 = -g c^2. Pure non-Abelian steering.
    """
    dim = n * n - 1

    def a(x: np.ndarray) -> np.ndarray:
        out = np.zeros((4, dim))
        out[1, 0] = c
        out[2, 1] = c
        return out

    return GaugeField("nonabelian-crossed", a, g, n)


def helical(
    b: float = 2.0e-4, c: float = 3.0e-3, wavelength_km: float = 120.0, n: int = 3, g: float = 1.0
) -> GaugeField:
    """Chromomagnetic guide field plus a helically wound color-1 potential."""
    dim = n * n - 1
    k = 2 * np.pi / wavelength_km

    def a(x: np.ndarray) -> np.ndarray:
        out = np.zeros((4, dim))
        out[2, 2] = b * x[1]
        out[3, 0] = c * np.cos(k * x[1])
        out[3, 1] = c * np.sin(k * x[1])
        return out

    return GaugeField("helical", a, g, n)


GAUGE_FIELDS: dict[str, Callable[..., GaugeField]] = {
    "off": _zero_field,
    "chromomagnetic": chromomagnetic,
    "chromoelectric": chromoelectric,
    "nonabelian-crossed": nonabelian_crossed,
    "helical": helical,
}
