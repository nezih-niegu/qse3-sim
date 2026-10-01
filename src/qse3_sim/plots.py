"""Figure generation. Uses the Agg backend so it runs headless."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same look as the swarm figures: grey panel, white grid (ggplot).
plt.style.use("ggplot")

from .dynamics import Trajectory
from .singularity import ResonanceReport

FLAVORS = ("nu_e", "nu_mu", "nu_tau")


def _flavor_labels(n: int) -> list[str]:
    return list(FLAVORS[:n]) if n <= 3 else [f"state_{i+1}" for i in range(n)]


def plot_trajectory(traj: Trajectory, out: Path) -> Path:
    fig = plt.figure(figsize=(11, 4.2))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(traj.x[:, 1], traj.x[:, 2], traj.x[:, 3], lw=1.4)
    ax.scatter(*traj.x[0, 1:], marker="o", s=30, label="start")
    ax.scatter(*traj.x[-1, 1:], marker="^", s=30, label="end")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.set_title("Worldline projection (Q_gauge + Q_topological)")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(traj.path_length, traj.u[:, 0], lw=1.4, label="u^0 (gamma)")
    ax2.set_xlabel("path length [km]")
    ax2.set_ylabel("gamma")
    ax2b = ax2.twinx()
    ax2b.plot(traj.path_length, traj.rho, color="tab:red", ls="--", lw=1.2, label="rho")
    ax2b.set_ylabel("density [g/cm^3]", color="tab:red")
    ax2.set_title("Lorentz factor and matter density")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_flavor(traj: Trajectory, out: Path) -> Path:
    n = traj.probs.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    s = traj.path_length
    for i, lab in enumerate(_flavor_labels(n)):
        axes[0].plot(s, traj.probs[:, i], lw=1.4, label=f"P({lab})")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=n, fontsize=8)
    axes[0].set_title("Internal state occupation along the chain")

    for a in range(traj.charge.shape[1]):
        axes[1].plot(s, traj.charge[:, a], lw=1.0, label=f"I^{a+1}")
    axes[1].set_xlabel("path length [km]")
    axes[1].set_ylabel("I^a = <psi|T_a|psi>")
    axes[1].legend(ncol=4, fontsize=7)
    axes[1].set_title("SU(N) color charge precession")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_resonance(rep: ResonanceReport, out: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for k in range(rep.eigenvalues.shape[1]):
        axes[0].plot(rep.s, rep.eigenvalues[:, k], lw=1.4, label=f"E_{k+1}")
    xi = rep.resonant_pair.crossing_index
    if xi >= 0:
        axes[0].axvline(rep.s[xi], ls="--", color="k", lw=0.9)
    axes[0].set_ylabel("eigenvalue [km^-1]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Instantaneous eigenvalues of H(V(s))")

    axes[1].plot(rep.s, rep.effective_theta13, lw=1.4, color="tab:purple")
    axes[1].axhline(45.0, ls=":", color="k", lw=1.0)
    axes[1].set_ylabel("theta_m [deg]")
    axes[1].set_title("Effective matter mixing angle (45 deg = resonance)")
    ax1b = axes[1].twinx()
    ax1b.plot(rep.s, rep.rho, color="tab:red", ls="--", lw=1.0)
    ax1b.axhline(rep.resonance_rho, color="tab:red", ls=":", lw=1.0)
    ax1b.set_ylabel("rho [g/cm^3]", color="tab:red")

    res = rep.resonant_pair
    axes[2].semilogy(
        rep.s,
        np.clip(res.adiabaticity, 1e-3, 1e6),
        lw=1.4,
        color="tab:green",
        label=f"levels {res.levels[0]+1}-{res.levels[1]+1}",
    )
    axes[2].axhline(1.0, ls=":", color="k", lw=1.0)
    if rep.danger_mask.any():
        axes[2].fill_between(
            rep.s,
            1e-3,
            1e3,
            where=rep.danger_mask,
            color="tab:red",
            alpha=0.15,
            label="control breakdown",
        )
        axes[2].legend(fontsize=8)
    axes[2].set_ylabel("adiabaticity gamma")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].set_xlabel("path length [km]")
    axes[2].set_title("Adiabaticity ratio; gamma < 1 means the state stops tracking")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_jacobian(diag: dict, out: Path) -> Path:
    ev = diag["eigenvalues"]
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = range(1, len(ev) + 1)
    ax.semilogy(idx, [max(v, 1e-30) for v in ev], "o-", lw=1.2)
    ax.axvline(diag["max_rank"] + 0.5, ls=":", color="k")
    ax.text(
        diag["max_rank"] + 0.6,
        max(ev[0], 1e-12),
        f"dim CP^(N-1) = {diag['max_rank']}",
        fontsize=8,
        va="top",
    )
    ax.set_xlabel("mode")
    ax.set_ylabel("Fubini-Study eigenvalue")
    ax.set_title("Geometric Jacobian spectrum (rank deficiency = singularity)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_control(res, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    it = np.arange(len(res.infidelity))

    axes[0].semilogy(it, np.maximum(res.infidelity, 1e-16), lw=1.5)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("1 - |<chi|psi>|^2")
    axes[0].set_title("Convergence onto the target state")

    axes[1].semilogy(it, np.maximum(res.sigma_min, 1e-16), lw=1.4, label="sigma_min")
    axes[1].semilogy(it, np.maximum(res.damping, 1e-16), lw=1.4, ls="--", label="lambda")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("magnitude")
    axes[1].set_title("Conditioning and applied damping")
    axes[1].legend(fontsize=8)

    rows = res.command_table()
    labels = [r[0] for r in rows]
    deltas = [r[3] for r in rows]
    axes[2].barh(range(len(rows)), deltas)
    axes[2].set_yticks(range(len(rows)))
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].axvline(0.0, color="k", lw=0.8)
    axes[2].set_xlabel("commanded change")
    axes[2].set_title("Joint schedule")
    axes[2].invert_yaxis()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
