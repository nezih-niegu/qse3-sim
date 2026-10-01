"""ggplot-style figures for the entangled swarm.

Every figure is drawn inside matplotlib's "ggplot" style (grey panel, white
grid) and uses ggplot2's default discrete palette, `hue_pal()`: n hues evenly
spaced on the HCL wheel at chroma 100 and luminance 65, starting at 15 degrees.
It is computed here from the CIE LCh(uv) definition, so robot 1 of 2 is exactly
ggplot2's #F8766D and robot 2 is #00BFC4.

Signed quantities (the finite-time Lyapunov exponent) use a diverging map
centred on zero: blue is contracting, red is expanding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .swarm import SwarmTrajectory

GG_PANEL = "#EBEBEB"
GG_GRID = "#FFFFFF"
GG_INK = "#4D4D4D"
# Diverging map with a mid-grey neutral: a white midpoint would vanish into
# ggplot's white grid lines, and a zero exponent is "neutral", not "empty".
DIVERGING = LinearSegmentedColormap.from_list(
    "lyap", ["#2166AC", "#6A9AC4", "#8C8C8C", "#D6604D", "#B2182B"]
)


# ---------------------------------------------------------------------------
# ggplot2 palette
# ---------------------------------------------------------------------------


def _hcl_to_hex(h: float, c: float = 100.0, lum: float = 65.0) -> str:
    """CIE LCh(uv) with D65 white -> sRGB hex (what ggplot2's hcl() does)."""
    xn, yn, zn = 95.047, 100.0, 108.883
    un = 4 * xn / (xn + 15 * yn + 3 * zn)
    vn = 9 * yn / (xn + 15 * yn + 3 * zn)
    u = c * np.cos(np.radians(h))
    v = c * np.sin(np.radians(h))
    y = yn * ((lum + 16) / 116) ** 3 if lum > 8 else yn * lum / 903.3
    up = u / (13 * lum) + un
    vp = v / (13 * lum) + vn
    x = y * 9 * up / (4 * vp)
    z = y * (12 - 3 * up - 20 * vp) / (4 * vp)
    m = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]]
    )
    rgb = m @ (np.array([x, y, z]) / 100.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.clip(rgb, 0, None), 1 / 2.4) - 0.055)
    rgb = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def hue_pal(n: int) -> list[str]:
    """ggplot2::hue_pal()(n)."""
    hues = np.linspace(15, 375, n + 1)[:-1]
    return [_hcl_to_hex(h % 360) for h in hues]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _style():
    return plt.style.context("ggplot")


def _gg3d(ax) -> None:
    """Give a 3D axis the ggplot look: grey panes, white grid."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(GG_PANEL))
        axis._axinfo["grid"]["color"] = GG_GRID
        axis._axinfo["grid"]["linewidth"] = 0.9
        axis.label.set_color(GG_INK)
    ax.tick_params(colors=GG_INK, labelsize=7)


def _colored_line3d(ax, pts: np.ndarray, values: np.ndarray, norm, cmap, lw: float = 1.8):
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=cmap, norm=norm, linewidths=lw)
    lc.set_array(values[:-1])
    ax.add_collection3d(lc)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.05 * np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    return lc


def _ftle_on_grid(traj: SwarmTrajectory) -> np.ndarray:
    """Guidance-layer local exponent per robot on the trajectory time grid, (T, R).

    Each exponent belongs to the window that ends at lyap_t, so every sample is
    assigned the window it falls in.
    """
    out = np.zeros((len(traj.t), traj.n_robots))
    if traj.lyap_guidance.size == 0:
        return out
    idx = np.clip(np.searchsorted(traj.lyap_t, traj.t, side="left"), 0, len(traj.lyap_t) - 1)
    return traj.lyap_guidance[idx]


def _ftle_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Diverging norm pinned at zero, each side scaled to its own range.

    Contracting exponents are typically an order of magnitude larger than the
    brief positive excursions, so a symmetric scale would paint every locally
    unstable stretch a barely-warm grey. Scaling the two sides separately
    keeps zero neutral and makes the largest positive excursion full red; the
    colorbar shows the asymmetry.
    """
    lo = min(float(np.percentile(values, 1)), -1e-9)
    hi = float(np.percentile(values, 99.5))
    hi = max(hi, 0.1 * abs(lo))
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def _grid(n: int) -> tuple[int, int]:
    cols = min(3, n)
    return int(np.ceil(n / cols)), cols


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_swarm_paths(traj: SwarmTrajectory, out: Path, noun: str = "robot") -> Path:
    cols = hue_pal(traj.n_robots)
    with _style():
        fig = plt.figure(figsize=(12.5, 5.4))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        _gg3d(ax)
        ax.plot(*traj.ref.T, color="#7F7F7F", lw=1.0, ls="--", label="Lorenz reference")
        for r in range(traj.n_robots):
            p = traj.pos[:, r]
            ax.plot(*p.T, color=cols[r], lw=1.1, label=f"{noun} {r + 1}")
            ax.scatter(*p[0], color=cols[r], s=18, marker="o")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_zlabel("z [km]")
        ax.set_title("Swarm on a chaotic global trajectory", fontsize=11)
        ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

        # Formation seen from the reference: p_r - p_ref. A held formation is a
        # set of tight clouds sitting on the design slots.
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _gg3d(ax2)
        rel = traj.pos - traj.ref[:, None, :]
        settle = len(traj.t) // 3
        a_graph = traj.meta.get("graph")
        if a_graph is not None:
            a_graph = np.asarray(a_graph)
            for i in range(traj.n_robots):
                for j in range(i + 1, traj.n_robots):
                    if a_graph[i, j]:
                        seg = traj.offsets[[i, j]]
                        ax2.plot(*seg.T, color="#9A9A9A", lw=0.8, ls=":")
        for r in range(traj.n_robots):
            ax2.plot(*rel[:settle, r].T, color=cols[r], lw=0.6, alpha=0.45)
            ax2.plot(*rel[settle:, r].T, color=cols[r], lw=1.4, label=f"{noun} {r + 1}")
        ax2.scatter(*traj.offsets.T, color=GG_INK, marker="x", s=40, linewidths=1.3,
                    label="design slot", depthshade=False)
        ax2.set_xlabel("x - x_ref [km]")
        ax2.set_ylabel("y - y_ref [km]")
        ax2.set_zlabel("z - z_ref [km]")
        ax2.set_title("Formation in the reference frame (faint: first third)", fontsize=11)
        ax2.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def plot_lyapunov_phase_3d(traj: SwarmTrajectory, out: Path, space: str = "position") -> Path:
    """One 3D panel per robot, its path coloured by its local Lyapunov exponent.

    The exponent is the robot's conditional one in the chaotic guidance layer:
    the local stretching rate of the Lorenz flow along the robot's own
    oscillator path, minus the rate m_r at which coupling pulls it back. Red
    stretches of path are where that robot, on its own, would drift out of
    formation; blue are where it is pulled in.

    space="position": the robot's worldline (x, y, z).
    space="sync":     its synchronization error L (s_r - s_ref) in km: the
                      phase portrait of how far its chaotic oscillator sits
                      from the global trajectory.
    """
    lam = _ftle_on_grid(traj)
    norm = _ftle_norm(lam)
    rows, cols = _grid(traj.n_robots)
    lam_bar = traj.guidance_lyapunov
    anchors = set(traj.meta.get("anchors", []))
    sync = traj.sync_error
    with _style():
        fig = plt.figure(figsize=(4.5 * cols, 4.5 * rows + 0.8))
        lc = None
        for r in range(traj.n_robots):
            ax = fig.add_subplot(rows, cols, r + 1, projection="3d")
            _gg3d(ax)
            if space == "sync":
                pts = sync[:, r]
                ax.scatter(0, 0, 0, color=GG_INK, marker="+", s=70, linewidths=1.3)
                labels = ("dx [km]", "dy [km]", "dz [km]")
            else:
                pts = traj.pos[:, r]
                labels = ("x [km]", "y [km]", "z [km]")
            lc = _colored_line3d(ax, pts, lam[:, r], norm, DIVERGING)
            ax.scatter(*pts[0], color=GG_INK, s=16, marker="o")
            ax.set_xlabel(labels[0], fontsize=8, labelpad=2)
            ax.set_ylabel(labels[1], fontsize=8, labelpad=2)
            ax.set_zlabel(labels[2], fontsize=8, labelpad=2)
            role = "anchor" if r in anchors else "follower"
            unstable = 100 * float(np.mean(traj.lyap_guidance[:, r] > 0))
            ax.set_title(
                f"robot {r + 1} ({role})\n" + r"$\bar\lambda$"
                + f" = {lam_bar[r]:+.2e} km$^{{-1}}$, locally unstable {unstable:.0f}% of time",
                fontsize=9, color=GG_INK,
            )
        what = ("synchronization-error phase space  L(s_r - s_ref)" if space == "sync"
                else "position phase space")
        fig.suptitle(
            f"Local conditional Lyapunov exponent in each robot's {what}\n"
            "blue: coupling beats chaotic stretching    red: stretching wins locally",
            fontsize=11.5, color=GG_INK,
        )
        # 3D axes and tight_layout do not mix well; place the grid by hand
        fig.subplots_adjust(left=0.01, right=0.865, bottom=0.03,
                            top=0.86 if rows == 1 else 0.9, wspace=0.04, hspace=0.22)
        cax = fig.add_axes((0.915, 0.2, 0.014, 0.6))
        cb = fig.colorbar(lc, cax=cax)
        ticks = np.concatenate([np.linspace(norm.vmin, 0, 4), np.linspace(0, norm.vmax, 3)[1:]])
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:+.1e}" if v else "0" for v in ticks])
        cb.set_label(r"local $\lambda_r$ [km$^{-1}$]  (sides scaled separately)", fontsize=9)
        cb.ax.tick_params(labelsize=7)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def plot_swarm_dashboard(
    traj: SwarmTrajectory,
    out: Path,
    lambda_ref: float,
    network_exponent: float,
    eta_floor: float,
    comparisons: dict[str, SwarmTrajectory] | None = None,
) -> Path:
    """Small multiples: formation, synchronization, entanglement, Lyapunov."""
    rcols = hue_pal(traj.n_robots)
    t = traj.t / 1000.0  # 10^3 km of coordinate time reads better
    tl = traj.lyap_t / 1000.0
    anchors = set(traj.meta.get("anchors", []))

    def lbl(r: int) -> str:
        return f"robot {r + 1}" + (" (anchor)" if r in anchors else "")

    with _style():
        fig, axes = plt.subplots(3, 2, figsize=(13, 11.5))

        ax = axes[0, 0]
        runs = {"entanglement-aware (main run)": traj, **(comparisons or {})}
        ccols = hue_pal(len(runs))
        for (name, tr), c in zip(runs.items(), ccols):
            ax.plot(tr.t / 1000.0, 100 * tr.distance_error, color=c, lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_ylabel("worst distance error [%]")
        ax.set_title("Formation: max over pairs of |d_rs - d*_rs| / d*_rs", fontsize=10.5)
        ax.legend(fontsize=7.5)

        ax = axes[0, 1]
        for r in range(traj.n_robots):
            ax.plot(t, np.linalg.norm(traj.sync_error[:, r], axis=1), color=rcols[r], lw=1.1,
                    label=lbl(r))
        ax.set_yscale("log")
        ax.set_ylabel("|L (s_r - s_ref)| [km]")
        ax.set_title("Chaotic guidance: desynchronization from the global path", fontsize=10.5)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1, 0]
        for r in range(traj.n_robots):
            ax.plot(t, traj.entropy[:, r], color=rcols[r], lw=1.1, label=lbl(r))
        smax = np.log2(traj.probs.shape[2])
        ax.axhline(smax, color=GG_INK, ls=":", lw=1)
        ax.text(t[-1], smax, "maximal  ", ha="right", va="bottom", fontsize=7.5, color=GG_INK)
        ax.set_ylabel("S(rho_r) [bits]")
        ax.set_title("Entanglement entropy of each robot with the rest", fontsize=10.5)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1, 1]
        for r in range(traj.n_robots):
            ax.plot(t, traj.eta[:, r], color=rcols[r], lw=1.1, label=lbl(r))
        ax.plot(t, traj.throttle, color="#7F7F7F", lw=1.1, ls="-.", label="entangler throttle")
        ax.axhline(eta_floor, color=GG_INK, ls="--", lw=1)
        ax.text(t[0], eta_floor, "  governor floor", va="bottom", fontsize=7.5, color=GG_INK)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("eta_r = |I_r| / |I|_pure")
        ax.set_title("Actuation efficiency left after entanglement", fontsize=10.5)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[2, 0]
        for r in range(traj.n_robots):
            run = np.cumsum(traj.lyap_guidance[:, r]) / np.arange(1, len(tl) + 1)
            ax.plot(tl, run, color=rcols[r], lw=1.2, label=lbl(r))
        ax.axhline(lambda_ref, color="#7F7F7F", ls="--", lw=1.1)
        ax.text(tl[-1], lambda_ref, f"free Lorenz flow  {lambda_ref:+.1e}  ", ha="right",
                va="bottom", fontsize=7.5, color=GG_INK)
        ax.axhline(network_exponent, color=GG_INK, ls=":", lw=1.1)
        ax.text(tl[-1], network_exponent, f"network transverse  {network_exponent:+.1e}  ",
                ha="right", va="top", fontsize=7.5, color=GG_INK)
        ax.axhline(0.0, color=GG_INK, lw=0.8)
        ax.set_ylabel("running mean lambda_r [km^-1]")
        ax.set_xlabel("coordinate time [10^3 km]")
        ax.set_title("Conditional Lyapunov exponents, guidance layer (< 0: held)",
                     fontsize=10.5)
        ax.legend(fontsize=7, ncol=2, loc="lower right")

        ax = axes[2, 1]
        ax.plot(tl, traj.lyap_reference, color="#7F7F7F", lw=1.0,
                label="local stretching of the free flow")
        m_r = np.asarray(traj.meta["self_gain"])
        for g in np.unique(np.round(m_r, 12)):
            who = [r for r in range(traj.n_robots) if abs(m_r[r] - g) < 1e-12]
            kind = "anchors" if who[0] in anchors else "followers"
            names = ", ".join(str(w + 1) for w in who)
            ax.axhline(g, color=rcols[who[0]], lw=1.3, ls="--",
                       label=f"m_r = {g:.1e}  ({kind} {names})")
        ax.axhline(0.0, color=GG_INK, lw=0.8)
        ax.set_ylabel("rate [km^-1]")
        ax.set_xlabel("coordinate time [10^3 km]")
        ax.set_title("Where chaos wins: stretching above a robot's pull-back m_r",
                     fontsize=10.5)
        ax.legend(fontsize=7.5, loc="lower right")

        for a in axes[:2].ravel():
            a.set_xlabel("coordinate time [10^3 km]")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def plot_flavor_swarm(traj: SwarmTrajectory, out: Path, names: list[str]) -> Path:
    """Flavor occupation of every robot's reduced state, small multiples."""
    n = traj.probs.shape[2]
    fcols = hue_pal(n)
    rows, cols = _grid(traj.n_robots)
    t = traj.t / 1000.0
    with _style():
        fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 2.7 * rows), sharex=True,
                                 sharey=True, squeeze=False)
        for r in range(rows * cols):
            ax = axes.ravel()[r]
            if r >= traj.n_robots:
                ax.set_visible(False)
                continue
            for f in range(n):
                ax.plot(t, traj.probs[:, r, f], color=fcols[f], lw=1.0, label=names[f])
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"robot {r + 1}", fontsize=9.5)
            if r == 0:
                ax.legend(fontsize=7, ncol=n, loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("coordinate time [10^3 km]")
        for ax in axes[:, 0]:
            ax.set_ylabel("Tr(rho_r P_f)")
        fig.suptitle("Flavor content of each robot's reduced state", fontsize=11)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out
