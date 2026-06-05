"""
Lesion study: Line Attractor vs Feedforward Network (ε = 0)

Compares the two memory architectures under random neuron lesions.
Uses the perfectly-tuned case (epsilon = 0) throughout.

Line attractor lesion
---------------------
Remove neuron i by zeroing b_i, c_i, and r_i(0).  The remaining sub-network
has effective eigenvalue (b'·c' − 1)/τ < 0, so the stored memory decays
at a rate proportional to the fraction of neurons removed.

Feedforward lesion
------------------
Remove stage n by zeroing G_n(t).  Readout weights are re-optimised on the
remaining stages via constrained least squares (|W| ≤ 5).

The same randomly-drawn lesion indices are used in both models so the
comparison is controlled.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaln
from scipy.optimize import lsq_linear

from reduced_autapse_line_attractor import AveragedAutapseParams

# ── Parameters ────────────────────────────────────────────────────────────────
N       = 100
params  = AveragedAutapseParams()
tau     = params.tau   # 100 ms
epsilon = 0.0          # perfectly tuned

T_ms        = 12_000.0
dt_ms       = 1.0
t_ms        = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s         = t_ms / 1000.0

T_target_ms = T_ms
LESION_COUNTS = [0, 1, 5, 10]
SEED          = 42


# ── RK4 step ──────────────────────────────────────────────────────────────────
def rk4_step(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ── Line attractor ────────────────────────────────────────────────────────────
def build_line_attractor(seed: int = SEED):
    rng = np.random.default_rng(seed)
    b = np.abs(rng.standard_normal(N)) + 0.1
    b /= b.sum()
    c_vec = np.abs(rng.standard_normal(N)) + 0.1
    c_vec *= (1.0 + epsilon) / (b @ c_vec)
    a = np.abs(rng.standard_normal(N)) + 0.1
    a *= (1.0 + epsilon) / (c_vec @ a)
    return b, c_vec, a


def simulate_line_attractor(b, c_vec, a, lesion_idx):
    b = b.copy(); c_vec = c_vec.copy(); a = a.copy()
    if len(lesion_idx) > 0:
        b[lesion_idx]     = 0.0
        c_vec[lesion_idx] = 0.0
        a[lesion_idx]     = 0.0

    def drdt(r: np.ndarray) -> np.ndarray:
        return (-r + b * (c_vec @ r)) / tau

    R = np.zeros((N, len(t_ms)))
    R[:, 0] = a
    for k in range(len(t_ms) - 1):
        R[:, k + 1] = rk4_step(drdt, R[:, k], dt_ms)
    return R


# ── Feedforward network ───────────────────────────────────────────────────────
def gn_fn(n: int, th: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g = np.where(
            th > 0,
            n * np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n + 1),
            0.0 if n == 0 else -np.inf,
        )
    return np.exp(log_g)


def build_feedforward_G() -> np.ndarray:
    """Numerically integrate the delta-pulse feedforward chain (epsilon=0)."""
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    G[0, 0] = 1.0
    for k in range(len(t_ms) - 1):
        src    = np.empty(N)
        src[0] = 0.0
        src[1:] = G[:N-1, k]
        G[:, k+1] = decay * G[:, k] + rise * src
    return G


def compute_feedforward_output(G_full: np.ndarray, lesion_idx: np.ndarray):
    G = G_full.copy()
    if len(lesion_idx) > 0:
        G[lesion_idx] = 0.0
    mask_fit = (t_ms > 0) & (t_ms <= T_target_ms)
    b_fit = np.ones(mask_fit.sum())
    res = lsq_linear(G[:, mask_fit].T, b_fit, bounds=(-5.0, 5.0))
    return G, res.x


# ── Simulate ──────────────────────────────────────────────────────────────────
print("Simulating lesion study…")

# Draw the maximum number of lesion indices once; subsets are the first n.
rng_lesion   = np.random.default_rng(SEED + 999)
all_lesion_idx = rng_lesion.choice(N, size=max(LESION_COUNTS), replace=False)

b_base, c_base, a_base = build_line_attractor()
G_full = build_feedforward_G()

la_results: list[np.ndarray]               = []
ff_results: list[tuple[np.ndarray, np.ndarray]] = []

for n_lesion in LESION_COUNTS:
    idx = all_lesion_idx[:n_lesion]
    la_results.append(simulate_line_attractor(b_base, c_base, a_base, idx))
    ff_results.append(compute_feedforward_output(G_full, idx))
    print(f"  n_lesion={n_lesion}  lesioned stages: {sorted(idx.tolist())}  done")


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    r"Lesion Study: Line Attractor vs Feedforward Network  ($\varepsilon=0$, "
    r"$N=100$, $\tau=100$ ms)"
    "\nSame random neuron indices lesioned in both models  |  "
    "dashed: target = 1  |  dotted: ±5%  |  shaded: 2 s fit window",
    fontsize=9,
)

gs_la = gridspec.GridSpec(
    1, 4, figure=fig, left=0.08, right=0.98, top=0.88, bottom=0.52, wspace=0.28,
)
gs_ff = gridspec.GridSpec(
    1, 4, figure=fig, left=0.08, right=0.98, top=0.47, bottom=0.06, wspace=0.28,
)

fig.text(0.027, 0.70, "Line\nAttractor", va='center', ha='center',
         rotation=90, fontsize=10, fontweight='bold', color='steelblue')
fig.text(0.027, 0.27, "Feedforward", va='center', ha='center',
         rotation=90, fontsize=10, fontweight='bold', color='darkorange')


def decorate_output(ax, ylim=None):
    ax.axhline(1.00, color='k',    ls='--', lw=0.9, zorder=3)
    ax.axhline(0.95, color='gray', ls=':',  lw=0.7, zorder=3)
    ax.axhline(1.05, color='gray', ls=':',  lw=0.7, zorder=3)
    ax.axvspan(0, T_target_ms / 1000, alpha=0.07, color='limegreen', zorder=0)
    ax.set_xlim([0, T_ms / 1000])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)


N_SHOW_EACH  = 3
idx_la_show  = np.argsort(b_base)[::-1][
    np.linspace(0, N - 1, N_SHOW_EACH * 2, dtype=int)
]
idx_ff_show  = np.linspace(0, N - 1, 6, dtype=int)

for col, n_lesion in enumerate(LESION_COUNTS):
    R_la           = la_results[col]
    G_lesion, W_opt = ff_results[col]
    lesion_idx      = all_lesion_idx[:n_lesion]
    title = "Intact" if n_lesion == 0 else (
        f"1 neuron lesioned" if n_lesion == 1 else f"{n_lesion} neurons lesioned"
    )

    # ── Line attractor ────────────────────────────────────────────────────
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_la[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_r = fig.add_subplot(gs_sub[0])
    ax_o = fig.add_subplot(gs_sub[1])

    for i, ni in enumerate(idx_la_show):
        shade = 0.35 + 0.55 * i / (N_SHOW_EACH * 2 - 1)
        lw    = 0.7 if (n_lesion > 0 and ni in lesion_idx) else 0.9
        ls    = ':'  if (n_lesion > 0 and ni in lesion_idx) else '-'
        ax_r.plot(t_s, R_la[ni], color=plt.cm.Blues(shade), lw=lw, ls=ls, alpha=0.9)

    ax_o.plot(t_s, R_la.sum(axis=0), color='steelblue', lw=2)
    decorate_output(ax_o, ylim=[-0.05, 1.35])

    ax_r.set_title(title, fontsize=9, pad=3)
    ax_r.set_xlim([0, T_ms / 1000])
    shown_max = R_la[idx_la_show].max()
    ax_r.set_ylim([0, max(shown_max * 1.2, 0.01)])
    ax_r.set_xticks([])
    ax_r.tick_params(labelsize=7)
    if col == 0:
        ax_r.set_ylabel("Neuronal\nactivity", fontsize=8)
        ax_o.set_ylabel("Summed\noutput", fontsize=8)
    ax_o.set_xlabel("Time (sec)", fontsize=8)
    ax_r.text(0.03, 0.92, chr(ord('A') + col),
              transform=ax_r.transAxes, fontsize=12, fontweight='bold', va='top')

    # Annotate which neurons are shown as lesioned in this panel
    if n_lesion > 0:
        hit = [ni for ni in idx_la_show if ni in lesion_idx]
        if hit:
            ax_r.text(0.97, 0.92, f"(lesioned: {hit})",
                      transform=ax_r.transAxes, fontsize=6, va='top', ha='right',
                      color='gray')

    # ── Feedforward ───────────────────────────────────────────────────────
    gs_sub_ff = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_ff[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_ff_r = fig.add_subplot(gs_sub_ff[0])
    ax_ff_o = fig.add_subplot(gs_sub_ff[1])

    for i, ni in enumerate(idx_ff_show):
        c  = plt.cm.Oranges(0.35 + 0.55 * i / (len(idx_ff_show) - 1))
        lw = 0.7 if (n_lesion > 0 and ni in lesion_idx) else 0.9
        ls = ':'  if (n_lesion > 0 and ni in lesion_idx) else '-'
        ax_ff_r.plot(t_s, G_lesion[ni], color=c, lw=lw, ls=ls, alpha=0.9)

    out_ff_opt = G_lesion.T @ W_opt
    out_ff_eq  = G_lesion.sum(axis=0)

    ax_ff_o.plot(t_s, out_ff_eq,  color='gray',      lw=1.2, ls='-',
                 alpha=0.6, label="W's = 1")
    ax_ff_o.plot(t_s, out_ff_opt, color='darkorange', lw=2,
                 label="W's opt")
    decorate_output(ax_ff_o, ylim=[-0.05, 1.35])

    ax_ff_r.set_xlim([0, T_ms / 1000])
    ax_ff_r.set_xticks([])
    ax_ff_r.tick_params(labelsize=7)
    if col == 0:
        ax_ff_r.set_ylabel("Stage\nactivity", fontsize=8)
        ax_ff_o.set_ylabel("Summed\noutput", fontsize=8)
        ax_ff_o.legend(fontsize=6.5, loc='upper right', framealpha=0.85)
    ax_ff_o.set_xlabel("Time (sec)", fontsize=8)
    ax_ff_r.text(0.03, 0.92, chr(ord('E') + col),
                 transform=ax_ff_r.transAxes, fontsize=12, fontweight='bold', va='top')

out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\lesion.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
