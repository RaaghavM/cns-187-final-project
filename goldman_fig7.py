"""
Goldman (2009) Figure 7: Line Attractor vs Feedforward Network
Pulse-to-step integration with four mistuning levels.

Line attractor
--------------
Rank-1 network: w_ij = b_i * c_j, with b·c = 1 + epsilon.
Each neuron obeys:
    tau dr_i/dt = -r_i + b_i * (c · r)

The scalar memory s = c·r satisfies tau ds/dt = epsilon * s — identical
to the Seung autapse.  The N-1 fast eigenmodes (eigenvalue -1/tau) decay
within a few tau, leaving only the slow eigenmode visible at long times.
Neurons converge to different persistent amplitudes b_i while all
tracking the same slow eigenvalue epsilon/tau.

Pulse initialization: r_i(0) = a_i (random, c·a = 1+epsilon) so the
summed output equals 1 at convergence for epsilon=0.

Feedforward network
-------------------
Stage n responds to a unit pulse at stage 0 with:
    G_n(t) = (1+epsilon)^n * g_n(t/tau)
where g_n is the standard Erlang basis function (same as goldman_fig1_feedforward.py).
An optimal readout (constrained |W_n| <= 5) approximates a unit step over 0–2 s.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaln
from scipy.optimize import lsq_linear

from reduced_autapse_line_attractor import AveragedAutapseParams

# ── Shared parameters (time in ms to match autapse convention) ───────────────
N      = 100
params = AveragedAutapseParams()   # tau=100 ms, F1=0.5314, …
tau    = params.tau                # 100 ms

T_ms   = 12_000.0                  # total simulation duration (ms)
dt_ms  = 1.0                       # time step (ms)
t_ms   = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s    = t_ms / 1000.0             # seconds — used for all axis labels


# ── RK4 step (vector form, same logic as rk4_step_scalar) ────────────────────
def rk4_step(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ── Line attractor simulation ─────────────────────────────────────────────────
def simulate_line_attractor(
    epsilon: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the rank-1 line attractor with pulse initialization.

    Connectivity: w_ij = b_i * c_j, with b·c = 1 + epsilon.
    Dynamics:     tau dr_i/dt = -r_i + b_i * (c · r)

    The scalar memory s = c·r obeys tau ds/dt = epsilon * s — identical to
    the Seung autapse.  The N-1 fast eigenmodes (eigenvalue -1/tau) decay
    within a few tau; afterwards each neuron tracks b_i * exp(epsilon*t/tau).

    Pulse initial condition: r_i(0) = a_i (random, independent of b),
    normalized so c·a = 1+epsilon.  This ensures sum(r) → 1 for epsilon=0
    after the fast transient, while keeping a_i ≠ b_i so the convergence
    is visible — neurons start at different heights and settle to b_i.

    Returns
    -------
    R : (N, T) firing-rate array
    b : (N,)  slow-mode weights (persistent amplitude of each neuron)
    """
    rng = np.random.default_rng(seed)

    b = np.abs(rng.standard_normal(N)) + 0.1
    b /= b.sum()                             # sum(b) = 1

    c_vec = np.abs(rng.standard_normal(N)) + 0.1
    c_vec *= (1.0 + epsilon) / (b @ c_vec)   # b·c = 1 + epsilon

    # Random pulse: normalize so c·a = 1+epsilon
    # → slow-mode coefficient α = c·a/(1+ε) = 1 → sum(r) → 1 after convergence
    a = np.abs(rng.standard_normal(N)) + 0.1
    a *= (1.0 + epsilon) / (c_vec @ a)

    def drdt(r: np.ndarray) -> np.ndarray:
        return (-r + b * (c_vec @ r)) / tau

    R       = np.zeros((N, len(t_ms)))
    R[:, 0] = a
    for k in range(len(t_ms) - 1):
        R[:, k + 1] = rk4_step(drdt, R[:, k], dt_ms)

    return R, b


# ── Feedforward basis functions ───────────────────────────────────────────────
def gn_fn(n: int, th: np.ndarray) -> np.ndarray:
    """g_n(t̂) = (1/n!) · t̂^n · exp(-t̂), computed in log-space to avoid overflow."""
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g = np.where(
            th > 0,
            n * np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n + 1),
            0.0 if n == 0 else -np.inf,
        )
    return np.exp(log_g)


def compute_feedforward_G(epsilon: float) -> np.ndarray:
    """
    Numerically integrate tau dG_n/dt = -G_n + (1+epsilon)*G_{n-1}.
    Delta-pulse initial condition: G[0,0] = 1, all others 0.
    Uses exact piecewise-constant (ZOH) step for accuracy across all N stages.
    """
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    w     = 1.0 + epsilon
    G[0, 0] = 1.0                      # delta-pulse IC
    for k in range(len(t_ms) - 1):
        src    = np.empty(N)
        src[0] = 0.0                   # no external input after t = 0
        src[1:] = w * G[:N-1, k]
        G[:, k+1] = decay * G[:, k] + rise * src
    return G


# ── Cases matching Fig 7 panels A–H ──────────────────────────────────────────
cases = [
    ("A / E", "Mistune −6%",    -0.06),
    ("B / F", "Mistune −0.5%",  -0.005),
    ("C / G", "Perfectly tuned", 0.0),
    ("D / H", "Mistune +2%",   +0.02),
]

# Optimal readout: constrained least squares fit to unit step over 0–2 s
T_target_ms = T_ms
mask_fit    = (t_ms > 0) & (t_ms <= T_target_ms)
b_fit       = np.ones(mask_fit.sum())

# ── Pre-compute all simulations ───────────────────────────────────────────────
print("Simulating…")
results = []
for _, _, epsilon in cases:
    R_la, b_la   = simulate_line_attractor(epsilon)
    G_ff  = compute_feedforward_G(epsilon)
    res   = lsq_linear(G_ff[:, mask_fit].T, b_fit, bounds=(-5.0, 5.0))
    results.append((R_la, b_la, G_ff, res.x))
    print(f"  epsilon={epsilon:+.3f}  done")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Goldman (2009) — Figure 7: Line Attractor vs Feedforward Network\n"
    r"N = 100, $\tau$ = 100 ms  |  dashed: $\pm$5% tolerance  |  "
    r"shaded: 2 s fit window",
    fontsize=10,
)

gs_la = gridspec.GridSpec(
    1, 4, figure=fig, left=0.08, right=0.98,
    top=0.88, bottom=0.52, wspace=0.28,
)
gs_ff = gridspec.GridSpec(
    1, 4, figure=fig, left=0.08, right=0.98,
    top=0.47, bottom=0.06, wspace=0.28,
)

fig.text(0.027, 0.70, "Line\nAttractor", va='center', ha='center',
         rotation=90, fontsize=10, fontweight='bold', color='steelblue')
fig.text(0.027, 0.27, "Feedforward",    va='center', ha='center',
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


N_SHOW_EACH = 3   # neurons shown from each bipolar group

for col, ((label, title, epsilon), (R_la, b_la, G_ff, W_opt)) in enumerate(
    zip(cases, results)
):
    # ── Line attractor ────────────────────────────────────────────────────────
    gs_sub_la = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_la[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_la_r = fig.add_subplot(gs_sub_la[0])
    ax_la_o = fig.add_subplot(gs_sub_la[1])

    # Show N_SHOW_EACH*2 neurons spanning the full b_la amplitude range
    idx_sorted = np.argsort(b_la)[::-1]
    idx_show   = idx_sorted[np.linspace(0, N - 1, N_SHOW_EACH * 2, dtype=int)]
    for i, ni in enumerate(idx_show):
        shade = 0.35 + 0.55 * i / (N_SHOW_EACH * 2 - 1)
        ax_la_r.plot(t_s, R_la[ni], color=plt.cm.Blues(shade), lw=0.9, alpha=0.9)

    out_la = R_la.sum(axis=0)          # summed output: → exp(ε t/τ) after convergence
    ax_la_o.plot(t_s, out_la, color='steelblue', lw=2)

    ylim_la = [-0.05, 5.5] if epsilon > 0 else [-0.05, 1.35]
    decorate_output(ax_la_o, ylim=ylim_la)

    ax_la_r.set_title(title, fontsize=9, pad=3)
    ax_la_r.set_xlim([0, T_ms / 1000])
    shown_max = np.array([R_la[ni] for ni in idx_show]).max()
    ax_la_r.set_ylim([0, shown_max * 1.2])
    ax_la_r.set_xticks([])
    ax_la_r.tick_params(labelsize=7)
    if col == 0:
        ax_la_r.set_ylabel("Neuronal\nactivity", fontsize=8)
        ax_la_o.set_ylabel("Summed\noutput", fontsize=8)
    ax_la_o.set_xlabel("Time (sec)", fontsize=8)

    ax_la_r.text(0.03, 0.92, chr(ord('A') + col),
                 transform=ax_la_r.transAxes,
                 fontsize=12, fontweight='bold', va='top')

    # ── Feedforward ───────────────────────────────────────────────────────────
    gs_sub_ff = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_ff[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_ff_r = fig.add_subplot(gs_sub_ff[0])
    ax_ff_o = fig.add_subplot(gs_sub_ff[1])

    idx_ff_show = np.linspace(0, N - 1, 6, dtype=int)
    for i, ni in enumerate(idx_ff_show):
        c = plt.cm.Oranges(0.35 + 0.55 * i / (len(idx_ff_show) - 1))
        ax_ff_r.plot(t_s, G_ff[ni], color=c, lw=0.9, alpha=0.9)

    out_ff_opt = G_ff.T @ W_opt
    out_ff_eq  = G_ff.sum(axis=0)

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
                 transform=ax_ff_r.transAxes,
                 fontsize=12, fontweight='bold', va='top')

out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\goldman_fig7.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
