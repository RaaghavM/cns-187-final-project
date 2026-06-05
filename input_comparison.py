"""
Input comparison: Sinusoidal vs IPSP drive
Line Attractor and Feedforward Network, epsilon = 0 (perfectly tuned)

Sinusoidal input:  I(t) = A * sin(omega * t)
  Amplitude A = tau * omega so the equal-weights sum oscillates between 0 and 2.
  LA and FF both act as integrators: Y(t) ≈ 1 - cos(omega*t).
  Key visual contrast: LA neurons all oscillate in phase (b_i scaling);
  FF stages form a phase-shifted traveling wave across the chain.

IPSP input:  I(t) = g_1(t/tau)  [alpha function, time constant tau]
  Amplitude normalised so the LA slow mode converges to s=1 for epsilon=0.
  Analytical FF response: G_n^{IPSP}(t) = g_{n+2}(t/tau).
  Both outputs follow 1 - (1 + t/tau)*exp(-t/tau) → 1.
  Key visual contrast: LA neurons rise together to b_i;
  FF stage n peaks later as n increases (cascade of delayed humps).

No mistuning variants.  Feedforward readout W_opt fit over full T_ms window.
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
epsilon = 0.0

T_ms        = 12_000.0
dt_ms       = 1.0
t_ms        = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s         = t_ms / 1000.0
T_target_ms = T_ms
SEED        = 42

# Sinusoidal
F_HZ  = 0.25                          # 4-s period — 3 full cycles visible
OMEGA = 2.0 * np.pi * F_HZ / 1000.0  # rad/ms
A_SIN = tau * OMEGA                   # normalised amplitude

# IPSP (alpha function = g_1)
# Drive: c_i * g_1(t/tau) / (c·c)  → slow mode integral = 1 - (1+t/tau)*e^{-t/tau} → 1


# ── Utilities ─────────────────────────────────────────────────────────────────
def rk4_step(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gn_fn(n: int, th: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g = np.where(
            th > 0,
            n * np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n + 1),
            0.0 if n == 0 else -np.inf,
        )
    return np.exp(log_g)


# ── Network (shared across both inputs) ───────────────────────────────────────
def build_network(seed: int = SEED):
    rng   = np.random.default_rng(seed)
    b     = np.abs(rng.standard_normal(N)) + 0.1
    b    /= b.sum()
    c_vec = np.abs(rng.standard_normal(N)) + 0.1
    c_vec *= (1.0 + epsilon) / (b @ c_vec)
    return b, c_vec


# ── Line attractor simulations ────────────────────────────────────────────────
def simulate_la(b: np.ndarray, c_vec: np.ndarray, drive: np.ndarray,
                r0: np.ndarray | None = None) -> np.ndarray:
    """drive: (N, T) pre-computed external current.  r0: optional IC (default 0)."""
    R = np.zeros((N, len(t_ms)))
    if r0 is not None:
        R[:, 0] = r0
    for k in range(len(t_ms) - 1):
        I_k = drive[:, k]
        def drdt(r, _I=I_k):
            return (-r + b * (c_vec @ r) + _I) / tau
        R[:, k + 1] = rk4_step(drdt, R[:, k], dt_ms)
    return R


def la_sin_drive(c_vec: np.ndarray) -> np.ndarray:
    """A_SIN * sin(omega*t) * c_i / (c·c)  → slow mode = 1 - cos(omega*t)."""
    c2 = c_vec @ c_vec
    return (A_SIN * np.sin(OMEGA * t_ms))[None, :] * c_vec[:, None] / c2


def la_ipsp_drive(c_vec: np.ndarray) -> np.ndarray:
    """Inhibitory IPSP: -g_1(t/tau)*c_i/(c·c) — drives slow mode down by 1."""
    c2 = c_vec @ c_vec
    return -gn_fn(1, t_ms / tau)[None, :] * c_vec[:, None] / c2


# ── Feedforward chain (numerical, exact ZOH step) ────────────────────────────
def ff_chain(input_signal: np.ndarray) -> np.ndarray:
    """
    Integrate tau dG_n/dt = -G_n + G_{n-1}  (epsilon=0) with exact ZOH step.
    G_{-1}(t) = input_signal[t].  Returns G: (N, T).
    """
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    for k in range(len(t_ms) - 1):
        src    = np.empty(N)
        src[0] = input_signal[k]
        src[1:] = G[:N - 1, k]
        G[:, k + 1] = decay * G[:, k] + rise * src
    return G


def ff_sin_response() -> np.ndarray:
    return ff_chain(A_SIN * np.sin(OMEGA * t_ms))


def ff_ipsp_response() -> np.ndarray:
    return ff_chain(-gn_fn(1, t_ms / tau))   # inhibitory: negative drive


# ── Optimal readout (constrained least squares, |W| ≤ 5) ─────────────────────
def fit_readout(G: np.ndarray, target: np.ndarray) -> np.ndarray:
    mask = t_ms > 0
    res  = lsq_linear(G[:, mask].T, target[mask], bounds=(-5.0, 5.0))
    return res.x


# ── Run ───────────────────────────────────────────────────────────────────────
print("Simulating…")
b, c_vec = build_network()

R_sin  = simulate_la(b, c_vec, la_sin_drive(c_vec))
R_ipsp = simulate_la(b, c_vec, la_ipsp_drive(c_vec))   # start silent — same IC as FF
print("  line attractor done")

G_sin  = ff_sin_response()
G_ipsp = ff_ipsp_response()
print("  feedforward done")

# Targets: the ideal output for each input type
target_sin  = 1.0 - np.cos(OMEGA * t_ms)   # integral of sin = 1-cos (0→2)
target_ipsp = -np.ones(len(t_ms))            # negative step (inhibitory memory)

W_sin  = fit_readout(G_sin,  target_sin)
W_ipsp = fit_readout(G_ipsp, target_ipsp)
print("  readout fit done")


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
fig.suptitle(
    (r"Input comparison: Line Attractor vs Feedforward"
     r"  ($\varepsilon=0$, $N=100$, $\tau=100$ ms)" + "\n" +
     r"Left: sinusoidal drive $A\sin(\omega t)$, $f={:.2f}$ Hz  |  ".format(F_HZ) +
     r"Right: IPSP alpha-function drive $g_1(t/\tau)$"),
    fontsize=9,
)

gs_la = gridspec.GridSpec(
    1, 2, figure=fig, left=0.10, right=0.97, top=0.88, bottom=0.52, wspace=0.32,
)
gs_ff = gridspec.GridSpec(
    1, 2, figure=fig, left=0.10, right=0.97, top=0.47, bottom=0.06, wspace=0.32,
)

fig.text(0.030, 0.70, "Line\nAttractor", va='center', ha='center',
         rotation=90, fontsize=10, fontweight='bold', color='steelblue')
fig.text(0.030, 0.27, "Feedforward",    va='center', ha='center',
         rotation=90, fontsize=10, fontweight='bold', color='darkorange')
fig.text(0.37, 0.93, "Sinusoidal input", ha='center', fontsize=10, fontweight='bold')
fig.text(0.77, 0.93, "IPSP input",       ha='center', fontsize=10, fontweight='bold')


def decorate_output(ax, ylim):
    ax.axhline(1.00, color='k',    ls='--', lw=0.9, zorder=3)
    ax.axhline(0.95, color='gray', ls=':',  lw=0.7, zorder=3)
    ax.axhline(1.05, color='gray', ls=':',  lw=0.7, zorder=3)
    ax.axvspan(0, T_target_ms / 1000, alpha=0.07, color='limegreen', zorder=0)
    ax.set_xlim([0, T_ms / 1000])
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)


N_SHOW_EACH  = 3
idx_la_show  = np.argsort(b)[::-1][np.linspace(0, N - 1, N_SHOW_EACH * 2, dtype=int)]
# For FF: show 6 evenly spaced stages to see the traveling wave / cascade
idx_ff_show  = np.linspace(0, N - 1, 6, dtype=int)

for col, (R_la, G_ff, W_ff, target, ylim_la, ylim_ff, label_r, label_o, letter_la, letter_ff) in enumerate([
    (R_sin,  G_sin,  W_sin,  target_sin,  [-0.05, 2.2],  [-0.05, 2.2],
     "Neuronal\nactivity", "Summed\noutput", 'A', 'C'),
    # IPSP: both start at 0 and are driven negative — same initial condition
    (R_ipsp, G_ipsp, W_ipsp, target_ipsp, [-1.35, 0.05], [-1.35, 0.05],
     "Neuronal\nactivity", "Summed\noutput", 'B', 'D'),
]):
    # ── Line attractor ────────────────────────────────────────────────────
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_la[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_r = fig.add_subplot(gs_sub[0])
    ax_o = fig.add_subplot(gs_sub[1])

    for i, ni in enumerate(idx_la_show):
        shade = 0.35 + 0.55 * i / (N_SHOW_EACH * 2 - 1)
        ax_r.plot(t_s, R_la[ni], color=plt.cm.Blues(shade), lw=0.9, alpha=0.9)

    ax_o.plot(t_s, R_la.sum(axis=0), color='steelblue', lw=2)
    # overlay the analytical target for reference
    ax_o.plot(t_s, target, color='k', lw=1.0, ls=':', alpha=0.5, label='ideal')

    decorate_output(ax_o, ylim=ylim_la)
    ax_r.set_xlim([0, T_ms / 1000])
    shown_min = R_la[idx_la_show].min()
    shown_max = R_la[idx_la_show].max()
    ax_r.set_ylim([min(shown_min * 1.2, -0.001), max(shown_max * 1.2, 0.001)])
    ax_r.set_xticks([])
    ax_r.tick_params(labelsize=7)
    ax_r.set_ylabel(label_r, fontsize=8)
    ax_o.set_ylabel(label_o, fontsize=8)
    ax_o.set_xlabel("Time (sec)", fontsize=8)
    if col == 1:
        ax_o.legend(fontsize=6.5, loc='lower right', framealpha=0.85)
    ax_r.text(0.03, 0.92, letter_la,
              transform=ax_r.transAxes, fontsize=12, fontweight='bold', va='top')

    # ── Feedforward ───────────────────────────────────────────────────────
    gs_sub_ff = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_ff[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_ff_r = fig.add_subplot(gs_sub_ff[0])
    ax_ff_o = fig.add_subplot(gs_sub_ff[1])

    for i, ni in enumerate(idx_ff_show):
        c = plt.cm.Oranges(0.35 + 0.55 * i / (len(idx_ff_show) - 1))
        ax_ff_r.plot(t_s, G_ff[ni], color=c, lw=0.9, alpha=0.9)

    out_eq  = G_ff.sum(axis=0)
    out_opt = G_ff.T @ W_ff

    ax_ff_o.plot(t_s, out_eq,  color='gray',      lw=1.2, ls='-', alpha=0.6, label="W=1")
    ax_ff_o.plot(t_s, out_opt, color='darkorange', lw=2,           label="W opt")
    ax_ff_o.plot(t_s, target,  color='k',          lw=1.0, ls=':', alpha=0.5, label='ideal')
    decorate_output(ax_ff_o, ylim=ylim_ff)

    ax_ff_r.set_xlim([0, T_ms / 1000])
    ax_ff_r.set_xticks([])
    ax_ff_r.tick_params(labelsize=7)
    ax_ff_r.set_ylabel("Stage\nactivity", fontsize=8)
    ax_ff_o.set_ylabel("Summed\noutput", fontsize=8)
    ax_ff_o.set_xlabel("Time (sec)", fontsize=8)
    if col == 0:
        ax_ff_o.legend(fontsize=6.5, loc='upper left', framealpha=0.85)
    ax_ff_r.text(0.03, 0.92, letter_ff,
                 transform=ax_ff_r.transAxes, fontsize=12, fontweight='bold', va='top')

out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\input_comparison.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
