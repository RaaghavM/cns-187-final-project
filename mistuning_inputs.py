"""
Mistuning × Inputs: Line Attractor vs Feedforward
Four epsilon mistunings × two input types (sinusoidal, IPSP).

Layout (top→bottom):
  Row block 1 — Sinusoidal,  Line Attractor   (panels A–D)
  Row block 2 — Sinusoidal,  Feedforward      (panels E–H)
  Row block 3 — IPSP,        Line Attractor   (panels I–L)
  Row block 4 — IPSP,        Feedforward      (panels M–P)

Sinusoidal drive: I(t) = A * sin(omega*t) * c_i / (c·c)
  A = tau*omega  → slow-mode integral = 1 - cos(omega*t), range [0,2] for eps=0.
  For eps≠0 the integral grows/decays as in goldman_fig7.

IPSP drive (alpha function): I(t) = g_1(t/tau) * c_i / (c·c)
  Normalised so slow mode → 1 for eps=0.
  Feedforward: G_n^{IPSP}(t) = (1+eps)^n * g_{n+2}(t/tau).
  Sinusoidal feedforward: solved via vectorised Euler chain ODE.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaln
from scipy.optimize import lsq_linear
from reduced_autapse_line_attractor import AveragedAutapseParams

# ── Parameters ────────────────────────────────────────────────────────────────
N      = 100
params = AveragedAutapseParams()
tau    = params.tau   # 100 ms

T_ms        = 12_000.0
dt_ms       = 1.0
t_ms        = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s         = t_ms / 1000.0
T_target_ms = T_ms
SEED        = 42

F_HZ  = 0.25
OMEGA = 2.0 * np.pi * F_HZ / 1000.0
A_SIN = tau * OMEGA          # normalised: slow-mode amplitude = 1

cases = [
    ("Mistune −6%",    -0.06),
    ("Mistune −0.5%",  -0.005),
    ("Perfectly tuned", 0.0),
    ("Mistune +2%",   +0.02),
]


# ── Utilities ─────────────────────────────────────────────────────────────────
def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5*dt*k1)
    k3 = f(y + 0.5*dt*k2)
    k4 = f(y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def gn_fn(n, th):
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g = np.where(
            th > 0,
            n * np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n + 1),
            0.0 if n == 0 else -np.inf,
        )
    return np.exp(log_g)


# ── Network ───────────────────────────────────────────────────────────────────
def build_network(epsilon, seed=SEED):
    rng   = np.random.default_rng(seed)
    b     = np.abs(rng.standard_normal(N)) + 0.1
    b    /= b.sum()
    c_vec = np.abs(rng.standard_normal(N)) + 0.1
    c_vec *= (1.0 + epsilon) / (b @ c_vec)
    return b, c_vec


# ── Line attractor ────────────────────────────────────────────────────────────
def simulate_la(b, c_vec, drive, r0=None):
    R = np.zeros((N, len(t_ms)))
    if r0 is not None:
        R[:, 0] = r0
    for k in range(len(t_ms) - 1):
        I_k = drive[:, k]
        def drdt(r, _I=I_k):
            return (-r + b * (c_vec @ r) + _I) / tau
        R[:, k+1] = rk4_step(drdt, R[:, k], dt_ms)
    return R


def sin_drive(c_vec):
    c2 = c_vec @ c_vec
    return (A_SIN * np.sin(OMEGA * t_ms))[None, :] * c_vec[:, None] / c2


def ipsp_drive(c_vec):
    c2 = c_vec @ c_vec
    return -gn_fn(1, t_ms / tau)[None, :] * c_vec[:, None] / c2   # inhibitory


# ── Feedforward chain (numerical, exact ZOH step) ────────────────────────────
def ff_chain(epsilon: float, input_signal: np.ndarray | None = None) -> np.ndarray:
    """
    Integrate tau dG_n/dt = -G_n + (1+epsilon)*G_{n-1} with exact ZOH step.
    input_signal: (T,) drive to stage 0.  None = delta-pulse IC (G[0,0]=1).
    """
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    w     = 1.0 + epsilon
    if input_signal is None:
        G[0, 0] = 1.0
    for k in range(len(t_ms) - 1):
        src    = np.empty(N)
        src[0] = 0.0 if input_signal is None else input_signal[k]
        src[1:] = w * G[:N-1, k]
        G[:, k+1] = decay * G[:, k] + rise * src
    return G


def ff_sin(epsilon):
    return ff_chain(epsilon, A_SIN * np.sin(OMEGA * t_ms))


def ff_ipsp(epsilon):
    return ff_chain(epsilon, -gn_fn(1, t_ms / tau))   # inhibitory


def fit_w(G, target):
    mask = t_ms > 0
    res  = lsq_linear(G[:, mask].T, target[mask], bounds=(-5.0, 5.0))
    return res.x


# ── Pre-compute ───────────────────────────────────────────────────────────────
target_sin  = 1.0 - np.cos(OMEGA * t_ms)   # ideal LA output for sin drive
target_ipsp = -np.ones(len(t_ms))           # negative step (inhibitory memory)

print("Simulating…")
results = []
for title, epsilon in cases:
    b, c_vec = build_network(epsilon)

    R_sin  = simulate_la(b, c_vec, sin_drive(c_vec))
    R_ipsp = simulate_la(b, c_vec, ipsp_drive(c_vec))   # start silent — same IC as FF

    G_sin  = ff_sin(epsilon)
    G_ipsp = ff_ipsp(epsilon)

    W_sin  = fit_w(G_sin,  target_sin)
    W_ipsp = fit_w(G_ipsp, target_ipsp)

    results.append(dict(
        epsilon=epsilon, title=title, b=b,
        R_sin=R_sin, R_ipsp=R_ipsp,
        G_sin=G_sin, G_ipsp=G_ipsp,
        W_sin=W_sin, W_ipsp=W_ipsp,
    ))
    print(f"  epsilon={epsilon:+.3f}  done")


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
fig.suptitle(
    (r"Mistuning $\times$ Input: Line Attractor vs Feedforward"
     r"  ($N=100$, $\tau=100$ ms)" + "\n" +
     r"Top: sinusoidal drive $A\sin(\omega t)$, $f={:.2f}$ Hz  |  ".format(F_HZ) +
     r"Bottom: IPSP alpha-function $g_1(t/\tau)$"),
    fontsize=10,
)

# Vertical layout: 4 strips, each [traces(2) + output(1)] sub-height ratio
#   sin_la  0.955 → 0.765
#   sin_ff  0.740 → 0.550
#   ipsp_la 0.495 → 0.305
#   ipsp_ff 0.280 → 0.035
KWGS = dict(left=0.07, right=0.98, wspace=0.28)

gs_sin_la  = gridspec.GridSpec(1, 4, figure=fig, top=0.955, bottom=0.765, **KWGS)
gs_sin_ff  = gridspec.GridSpec(1, 4, figure=fig, top=0.740, bottom=0.550, **KWGS)
gs_ipsp_la = gridspec.GridSpec(1, 4, figure=fig, top=0.495, bottom=0.305, **KWGS)
gs_ipsp_ff = gridspec.GridSpec(1, 4, figure=fig, top=0.280, bottom=0.035, **KWGS)

# Section labels
kw_lbl = dict(va='center', ha='center', rotation=90, fontsize=9, fontweight='bold')
fig.text(0.012, 0.860, "Sin\nLA",   color='steelblue',  **kw_lbl)
fig.text(0.012, 0.645, "Sin\nFF",   color='darkorange', **kw_lbl)
fig.text(0.012, 0.400, "IPSP\nLA",  color='steelblue',  **kw_lbl)
fig.text(0.012, 0.160, "IPSP\nFF",  color='darkorange', **kw_lbl)

# Horizontal separator between sin and IPSP blocks
fig.add_artist(plt.Line2D([0.04, 0.99], [0.525, 0.525],
                           transform=fig.transFigure, color='gray',
                           lw=0.8, ls='--', alpha=0.6))


def decorate_output(ax, ylim):
    ax.axhline(1.00, color='k',    ls='--', lw=0.8, zorder=3)
    ax.axhline(0.95, color='gray', ls=':',  lw=0.6, zorder=3)
    ax.axhline(1.05, color='gray', ls=':',  lw=0.6, zorder=3)
    ax.axvspan(0, T_target_ms / 1000, alpha=0.07, color='limegreen', zorder=0)
    ax.set_xlim([0, T_ms / 1000])
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=6)


N_SHOW = 3
IDX_FF = np.linspace(0, N - 1, 6, dtype=int)

PANEL_LETTERS = iter('ABCDEFGHIJKLMNOP')

for col, res in enumerate(results):
    epsilon = res['epsilon']
    title   = res['title']
    b       = res['b']

    idx_la = np.argsort(b)[::-1][np.linspace(0, N-1, N_SHOW*2, dtype=int)]

    # Per-epsilon ylims for summed output
    # Sinusoidal: base range [0,2]; eps>0 → exponential growth
    ylim_sin      = [-0.5,  5.5]  if epsilon > 0 else [-0.05, 2.4]
    # LA starts at 1 and is driven down; FF starts at 0 and goes negative
    # Both start at 0 and are driven negative — same scale for LA and FF
    ylim_ipsp_la  = [-5.5, 0.05] if epsilon > 0 else [-1.35, 0.05]
    ylim_ipsp_ff  = [-5.5, 0.05] if epsilon > 0 else [-1.35, 0.05]

    for (gs, R_la, G_ff, W_ff, target, ylim, cmap_r, cmap_ff, ideal_lbl) in [
        (gs_sin_la,  res['R_sin'],  res['G_sin'],  res['W_sin'],
         target_sin,  ylim_sin,  plt.cm.Blues,   None,         "ideal"),
        (gs_sin_ff,  None,          res['G_sin'],  res['W_sin'],
         target_sin,  ylim_sin,  None,            plt.cm.Oranges, "ideal"),
        (gs_ipsp_la, res['R_ipsp'], res['G_ipsp'], res['W_ipsp'],
         target_ipsp, ylim_ipsp_la, plt.cm.Blues,   None,         "ideal"),
        (gs_ipsp_ff, None,          res['G_ipsp'], res['W_ipsp'],
         target_ipsp, ylim_ipsp_ff, None,            plt.cm.Oranges, "ideal"),
    ]:
        letter = next(PANEL_LETTERS)

        gs_sub = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[col], hspace=0.08, height_ratios=[2, 1],
        )
        ax_top = fig.add_subplot(gs_sub[0])
        ax_bot = fig.add_subplot(gs_sub[1])

        if R_la is not None:
            # Line attractor: individual neuron traces
            for i, ni in enumerate(idx_la):
                shade = 0.35 + 0.55 * i / (N_SHOW*2 - 1)
                ax_top.plot(t_s, R_la[ni], color=cmap_r(shade), lw=0.8, alpha=0.9)
            out_la = R_la.sum(axis=0)
            ax_bot.plot(t_s, out_la,  color='steelblue', lw=1.8)
            ax_bot.plot(t_s, target,  color='k', lw=0.9, ls=':', alpha=0.5,
                        label=ideal_lbl)
            decorate_output(ax_bot, ylim)
            ax_top.set_xlim([0, T_ms/1000])
            ax_top.set_ylim([0, max(R_la[idx_la].max()*1.2, 0.001)])
            if col == 0:
                ax_top.set_ylabel("Neuronal\nactivity", fontsize=7)
                ax_bot.set_ylabel("Summed\noutput",     fontsize=7)
                ax_bot.legend(fontsize=5.5, loc='lower right', framealpha=0.8)
        else:
            # Feedforward: stage activity traces
            for i, ni in enumerate(IDX_FF):
                c = cmap_ff(0.35 + 0.55 * i / (len(IDX_FF) - 1))
                ax_top.plot(t_s, G_ff[ni], color=c, lw=0.8, alpha=0.9)
            out_eq  = G_ff.sum(axis=0)
            out_opt = G_ff.T @ W_ff
            ax_bot.plot(t_s, out_eq,  color='gray',      lw=1.0, ls='-',
                        alpha=0.6, label="W=1")
            ax_bot.plot(t_s, out_opt, color='darkorange', lw=1.8,
                        label="W opt")
            ax_bot.plot(t_s, target,  color='k', lw=0.9, ls=':', alpha=0.5,
                        label=ideal_lbl)
            decorate_output(ax_bot, ylim)
            ax_top.set_xlim([0, T_ms/1000])
            # robust ylim: clip extreme values from mistuned stages
            G_vis = G_ff[IDX_FF]
            ax_top.set_ylim([G_vis.min()*1.1 - 0.01,
                             max(G_vis.max()*1.2, 0.01)])
            if col == 0:
                ax_top.set_ylabel("Stage\nactivity", fontsize=7)
                ax_bot.set_ylabel("Summed\noutput",  fontsize=7)
                ax_bot.legend(fontsize=5.5, loc='upper left', framealpha=0.8)

        ax_top.set_xticks([])
        ax_top.tick_params(labelsize=6)
        ax_bot.set_xlabel("Time (sec)", fontsize=7)

        # Column title only on the topmost row block
        if gs is gs_sin_la:
            ax_top.set_title(title, fontsize=8, pad=2)

        ax_top.text(0.03, 0.93, letter,
                    transform=ax_top.transAxes, fontsize=10,
                    fontweight='bold', va='top')

out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project"
    r"\mistuning_inputs.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
