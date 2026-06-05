"""
IPSP + Noise: Line Attractor vs Feedforward Network

Replaces the instantaneous pulse of goldman_fig7.py with:
  (1) an alpha-function (IPSP) input current: I(t) ∝ g_1(t/tau) = (t/tau)*exp(-t/tau)
  (2) an Ornstein-Uhlenbeck fluctuating background current throughout

Line attractor
--------------
  Starts silent (r=0).  The IPSP current smoothly drives the network up:
    tau dr_i/dt = -r_i + b_i*(c·r)  +  g_1(t/tau)*c_i/(c·c)  +  sigma*eta_i(t)
  Amplitude is normalised so s = c·r converges to 1 for epsilon=0 (same
  long-run target as the original pulse initialisation).
  The slow-mode noise does a random walk for epsilon=0 — visible at long
  times — whereas the line attractor decays/grows for epsilon != 0.

Feedforward
-----------
  Convolving the delta-pulse Erlang basis g_n with the IPSP waveform g_1
  shifts the response by two stages:
    G_n^{IPSP}(t) = (1+epsilon)^n * g_{n+2}(t/tau)
  OU noise is added to the summed output at the same per-unit amplitude
  as the line attractor for a like-for-like comparison.
  Readout W is re-optimised for the shifted basis over the full T_ms window.

Columns match goldman_fig7: four epsilon (mistuning) levels.
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

SIGMA_NOISE = 0.01   # OU noise amplitude (same units as firing rate)
TAU_NOISE   = tau    # OU noise correlation time (ms)
SEED        = 42

cases = [
    ("A / E", "Mistune −6%",    -0.06),
    ("B / F", "Mistune −0.5%",  -0.005),
    ("C / G", "Perfectly tuned", 0.0),
    ("D / H", "Mistune +2%",   +0.02),
]


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


def generate_ou_noise(n_neurons: int, seed: int) -> np.ndarray:
    """
    Vectorised Euler-Maruyama OU: tau_n dη/dt = -η + sqrt(2*tau_n) * xi(t)
    Pre-draw all white-noise increments at once, then scan forward.
    Steady-state std = 1.  Shape: (n_neurons, len(t_ms)).
    """
    rng    = np.random.default_rng(seed)
    T      = len(t_ms)
    c_dt   = dt_ms / TAU_NOISE
    s_dt   = np.sqrt(2.0 * dt_ms / TAU_NOISE)
    xi     = rng.standard_normal((n_neurons, T))   # all increments at once
    out    = np.zeros((n_neurons, T))
    eta    = np.zeros(n_neurons)
    for k in range(T - 1):
        eta           = eta * (1.0 - c_dt) + s_dt * xi[:, k]
        out[:, k + 1] = eta
    return out                        # std ≈ 1 in steady state


# ── Line attractor with IPSP drive ────────────────────────────────────────────
def simulate_line_attractor_ipsp(epsilon: float, seed: int = SEED):
    rng = np.random.default_rng(seed)

    b = np.abs(rng.standard_normal(N)) + 0.1
    b /= b.sum()
    c_vec = np.abs(rng.standard_normal(N)) + 0.1
    c_vec *= (1.0 + epsilon) / (b @ c_vec)

    c2 = c_vec @ c_vec   # normalisation factor so |Δs| = 1 for epsilon=0

    # Inhibitory IPSP: negative alpha-function drive, normalised so s drops by 1
    I_ipsp = -gn_fn(1, t_ms / tau) / c2   # shape (T,)

    # Independent OU noise, amplitude SIGMA_NOISE
    noise = SIGMA_NOISE * generate_ou_noise(N, seed + 1)   # (N, T)

    R = np.zeros((N, len(t_ms)))           # start silent — same IC as feedforward

    for k in range(len(t_ms) - 1):
        I_k = I_ipsp[k] * c_vec + noise[:, k]   # combined drive at step k

        def drdt(r, _I=I_k):                     # capture by default arg
            return (-r + b * (c_vec @ r) + _I) / tau

        R[:, k + 1] = rk4_step(drdt, R[:, k], dt_ms)

    return R, b


# ── Feedforward with IPSP-shifted basis ──────────────────────────────────────
def compute_feedforward_ipsp(epsilon: float, seed: int = SEED):
    """
    Numerically integrate the feedforward chain driven by inhibitory IPSP.
    tau dG_n/dt = -G_n + (1+epsilon)*G_{n-1},  G_{-1}(t) = -g_1(t/tau).
    OU noise is added to the summed-output trace at amplitude SIGMA_NOISE.
    """
    f_in  = -gn_fn(1, t_ms / tau)     # inhibitory IPSP: negative alpha function
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    w     = 1.0 + epsilon
    for k in range(len(t_ms) - 1):
        src    = np.empty(N)
        src[0] = f_in[k]
        src[1:] = w * G[:N-1, k]
        G[:, k+1] = decay * G[:, k] + rise * src

    mask_fit = (t_ms > 0) & (t_ms <= T_target_ms)
    res = lsq_linear(G[:, mask_fit].T, -np.ones(mask_fit.sum()), bounds=(-5.0, 5.0))
    W_opt = res.x

    # OU noise on the output (1 readout neuron) for fair comparison
    out_noise = SIGMA_NOISE * generate_ou_noise(1, seed + 2)[0]  # shape (T,)

    return G, W_opt, out_noise


# ── Pre-compute all cases ─────────────────────────────────────────────────────
print("Simulating IPSP + noise study…")
results = []
for _, _, epsilon in cases:
    R_la, b_la          = simulate_line_attractor_ipsp(epsilon)
    G_ff, W_opt, ou_out = compute_feedforward_ipsp(epsilon)
    results.append((R_la, b_la, G_ff, W_opt, ou_out))
    print(f"  epsilon={epsilon:+.3f}  done")


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    r"IPSP + Noise: Line Attractor vs Feedforward  ($N=100$, $\tau=100$ ms)"
    "\n"
    r"Input: $g_1(t/\tau)$ alpha-function drive + OU noise ($\sigma="
    + f"{SIGMA_NOISE}"
    + r"$, $\tau_{\rm noise}=\tau$)  |  shaded: full fit window",
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


N_SHOW_EACH = 3

for col, ((label, title, epsilon), (R_la, b_la, G_ff, W_opt, ou_out)) in enumerate(
    zip(cases, results)
):
    # ── Line attractor ────────────────────────────────────────────────────
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_la[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_r = fig.add_subplot(gs_sub[0])
    ax_o = fig.add_subplot(gs_sub[1])

    idx_sorted = np.argsort(b_la)[::-1]
    idx_show   = idx_sorted[np.linspace(0, N - 1, N_SHOW_EACH * 2, dtype=int)]
    for i, ni in enumerate(idx_show):
        shade = 0.35 + 0.55 * i / (N_SHOW_EACH * 2 - 1)
        ax_r.plot(t_s, R_la[ni], color=plt.cm.Blues(shade), lw=0.9, alpha=0.9)

    out_la = R_la.sum(axis=0)
    ax_o.plot(t_s, out_la, color='steelblue', lw=2)

    # Both start from 0; IPSP drives output negative
    ylim_la = [-5.5, 0.05] if epsilon > 0 else [-1.35, 0.05]
    decorate_output(ax_o, ylim=ylim_la)
    ax_r.set_title(title, fontsize=9, pad=3)
    ax_r.set_xlim([0, T_ms / 1000])
    shown_max = R_la[idx_show].max()
    ax_r.set_ylim([0, max(shown_max * 1.2, 0.01)])
    ax_r.set_xticks([])
    ax_r.tick_params(labelsize=7)
    if col == 0:
        ax_r.set_ylabel("Neuronal\nactivity", fontsize=8)
        ax_o.set_ylabel("Summed\noutput", fontsize=8)
    ax_o.set_xlabel("Time (sec)", fontsize=8)
    ax_r.text(0.03, 0.92, chr(ord('A') + col),
              transform=ax_r.transAxes, fontsize=12, fontweight='bold', va='top')

    # ── Feedforward ───────────────────────────────────────────────────────
    gs_sub_ff = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_ff[col], hspace=0.08, height_ratios=[2, 1],
    )
    ax_ff_r = fig.add_subplot(gs_sub_ff[0])
    ax_ff_o = fig.add_subplot(gs_sub_ff[1])

    idx_ff_show = np.linspace(0, N - 1, 6, dtype=int)
    for i, ni in enumerate(idx_ff_show):
        c = plt.cm.Oranges(0.35 + 0.55 * i / (len(idx_ff_show) - 1))
        ax_ff_r.plot(t_s, G_ff[ni], color=c, lw=0.9, alpha=0.9)

    out_ff_opt  = G_ff.T @ W_opt + ou_out
    out_ff_eq   = G_ff.sum(axis=0) + ou_out

    ax_ff_o.plot(t_s, out_ff_eq,  color='gray',      lw=1.2, ls='-',
                 alpha=0.6, label="W's = 1")
    ax_ff_o.plot(t_s, out_ff_opt, color='darkorange', lw=2,
                 label="W's opt")
    # FF driven from 0: inhibitory IPSP pushes output negative
    decorate_output(ax_ff_o, ylim=[-1.35, 0.05])

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
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\noise_ipsp.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
