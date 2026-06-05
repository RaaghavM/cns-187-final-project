"""
Memory evaluation: Line Attractor vs Feedforward Network

Metric: T_p% — the first time the output drifts more than p% away from target 1.
Three thresholds are reported side-by-side to show how the choice of tolerance
changes the result: T_1% (tight), T_5% (standard), T_10% (loose).

For the LA the exact formula is:
    T_p%(epsilon) = tau * |ln(1 - p/100)| / |epsilon|  (decay, epsilon < 0)
                  = tau *  ln(1 + p/100)  / epsilon     (growth, epsilon > 0)
This diverges as epsilon -> 0: the LA has infinite memory at perfect tuning.

For the FF the threshold time is measured numerically from the simulated output.
It saturates at a finite value as epsilon -> 0 (the basis functions all eventually
decay regardless of tuning) and that saturation value shifts with threshold choice.

Figure panels
-------------
  A  Log output traces, Line Attractor — threshold lines marked
  B  Log output traces, Feedforward (solid = W_opt, dashed = W=1)
  C  T_1%, T_5%, T_10% vs epsilon — LA analytical vs FF W_opt
  D  T_1%, T_5%, T_10% vs epsilon — FF W=1 vs FF W_opt (readout comparison)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.special import gammaln
from reduced_autapse_line_attractor import AveragedAutapseParams

# ── Parameters ────────────────────────────────────────────────────────────────
N     = 100
tau   = AveragedAutapseParams().tau      # 100 ms
tau_s = tau / 1000.0                     # 0.1 s

T_ms  = 12_000.0
dt_ms = 1.0
t_ms  = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s   = t_ms / 1000.0
SEED  = 42

T_SKIP = 500.0   # ms — skip initial fast-mode transient before checking threshold

THRESHOLDS = [
    (0.01, 'T_1%',  'royalblue'),
    (0.05, 'T_5%',  'darkorange'),
    (0.10, 'T_10%', 'crimson'),
]

eps_trace = [-0.06, -0.005, 0.0, +0.005, +0.02]
eps_sweep = np.sort(np.concatenate([
    -np.logspace(np.log10(0.001), np.log10(0.12), 30),
    [0.0],
     np.logspace(np.log10(0.001), np.log10(0.06),  30),
]))


# ── Utilities ─────────────────────────────────────────────────────────────────
def rk4_step(f, y, dt):
    k1 = f(y); k2 = f(y + 0.5*dt*k1)
    k3 = f(y + 0.5*dt*k2); k4 = f(y + dt*k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def gn_fn(n, th):
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.where(th > 0,
                      n*np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n+1),
                      0.0 if n == 0 else -np.inf)
    return np.exp(lg)


# ── Line attractor ────────────────────────────────────────────────────────────
def simulate_la(epsilon, seed=SEED):
    rng = np.random.default_rng(seed)
    b   = np.abs(rng.standard_normal(N)) + 0.1;  b /= b.sum()
    c   = np.abs(rng.standard_normal(N)) + 0.1;  c *= (1 + epsilon) / (b @ c)
    a   = np.abs(rng.standard_normal(N)) + 0.1;  a *= (1 + epsilon) / (c @ a)
    R       = np.zeros((N, len(t_ms)))
    R[:, 0] = a
    for k in range(len(t_ms) - 1):
        def drdt(r, _b=b, _c=c):
            return (-r + _b * (_c @ r)) / tau
        R[:, k+1] = rk4_step(drdt, R[:, k], dt_ms)
    return R.sum(axis=0)


# ── Feedforward chain (exact ZOH step) ───────────────────────────────────────
def ff_chain(epsilon):
    G     = np.zeros((N, len(t_ms)))
    decay = np.exp(-dt_ms / tau)
    rise  = 1.0 - decay
    w     = 1.0 + epsilon
    G[0, 0] = 1.0
    for k in range(len(t_ms) - 1):
        src     = np.empty(N)
        src[0]  = 0.0
        src[1:] = w * G[:N-1, k]
        G[:, k+1] = decay * G[:, k] + rise * src
    return G


def fit_w(G):
    mask = t_ms > 0
    return lsq_linear(G[:, mask].T, np.ones(mask.sum()), bounds=(-5.0, 5.0)).x


# ── Threshold times ───────────────────────────────────────────────────────────
def la_T(eps, p_frac):
    """
    Exact time for LA output exp(eps*t/tau) to drift by p_frac from 1.
    eps < 0: decay crosses (1 - p_frac);  eps > 0: growth crosses (1 + p_frac).
    Returns inf for eps == 0.
    """
    if eps == 0.0:
        return np.inf
    elif eps < 0:
        return tau_s * np.log(1.0 - p_frac) / eps   # both log and eps negative → positive
    else:
        return tau_s * np.log(1.0 + p_frac) / eps


def ff_T(out, p_frac):
    """
    First time after T_SKIP ms where |out(t) - 1| > p_frac.
    Returns inf if the output never crosses the threshold.
    Returns T_SKIP/1000 if it has already crossed at T_SKIP (very fast decay).
    """
    skip_i   = np.searchsorted(t_ms, T_SKIP)
    window   = out[skip_i:]
    t_window = t_ms[skip_i:]
    crossed  = np.abs(window - 1.0) > p_frac
    if not np.any(crossed):
        return np.inf
    return t_window[np.argmax(crossed)] / 1000.0   # seconds


# ── Simulate ─────────────────────────────────────────────────────────────────
print("Simulating trace cases...")
la_out     = {}
ff_eq_out  = {}
ff_opt_out = {}
for eps in eps_trace:
    G = ff_chain(eps);  W = fit_w(G)
    la_out[eps]     = simulate_la(eps)
    ff_eq_out[eps]  = G.sum(axis=0)
    ff_opt_out[eps] = G.T @ W
    print(f"  eps={eps:+.4f}  done")

print("Sweeping epsilon...")
# For each (epsilon, threshold) store the threshold crossing time
ff_eq_times  = {p: [] for p, _, _ in THRESHOLDS}
ff_opt_times = {p: [] for p, _, _ in THRESHOLDS}

for i, eps in enumerate(eps_sweep):
    G   = ff_chain(eps);  W = fit_w(G)
    eq  = G.sum(axis=0)
    opt = G.T @ W
    for p_frac, _, _ in THRESHOLDS:
        ff_eq_times[p_frac].append(ff_T(eq,  p_frac))
        ff_opt_times[p_frac].append(ff_T(opt, p_frac))
    print(f"  {i+1}/{len(eps_sweep)} eps={eps:+.4f}", end='\r')

# Convert to arrays; replace inf with NaN for cleaner plotting
for p_frac, _, _ in THRESHOLDS:
    ff_eq_times[p_frac]  = np.array(ff_eq_times[p_frac],  dtype=float)
    ff_opt_times[p_frac] = np.array(ff_opt_times[p_frac], dtype=float)

la_times = {p: np.array([la_T(e, p) for e in eps_sweep]) for p, _, _ in THRESHOLDS}

print("\nDone.")


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    r"Memory Evaluation: Line Attractor vs Feedforward  ($N=100$, $\tau=100$ ms)"
    "\nDelta-pulse IC  |  "
    r"$T_{p\%}$ = first time $|$output$-1| > p/100$"
    f"  |  threshold skip = {T_SKIP:.0f} ms",
    fontsize=10,
)

cmap     = plt.cm.RdBu_r
eps_arr  = np.array(eps_trace)
norm     = plt.Normalize(vmin=eps_arr.min(), vmax=eps_arr.max())
skip_s   = T_SKIP / 1000.0


# ── Panel A: LA output traces ─────────────────────────────────────────────────
ax = axes[0, 0]
for eps in eps_trace:
    out = la_out[eps]
    col = cmap(norm(eps))
    pos = out > 1e-6
    ax.semilogy(t_s[pos], out[pos], color=col, lw=1.8, label=f"eps={eps:+.3f}")
ax.axhline(1.00, color='k', ls='--', lw=0.8, alpha=0.5)
# Mark the three threshold levels
for p_frac, lbl, col in THRESHOLDS:
    ax.axhline(1.0 - p_frac, color=col, ls=':', lw=0.9, alpha=0.7, label=lbl)
ax.set_xlim([0, T_ms/1000]);  ax.set_ylim([1e-4, 1e2])
ax.set_xlabel("Time (s)");    ax.set_ylabel("Summed output (log)")
ax.set_title("A  Line Attractor output traces")
ax.legend(fontsize=7, loc='lower left', ncol=2)
ax.tick_params(labelsize=8)


# ── Panel B: FF output traces ─────────────────────────────────────────────────
ax = axes[0, 1]
for eps in eps_trace:
    col = cmap(norm(eps))
    eq  = ff_eq_out[eps];   pos_eq  = eq  > 1e-6
    opt = ff_opt_out[eps];  pos_opt = opt > 1e-6
    ax.semilogy(t_s[pos_eq],  eq[pos_eq],   color=col, lw=0.9, ls='--', alpha=0.6)
    ax.semilogy(t_s[pos_opt], opt[pos_opt], color=col, lw=1.8, ls='-',
                label=f"eps={eps:+.3f}")
ax.axhline(1.00, color='k', ls='--', lw=0.8, alpha=0.5)
for p_frac, lbl, col in THRESHOLDS:
    ax.axhline(1.0 - p_frac, color=col, ls=':', lw=0.9, alpha=0.7)
ax.plot([], [], 'k-',  lw=1.8, label="solid: W_opt")
ax.plot([], [], 'k--', lw=0.9, label="dash:  W=1",   alpha=0.6)
ax.set_xlim([0, T_ms/1000]);  ax.set_ylim([1e-4, 1e2])
ax.set_xlabel("Time (s)");    ax.set_ylabel("Summed output (log)")
ax.set_title("B  Feedforward output traces")
ax.legend(fontsize=7, loc='lower left', ncol=2)
ax.tick_params(labelsize=8)


# ── Shared helper for threshold panels ───────────────────────────────────────
def plot_threshold_panel(ax, series_list, title, ylabel=True):
    """
    series_list: list of (times_array, label, color, linestyle, marker)
    Plots T_p% vs epsilon on a semilogy axis.
    Inf values are dropped; the eps=0 point is annotated separately.
    """
    for times, label, color, ls, marker in series_list:
        finite = np.isfinite(times)
        ax.semilogy(eps_sweep[finite], times[finite],
                    color=color, ls=ls, marker=marker, ms=3,
                    lw=1.6, label=label)
    ax.axvline(0.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel("Mistuning epsilon")
    if ylabel:
        ax.set_ylabel("Threshold crossing time (s)")
    ax.set_title(title)
    ax.set_xlim([eps_sweep.min(), eps_sweep.max()])
    ax.set_ylim([0.01, 200])
    ax.legend(fontsize=7, loc='upper center')
    ax.tick_params(labelsize=8)


# ── Panel C: LA analytical vs FF W_opt, all three thresholds ─────────────────
ax = axes[1, 0]
series_C = []
for p_frac, lbl, col in THRESHOLDS:
    # LA analytical
    series_C.append((la_times[p_frac], f"LA {lbl}", col, '-', ''))
    # FF W_opt measured
    series_C.append((ff_opt_times[p_frac], f"FF W_opt {lbl}", col, '--', 'o'))

plot_threshold_panel(ax, series_C, "C  LA (analytical) vs FF W_opt")


# ── Panel D: FF W=1 vs FF W_opt, all three thresholds ────────────────────────
ax = axes[1, 1]
series_D = []
for p_frac, lbl, col in THRESHOLDS:
    series_D.append((ff_eq_times[p_frac],  f"FF W=1  {lbl}", col, '--', 's'))
    series_D.append((ff_opt_times[p_frac], f"FF Wopt {lbl}", col, '-',  'o'))

plot_threshold_panel(ax, series_D, "D  FF W=1 vs FF W_opt", ylabel=False)
axes[1, 1].set_ylabel("Threshold crossing time (s)")

plt.tight_layout()
out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\memory_eval.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
