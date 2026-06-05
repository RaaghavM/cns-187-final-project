"""
Improvements to Feedforward and Line Attractor Networks
=======================================================

FEEDFORWARD — problem: requires N = T/tau neurons
--------------------------------------------------
Fix: log-spaced time constants.
  Replace the homogeneous chain (all stages at tau=100ms) with M stages
  whose time constants grow geometrically: tau_n = tau_min * r^n,
  with r = (tau_max/tau_min)^(1/(M-1)).

  Why this helps: stage n covers the time interval [tau_{n-1}, tau_n] on a
  log scale.  With M stages spanning [tau_min, T], the number of stages
  needed is M = log_r(T/tau_min) instead of T/tau_min.  For T/tau = 100,
  log_2(100) ~ 7 stages suffice in principle.

  Biological plausibility: neurons across hippocampus, prefrontal and
  temporal cortex show heterogeneous intrinsic time constants spanning
  orders of magnitude (Cavanagh et al. 2016, Murray et al. 2014).

LINE ATTRACTOR — problem: fragile zero eigenvalue, unbounded noise random walk
-------------------------------------------------------------------------------
Fix: homeostatic feedback (proportional control).
  Add a weak restoring force toward a target amplitude S*:
    tau dr_i/dt = -r_i + b_i*(c.r) + alpha*(S* - sum_r)*b_i

  Effect on the slow mode:
    tau ds/dt = eps*s + alpha*(S* - s)
             = (eps - alpha)*s + alpha*S*

  * For eps=0:  effective eigenvalue = -alpha  (stable, time constant tau/alpha)
  * For eps>0:  effective eigenvalue = eps-alpha  (stable if alpha>eps)
  * For eps<0:  effective eigenvalue = eps-alpha  (more stable, decays faster)
  * Fixed point: s* = alpha*S* / (alpha - eps) ~ S*(1 + eps/alpha) for eps<<alpha

  Noise: converts unbounded random walk into OU process.
  Variance at steady state: Var(s) ~ sigma_r^2 * |c|^2 / (2*alpha*tau)
  (bounded, unlike the growing variance without feedback)

  Biological plausibility: divisive normalization by inhibitory interneurons,
  homeostatic synaptic scaling (Turrigiano 2008).

Figure layout
-------------
  A  Basis functions: log-spaced (M=10) vs homogeneous (N=100)
  B  T_p% vs number of stages M for log-spaced chain  [main result]
  C  Robust LA output traces: standard vs homeostatic feedback, several eps
  D  T_5% vs eps: standard LA vs robust LA for several alpha values
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.special import gammaln
from reduced_autapse_line_attractor import AveragedAutapseParams

# ── Shared parameters ─────────────────────────────────────────────────────────
N    = 100
tau  = AveragedAutapseParams().tau      # 100 ms
tau_s = tau / 1000.0

T_ms  = 12_000.0
dt_ms = 1.0
t_ms  = np.arange(0.0, T_ms + dt_ms, dt_ms)
t_s   = t_ms / 1000.0
SEED  = 42

T_SKIP = 500.0   # ms — skip initial transient
THRESHOLDS = [(0.01, 'T_1%', 'royalblue'),
              (0.05, 'T_5%', 'darkorange'),
              (0.10, 'T_10%', 'crimson')]


# ── Utilities ──────────────────────────────────────────────────────────────────
def rk4_step(f, y, dt):
    k1 = f(y); k2 = f(y + 0.5*dt*k1)
    k3 = f(y + 0.5*dt*k2); k4 = f(y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def fit_w(G, target=None):
    if target is None:
        target = np.ones(len(t_ms))
    mask = t_ms > 0
    return lsq_linear(G[:, mask].T, target[mask], bounds=(-5.0, 5.0)).x

def T_threshold(out, p_frac):
    """First time after T_SKIP where |out - 1| > p_frac. Returns inf if never."""
    skip_i = np.searchsorted(t_ms, T_SKIP)
    win = out[skip_i:]; t_win = t_ms[skip_i:]
    hit = np.abs(win - 1.0) > p_frac
    return t_win[np.argmax(hit)] / 1000.0 if np.any(hit) else np.inf


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Heterogeneous-tau feedforward
# ══════════════════════════════════════════════════════════════════════════════

def hetero_ff_chain(taus_ms: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Feedforward chain with per-stage time constants given by taus_ms (ms).
    Stage n: tau_n dG_n/dt = -G_n + (1+eps)*G_{n-1}
    Exact ZOH step.  Delta-pulse IC: G[0,0]=1.
    Returns G: (M, T).
    """
    M = len(taus_ms)
    decays = np.exp(-dt_ms / taus_ms)       # (M,)
    rises  = 1.0 - decays                    # (M,)
    w = 1.0 + epsilon
    G = np.zeros((M, len(t_ms)))
    G[0, 0] = 1.0
    for k in range(len(t_ms) - 1):
        G[0, k+1] = decays[0] * G[0, k]             # stage 0: no input after t=0
        G[1:, k+1] = (decays[1:] * G[1:, k]
                      + rises[1:] * w * G[:-1, k])  # stages 1..M-1
    return G

def homogeneous_ff_chain(N_stages: int, epsilon: float = 0.0) -> np.ndarray:
    """Standard homogeneous chain: all stages at tau=tau."""
    taus = np.full(N_stages, tau)
    return hetero_ff_chain(taus, epsilon)


# --- Evaluate T_p% vs number of stages M for log-spaced chain ----------------
M_vals = [5, 7, 10, 15, 20, 30, 50, 100]
tau_max_ms = T_ms   # slowest stage spans the full window

print("Evaluating log-spaced feedforward chains...")
hetero_results = {}   # M -> {p_frac: T_threshold}
for M in M_vals:
    taus = np.geomspace(tau, tau_max_ms, M)
    G    = hetero_ff_chain(taus)
    W    = fit_w(G)
    out  = G.T @ W
    hetero_results[M] = {p: T_threshold(out, p) for p, _, _ in THRESHOLDS}
    print(f"  M={M:3d}  T_5% = {hetero_results[M][0.05]:.2f} s")

# Homogeneous N=100 reference
G_hom = homogeneous_ff_chain(100)
W_hom = fit_w(G_hom)
out_hom = G_hom.T @ W_hom
hom_ref = {p: T_threshold(out_hom, p) for p, _, _ in THRESHOLDS}
print(f"  Homogeneous N=100  T_5% = {hom_ref[0.05]:.2f} s")

# Homogeneous T_p% vs M (for direct per-M comparison with log-spaced)
print("Evaluating homogeneous feedforward chains...")
hom_results = {}
for M in M_vals:
    G_m   = homogeneous_ff_chain(M)
    W_m   = fit_w(G_m)
    out_m = G_m.T @ W_m
    hom_results[M] = {p: T_threshold(out_m, p) for p, _, _ in THRESHOLDS}
    print(f"  M={M:3d}  T_5% = {hom_results[M][0.05]:.2f} s")

# Basis functions for the comparison plot
taus_demo = np.geomspace(tau, tau_max_ms, 50)
G_hetero_demo = hetero_ff_chain(taus_demo)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Robust line attractor with homeostatic feedback
# ══════════════════════════════════════════════════════════════════════════════

def simulate_la_feedback(epsilon: float, alpha: float,
                         seed: int = SEED) -> np.ndarray:
    """
    Line attractor + homeostatic feedback.

    tau dr_i/dt = -r_i + b_i*(c.r)  +  alpha*(S_target - sum_r)*b_i

    The extra term is a force in the b-direction proportional to how far the
    total activity is from S_target=1.  It modifies the slow-mode eigenvalue:
        effective eigenvalue = eps - alpha
    and turns the unbounded random walk (alpha=0) into an OU process.

    alpha = 0   : standard line attractor
    alpha > 0   : soft restoring force; memory time constant ~ tau/alpha
    """
    rng = np.random.default_rng(seed)
    b   = np.abs(rng.standard_normal(N)) + 0.1;  b /= b.sum()
    c   = np.abs(rng.standard_normal(N)) + 0.1;  c *= (1+epsilon) / (b@c)
    a   = np.abs(rng.standard_normal(N)) + 0.1;  a *= (1+epsilon) / (c@a)
    S_target = 1.0
    R = np.zeros((N, len(t_ms)));  R[:, 0] = a
    for k in range(len(t_ms) - 1):
        sum_r = R[:, k].sum()
        fb    = alpha * (S_target - sum_r) * b   # homeostatic restoring force
        def drdt(r, _b=b, _c=c, _f=fb):
            return (-r + _b*(_c@r) + _f) / tau
        R[:, k+1] = rk4_step(drdt, R[:, k], dt_ms)
    return R.sum(axis=0)


# --- Trace plots: a few (eps, alpha) combinations ----------------------------
eps_trace = [-0.02, 0.0, +0.02]
alpha_vals_trace = [0.0, 0.005, 0.02]
colors_trace = {'eps': {-0.02: 'steelblue', 0.0: 'k', +0.02: 'firebrick'},
                'alpha': {0.0: '-', 0.005: '--', 0.02: ':'}}

print("Simulating robust LA traces...")
la_traces = {}
for eps in eps_trace:
    for alpha in alpha_vals_trace:
        la_traces[(eps, alpha)] = simulate_la_feedback(eps, alpha)
        print(f"  eps={eps:+.3f}  alpha={alpha:.3f}  done")


# --- Sweep over epsilon for several alpha values ----------------------------
eps_sweep = np.sort(np.concatenate([
    -np.logspace(np.log10(0.001), np.log10(0.08), 25),
    [0.0],
     np.logspace(np.log10(0.001), np.log10(0.04), 25),
]))

alpha_vals_sweep = [0.0, 0.003, 0.01, 0.03]
sweep_colors     = ['k', 'green', 'darkorange', 'purple']

# Analytical threshold time for robust LA:
#   effective eigenvalue lambda = eps - alpha
#   slow-mode crosses 1-p at t = tau_s * log(1/(1-p)) / |lambda|
def la_feedback_T(eps, alpha, p_frac):
    lam = eps - alpha           # effective eigenvalue
    S_star = alpha / (alpha - eps) if alpha != eps else np.inf  # fixed point / S_target
    # Fixed point within tolerance?
    if alpha > 0 and abs(S_star - 1.0) <= p_frac:
        return np.inf           # steady state is within tolerance band -> T_p = inf
    if lam == 0.0:
        return np.inf
    elif lam < 0:               # decaying toward S_star
        # Output decays from 1 to S_star; check if S_star is within tolerance first
        # If S_star < 1-p, output will eventually leave the tolerance band
        if S_star >= 1.0 - p_frac:
            return np.inf       # converges to a point inside tolerance
        # Time to cross 1-p (decaying case)
        return tau_s * np.log((1.0 - S_star) / (1.0 - p_frac - S_star)) / abs(lam)
    else:                       # growing
        # Check upper threshold (1+p)
        if S_star <= 1.0 + p_frac:
            return np.inf
        return tau_s * np.log((1.0 + p_frac - S_star) / (1.0 - S_star)) / lam \
               if S_star < 1.0 else \
               tau_s * np.log(1.0 + p_frac) / lam

print("Computing sweep...")
la_sweep_T5 = {a: [] for a in alpha_vals_sweep}
for eps in eps_sweep:
    for alpha in alpha_vals_sweep:
        la_sweep_T5[alpha].append(la_feedback_T(eps, alpha, 0.05))

for alpha in alpha_vals_sweep:
    la_sweep_T5[alpha] = np.array(la_sweep_T5[alpha], dtype=float)
print("  done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Improvements: Log-spaced Feedforward  |  Homeostatic Line Attractor\n"
    r"$N=100$, $\tau=100$ ms, $T=12$ s",
    fontsize=11,
)

# ── Panel A: First 6 basis functions + optimised readout for each chain ───────
ax = axes[0, 0]
N_SHOW = 20
cmap_h = plt.cm.Oranges
cmap_l = plt.cm.Blues

# Individual basis functions (thin, semi-transparent)
for i in range(N_SHOW):
    shade = 0.35 + 0.55 * i / (N_SHOW - 1)
    ax.plot(t_s, G_hom[i],          color=cmap_h(shade), lw=0.9, alpha=0.7)
    ax.plot(t_s, G_hetero_demo[i],  color=cmap_l(shade), lw=0.9, alpha=0.7)

# Optimised readout using only the first 6 stages of each
W_hom6    = fit_w(G_hom[:N_SHOW])
W_hetero6 = fit_w(G_hetero_demo[:N_SHOW])
out_hom6    = G_hom[:N_SHOW].T    @ W_hom6
out_hetero6 = G_hetero_demo[:N_SHOW].T @ W_hetero6
ax.plot(t_s, out_hom6,    color='darkorange', lw=2.2,
        label=f'Homogeneous opt. readout (20 stages, tau={tau:.0f}ms each)')
ax.plot(t_s, out_hetero6, color='steelblue',  lw=2.2,
        label=f'Log-spaced opt. readout (20 stages, tau_min={tau:.0f}–{tau_max_ms:.0f}ms)')
# Legend proxies for individual curves
ax.plot([], [], color=cmap_h(0.6), lw=0.9, alpha=0.7, label='Homogeneous basis fns')
ax.plot([], [], color=cmap_l(0.6), lw=0.9, alpha=0.7, label='Log-spaced basis fns')

ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.set_xlim([0, T_ms/1000]);  ax.set_ylim([-0.2, 1.6])
ax.set_xlabel("Time (s)");  ax.set_ylabel("Activity / readout output")
ax.set_title("A  First 20 basis functions + optimised readout")
ax.legend(fontsize=6.5);  ax.tick_params(labelsize=8)

# ── Panel B: T_p% vs M — log-spaced vs homogeneous ───────────────────────────
ax = axes[0, 1]
for p_frac, lbl, col in THRESHOLDS:
    # Log-spaced curves
    times_h = [hetero_results[M][p_frac] for M in M_vals]
    fin_h   = [(M_vals[i], t) for i, t in enumerate(times_h) if np.isfinite(t)]
    if fin_h:
        ax.plot(*zip(*fin_h), 'o-', color=col, lw=1.8, ms=6,
                label=f'Log-spaced {lbl}')
    # Homogeneous curves (dashed, same colour)
    times_u = [hom_results[M][p_frac] for M in M_vals]
    fin_u   = [(M_vals[i], t) for i, t in enumerate(times_u) if np.isfinite(t)]
    if fin_u:
        ax.plot(*zip(*fin_u), 's--', color=col, lw=1.4, ms=5, alpha=0.75,
                label=f'Homogeneous {lbl}')

ax.set_xlabel("Number of stages M")
ax.set_ylabel("Threshold crossing time (s)")
ax.set_title("B  Memory duration vs M: log-spaced vs homogeneous")
ax.set_xlim([4, 110]);  ax.set_ylim([0, T_ms/1000 + 1])
ax.legend(fontsize=6.5, ncol=2);  ax.tick_params(labelsize=8)

# ── Panel C: Robust LA traces ─────────────────────────────────────────────────
ax = axes[1, 0]
from matplotlib.lines import Line2D
ls_map    = {0.0: '-', 0.005: '--', 0.02: ':'}
lw_map    = {0.0: 1.2, 0.005: 1.8, 0.02: 1.2}
col_map   = {-0.02: 'steelblue', 0.0: 'k', 0.02: 'firebrick'}

for eps in eps_trace:
    for alpha in alpha_vals_trace:
        out = la_traces[(eps, alpha)]
        ax.plot(t_s, out, color=col_map[eps], ls=ls_map[alpha],
                lw=lw_map[alpha], alpha=0.85)

ax.axhline(1.0,  color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axhline(0.95, color='gray', ls=':',  lw=0.7, alpha=0.5)
ax.axhline(1.05, color='gray', ls=':',  lw=0.7, alpha=0.5)

# Two-group legend: color = epsilon, style = alpha
color_handles = [
    Line2D([0],[0], color='steelblue', lw=2, label=r'$\varepsilon = -2\%$ (leaky)'),
    Line2D([0],[0], color='k',         lw=2, label=r'$\varepsilon = 0$ (tuned)'),
    Line2D([0],[0], color='firebrick', lw=2, label=r'$\varepsilon = +2\%$ (unstable)'),
]
style_handles = [
    Line2D([0],[0], color='gray', lw=1.2, ls='-',  label=r'$\alpha = 0$ (no feedback)'),
    Line2D([0],[0], color='gray', lw=1.8, ls='--', label=r'$\alpha = 0.005$'),
    Line2D([0],[0], color='gray', lw=1.2, ls=':',  label=r'$\alpha = 0.02$'),
]
ax.set_xlim([0, T_ms/1000]);  ax.set_ylim([-0.1, 5.5])
ax.set_xlabel("Time (s)");  ax.set_ylabel("Summed output")
ax.set_title("C  Homeostatic LA: color = mistuning, style = feedback strength")
ax.legend(handles=color_handles + style_handles, fontsize=7.5, ncol=2)
ax.tick_params(labelsize=8)

# ── Panel D: T_5% vs epsilon for different alpha ──────────────────────────────
ax = axes[1, 1]
for alpha, col in zip(alpha_vals_sweep, sweep_colors):
    times = la_sweep_T5[alpha]
    finite = np.isfinite(times)
    lbl = (f'alpha={alpha} (tau_mem={tau_s/alpha:.0f}s)' if alpha > 0
           else 'alpha=0 (standard LA)')
    # Plot finite values
    ax.semilogy(eps_sweep[finite], times[finite], '-', color=col, lw=1.8, label=lbl)
    # Mark where T_5% = inf (memory stays within 5% forever)
    inf_mask = ~finite
    if np.any(inf_mask):
        ax.scatter(eps_sweep[inf_mask], np.full(inf_mask.sum(), 200),
                   color=col, marker='^', s=40, zorder=5, alpha=0.7)

ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel("Mistuning epsilon")
ax.set_ylabel("T_5% (s)")
ax.set_title("D  Memory duration T_5% vs eps for homeostatic LA\n"
             "(^ markers = T_5% = inf: memory stays within 5% tolerance)")
ax.set_xlim([eps_sweep.min(), eps_sweep.max()])
ax.set_ylim([0.01, 500])
ax.legend(fontsize=7.5);  ax.tick_params(labelsize=8)

plt.tight_layout()
out_path = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\improvements.png"
)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"\nHomogeneous FF (N=100):  T_5% = {hom_ref[0.05]:.1f}s")
for M in [10, 15, 20, 30]:
    t5 = hetero_results[M][0.05]
    s = f"{t5:.1f}s" if np.isfinite(t5) else "inf"
    print(f"Log-spaced FF (M={M:3d}):  T_5% = {s}  ({100-M} neurons saved)")

print(f"\nStandard LA (alpha=0):   T_5% = inf at eps=0, degrades rapidly for eps!=0")
for alpha in [0.003, 0.01, 0.03]:
    n_inf = np.sum(~np.isfinite(la_sweep_T5[alpha]))
    eps_range = eps_sweep[~np.isfinite(la_sweep_T5[alpha])]
    r = f"[{eps_range.min():+.3f}, {eps_range.max():+.3f}]" if len(eps_range) else "none"
    print(f"Robust LA (alpha={alpha:.3f}): T_mem~{tau_s/alpha:.0f}s, "
          f"T_5%=inf for eps in {r}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Skip-connection feedforward chain
# ══════════════════════════════════════════════════════════════════════════════
# Biological motivation: cortical circuits have long-range horizontal connections
# skipping intermediate layers (Salin & Bullier 1995).  Here each stage n
# receives equally-weighted input from the min(k, n) preceding stages.
# k=1 is the standard chain; larger k asks whether cross-stage pooling allows
# fewer stages to cover the same time window.

def skip_ff_chain(taus_ms: np.ndarray, epsilon: float = 0.0, k: int = 1) -> np.ndarray:
    """
    Log-spaced chain; stage n receives equal-weight input from stages n-1, ..., n-k.
    Inner loop runs only k times per time step → nearly as fast as k=1.
    """
    M      = len(taus_ms)
    decays = np.exp(-dt_ms / taus_ms)
    rises  = 1.0 - decays
    w      = 1.0 + epsilon
    G      = np.zeros((M, len(t_ms)))
    G[0, 0] = 1.0
    spans  = np.maximum(np.minimum(np.arange(M, dtype=float), k), 1.0)
    for t_k in range(len(t_ms) - 1):
        src = np.zeros(M)
        for j in range(1, k + 1):
            if j < M:
                src[j:] += G[:M - j, t_k]
        src /= spans
        src[0] = 0.0
        G[:, t_k + 1] = decays * G[:, t_k] + rises * w * src
    return G


M_vals_skip = [5, 7, 10, 15, 20, 30, 50]
skip_k_vals = [1, 2, 3]
skip_p5     = {k: [] for k in skip_k_vals}

print("\nEvaluating skip-connection chains...")
for k in skip_k_vals:
    for M in M_vals_skip:
        taus_sk = np.geomspace(tau, tau_max_ms, M)
        G_sk    = skip_ff_chain(taus_sk, k=k)
        W_sk    = fit_w(G_sk)
        out_sk  = G_sk.T @ W_sk
        skip_p5[k].append(T_threshold(out_sk, 0.05))
    print(f"  k={k} done  (T_5% at M=20: {skip_p5[k][M_vals_skip.index(20)]:.2f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Mistuned LA: standard vs homeostatic (Goldman Fig-7 style)
# ══════════════════════════════════════════════════════════════════════════════
# For each of four mistuning levels, run both the standard LA (alpha=0) and a
# homeostatic LA (fixed alpha).  Directly shows how feedback tames mistuning.

ALPHA_HOM   = 0.02     # homeostatic strength used in panel F
eps_f_vals  = [-0.06, -0.005, 0.0, +0.02]   # same four cases as Goldman Fig 7
eps_f_colors = {
    -0.06:  'steelblue',
    -0.005: 'cornflowerblue',
     0.0:   'k',
    +0.02:  'firebrick',
}
eps_f_labels = {
    -0.06:  r'$\varepsilon = -6\%$',
    -0.005: r'$\varepsilon = -0.5\%$',
     0.0:   r'$\varepsilon = 0$',
    +0.02:  r'$\varepsilon = +2\%$',
}

print("\nSimulating mistuned LA (standard vs homeostatic)...")
mistuned_out = {}   # (eps, alpha) -> summed output array
for eps in eps_f_vals:
    for alpha in [0.0, ALPHA_HOM]:
        mistuned_out[(eps, alpha)] = simulate_la_feedback(eps, alpha)
        print(f"  eps={eps:+.3f}  alpha={alpha:.3f}  done")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Divisive normalization line attractor
# ══════════════════════════════════════════════════════════════════════════════
# Replace additive homeostatic feedback with multiplicative (divisive) inhibition:
#   tau dr_i/dt = -r_i + b_i*(c.r) / (1 + kappa*sum(r))
# Implemented by a single interneuron reading total population activity.
# Slow-mode equation: tau ds/dt = eps*s / (1 + kappa*s)
# Implicit solution: log(s) + kappa*(s-1) = eps*t/tau  (starting s(0)=1)
# Crossing time at s=1+p: T5_upper = tau*(log(1+p) + kappa*p) / eps
# Crossing time at s=1-p: T5_lower = tau*(log(1-p) - kappa*p) / eps
# Both extend T_5% by ~(1+kappa) vs standard at the operating point.
# Key property: for eps>0 divisive normalization converts exponential runaway
# into asymptotically LINEAR growth  (s ~ eps*t/(tau*kappa) for large t).


def simulate_la_divisive(epsilon: float, kappa: float, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b   = np.abs(rng.standard_normal(N)) + 0.1;  b /= b.sum()
    c   = np.abs(rng.standard_normal(N)) + 0.1;  c *= (1 + epsilon) / (b @ c)
    a   = np.abs(rng.standard_normal(N)) + 0.1;  a *= (1 + epsilon) / (c @ a)
    R   = np.zeros((N, len(t_ms)));  R[:, 0] = a
    for t_k in range(len(t_ms) - 1):
        nf = 1.0 / (1.0 + kappa * R[:, t_k].sum())
        def drdt(r, _b=b, _c=c, _nf=nf):
            return (-r + _b * (_c @ r) * _nf) / tau
        R[:, t_k + 1] = rk4_step(drdt, R[:, t_k], dt_ms)
    return R.sum(axis=0)


def div_T5_analytical(eps: float, kappa: float, p: float = 0.05) -> float:
    """T_5% from the implicit slow-mode equation (exact for the ODE, no noise)."""
    if eps == 0.0:
        return np.inf
    if eps > 0:
        return tau_s * (np.log(1.0 + p) + kappa * p) / eps
    else:
        # log(1-p) - kappa*p < 0  and  eps < 0  →  positive time
        return tau_s * (np.log(1.0 - p) - kappa * p) / eps


kappa_div_vals  = [0.0, 1.0, 5.0, 20.0]
div_sweep_colors = ['k', 'mediumblue', 'darkorange', 'crimson']

div_sweep_T5 = {}
for kappa in kappa_div_vals:
    div_sweep_T5[kappa] = np.array([div_T5_analytical(e, kappa) for e in eps_sweep])

# Simulate traces to visually confirm analytical prediction
eps_div_traces = [0.0, -0.01, +0.01]
kappa_trace    = 5.0
print("\nSimulating divisive normalization LA traces...")
div_traces = {}
for eps in eps_div_traces:
    for kap in [0.0, kappa_trace]:
        div_traces[(eps, kap)] = simulate_la_divisive(eps, kap)
        print(f"  eps={eps:+.3f}  kappa={kap:.0f}  done")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Three new improvements (panels E, F, G)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    "Further Improvements  |  "
    "E: Skip-conn FF   F: Noisy LA   G: Divisive Normalization\n"
    r"N=100, $\tau$=100 ms",
    fontsize=10,
)

# ── Panel E: T_5% vs M for skip-connection chains ────────────────────────────
ax = axes2[0]
skip_colors_map = {1: 'steelblue', 2: 'darkorange', 3: 'crimson'}
skip_ls_map     = {1: '-',         2: '--',          3: ':'}
for k in skip_k_vals:
    t5_list    = skip_p5[k]
    fin_mask   = [np.isfinite(v) for v in t5_list]
    M_fin      = [M_vals_skip[i] for i, m in enumerate(fin_mask) if m]
    t5_fin     = [t5_list[i]     for i, m in enumerate(fin_mask) if m]
    ax.semilogx(M_fin, t5_fin, 'o' + skip_ls_map[k],
                color=skip_colors_map[k], lw=1.8, ms=6,
                label=f'k={k} ({k-1} skip conn.)')
ax.axhline(hom_ref[0.05], color='gray', ls=':', lw=1.3,
           label=f'Homogeneous N=100 (T_5%={hom_ref[0.05]:.1f}s)')
ax.set_xlabel("Log-spaced stages M");  ax.set_ylabel("T_5% (s)")
ax.set_title("E  Skip-connection feedforward\n"
             "(k prev. stages feed each neuron)")
ax.set_xlim([4, 60]);  ax.set_ylim([0, T_ms / 1000 + 1])
ax.legend(fontsize=7.5);  ax.tick_params(labelsize=8)

# ── Panel F: Mistuned LA — standard vs homeostatic (Goldman Fig-7 style) ──────
ax = axes2[1]
for eps in eps_f_vals:
    col = eps_f_colors[eps]
    lbl = eps_f_labels[eps]
    ax.plot(t_s, mistuned_out[(eps, 0.0)],        color=col, ls='-',  lw=1.6,
            label=lbl)
    ax.plot(t_s, mistuned_out[(eps, ALPHA_HOM)],  color=col, ls='--', lw=1.6)

ax.axhline(1.00, color='gray', ls='--', lw=0.8, alpha=0.6)
ax.axhline(0.95, color='gray', ls=':',  lw=0.7, alpha=0.5)
ax.axhline(1.05, color='gray', ls=':',  lw=0.7, alpha=0.5)

# Legend: color = epsilon, style = standard vs homeostatic
color_handles_f = [
    Line2D([0],[0], color=eps_f_colors[e], lw=2, label=eps_f_labels[e])
    for e in eps_f_vals
]
style_handles_f = [
    Line2D([0],[0], color='gray', lw=2, ls='-',  label=r'Standard ($\alpha=0$)'),
    Line2D([0],[0], color='gray', lw=2, ls='--', label=fr'Homeostatic ($\alpha={ALPHA_HOM}$)'),
]
ax.set_xlim([0, T_ms/1000]);  ax.set_ylim([-0.1, 5.5])
ax.set_xlabel("Time (s)");  ax.set_ylabel("Summed output")
ax.set_title(f"F  Mistuned LA: standard vs homeostatic\n"
             fr"(solid = standard, dashed = $\alpha={ALPHA_HOM}$)")
ax.legend(handles=color_handles_f + style_handles_f, fontsize=7.5, ncol=2)
ax.tick_params(labelsize=8)

# ── Panel G: Divisive norm — T_5% vs epsilon (analytical) + trace inset ──────
ax = axes2[2]
for kappa, col in zip(kappa_div_vals, div_sweep_colors):
    t5  = div_sweep_T5[kappa]
    fin = np.isfinite(t5)
    lbl = (f'kappa={kappa:.0f}  (eps_eff / {1+kappa:.0f})' if kappa > 0
           else 'kappa=0 (standard)')
    if np.any(fin):
        ax.semilogy(eps_sweep[fin], t5[fin], '-', color=col, lw=1.8, label=lbl)
# Overlay simulation points for verification
sim_eps_check = [-0.01, 0.01]
sim_kap_check = [0.0, kappa_trace]
for eps in sim_eps_check:
    for kap in sim_kap_check:
        out_check = div_traces[(eps, kap)]
        t5_sim    = T_threshold(out_check, 0.05)
        if np.isfinite(t5_sim):
            col = div_sweep_colors[kappa_div_vals.index(kap)] if kap in kappa_div_vals else 'gray'
            ax.scatter([eps], [t5_sim], color=col, marker='x', s=60, zorder=5)

ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel("Mistuning epsilon");  ax.set_ylabel("T_5% (s)")
ax.set_title("G  Divisive normalization: T_5% vs epsilon\n"
             r"$\tau\dot{r}_i = -r_i + b_i(c \!\cdot\! r)/(1+\kappa \sum r)$"
             "   (x = simulation)")
ax.set_xlim([eps_sweep.min(), eps_sweep.max()]);  ax.set_ylim([0.01, 2000])
ax.legend(fontsize=7.5);  ax.tick_params(labelsize=8)

plt.tight_layout()
out_path2 = (
    r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\improvements2.png"
)
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path2}")

print("\n=== Summary: new improvements ===")
print("\nSkip-connection FF (log-spaced, T_5% at M=20):")
for k in skip_k_vals:
    t5 = skip_p5[k][M_vals_skip.index(20)]
    print(f"  k={k}: T_5% = {t5:.2f}s")

print(f"\nMistuned LA (standard vs homeostatic alpha={ALPHA_HOM}):")
for eps in eps_f_vals:
    t5_std = T_threshold(mistuned_out[(eps, 0.0)],       0.05)
    t5_hom = T_threshold(mistuned_out[(eps, ALPHA_HOM)], 0.05)
    s_std  = f"{t5_std:.2f}s" if np.isfinite(t5_std) else "inf"
    s_hom  = f"{t5_hom:.2f}s" if np.isfinite(t5_hom) else "inf"
    print(f"  eps={eps:+.3f}: standard T_5%={s_std}  homeostatic T_5%={s_hom}")

print("\nDivisive normalization T_5% at eps=±0.01:")
for kappa in kappa_div_vals:
    t5p = div_T5_analytical(+0.01, kappa)
    t5n = div_T5_analytical(-0.01, kappa)
    print(f"  kappa={kappa:.0f}: T_5%(eps=+0.01)={t5p:.1f}s  T_5%(eps=-0.01)={t5n:.1f}s")
