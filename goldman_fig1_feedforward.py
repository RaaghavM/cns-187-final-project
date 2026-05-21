"""
Goldman (2009) "Memory without Feedback in a Neural Network"
Replication of Figure 1B–E: Feedforward network integration

Network dynamics (Eq. 2 in paper):
    τ dR_n/dt = -R_n + R_{n-1}     n = 1, ..., N-1
    R_0(t) = x(t)  (external input)

Analytical basis function for stage n (response to unit pulse):
    g_n(t̂) = (1/n!) · t̂^n · e^{-t̂},   t̂ = t/τ

Output:  r_out(t) = Σ_n W_n · g_n(t)
Equal weights W_n = 1 → r_out ≈ 1 (step) for 0 < t < N·τ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaln
from scipy.optimize import lsq_linear

# ── Network parameters (from Fig 1B caption) ────────────────────
N   = 100       # feedforward stages
tau = 0.1       # intrinsic time constant τ = 100 ms

# ── Time axis ────────────────────────────────────────────────────
dt  = 0.001                      # 1 ms
t   = np.arange(0, 15.001, dt)  # 0–15 s
th  = t / tau                    # dimensionless t̂ = t/τ


# ── Analytical basis functions ───────────────────────────────────
def gn_fn(n, th):
    """g_n(t̂) = (1/n!) · t̂^n · exp(-t̂), computed in log-space."""
    th = np.asarray(th, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_g = np.where(
            th > 0,
            n * np.log(np.where(th > 0, th, 1.0)) - th - gammaln(n + 1),
            0.0 if n == 0 else -np.inf
        )
    return np.exp(log_g)


# Pre-compute all basis functions: G[n, t]
G = np.array([gn_fn(n, th) for n in range(N)])   # (N, T)


# ── Pulse response (output = weighted sum of basis functions) ────
out_pulse_eq = G.sum(axis=0)       # equal weights W_n = 1  → step ≈ 1

# Optimal weights, constrained |W_n| ≤ 5 (per paper)
mask_fit = (t > 0) & (t <= N * tau)
res      = lsq_linear(G[:, mask_fit].T,
                      np.ones(mask_fit.sum()),
                      bounds=(-5.0, 5.0))
W_opt         = res.x
out_pulse_opt = G.T @ W_opt


# ── Step response = cumulative integral of pulse response ────────
out_step_eq  = np.cumsum(out_pulse_eq)  * dt
out_step_opt = np.cumsum(out_pulse_opt) * dt


# ══════════════════════════════════════════════════════════════════
# Figure layout: 2×2 grid, each cell subdivided
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 11))
fig.suptitle(
    "Goldman (2009) — Figure 1B–E: Feedforward Network Integration\n"
    r"N = 100 stages, $\tau$ = 100 ms",
    fontsize=11, y=0.99
)

gs_main = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.42, wspace=0.30,
                             left=0.08, right=0.97,
                             top=0.90, bottom=0.06)

gs_b = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[0, 0], hspace=0.06)
ax_b_in, ax_b_fn, ax_b_out = [fig.add_subplot(gs_b[i]) for i in range(3)]

gs_c = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[0, 1], hspace=0.06)
ax_c = [fig.add_subplot(gs_c[i]) for i in range(4)]

gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1, 0], hspace=0.06)
ax_d_in, ax_d_out = [fig.add_subplot(gs_d[i]) for i in range(2)]

gs_e = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1, 1], hspace=0.06)
ax_e_in, ax_e_out = [fig.add_subplot(gs_e[i]) for i in range(2)]


# ── Fig 1B: Pulse → Step ─────────────────────────────────────────
ax_b_in.set_title("Fig 1B  Integration of pulse to step",
                   loc='left', fontsize=9, fontweight='bold', pad=3)

# Input: impulse
ax_b_in.annotate("", xy=(0, 1.0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color='k', lw=2.5))
ax_b_in.set_xlim([0, 15]); ax_b_in.set_ylim([0, 1.5])
ax_b_in.set_yticks([]); ax_b_in.set_xticks([])
ax_b_in.set_ylabel("Input", fontsize=8)
ax_b_in.text(0.4, 1.0, r"$\delta(t)$", fontsize=10, va='bottom')

# Basis functions (select evenly-spaced n)
n_vals = list(range(1, 85, 7))   # n = 1, 8, 15, 22, ...
cmap_b = plt.cm.rainbow
for i, n in enumerate(n_vals):
    c = cmap_b(i / (len(n_vals) - 1))
    ax_b_fn.plot(t, G[n], color=c, lw=1.1, alpha=0.9)
ax_b_fn.set_ylabel("Basis fns $g_n$", fontsize=8)
ax_b_fn.set_xlim([0, 15]); ax_b_fn.set_ylim([0, 0.45])
ax_b_fn.set_xticks([])
ax_b_fn.text(0.97, 0.93, r"$n = 1, 8, 15, 22, \ldots$",
             transform=ax_b_fn.transAxes, ha='right', va='top',
             fontsize=7.5, color='dimgray',
             bbox=dict(fc='white', ec='none', alpha=0.7))

# Output: weighted sum
ax_b_out.plot(t, out_pulse_eq,  'k-',  lw=2.0, label="W's = 1")
ax_b_out.plot(t, out_pulse_opt, 'r-',  lw=1.5, label="W's optimally fit")
ax_b_out.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax_b_out.set_xlim([0, 15]); ax_b_out.set_ylim([-0.05, 1.4])
ax_b_out.set_xlabel("Time (sec)", fontsize=8)
ax_b_out.set_ylabel("Output $r_{out}$", fontsize=8)
ax_b_out.legend(loc='upper right', fontsize=7.5, framealpha=0.9)


# ── Fig 1C: First 3 basis functions + sum ────────────────────────
t_c = np.linspace(0, 8, 3000)   # x in units of τ

ax_c[0].set_title("Fig 1C  First three basis functions $g_n$ and their sum",
                   loc='left', fontsize=9, fontweight='bold', pad=3)

colors_c = ['steelblue', 'darkorange', 'forestgreen']
for i, n in enumerate([0, 1, 2]):
    vals = gn_fn(n, t_c)
    ax_c[i].plot(t_c, vals, color=colors_c[i], lw=2)
    ax_c[i].set_ylabel(f"$g_{n}$", fontsize=9)
    ax_c[i].set_xlim([0, 8]); ax_c[i].set_xticks([])
    ax_c[i].set_ylim([0, max(vals.max() * 1.18, 0.05)])

# Sum panel: equal and optimal weights on only 3 stages
g012_eq  = sum(gn_fn(n, t_c) for n in range(3))
g012_opt = gn_fn(0, t_c) + 0.7 * gn_fn(1, t_c) + 2.0 * gn_fn(2, t_c)
ax_c[3].plot(t_c, g012_eq,  'k-', lw=2, label=r"$g_0 + g_1 + g_2$")
ax_c[3].plot(t_c, g012_opt, 'r-', lw=2, label=r"$g_0 + 0.7\,g_1 + 2\,g_2$")
ax_c[3].set_ylabel("sum", fontsize=8)
ax_c[3].set_xlabel(r"Time (units of $\tau$)", fontsize=8)
ax_c[3].set_xlim([0, 8])
ax_c[3].legend(fontsize=7, loc='upper right', framealpha=0.9)


# ── Fig 1D: Linear amplitude scaling ─────────────────────────────
ax_d_in.set_title("Fig 1D  Response scales linearly with pulse amplitude",
                   loc='left', fontsize=9, fontweight='bold', pad=3)

amps     = [2.0, 1.0, 0.5]
colors_d = ['tab:green', 'tab:blue', 'tab:red']
labels_d = ['×2', '×1', '×0.5']

for amp, c, lbl in zip(amps, colors_d, labels_d):
    ax_d_in.annotate("", xy=(0, amp), xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color=c, lw=2))
    ax_d_in.text(0.25, amp + 0.05, lbl, color=c, fontsize=9, va='bottom')
ax_d_in.set_xlim([-0.3, 15]); ax_d_in.set_ylim([0, 2.6])
ax_d_in.set_yticks([]); ax_d_in.set_xticks([])
ax_d_in.set_ylabel("Input", fontsize=8)

for amp, c, lbl in zip(amps, colors_d, labels_d):
    ax_d_out.plot(t, amp * out_pulse_eq, color=c, lw=2, label=lbl)
ax_d_out.set_xlim([0, 15]); ax_d_out.set_ylim([-0.05, 2.6])
ax_d_out.set_xlabel("Time (sec)", fontsize=8)
ax_d_out.set_ylabel("Output $r_{out}$", fontsize=8)
ax_d_out.legend(fontsize=7.5, loc='upper right')


# ── Fig 1E: Step → Ramp ──────────────────────────────────────────
ax_e_in.set_title("Fig 1E  Integration of step to ramp",
                   loc='left', fontsize=9, fontweight='bold', pad=3)

# Step input
ax_e_in.plot([0, 0, 15], [0, 1, 1], 'k-', lw=2)
ax_e_in.set_xlim([0, 15]); ax_e_in.set_ylim([0, 1.5])
ax_e_in.set_yticks([]); ax_e_in.set_xticks([])
ax_e_in.set_ylabel("Input", fontsize=8)
ax_e_in.text(7, 1.1, "Step", fontsize=9, ha='center')

# Ramp output
ax_e_out.plot(t, out_step_eq,  'k-', lw=2.0, label="W's = 1")
ax_e_out.plot(t, out_step_opt, 'r-', lw=1.5, label="W's optimally fit")
ax_e_out.set_xlim([0, 15])
ax_e_out.set_xlabel("Time (sec)", fontsize=8)
ax_e_out.set_ylabel("Output $r_{out}$", fontsize=8)
ax_e_out.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

# Save
out_path = r"c:\Users\ET USER\Documents\Caltech\Caltech CNS 187\Final project\goldman_fig1.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out_path}")
