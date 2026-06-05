"""
bio_plausible_memory_emergence.py

Tests whether attractor-like or functionally feedforward recurrent structure
emerges when a random rate network learns a finite-delay memory task.

Optimization: gradient descent (BPTT + Adam) on MSE + L2 regularization.
Architecture: biologically realistic rate network with tanh nonlinearity,
  no self-connections, fixed random input/readout vectors.

After training the network is analyzed via eigenvalue spectra, Schur
decomposition, and transient-amplification profiles to determine whether it
resembles a line-attractor or a functionally feedforward (non-normal) system.

Dependencies: numpy, scipy, matplotlib only.
Run:  python bio_plausible_memory_emergence.py
"""

import os
import sys
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use('Agg')          # headless backend -- no display required
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs('results', exist_ok=True)

# Force UTF-8 output on Windows to avoid charmap errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------------------------
N          = 30       # recurrent rate units
tau        = 0.1      # membrane time constant (s)
dt         = 0.005    # Euler step (s)
T_trial    = 2.0      # trial duration (s)
t_on       = 0.05     # input pulse onset (s)
t_off      = 0.10     # input pulse offset (s)
n_steps    = int(T_trial / dt)     # 400 integration steps
t_arr      = np.arange(n_steps) * dt

N_TRIALS      = 2000    # number of Adam update steps (each uses batch_size samples)
batch_size    = 32      # mini-batch size: averages out stochastic gradient noise
lr            = 2e-3    # Adam learning rate (higher works since batch grad is lower variance)
beta1         = 0.9     # Adam first-moment decay
beta2         = 0.999   # Adam second-moment decay
eps_adam      = 1e-8    # Adam numerical stability
lambda_W      = 1e-4    # L2 weight regularization (light; don't fight the attractor)
W_clip        = 3.0     # hard weight clipping bound
SR_MAX        = 0.999   # spectral radius ceiling -- train and eval in the same regime
                        # (SR checked every batch so no drift above this value)
max_grad_norm = 5.0     # gradient clipping threshold

# -------------------------------------------------------------------------------
# NETWORK INITIALIZATION
# -------------------------------------------------------------------------------
# Random symmetric W scaled to SR = 0.95.
# Symmetric -> all real eigenvalues -> prevents the "phase-tuned oscillation"
# trick where complex eigenvalues at the SR ceiling memorize A via frequency tuning
# rather than genuine persistent activity.
# SR = 0.95 -> Euler-step Jacobian rho = 0.95 + 0.05*0.95 = 0.9975
# -> gradient survives 380 steps as 0.9975^380 ~ 0.39 (workable).
# No u-direction bias: the optimizer must discover the line attractor structure
# on its own from the MSE loss alone.
W = np.random.randn(N, N) / np.sqrt(N)
W = (W + W.T) / 2.0                # symmetric -> all real eigenvalues
np.fill_diagonal(W, 0.0)

# Fixed random input vector
u = np.random.randn(N);  u /= np.linalg.norm(u)

# Readout aligned with the input direction: v = u
# Biological rationale: probing the same axis that was driven by the stimulus
# is the most natural memory readout and makes the task tractable for a
# biologically plausible learning rule.  The emergent structure that solves this
# is a line attractor along u.
v = u.copy()

_sr = np.max(np.abs(np.linalg.eigvals(W)))
W  *= 0.95 / _sr                    # scale to SR = 0.95

# Adam optimizer state for W
adam_m = np.zeros((N, N))   # first moment
adam_v = np.zeros((N, N))   # second moment

# History buffers
hist_mse    = np.zeros(N_TRIALS)
hist_energy = np.zeros(N_TRIALS)   # repurposed: total loss (MSE + regularization)
hist_reward = np.zeros(N_TRIALS)   # repurposed: gradient norm (training diagnostic)


# -------------------------------------------------------------------------------
# TRIAL SIMULATION
# -------------------------------------------------------------------------------
def run_trial(W_mat, A, store=False):
    """
    Euler-integrate the rate network for one trial.

    Dynamics:  tau * dr/dt = -r + W @ r + u * x(t)
    After each step r is passed through tanh for soft bounding.

    Returns (r_final, None, energy, r_hist, r_mean) — compatible with analyze_W.
    """
    r      = np.zeros(N)
    energy = 0.0
    r_sum  = np.zeros(N)
    r_hist = np.zeros((n_steps, N)) if store else None

    for k in range(n_steps):
        t = k * dt
        x = A if t_on <= t < t_off else 0.0
        dr = (-r + W_mat @ r + u * x) / tau
        r  = np.tanh(r + dt * dr)
        energy += np.mean(r ** 2) * dt
        r_sum  += r
        if store:
            r_hist[k] = r

    return r, None, energy, r_hist, r_sum / n_steps


def forward_pass(W_mat, A):
    """Forward pass storing all states for BPTT.

    Returns r_hist (n_steps+1, N): r_hist[0]=zeros, r_hist[k+1]=r after step k.
    """
    r_hist = np.zeros((n_steps + 1, N))
    r = np.zeros(N)
    for k in range(n_steps):
        t = k * dt
        x = A if t_on <= t < t_off else 0.0
        dr = (-r + W_mat @ r + u * x) / tau
        r  = np.tanh(r + dt * dr)
        r_hist[k + 1] = r
    return r_hist


def bptt_gradient(W_mat, A, r_hist):
    """Backpropagation through time for the MSE + L2 loss.

    Loss = (A - v@r_final)^2 + lambda_W * mean(W^2)

    Forward step:  h_{k+1} = r_k*(1-dt/tau) + (dt/tau)*(W@r_k + u*x_k)
                   r_{k+1} = tanh(h_{k+1})

    dL/dW_{ij} = sum_k (dt/tau) * delta_{k+1,i} * r_{k,j}
    where delta_{k+1} = sech^2(h_{k+1}) * dL/dr_{k+1}
    and   dL/dr_k     = (1-dt/tau)*delta_{k+1} + (dt/tau)*W^T @ delta_{k+1}
    """
    r_final = r_hist[n_steps]
    y = float(v @ r_final)
    mse  = (A - y) ** 2
    loss = mse + lambda_W * float(np.mean(W_mat ** 2))

    dL_dr = -2.0 * (A - y) * v          # dL/dr_final from MSE
    dL_dW = np.zeros((N, N))

    for k in range(n_steps - 1, -1, -1):
        r_prev = r_hist[k]
        r_curr = r_hist[k + 1]
        sech2  = 1.0 - r_curr ** 2      # element-wise sech^2
        delta  = sech2 * dL_dr
        dL_dW += (dt / tau) * np.outer(delta, r_prev)
        dL_dr  = (1.0 - dt / tau) * delta + (dt / tau) * (W_mat.T @ delta)

    # L2 regularization gradient: d/dW lambda_W * mean(W^2) = 2*lambda_W/N^2 * W
    dL_dW += (2.0 * lambda_W / (N * N)) * W_mat
    np.fill_diagonal(dL_dW, 0.0)

    return dL_dW, loss, mse, y


# -------------------------------------------------------------------------------
# TRAINING LOOP  (mini-batch BPTT + Adam)
# -------------------------------------------------------------------------------
# Mini-batch design: each "trial" is one Adam update step using the average
# gradient over batch_size independent rollouts.  Benefits:
#   - Gradient variance reduced by sqrt(batch_size); allows higher lr
#   - SR is checked after every update step, so training and eval always run
#     with the same W (no more double-well / oscillatory exploitation between
#     infrequent SR checks)
print('Training for {} batches x {} trials (BPTT + Adam)...'.format(
    N_TRIALS, batch_size))
for trial in range(N_TRIALS):
    batch_As = np.random.uniform(-1.0, 1.0, batch_size)

    batch_grad = np.zeros((N, N))
    batch_mse  = 0.0
    batch_loss = 0.0
    for A in batch_As:
        r_hist = forward_pass(W, A)
        g, loss, mse, _ = bptt_gradient(W, A, r_hist)
        batch_grad += g
        batch_mse  += mse
        batch_loss += loss
    batch_grad /= batch_size
    batch_mse  /= batch_size
    batch_loss /= batch_size

    # Gradient clipping
    grad_norm = float(np.linalg.norm(batch_grad))
    if grad_norm > max_grad_norm:
        batch_grad *= max_grad_norm / grad_norm

    # Adam update
    t_adam = trial + 1
    adam_m[:] = beta1 * adam_m + (1.0 - beta1) * batch_grad
    adam_v[:] = beta2 * adam_v + (1.0 - beta2) * batch_grad ** 2
    m_hat = adam_m / (1.0 - beta1 ** t_adam)
    v_hat = adam_v / (1.0 - beta2 ** t_adam)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    # Symmetrize: keeps all eigenvalues real, prevents phase-tuned oscillatory
    # solutions from masquerading as memory via complex eigenvalues at SR ceiling.
    W = (W + W.T) / 2.0
    np.clip(W, -W_clip, W_clip, out=W)
    np.fill_diagonal(W, 0.0)

    # SR check every batch -- training and eval use the same W regime
    sr_now = np.max(np.abs(np.linalg.eigvals(W)))
    if sr_now > SR_MAX:
        W *= SR_MAX / sr_now

    hist_mse[trial]    = batch_mse
    hist_energy[trial] = batch_loss
    hist_reward[trial] = grad_norm

    if (trial + 1) % 200 == 0:
        sl = slice(max(0, trial - 99), trial + 1)
        print('  [{:4d}]  MSE={:.4f}  loss={:.4f}  grad_norm={:.4f}  u@W@u={:.3f}  SR={:.4f}'.format(
            trial + 1,
            np.mean(hist_mse[sl]),
            np.mean(hist_energy[sl]),
            np.mean(hist_reward[sl]),
            float(u @ W @ u),
            sr_now))

print('Training complete.\n')

# Quick held-out eval: deterministic A sweep, fixed W
A_eval   = np.linspace(-1.0, 1.0, 100)
y_eval   = np.array([v @ run_trial(W, A)[0] for A in A_eval])
corr_eval = float(np.corrcoef(y_eval, A_eval)[0, 1])
mse_eval  = float(np.mean((A_eval - y_eval) ** 2))
slope     = float(np.polyfit(A_eval, y_eval, 1)[0])
print('Post-training held-out eval (100 det. trials):')
print('  MSE={:.4f}  corr(y,A)={:.4f}  slope y/A={:.4f}  u@W@u={:.4f}'.format(
    mse_eval, corr_eval, slope, float(u @ W @ u)))
print('  (chance MSE=0.333; slope=1 means perfect memory amplitude)\n')


# -------------------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------------------
def smooth(x, w=50):
    """Moving-average smoothing with window w."""
    return np.convolve(x, np.ones(w) / w, mode='valid')


# -------------------------------------------------------------------------------
# PLOT 1 -- LEARNING CURVES
# -------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Learning Curves', fontsize=14, fontweight='bold')

for ax, data, ylabel, color, title in zip(
        axes,
        [hist_mse, hist_energy, hist_reward],
        ['MSE', 'Total loss', 'Grad norm'],
        ['#c0392b', '#2980b9', '#27ae60'],
        ['Memory MSE  (smoothed)', 'Total loss = MSE + L2  (smoothed)',
         'Gradient norm  (smoothed)']):
    ax.plot(smooth(data), color=color, lw=1.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)

axes[-1].set_xlabel('Trial')
plt.tight_layout()
plt.savefig('results/01_learning_curves.png', dpi=150)
plt.close()
print('Saved  results/01_learning_curves.png')


# -------------------------------------------------------------------------------
# PLOT 2 -- EXAMPLE NEURAL ACTIVITY AFTER TRAINING
# -------------------------------------------------------------------------------
N_SHOW = 8
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
fig.suptitle('Learned Network Activity', fontsize=13, fontweight='bold')

for ai, (A_demo, color, label) in enumerate([
        ( 0.8, '#2980b9', 'Positive pulse  A = +0.8'),
        (-0.8, '#c0392b', 'Negative pulse  A = -0.8')]):
    _, _, _, r_hist, _ = run_trial(W, A_demo, store=True)
    ax = axes[ai]
    for ni in range(N_SHOW):
        ax.plot(t_arr, r_hist[:, ni], lw=0.9, alpha=0.75)
    ax.axvspan(t_on, t_off, color='gold', alpha=0.2, label='Input pulse')
    ax.set_ylabel('r(t)')
    ax.set_title(label)
    ax.legend(fontsize=8, loc='upper right')

ax = axes[2]
for A_demo, color, label in [( 0.8, '#2980b9', 'A=+0.8'),
                               (-0.8, '#c0392b', 'A=-0.8')]:
    _, _, _, r_hist, _ = run_trial(W, A_demo, store=True)
    ax.plot(t_arr, r_hist @ v, lw=1.6, color=color, label=label)
ax.axhline( 0.8, ls='--', color='#2980b9', alpha=0.4, lw=1)
ax.axhline(-0.8, ls='--', color='#c0392b', alpha=0.4, lw=1)
ax.axvspan(t_on, t_off, color='gold', alpha=0.2, label='Input pulse')
ax.set_xlabel('Time (s)')
ax.set_ylabel('y(t) = v . r(t)')
ax.set_title('Fixed readout over time  (dashed lines = target values)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('results/02_neural_activity.png', dpi=150)
plt.close()
print('Saved  results/02_neural_activity.png')


# -------------------------------------------------------------------------------
# PLOT 2b -- HELD-OUT READOUT y vs A (memory amplitude diagnostic)
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(A_eval, y_eval, s=20, color='steelblue', alpha=0.7)
fit = np.polyfit(A_eval, y_eval, 1)
ax.plot(A_eval, np.polyval(fit, A_eval), 'r--', lw=1.5,
        label='slope={:.3f}'.format(fit[0]))
ax.plot([-1, 1], [-1, 1], 'k:', lw=0.8, alpha=0.4, label='ideal (slope=1)')
ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
ax.set_xlabel('Stimulus amplitude A')
ax.set_ylabel('Readout y = v @ r_final')
ax.set_title('Memory amplitude: corr={:.3f}, slope={:.3f}\n'
             'MSE={:.4f}  u@W@u={:.3f}'.format(
                 corr_eval, fit[0], mse_eval, float(u @ W @ u)))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/02b_readout_scatter.png', dpi=150)
plt.close()
print('Saved  results/02b_readout_scatter.png')


# -------------------------------------------------------------------------------
# PLOT 3 -- EIGENVALUE SPECTRUM
# -------------------------------------------------------------------------------
eigvals         = np.linalg.eigvals(W)
spectral_radius = float(np.max(np.abs(eigvals)))

# Separate truly real eigenvalues from complex ones.
# np.max(eigvals.real) is WRONG for an asymmetric W: it returns the max real
# *part* of all eigenvalues, including complex pairs like 0.95+0.28i whose
# real part 0.95 is not an actual memory mode -- the complex pair oscillates
# and decays at rate |lambda|, not at rate Re(lambda).
real_mask    = np.abs(eigvals.imag) < 1e-8
max_real_eig = float(np.max(eigvals[real_mask].real)) if real_mask.any() else float('nan')
max_complex_mag = float(np.max(np.abs(eigvals[~real_mask]))) if (~real_mask).any() else 0.0
# u@W@u: self-feedback of the input/readout direction (should approach ~1 for line attractor)
u_feedback   = float(u @ W @ u)

theta = np.linspace(0, 2 * np.pi, 300)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.7, alpha=0.4,
        label='Unit circle')
ax.scatter(eigvals.real, eigvals.imag, s=55, color='steelblue',
           zorder=3, label='Eigenvalues of W')
ax.axvline(0, color='gray', lw=0.5)
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Re(lambda)')
ax.set_ylabel('Im(lambda)')
ax.set_title('Eigenvalue Spectrum of Learned W\n'
             'Max real eig={:.3f}   Max |complex| eig={:.3f}   rho={:.3f}'.format(
                 max_real_eig, max_complex_mag, spectral_radius))
ax.legend(fontsize=9)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('results/03_eigenvalues.png', dpi=150)
plt.close()
print('Saved  results/03_eigenvalues.png')
print('  Max REAL eigenvalue  : {:.4f}  (memory mode; needs ~0.97 for 2-s delay)'.format(max_real_eig))
print('  Max |complex| eig    : {:.4f}  (oscillatory; decays as |lam|^steps)'.format(max_complex_mag))
print('  Spectral radius      : {:.4f}'.format(spectral_radius))
print('  u @ W @ u            : {:.4f}  (u-direction self-feedback; ideal ~0.97)'.format(u_feedback))


# -------------------------------------------------------------------------------
# SCHUR DECOMPOSITION / NON-NORMALITY ANALYSIS
# -------------------------------------------------------------------------------
# scipy.linalg.schur with output='real' returns W = Z T Z^T where T is upper
# quasi-triangular (real Schur form).  For a normal matrix T would be
# block-diagonal (only eigenvalues on the diagonal).  The strictly upper-
# triangular part of T quantifies non-normality:  large off-diagonal mass means
# distinct modes can constructively interfere -> transient amplification even
# when all eigenvalues are sub-unit (characteristic of feedforward networks).
T_schur, Z_schur = scipy.linalg.schur(W, output='real')
norm_T        = np.linalg.norm(T_schur, 'fro')
upper_offdiag = np.triu(T_schur, k=1)   # strictly upper triangular
offdiag_ratio = float(np.linalg.norm(upper_offdiag, 'fro') / (norm_T + 1e-12))
print('  Schur off-diag ratio : {:.4f}'.format(offdiag_ratio))
print('  (High value -> non-normal / functionally feedforward dynamics)')

fig, ax = plt.subplots(figsize=(6, 5))
vmax = max(np.max(np.abs(T_schur)), 1e-6)
im   = ax.imshow(T_schur, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax)
ax.set_title(
    'Real Schur Form  T   (off-diag ratio = {:.3f})\n'
    'Upper-triangular mass = non-normality = feedforward amplification'.format(
        offdiag_ratio))
ax.set_xlabel('Column')
ax.set_ylabel('Row')
plt.tight_layout()
plt.savefig('results/04_schur_matrix.png', dpi=150)
plt.close()
print('Saved  results/04_schur_matrix.png')


# -------------------------------------------------------------------------------
# PLOT 5 -- TRANSIENT AMPLIFICATION  (no input, random initial states)
# -------------------------------------------------------------------------------
# A normal network (e.g. stable attractor) should show monotone decay of ||r||.
# A non-normal / feedforward network can transiently amplify activity before it
# decays -- information is "passed forward" through the network even with no
# sustained drive.
fig, ax = plt.subplots(figsize=(9, 5))
for _ in range(14):
    r      = 0.1 * np.random.randn(N)
    norms  = []
    for _ in range(n_steps):
        dr = (-r + W @ r) / tau
        r  = np.tanh(r + dt * dr)
        norms.append(np.linalg.norm(r))
    ax.plot(t_arr, norms, lw=1.0, alpha=0.55)

ax.set_xlabel('Time (s)')
ax.set_ylabel('||r(t)||')
ax.set_title('Transient Amplification -- No Input\n'
             'Growth then decay -> non-normal / feedforward;   '
             'sustained plateau -> attractor')
plt.tight_layout()
plt.savefig('results/05_transient_amplification.png', dpi=150)
plt.close()
print('Saved  results/05_transient_amplification.png')


# -------------------------------------------------------------------------------
# BASELINE COMPARISON
# -------------------------------------------------------------------------------
def analyze_W(W_test, label, n_eval=300):
    """Compute spectral stats and memory performance for a weight matrix."""
    eigs     = np.linalg.eigvals(W_test)
    Ts, _    = scipy.linalg.schur(W_test, output='real')
    nT       = np.linalg.norm(Ts, 'fro')
    odr      = float(np.linalg.norm(np.triu(Ts, k=1), 'fro') / (nT + 1e-12))
    mse_vals = []
    en_vals  = []
    for _ in range(n_eval):
        A  = np.random.uniform(-1.0, 1.0)
        rf, _, en, _, _ = run_trial(W_test, A)
        mse_vals.append((A - v @ rf) ** 2)
        en_vals.append(en)
    return dict(
        label  = label,
        max_re = float(np.max(eigs.real)),
        sr     = float(np.max(np.abs(eigs))),
        odr    = odr,
        mse    = float(np.mean(mse_vals)),
        energy = float(np.mean(en_vals)),
    )

# Baseline 1: random untrained weights
W_rand = 0.1 * np.random.randn(N, N) / np.sqrt(N)
np.fill_diagonal(W_rand, 0.0)

# Baseline 2: hand-built line attractor
# One mode q has eigenvalue (1-eps) ~= 1; all other modes have eigenvalue -0.3.
# Construction: W = -0.3*I + [(1-eps) - (-0.3)] * outer(q, q)
#             = -0.3*I + (1-eps+0.3) * outer(q, q)
# We keep the diagonal as designed (do not zero it) so the spectral property is
# exactly as intended.
eps_att = 0.03
# Align the attractor mode with u so the input can excite it and the
# readout (v=u) can read it out.  A random q would make the attractor
# orthogonal to u/v and give chance-level memory, defeating the comparison.
q       = u.copy()
W_att   = -0.3 * np.eye(N) + (1.0 - eps_att + 0.3) * np.outer(q, q)

# Baseline 3: hand-built feedforward chain
# W[i, i-1] = c (strictly sub-diagonal), zero elsewhere.
# All eigenvalues are 0 (maximally non-normal); activity cascades down the chain.
c_ff = 0.85
W_ff = np.zeros((N, N))
for i in range(1, N):
    W_ff[i, i - 1] = c_ff

baselines = [
    analyze_W(W_rand, 'Random untrained'),
    analyze_W(W_att,  'Line attractor'),
    analyze_W(W_ff,   'Feedforward chain'),
    analyze_W(W,      'Learned network'),
]

print('\n' + '=' * 72)
print('{:<22} {:>8} {:>8} {:>11} {:>10} {:>10}'.format(
    'Network', 'MaxRe', 'rho', 'Schur ODR', 'Mem. MSE', 'Energy'))
print('-' * 72)
for b in baselines:
    print('{:<22} {:>8.3f} {:>8.3f} {:>11.3f} {:>10.4f} {:>10.4f}'.format(
        b['label'], b['max_re'], b['sr'], b['odr'], b['mse'], b['energy']))
print('=' * 72)

# Baseline comparison bar chart
metrics = [('mse',    'Memory MSE'),
           ('odr',    'Schur Off-diag Ratio'),
           ('sr',     'Spectral Radius')]
colors  = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71']
names   = [b['label'] for b in baselines]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('Network Comparison: Learned vs. Baselines', fontsize=13,
             fontweight='bold')
for ax, (key, title) in zip(axes, metrics):
    vals = [b[key] for b in baselines]
    bars = ax.bar(range(len(names)), vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01 + 1e-4,
                '{:.3f}'.format(val), ha='center', va='bottom', fontsize=8)
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig('results/06_baseline_comparison.png', dpi=150)
plt.close()
print('Saved  results/06_baseline_comparison.png')


# -------------------------------------------------------------------------------
# SCHUR MODE ANALYSIS: SELF-FEEDBACK vs FEEDFORWARD COUPLING
# -------------------------------------------------------------------------------
# Complex Schur decomposition: W = Z T Z^H
#   T is strictly upper triangular; diagonal = eigenvalues (self-feedback strength)
#   T[i,j] for j > i = coupling from Schur mode i into mode j (feedforward)
#
# Per-mode decomposition:
#   self_i    = |T[i,i]|              eigenvalue magnitude — how strongly mode i
#                                     reinforces itself each time step
#   ff_out_i  = ||T[i, i+1:]||        total feedforward output: how strongly
#                                     mode i drives all downstream Schur modes
#   ff_ratio_i = ff_out_i /           1 = pure feedforward node (passes activity
#               (self_i + ff_out_i)       on, doesn't store it)
#                                     0 = pure self-sustaining attractor mode
#
# Superdiagonal cascade:
#   ||diag(T, k)|| for k=1,2,...      how quickly feedforward coupling decays
#                                     with "Schur distance" k between modes.
#   Slow decay -> long-range cascade (feedforward-chain-like)
#   Fast decay -> local coupling only (attractor-like background noise)

def schur_feedback_profile(W_mat):
    """Complex Schur: T[i,i]=eigenvalue (self-feedback), T[i,j>i]=feedforward."""
    T_c, Z_c = scipy.linalg.schur(W_mat, output='complex')
    n = W_mat.shape[0]

    diag        = np.diag(T_c)
    mode_self   = np.abs(diag)
    mode_ff_out = np.array([np.linalg.norm(T_c[i, i+1:]) for i in range(n)])
    mode_ff_in  = np.array([np.linalg.norm(T_c[:i,   i]) for i in range(n)])
    ff_ratio    = mode_ff_out / (mode_self + mode_ff_out + 1e-12)

    # Sort modes by descending |eigenvalue| for consistent visualization
    order = np.argsort(-mode_self)

    # Superdiagonal cascade: norm of k-th superdiagonal
    cascade = np.array([np.linalg.norm(np.diag(T_c, k)) for k in range(1, n)])

    total_norm = np.linalg.norm(T_c, 'fro')
    diag_norm  = np.linalg.norm(diag)
    ff_norm    = np.linalg.norm(np.triu(T_c, k=1), 'fro')

    return dict(
        T_c       = T_c,
        order     = order,
        mode_self   = mode_self[order],
        mode_ff_out = mode_ff_out[order],
        mode_ff_in  = mode_ff_in[order],
        ff_ratio    = ff_ratio[order],
        cascade   = cascade,
        self_frac = float(diag_norm / (total_norm + 1e-12)),
        ff_frac   = float(ff_norm   / (total_norm + 1e-12)),
        dom_eig   = float(np.max(mode_self)),
    )


nets_schur = [
    (W,      'Learned',         '#2ecc71'),
    (W_att,  'Line attractor',  '#e74c3c'),
    (W_ff,   'Feedforward',     '#3498db'),
    (W_rand, 'Random',          '#95a5a6'),
]
profiles = {label: schur_feedback_profile(Wn) for Wn, label, _ in nets_schur}

print('\n--- Schur self-feedback vs feedforward ---')
print('{:<20} {:>12} {:>10} {:>14} {:>14}'.format(
    'Network', 'Self-frac', 'FF-frac', 'Dom. |eig|', 'FF ratio (dom)'))
for _, label, _ in nets_schur:
    p = profiles[label]
    # feedforward ratio of the dominant (highest |eig|) mode
    dom_ff_ratio = p['ff_ratio'][0]
    print('{:<20} {:>12.3f} {:>10.3f} {:>14.3f} {:>14.3f}'.format(
        label, p['self_frac'], p['ff_frac'], p['dom_eig'], dom_ff_ratio))

# ── Plot 07: |T_c| heatmaps + per-mode self vs feedforward bars ──────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Schur Mode Analysis — Self-feedback vs Feedforward Coupling',
             fontsize=13, fontweight='bold')

for col, (Wn, label, color) in enumerate(nets_schur):
    p = profiles[label]

    # Reorder rows/cols by sorted mode order so dominant modes are top-left
    T_abs = np.abs(p['T_c'])[np.ix_(p['order'], p['order'])]

    ax = axes[0, col]
    vmax = float(np.percentile(T_abs[T_abs > 0], 98)) if T_abs.max() > 0 else 1.0
    im = ax.imshow(T_abs, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    ax.set_title('{}\nself={:.2f}  ff={:.2f}'.format(
                     label, p['self_frac'], p['ff_frac']), fontsize=9)
    ax.set_xlabel('Mode j'); ax.set_ylabel('Mode i')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stacked bar: self-feedback | feedforward-out, sorted by |eigenvalue|
    ax = axes[1, col]
    x = np.arange(N)
    ax.bar(x, p['mode_self'],   color='steelblue',  label='Self  |lambda_i|')
    ax.bar(x, p['mode_ff_out'], color='darkorange', label='FF out ||T[i,j>i]||',
           bottom=p['mode_self'], alpha=0.85)
    ax.set_xlabel('Mode  (sorted by |lambda|, dominant first)')
    ax.set_ylabel('Strength')
    if col == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig('results/07_schur_mode_profiles.png', dpi=150)
plt.close()
print('Saved  results/07_schur_mode_profiles.png')

# ── Plot 08: feedforward cascade depth ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for Wn, label, color in nets_schur:
    p = profiles[label]
    c = p['cascade']
    norm = c[0] if c[0] > 1e-12 else 1.0
    ax.plot(range(1, N), c / norm, color=color, lw=1.8, label=label)
ax.set_xlabel('Schur superdiagonal index  k  (feedforward distance)')
ax.set_ylabel('Cascade power  (normalized to k = 1)')
ax.set_title('Feedforward Cascade Depth\n'
             'Slow decay -> long-range feedforward chain;  '
             'Fast decay -> local coupling / attractor')
ax.legend()
plt.tight_layout()
plt.savefig('results/08_feedforward_cascade.png', dpi=150)
plt.close()
print('Saved  results/08_feedforward_cascade.png')

# ── Plot 09: per-mode feedforward ratio ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Per-mode Feedforward Ratio   ff_out / (|lambda| + ff_out)',
             fontsize=12, fontweight='bold')
for ax, (Wn, label, color) in zip(axes, nets_schur):
    p = profiles[label]
    ax.bar(np.arange(N), p['ff_ratio'], color=color, alpha=0.85)
    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Mode (dominant first)')
ax.set_ylabel('FF ratio  (1=pure FF,  0=pure attractor)')
axes[0].set_ylabel('FF ratio  (1=pure FF,  0=pure attractor)')
plt.tight_layout()
plt.savefig('results/09_mode_ff_ratio.png', dpi=150)
plt.close()
print('Saved  results/09_mode_ff_ratio.png')


# -------------------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------------------
final_mse  = float(np.mean(hist_mse[-500:]))
final_loss = float(np.mean(hist_energy[-500:]))

# Interpretation heuristic:
#   attractor-like     : a real eigenvalue is near 1 (persistent attractor mode)
#   feedforward-like   : all eigenvalues clearly sub-unit + high Schur off-diag
#                        (transient amplification without sustained attractor)
#   unclear / mixed    : neither criterion met strongly
# Use strictly real max eigenvalue for interpretation
if max_real_eig > 0.80:
    interpretation = 'attractor-like  (real eigenvalue near 1 -- persistent memory mode)'
elif offdiag_ratio > 0.35 and spectral_radius < 0.85:
    interpretation = ('functionally-feedforward-like  '
                      '(high non-normality, sub-unit spectral radius)')
elif offdiag_ratio > 0.25:
    interpretation = ('mixed / mildly non-normal  '
                      '(moderate feedforward structure)')
else:
    interpretation = ('unclear  '
                      '(neither strong attractor nor clear feedforward structure)')

print('\n' + '=' * 60)
print('Learned network summary:')
print('  Memory MSE (online train) : {:.4f}  (chance = 0.333)'.format(final_mse))
print('  Memory MSE (held-out eval): {:.4f}  (slope y/A = {:.3f})'.format(
    mse_eval, slope))
print('  Total loss (MSE + L2):     {:.4f}'.format(final_loss))
print('  u @ W @ u:                 {:.4f}  (line attractor ideal: ~0.97)'.format(u_feedback))
print('  Max REAL eigenvalue:       {:.4f}  (complex eigs max |lam|={:.4f})'.format(
    max_real_eig, max_complex_mag))
print('  Spectral radius:           {:.4f}'.format(spectral_radius))
print('  Schur offdiag ratio:       {:.4f}'.format(offdiag_ratio))
print('  Interpretation:            {}'.format(interpretation))
print('=' * 60)
