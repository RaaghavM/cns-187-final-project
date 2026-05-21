"""
Goldman Fig. 7A-D style line-attractor simulation using Seung et al.'s
reduced averaged autapse feedback equation, with mixed-sign neuronal responses.

Why mixed signs?
----------------
In Goldman Fig. 7A-D, the line-attractor network is low-dimensional at long
times, but individual neurons need not all have positive loadings on the slow
mode. After a single pulse, all neurons share the same slow temporal waveform,
but neurons whose loading on the slow mode is positive increase while neurons
whose loading is negative decrease. The summed/readout output can still be a
positive step.

This script implements that using Seung et al.'s reduced averaged autapse
equation for every neuron:

    tau ds_i/dt = -s_i + alpha f(W_i s_i + B_i) (1 - s_i)

with

    F(g_E) = alpha f(g_E)/(1 + alpha f(g_E))
    F(g_E) ≈ F1 g_E + F0

and the local feedback slope

    W_i = (1 + epsilon)/F1

so each neuron's deviation from its baseline fixed point satisfies, near the
baseline,

    d(delta s_i)/dt ≈ epsilon * delta s_i / tau

The pulse kicks the population along a mixed-sign mode q_i. Thus some neurons
increase and some decrease, while the readout sums the mixed responses into a
single pulse-to-step output.

This reproduces the qualitative structure of Goldman Fig. 7A-D:
    top row:    single pulse input
    middle row: individual neuronal responses, with mixed increases/decreases
    bottom row: summed/readout output
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Seung et al. reduced model
# -----------------------------

@dataclass(frozen=True)
class SeungReducedParams:
    tau: float = 100.0      # ms
    alpha: float = 1.0
    F1: float = 0.5314      # Eq. (13) slope
    F0: float = -0.01878    # Eq. (13) intercept

    @property
    def tuned_W(self) -> float:
        return 1.0 / self.F1

    @property
    def tuned_B(self) -> float:
        return -self.F0 / self.F1


def F_linear(g_E: np.ndarray | float, p: SeungReducedParams) -> np.ndarray | float:
    """Linear approximation F(g_E) = F1*g_E + F0 from Seung et al."""
    return p.F1 * np.asarray(g_E) + p.F0


def alpha_f_from_F(F: np.ndarray | float) -> np.ndarray | float:
    """
    Convert F = alpha*f/(1 + alpha*f) into alpha*f = F/(1 - F).

    Clipping keeps the simulation well-defined outside the narrow range where
    the linear approximation is intended to be used.
    """
    F = np.clip(F, 0.0, 0.95)
    return F / (1.0 - F)


def seung_autapse_dsdt(
    s: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    p: SeungReducedParams,
) -> np.ndarray:
    """Vectorized Seung reduced averaged autapse equation."""
    g_E = W * s + B
    F = F_linear(g_E, p)
    alpha_f = alpha_f_from_F(F)
    return (-s + alpha_f * (1.0 - s)) / p.tau


# -----------------------------
# Fig. 7-style population model
# -----------------------------

@dataclass(frozen=True)
class Fig7AutapsePopulationParams:
    n_neurons: int = 12
    T: float = 12_000.0          # ms
    dt: float = 0.5              # ms

    pulse_time: float = 100.0    # ms
    pulse_width: float = 1.0     # ms
    x_kick: float = 0.010        # memory-coordinate kick

    # Baseline around which each reduced autapse is linearized.
    # This is kept away from 0 and 1 so mixed positive/negative deviations
    # remain physical.
    s_base: float = 0.030

    # Bounds avoid unphysical s values under unstable growth.
    s_min: float = 0.0
    s_max: float = 0.09


def mixed_sign_memory_mode(n: int) -> np.ndarray:
    """
    Mixed-sign slow mode q_i.

    Positive entries are neurons that increase after the pulse; negative entries
    are neurons that decrease. Normalize max(abs(q)) = 1.
    """
    q = np.linspace(-1.0, 1.0, n)

    # Avoid a neuron with exactly zero loading for even/odd n differences.
    q[np.abs(q) < 1e-12] = 0.15

    return q / np.max(np.abs(q))


def make_feedback_parameters_for_baseline(
    s_star: np.ndarray,
    mistuning: float,
    p: SeungReducedParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose W_i and B_i so s_star_i is each neuron's fixed point.

    Fixed point condition:
        s_star = F(W*s_star + B)

    With F(g) = F1*g + F0 and W = (1+epsilon)/F1:
        B_i = (s_star_i - F0)/F1 - W_i*s_star_i
    """
    W = np.full_like(s_star, (1.0 + mistuning) / p.F1)
    B = (s_star - p.F0) / p.F1 - W * s_star
    return W, B


def rk4_step_vector(func, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = func(y)
    k2 = func(y + 0.5 * dt * k1)
    k3 = func(y + 0.5 * dt * k2)
    k4 = func(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_fig7_seung_population(
    mistuning: float,
    model_params: SeungReducedParams = SeungReducedParams(),
    sim_params: Fig7AutapsePopulationParams = Fig7AutapsePopulationParams(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a population of Seung reduced autapses.

    The pulse is a single brief kick along a mixed-sign line-attractor mode.
    """
    p = model_params
    sp = sim_params

    t = np.arange(0.0, sp.T + sp.dt, sp.dt)
    n = sp.n_neurons

    q = mixed_sign_memory_mode(n)
    s_star = np.full(n, sp.s_base)

    W, B = make_feedback_parameters_for_baseline(s_star, mistuning, p)

    s = np.empty((len(t), n))
    s[0] = s_star.copy()

    pulse = np.zeros_like(t)

    # Readout sums along the same mixed-sign mode. A neuron that decreases
    # after the pulse has a negative readout weight, so it still contributes
    # positively to the remembered scalar.
    readout_weights = q / np.dot(q, q)

    for k in range(len(t) - 1):
        current_s = s[k].copy()

        if sp.pulse_time <= t[k] < sp.pulse_time + sp.pulse_width:
            current_s += q * sp.x_kick
            pulse[k] = 1.0

        step = lambda y: seung_autapse_dsdt(y, W=W, B=B, p=p)
        next_s = rk4_step_vector(step, current_s, sp.dt)
        s[k + 1] = np.clip(next_s, sp.s_min, sp.s_max)

    responses = s - s_star[None, :]
    output = (responses @ readout_weights) / sp.x_kick

    return t, pulse, responses, output


def plot_goldman_fig7_using_seung_reduced_model() -> None:
    cases = [
        ("A: W mistuned by -6%", -0.06, 5.0),
        ("B: W mistuned by -0.5%", -0.005, 5.0),
        ("C: W perfectly tuned", 0.0, 12.0),
        ("D: W mistuned by +2%", +0.02, 12.0),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [0.6, 2.0, 1.3]},
    )

    for col, (title, eps, xmax_seconds) in enumerate(cases):
        t, pulse, responses, output = simulate_fig7_seung_population(eps)
        seconds = t / 1000.0
        visible = seconds <= xmax_seconds

        # Top: single input spike/pulse.
        axes[0, col].plot(seconds[visible], pulse[visible], linewidth=1.2)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].set_ylim(-0.1, 1.2)
        if col == 0:
            axes[0, col].set_ylabel("input")

        # Middle: individual neuronal responses.
        # These are signed deviations from baseline. Positive traces increase;
        # negative traces decrease, matching the qualitative behavior in Fig. 7.
        scale = 1.0 / max(np.max(np.abs(responses[visible])), 1e-12)
        for j in range(responses.shape[1]):
            axes[1, col].plot(
                seconds[visible],
                0.25 * scale * responses[visible, j],
                linewidth=1.0,
            )
        axes[1, col].axhline(0.0, linewidth=0.8)
        if col == 0:
            axes[1, col].set_ylabel("neuronal\nresponses")

        # Bottom: summed/readout output.
        axes[2, col].plot(seconds[visible], output[visible], linewidth=1.5)
        axes[2, col].axhline(1.0, linestyle="--", linewidth=0.8)
        axes[2, col].axhline(0.95, linestyle=":", linewidth=0.8)
        axes[2, col].axhline(1.05, linestyle=":", linewidth=0.8)
        axes[2, col].set_xlabel("time (s)")
        if col == 0:
            axes[2, col].set_ylabel("summed\noutput")

        if eps > 0:
            axes[2, col].set_ylim(-0.2, min(6.0, np.max(output[visible]) * 1.05))
        else:
            axes[2, col].set_ylim(-0.2, 1.3)

    fig.suptitle(
        "Goldman Fig. 7A-D style line attractor using mixed-sign Seung reduced autapses",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_goldman_fig7_using_seung_reduced_model()
