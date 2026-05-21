"""
Reduced averaged autapse / line-attractor model from Seung et al. (2000),
"The Autapse: A Simple Illustration of Short-Term Analog Memory Storage
by Tuned Synaptic Feedback".

This script keeps only the method-of-averaging model from Sections 3-4.

Model
-----
The slow synaptic activation s obeys

    tau ds/dt = -s + alpha f(g_E) (1 - s)
    g_E = W s + B

where W is the recurrent autapse strength and B is the feedforward bias.

The paper defines

    F(g_E) = alpha f(g_E) / (1 + alpha f(g_E))

and shows that analog memory storage is possible when F is approximately
linear and the feedback parameters are tuned:

    F(g_E) ≈ F1 g_E + F0
    W = 1/F1
    B = -F0/F1

Using the paper's fit,

    F(g_E) ≈ 0.5314 g_E - 0.01878

which gives

    W ≈ 1.882
    B ≈ 0.03534

Units
-----
time: ms
conductance: mS/cm^2
s: dimensionless synaptic activation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AveragedAutapseParams:
    """Parameters for the reduced averaged autapse model."""

    tau: float = 100.0       # synaptic time constant, ms
    alpha: float = 1.0       # synaptic saturation parameter
    F1: float = 0.5314       # slope of paper's linear F(g_E) fit
    F0: float = -0.01878     # intercept of paper's linear F(g_E) fit

    @property
    def tuned_W(self) -> float:
        """Autapse strength satisfying W = 1/F1."""
        return 1.0 / self.F1

    @property
    def tuned_B(self) -> float:
        """Feedforward bias satisfying B = -F0/F1."""
        return -self.F0 / self.F1


@dataclass(frozen=True)
class Pulse:
    """Instantaneous perturbation to s, approximating a transient burst input."""

    time: float       # ms
    delta_s: float    # dimensionless increment to s


def F_linear(g_E: np.ndarray | float, params: AveragedAutapseParams) -> np.ndarray | float:
    """Linear approximation to F(g_E) from the paper."""
    return params.F1 * np.asarray(g_E) + params.F0


def alpha_f_from_F(F: np.ndarray | float) -> np.ndarray | float:
    """
    Convert F = alpha f/(1 + alpha f) into alpha f = F/(1 - F).

    F is clipped to keep the reduced model numerically well-defined outside
    the range where the paper's linear approximation is meant to apply.
    """
    F = np.clip(F, 0.0, 0.95)
    return F / (1.0 - F)


def dsdt(
    s: float,
    W: float,
    B: float,
    params: AveragedAutapseParams = AveragedAutapseParams(),
) -> float:
    """
    Drift of synaptic activation in the reduced averaged model.

        ds/dt = [-s + alpha f(Ws + B)(1 - s)] / tau
    """
    g_E = W * s + B
    F = F_linear(g_E, params)
    alpha_f = alpha_f_from_F(F)
    return float((-s + alpha_f * (1.0 - s)) / params.tau)


def rk4_step_scalar(func, y: float, dt: float) -> float:
    """One fourth-order Runge-Kutta step for a scalar ODE."""
    k1 = func(y)
    k2 = func(y + 0.5 * dt * k1)
    k3 = func(y + 0.5 * dt * k2)
    k4 = func(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_reduced_autapse(
    T: float = 4000.0,
    dt: float = 0.1,
    s0: float = 0.010,
    W: float | None = None,
    B: float | None = None,
    params: AveragedAutapseParams = AveragedAutapseParams(),
    pulses: Iterable[Pulse] = (
        Pulse(1000.0, +0.010),
        Pulse(2000.0, -0.008),
        Pulse(3000.0, +0.010),
    ),
    s_bounds: tuple[float, float] = (0.0, 0.05),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the reduced averaged autapse model.

    Parameters
    ----------
    T:
        Total simulation time in ms.
    dt:
        Time step in ms.
    s0:
        Initial synaptic activation.
    W:
        Autapse strength. Defaults to tuned W = 1/F1.
    B:
        Feedforward bias. Defaults to tuned B = -F0/F1.
    params:
        Model parameters.
    pulses:
        Instantaneous perturbations to s, approximating transient excitatory
        or inhibitory burst inputs.
    s_bounds:
        Lower and upper clipping bounds for s.

    Returns
    -------
    t, s:
        Time vector and synaptic activation trajectory.
    """
    W = params.tuned_W if W is None else W
    B = params.tuned_B if B is None else B

    t = np.arange(0.0, T + dt, dt)
    s = np.empty_like(t)
    s[0] = np.clip(s0, *s_bounds)

    pulse_by_index = {
        int(round(pulse.time / dt)): pulse.delta_s
        for pulse in pulses
    }

    for k in range(len(t) - 1):
        current_s = s[k]

        if k in pulse_by_index:
            current_s += pulse_by_index[k]
            current_s = np.clip(current_s, *s_bounds)

        step = lambda x: dsdt(x, W=W, B=B, params=params)
        next_s = rk4_step_scalar(step, current_s, dt)
        s[k + 1] = np.clip(next_s, *s_bounds)

    return t, s


def linear_feedback_time_constant(
    W: float,
    params: AveragedAutapseParams = AveragedAutapseParams(),
) -> float:
    """
    Time constant predicted by the paper's linear feedback approximation.

        tau_eff = tau / |1 - W F1|

    Infinite if W is exactly tuned.
    """
    denominator = abs(1.0 - W * params.F1)
    if denominator == 0:
        return np.inf
    return params.tau / denominator


def plot_demo() -> None:
    """Compare tuned, leaky, and unstable averaged autapses."""
    params = AveragedAutapseParams()

    cases = [
        ("tuned", params.tuned_W, params.tuned_B),
        ("leaky: W = 0.75 tuned", 0.75 * params.tuned_W, params.tuned_B),
        ("unstable: W = 1.25 tuned", 1.25 * params.tuned_W, params.tuned_B),
    ]

    plt.figure(figsize=(8, 4.5))

    for label, W, B in cases:
        t, s = simulate_reduced_autapse(W=W, B=B, params=params)
        tau_eff = linear_feedback_time_constant(W, params)
        suffix = "tau_eff = inf" if np.isinf(tau_eff) else f"tau_eff = {tau_eff:.0f} ms"
        plt.plot(t / 1000.0, s, label=f"{label} ({suffix})")

    plt.xlabel("time (s)")
    plt.ylabel("synaptic activation s")
    plt.title("Reduced averaged autapse / line-attractor model")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_goldman_fig7_line_attractor():
    """
    Qualitative replication of Goldman Fig. 7A-D:
    pulse-to-step integration in a mistuned line attractor.

    This uses the reduced averaged autapse as a one-dimensional
    line attractor. The plotted variable x is the deviation from
    the baseline memory state.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    params = AveragedAutapseParams()
    tau = params.tau  # 100 ms, as in the Goldman figure caption

    # Goldman Fig. 7A-D mistunings
    cases = [
        ("A: -6%", -0.06),
        ("B: -0.5%", -0.005),
        ("C: perfect", 0.0),
        ("D: +2%", 0.02),
    ]

    T = 12000.0  # ms
    dt = 1.0
    t = np.arange(0.0, T + dt, dt)

    pulse_time = 100.0
    pulse_size = 1.0

    plt.figure(figsize=(10, 6))

    for i, (label, epsilon) in enumerate(cases, start=1):
        x = np.zeros_like(t)

        for k in range(len(t) - 1):
            if t[k] == pulse_time:
                x[k] += pulse_size

            # linearized averaged attractor:
            # dx/dt = epsilon*x/tau
            x[k + 1] = x[k] + dt * (epsilon * x[k] / tau)

        plt.subplot(2, 2, i)
        plt.plot(t / 1000.0, x)
        plt.axhline(1.0, linestyle="--", linewidth=0.8)
        plt.axhline(0.95, linestyle=":", linewidth=0.8)
        plt.axhline(1.05, linestyle=":", linewidth=0.8)
        plt.title(label)
        plt.xlabel("time (s)")
        plt.ylabel("summed output")
        plt.ylim(-0.1, 5 if epsilon > 0 else 1.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_demo()
    plot_goldman_fig7_line_attractor()
