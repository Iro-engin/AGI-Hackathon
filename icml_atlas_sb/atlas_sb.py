from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


Array = np.ndarray


def sym(a: Array) -> Array:
    return (a + a.T) / 2.0


def opnorm(a: Array) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(sym(a)))))


def radial_clip(x: Array, tau: float) -> Array:
    norm = float(np.linalg.norm(x))
    if norm == 0.0 or norm <= tau:
        return x.copy()
    return x * (tau / norm)


def spectral_clip(a: Array, lower: float, upper: float) -> Array:
    if not lower <= upper:
        raise ValueError("lower must not exceed upper")
    values, vectors = np.linalg.eigh(sym(a))
    values = np.clip(values, lower, upper)
    return sym((vectors * values) @ vectors.T)


def expected_gaussian_nll_excess(forecast: Array, truth: Array) -> float:
    sign_f, logdet_f = np.linalg.slogdet(forecast)
    sign_t, logdet_t = np.linalg.slogdet(truth)
    if sign_f <= 0 or sign_t <= 0:
        raise ValueError("forecast and truth must be positive definite")
    return float(
        logdet_f
        + np.trace(np.linalg.solve(forecast, truth))
        - logdet_t
        - truth.shape[0]
    )


def realized_gaussian_nll(forecast: Array, observation: Array) -> float:
    sign, logdet = np.linalg.slogdet(forecast)
    if sign <= 0:
        raise ValueError("forecast must be positive definite")
    return float(logdet + observation @ np.linalg.solve(forecast, observation))


@dataclass(frozen=True)
class RadiusConfig:
    """Parameters declared before a deployment episode.

    c_det is the constant in the deterministic sub-Gaussian matrix radius.
    kappa stabilizes the observable radius from below.  The stochastic
    certificate remains valid because the final radius is at least the minimum
    of two valid radii; the floor prevents data-dependent undersmoothing.
    """

    K: float
    M: float
    alpha: float
    horizon: int
    n_windows: int
    gammas: Tuple[float, ...]
    c_tau: float = 1.0
    c_det: float = 2.0
    kappa: float = 0.25
    c_bias: float = 4.0

    def validate(self) -> None:
        if self.K <= 0 or self.M <= 1:
            raise ValueError("K must be positive and M must exceed one")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        if self.horizon <= 1 or self.n_windows <= 0:
            raise ValueError("invalid horizon or window count")
        if not self.gammas or any(not 0 < g < 1 for g in self.gammas):
            raise ValueError("gammas must be a nonempty subset of (0,1)")
        if not 0 < self.kappa <= 1:
            raise ValueError("kappa must lie in (0,1]")


def theoretical_tau(m: int, config: RadiusConfig) -> float:
    """Sub-Gaussian radial clipping threshold fixed before evaluation.

    The expression follows the standard norm concentration envelope
    m + 2 sqrt(m x) + 2 x with x=log(8T/alpha).  c_tau permits a declared
    safety multiplier; the manuscript uses the value in the configuration.
    """

    config.validate()
    x = math.log(8.0 * config.horizon / config.alpha)
    envelope = m + 2.0 * math.sqrt(m * x) + 2.0 * x
    return math.sqrt(config.c_tau * config.K**2 * config.M * envelope)


def clipping_bias_bound(m: int, config: RadiusConfig) -> float:
    """Declared high-probability clipping-bias envelope.

    This term is negligible for the predeclared sub-Gaussian threshold.  It is
    kept in every selector threshold, rather than silently omitted.
    """

    x = math.log(8.0 * config.horizon / config.alpha)
    return (
        config.c_bias
        * config.K**2
        * config.M
        * (m + x + 1.0)
        * math.exp(-x)
    )


def deterministic_radius(n: int, m: int, config: RadiusConfig) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    u = math.log(
        8.0
        * m
        * config.horizon
        * config.n_windows
        / config.alpha
    )
    d = m + u
    return config.c_det * config.K**2 * config.M * (
        math.sqrt(d / n) + d / n
    )


def empirical_bernstein_radius(
    variance_process: Array,
    n: int,
    tau: float,
    m: int,
    config: RadiusConfig,
) -> float:
    """Observable rolling-window matrix empirical-Bernstein radius.

    variance_process is sum Z_s^2 over the window, where
    Z_s = psi_tau(y_s) psi_tau(y_s)' / tau^2 and 0 <= Z_s <= I.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    ell = math.log(
        8.0
        * m
        * config.horizon
        * config.n_windows
        * len(config.gammas)
        / config.alpha
    )
    vmax = max(0.0, float(np.max(np.linalg.eigvalsh(sym(variance_process)))))
    candidates: List[float] = []
    for gamma in config.gammas:
        phi = -math.log1p(-gamma) - gamma
        candidates.append(
            2.0
            * tau**2
            / (n * gamma)
            * (ell + 0.25 * phi * vmax)
        )
    return float(min(candidates))


def stabilized_radius(
    empirical: float,
    deterministic: float,
    bias: float,
    config: RadiusConfig,
) -> float:
    """The theorem-matched stabilized radius.

    On the intersection of the empirical- and deterministic-certificate events,
    min(empirical, deterministic) remains a valid radius.  The kappa floor is
    needed for the Lepski oracle argument and never makes the certificate smaller.
    """

    return max(config.kappa * deterministic, min(deterministic, empirical)) + bias


@dataclass
class RollingStatistics:
    clipped: Array
    cumulative_outer: Array
    cumulative_z2: Array
    tau: float


def prepare_rolling_statistics(observations: Array, tau: float) -> RollingStatistics:
    if observations.ndim != 2:
        raise ValueError("observations must be two-dimensional")
    n, m = observations.shape
    clipped = np.empty_like(observations, dtype=float)
    outer = np.empty((n, m, m), dtype=float)
    z2 = np.empty((n, m, m), dtype=float)
    tau2 = tau**2
    tau4 = tau2**2
    for i, row in enumerate(observations):
        c = radial_clip(np.asarray(row, dtype=float), tau)
        clipped[i] = c
        oo = np.outer(c, c)
        outer[i] = oo
        z2[i] = (float(c @ c) / tau4) * oo
    cumulative_outer = np.concatenate(
        [np.zeros((1, m, m)), np.cumsum(outer, axis=0)], axis=0
    )
    cumulative_z2 = np.concatenate(
        [np.zeros((1, m, m)), np.cumsum(z2, axis=0)], axis=0
    )
    return RollingStatistics(clipped, cumulative_outer, cumulative_z2, tau)


def window_statistics(stats: RollingStatistics, end: int, window: int) -> Tuple[Array, Array]:
    """Return scatter and sum Z_s^2 for observations [end-window, end).

    end is an exclusive Python index.  At forecast origin t (0-indexed), pass
    end=t+1 so no future observation is used.
    """

    if window <= 0 or end < window or end > len(stats.clipped):
        raise ValueError("invalid rolling window")
    start = end - window
    scatter = sym(
        (stats.cumulative_outer[end] - stats.cumulative_outer[start]) / window
    )
    variance = sym(stats.cumulative_z2[end] - stats.cumulative_z2[start])
    return scatter, variance


@dataclass
class AtlasForecast:
    forecast: Array
    selected_window: int
    scatters: Dict[int, Array]
    empirical_radii: Dict[int, float]
    deterministic_radii: Dict[int, float]
    stabilized_radii: Dict[int, float]
    monotone_radii: Dict[int, float]
    floor_radii: Dict[int, float]


def atlas_sb_from_statistics(
    stats: RollingStatistics,
    end: int,
    windows: Sequence[int],
    config: RadiusConfig,
    lower: float = 1.0,
) -> AtlasForecast:
    config.validate()
    available = sorted({int(h) for h in windows if 0 < int(h) <= end})
    if not available:
        raise ValueError("no candidate window is available")
    m = stats.clipped.shape[1]
    bias = clipping_bias_bound(m, config)
    scatters: Dict[int, Array] = {}
    empirical: Dict[int, float] = {}
    deterministic: Dict[int, float] = {}
    stabilized: Dict[int, float] = {}
    floors: Dict[int, float] = {}
    for h in available:
        scatter, variance = window_statistics(stats, end, h)
        e = empirical_bernstein_radius(variance, h, stats.tau, m, config)
        r = deterministic_radius(h, m, config)
        scatters[h] = scatter
        empirical[h] = e
        deterministic[h] = r
        floors[h] = config.kappa * r
        stabilized[h] = stabilized_radius(e, r, bias, config)

    # Nonincreasing majorant required by the Lepski proof.
    monotone: Dict[int, float] = {}
    running = 0.0
    for h in reversed(available):
        running = max(running, stabilized[h])
        monotone[h] = running

    selected = available[0]
    for h in available:
        admissible = True
        for short_h in available:
            if short_h > h:
                break
            difference = opnorm(scatters[h] - scatters[short_h])
            # The extra floor term pays for the two local-drift biases in the
            # oracle-window admissibility proof and is smaller than the older
            # 2rho(H)+2rho(h) rule.
            threshold = (
                monotone[h]
                + monotone[short_h]
                + 2.0 * floors[short_h]
            )
            if difference > threshold:
                admissible = False
                break
        if admissible:
            selected = h

    forecast = spectral_clip(scatters[selected], lower, config.M)
    return AtlasForecast(
        forecast=forecast,
        selected_window=selected,
        scatters=scatters,
        empirical_radii=empirical,
        deterministic_radii=deterministic,
        stabilized_radii=stabilized,
        monotone_radii=monotone,
        floor_radii=floors,
    )


def atlas_sb(
    observations: Array,
    windows: Sequence[int],
    config: RadiusConfig,
    tau: float | None = None,
    lower: float = 1.0,
) -> AtlasForecast:
    if observations.ndim != 2:
        raise ValueError("observations must be two-dimensional")
    if tau is None:
        tau = theoretical_tau(observations.shape[1], config)
    stats = prepare_rolling_statistics(observations, tau)
    return atlas_sb_from_statistics(stats, len(observations), windows, config, lower)


def covariance_of_rows(rows: Array) -> Array:
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError("rows must be a nonempty matrix")
    return sym(rows.T @ rows / len(rows))
