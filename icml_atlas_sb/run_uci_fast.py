from __future__ import annotations

import numpy as np

import run_uci as base


def vectorized_mean_nll(forecast: np.ndarray, rows: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(forecast)
    if sign <= 0:
        raise ValueError("forecast must be positive definite")
    inverse = np.linalg.inv(forecast)
    quadratic = np.einsum("ni,ij,nj->n", rows, inverse, rows, optimize=True)
    return float(logdet + quadratic.mean())


base.mean_nll = vectorized_mean_nll


if __name__ == "__main__":
    base.main()
