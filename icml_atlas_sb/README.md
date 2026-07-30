# ATLAS-SB ICML replication build

This directory contains the anonymous ICML manuscript, complete proofs, a single implementation of the stabilized certificate, strict multidimensional Monte Carlo, and a public-data evaluation on the UCI Gas Sensor Array Drift at Different Concentrations dataset.

## Scientific information clock

- Every Monte Carlo forecast uses observations through time `t` and is evaluated only on `C_{t+1}` and `y_{t+1}`.
- All fixed windows used by ATLAS-SB appear in the comparison table.
- The predictive output is a precision-space exponential-weights aggregate.
- The uniform empirical-Bernstein/Lepski output is reported separately as `Certified selector`.
- UCI batches 1–4 fix preprocessing, calibration envelopes, and the frozen pilot. Batches 5–10 are predicted once in chronological order.
- Row order inside a UCI batch is not treated as measurement time.
- The local oracle uses target covariance information and is explicitly labeled infeasible.

## Reproduction

```bash
python -m pip install -r requirements.txt
python run_monte_carlo.py --out generated/monte_carlo --seeds 6 --T 6000 --c-det 2.0 --kappa 0.25
python run_uci_fast.py --archive data/gas_sensor_array_drift.zip --out generated/uci --bootstrap 500 --c-det 2.0 --kappa 0.25
python make_assets.py --mc generated/monte_carlo --uci generated/uci --out paper/generated
```

`run_uci_fast.py` downloads the public archive when it is absent. The workflow records all numerical metadata, radius coverage, the maximum expert-loss range, PDF page count, fonts, qpdf output, rendered pages, and SHA-256 digests.

## Claims discipline

- The Hӧlder minimax statement concerns the certified selector in the stated sub-Gaussian interior regime.
- The predictive aggregate receives a pathwise Gaussian log-loss regret guarantee against the best fixed-memory expert.
- Public-data calibration envelopes do not create a formal coverage guarantee.
- Bootstrap intervals that cross zero are reported as inconclusive rather than as improvements.
