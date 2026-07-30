# ATLAS-SB ICML replication build

Isolated source, experiments, and GitHub Actions workflow for the ATLAS-SB ICML manuscript. The workflow runs strict one-step Monte Carlo, downloads and analyzes the public UCI Gas Sensor Array Drift dataset, compiles the anonymous ICML PDF, and uploads all audited artifacts.

The scientific information clock is fixed: every Monte Carlo forecast uses observations through time t and is evaluated on time t+1; UCI batches 1-2 fit preprocessing and the frozen pilot, while batches 3-10 are predicted once in chronological order.
