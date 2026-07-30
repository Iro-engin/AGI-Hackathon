# ATLAS-SB ICML replication build

Isolated source, experiments, and GitHub Actions workflow for the ATLAS-SB ICML manuscript. The workflow runs strict one-step Monte Carlo, downloads and analyzes the public UCI Gas Sensor Array Drift dataset, compiles the anonymous ICML PDF, and uploads all audited artifacts.

The scientific information clock is fixed: every Monte Carlo forecast uses observations through time t and is evaluated on time t+1; UCI batches 1-4 fix preprocessing, envelopes, and the frozen pilot, while batches 5-10 are predicted once in chronological order.

This branch exists only to expose the complete CI run, logs, and build artifact for inspection before the final release is committed.
