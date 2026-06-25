# QFDIA Project Checklist

This checklist keeps the detector-comparison track and the quantum RL/IBM track aligned.

## 1. Data And Public Benchmarks

- [x] Verify local QGrid-Synth datasets for IEEE 30/57/118 bus systems.
- [x] Verify Ruan/STGDL public FDIA repository and bundled admittance matrices.
- [x] Normalize Ruan CAISO 30/118 data into the shared detector Parquet schema.
- [x] Keep QGrid-Synth and Ruan datasets separate in figures and tables.
- [x] Add a written dataset table with sample counts, feature dimensions, labels, and topology source.

## 2. Classical Detector Baselines

- [x] Train 1D-CNN on QGrid-Synth 30/57/118 and Ruan CAISO 30/118.
- [x] Train MLP on the same five datasets.
- [x] Log `config.json`, `history.csv`, `metrics.json`, `model.pt`, `architecture.json`, and `architecture.txt`.
- [x] Generate combined comparison plots.
- [x] Generate individual metric plots and individual learning-curve plots.

## 3. Graph Detector Baselines

- [x] Train topology-aware GCN on the same five datasets.
- [x] Train masked graph-attention GAT on the same five datasets.
- [x] Include graph model architecture summaries with node count, edge count, feature dimension, and parameter count.
- [x] Add edge-weighted graph model using admittance magnitude instead of binary topology only.
- [ ] Add ablation: raw `z` node features vs `z + |a|` or residual-aware features where labels permit.

## 4. QGNN Detector Track

- [x] Define a reduced quantum graph detector architecture that can run on simulator and IBM hardware.
- [x] Map classical GCN/GAT concepts to a quantum architecture:
  - node feature encoding via angle/amplitude encoding,
  - graph edges via entangling gates over physical topology or reduced topology,
  - graph readout via Pauli expectation pooling,
  - classical head for binary FDIA detection.
- [x] Start with small/reduced graphs before attempting 30/57/118 full systems.
- [x] Compare QGNN-style detector against CNN, MLP, GCN, and GAT using the same metrics.
- [ ] Scale QGNN experiments beyond the 4-qubit/2k-sample pilot.
- [ ] Add QGNN threshold calibration because the first Ruan pilot is recall-heavy and over-flags.

## 5. Quantum RL / IBM Quantum Track

- [x] Existing Q-NPG-FDIA policy uses a VQC actor:
  `AngleEmbedding(Y) + StronglyEntanglingLayers + PauliZ expectation readout`.
- [x] Existing QNPG trainer uses QFIM/natural-gradient update for `theta_q`.
- [x] Existing `verify_ibm.py` supports simulator, Aer, noisy Aer, and IBM hardware verification.
- [x] Generate explicit quantum circuit architecture summaries for each bus.
- [ ] Verify trained policies with:
  - [x] noiseless simulator,
  - [x] Qiskit Aer,
  - [x] fake/noisy backend,
  - [ ] selected IBM backend.
- [x] Log IBM verification metadata:
  backend, shots, qubits, logical circuit depth, device result, simulator result, stealth, SDS, flagged rate.
- [x] Keep real QPU work as inference/verification first; full QNPG training on queued hardware is not practical.
- [x] Add clean IBM/Qiskit environment setup script for latest IBM hardware verification.
- [x] Add quantum stack checker and Hellbender verification Slurm wrapper.

## 6. Permissions / Operations

For hands-off progress, approve persistent prompts for:

- `ssh` so Codex can run commands on Hellbender.
- `scp` so Codex can copy plots/results back locally.
- `git` so Codex can commit and push code changes.

Keep at least one Hellbender GPU allocation running for training. For IBM Quantum, do not paste API tokens into chat; configure the account in the environment/session instead.
