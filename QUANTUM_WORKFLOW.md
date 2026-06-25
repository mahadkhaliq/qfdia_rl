# Quantum Workflow

This project has two quantum tracks:

1. Q-NPG-FDIA attacker policy, already implemented with a variational quantum circuit.
2. QGNN-style detector research, which should begin as a reduced quantum graph detector before scaling.

## Q-NPG-FDIA Circuit

The implemented policy circuit is:

```text
classical observation -> W_enc -> angle vector
AngleEmbedding(rotation="Y")
StronglyEntanglingLayers
PauliZ expectation readout
W_act + tanh -> FDIA attack vector
```

Current bus presets:

```text
IEEE 30-bus:  4 qubits, 3 layers, 36 VQC parameters
IEEE 57-bus:  6 qubits, 4 layers, 72 VQC parameters
IEEE 118-bus: 8 qubits, 4 layers, 96 VQC parameters
```

Training uses simulator-based QNPG because the QFIM/natural-gradient loop requires many circuit evaluations. IBM hardware should be used first for inference/noise verification of trained policies.

## IBM-Compatible Environment

Keep training/detector work in the stable research environment, and use a clean environment for IBM hardware verification. This avoids dependency conflicts between the detector stack and the fast-moving Qiskit Runtime stack.

The current Hellbender research environment can import the IBM-facing stack:

```text
pennylane: 0.42.3
pennylane-qiskit: 0.42.0
qiskit: 2.4.1
qiskit-aer: 0.17.2
qiskit-ibm-runtime: 0.47.0
```

That is suitable for simulator, Aer, noisy-Aer, and small PennyLane/Qiskit experiments. For real IBM hardware, prefer the clean `qfdia_ibm_latest` environment and keep hardware runs as inference/verification jobs. If the PennyLane plugin and the newest Qiskit Runtime release diverge, use `verify_ibm.py` through direct Qiskit Runtime first, then bring the result back into the shared JSON/plot workflow.

On Hellbender:

```bash
cd ~/qfdia_rl_git
```

Do not run Python or conda-heavy work on the Hellbender login node. Use Open
OnDemand, an interactive Slurm allocation, or an `sbatch` wrapper. Login-node
use should be limited to lightweight file checks and job submission commands.

Inside an interactive allocation or Open OnDemand session, create/update the
IBM environment with:

```bash
bash scripts/setup_ibm_quantum_env.sh
```

Default environment name:

```text
qfdia_ibm_latest
```

To use a different name:

```bash
ENV_NAME=qfdia_ibm_2026 bash scripts/setup_ibm_quantum_env.sh
```

Check the active quantum stack:

```bash
sbatch scripts/run_ibm_readiness_hellbender.sbatch
```

This writes:

```text
runs/quantum_architectures/quantum_stack_versions.json
```

The same Slurm readiness job also checks IBM Runtime account/backend readiness
without exposing tokens:

```bash
ENV_NAME=qfdia_ibm_latest MIN_QUBITS=4 SHOTS=1024 N_POINTS=4 \
  sbatch scripts/run_ibm_readiness_hellbender.sbatch
```

If credentials and a hardware backend are available, the preflight JSON includes
`ready_for_hardware: true` and a recommended `sbatch` command.
The current hardware-readiness note is tracked in `IBM_HARDWARE_READINESS.md`.

If a token is ever printed in terminal output or logs, rotate/recreate that IBM
API key before submitting a real hardware job. The preflight script is designed
to write only `token: <hidden>` in its JSON report.

The readiness job creates a token-free verification manifest before hardware
submission. To run only this logic inside an existing allocation, use:

```bash
python scripts/ibm_verification_manifest.py \
  --preflight runs/quantum_architectures/ibm_preflight_synthgrad.json \
  --env-name qfdia_ibm_latest \
  --shots 1024 \
  --n-points 4
```

This writes:

```text
runs/quantum_architectures/ibm_verification_manifest.json
IBM_VERIFICATION_MANIFEST.md
```

The manifest records existing simulator/Aer/noisy-Aer verification, selected backend, exact Slurm command, expected output JSON, and the token-rotation safety gate.

Export the paper-ready quantum verification table locally or inside an existing
allocation:

```bash
python scripts/export_quantum_verification_results.py
```

This writes:

```text
paper_tables/quantum_verification_results.csv
QUANTUM_VERIFICATION_RESULTS.md
```

Before IBM hardware submission, the table should show completed `sim`, `aer`,
and `aer_noisy` rows plus an `ibm` row with status
`planned_after_token_rotation`. After the QPU job completes, rerun the exporter
so the `ibm` row becomes a completed hardware result.

## Generate Architecture Artifacts

```bash
python scripts/quantum_architecture_summary.py \
  --out-dir runs/quantum_architectures
```

Outputs:

```text
runs/quantum_architectures/quantum_architecture_summary.json
runs/quantum_architectures/quantum_architecture_summary.md
runs/quantum_architectures/qnpg_vqc_30_bus_circuit.txt
runs/quantum_architectures/qnpg_vqc_57_bus_circuit.txt
runs/quantum_architectures/qnpg_vqc_118_bus_circuit.txt
QUANTUM_ARCHITECTURE_REGISTRY.md
```

## Verification Order

Start with the 30-bus trained policy because it uses only 4 qubits.

Simulator sanity check:

```bash
python verify_ibm.py \
  --bus 30 \
  --load outputs/qnpg_30_policy.npz \
  --device sim \
  --n-points 16 \
  --shots 4096 \
  --out runs/quantum_architectures/verify_sim_30.json
```

Noiseless Qiskit/Aer check:

```bash
python verify_ibm.py \
  --bus 30 \
  --load outputs/qnpg_30_policy.npz \
  --device aer \
  --n-points 16 \
  --shots 4096 \
  --out runs/quantum_architectures/verify_aer_30.json
```

Noisy fake-backend check:

```bash
python verify_ibm.py \
  --bus 30 \
  --load outputs/qnpg_30_policy.npz \
  --device aer_noisy \
  --fake-backend FakeKolkataV2 \
  --n-points 16 \
  --shots 4096 \
  --out runs/quantum_architectures/verify_aer_noisy_30.json
```

IBM hardware check:

```bash
conda run -n qfdia_ibm_latest python verify_ibm.py \
  --bus 30 \
  --load outputs/qnpg_30_policy.npz \
  --device ibm \
  --ibm-backend <backend-name> \
  --n-points 16 \
  --shots 4096 \
  --out runs/quantum_architectures/verify_ibm_30.json
```

Do not paste IBM API tokens into chat. Configure the IBM account in the shell/session using the IBM/Qiskit account mechanism, then run the command.

Hellbender Slurm wrapper:

```bash
BUS=30 DEVICE=aer ENV_NAME=qfdia_ibm_latest sbatch scripts/run_ibm_verification_hellbender.sbatch
BUS=30 DEVICE=aer_noisy ENV_NAME=qfdia_ibm_latest sbatch scripts/run_ibm_verification_hellbender.sbatch
BUS=30 DEVICE=ibm IBM_BACKEND=<backend-name> ENV_NAME=qfdia_ibm_latest sbatch scripts/run_ibm_verification_hellbender.sbatch
```

The wrapper writes logs to:

```text
runs/logs/ibm_verify_<jobid>.out
runs/logs/ibm_verify_<jobid>.err
```

and tagged metrics to:

```text
runs/quantum_architectures/verify_<device>_<bus>_<result_tag_or_jobid>.json
```

It also refreshes the latest-result alias:

```text
runs/quantum_architectures/verify_<device>_<bus>.json
```

## QGNN Detector Direction

The detector-side quantum architecture is implemented first as a reduced, IBM-feasible pilot:

```text
graph node features -> reduced q-qubit feature vector
RY/RZ encoding
edge-inspired entanglers over the reduced topology
PauliZ readout
classical binary FDIA head
```

The first completed pilot uses:

```text
datasets: QGrid-Synth 30-bus and Ruan CAISO 30-bus
samples: 2,000 per dataset
qubits: 4
quantum layers: 2
selected nodes: [3, 5, 9, 11]
reduced edges: [(0, 1), (0, 3), (1, 2)]
```

The current stronger Ruan 30-bus reduced QGNN uses:

```text
variant: enhanced6 / enhanced6_balacc
qubits: 6
quantum layers: 2
node selection: hybrid topology + label-shift
feature mode: raw_plus_diffused
selected nodes: [3, 5, 9, 11, 14, 26]
reduced edges: [(0, 1), (0, 3), (1, 2), (3, 4)]
encoding: raw selected node features + graph-diffused features -> tanh linear encoder -> RY angles
entanglers: CNOT gates on reduced physical topology
readout: PauliZ expectations on all qubits -> classical binary head
```

Compare QGNN against CNN, MLP, GCN, W-GCN, and GAT using the same detector metrics. Treat this as a proof of equivalent quantum detector architecture first; performance tuning should come after threshold calibration and richer reduced-node selection.
