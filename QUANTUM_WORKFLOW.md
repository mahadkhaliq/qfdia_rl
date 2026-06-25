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

On Hellbender:

```bash
cd ~/qfdia_rl_git
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
conda run -n qfdia_ibm_latest python scripts/check_quantum_stack.py
```

This writes:

```text
runs/quantum_architectures/quantum_stack_versions.json
```

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

and metrics to:

```text
runs/quantum_architectures/verify_<device>_<bus>.json
```

## QGNN Detector Direction

The next detector-side quantum architecture should be reduced and IBM-feasible:

```text
graph node features -> reduced q-qubit feature vector
RY/RZ encoding
edge-inspired entanglers or compressed VQC block
PauliZ readout
classical binary FDIA head
```

The first target should be 4-8 qubits, matching the current Q-NPG bus presets. Compare against CNN, MLP, GCN, and GAT using the same detector metrics.
