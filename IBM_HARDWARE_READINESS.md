# IBM Hardware Readiness

Status as of 2026-06-25:

- IBM Runtime package imports successfully on Hellbender in `synthgrad`.
- Saved IBM account is visible to Qiskit Runtime.
- Redacted preflight JSON was written to:

```text
runs/quantum_architectures/ibm_preflight_synthgrad.json
```

- The preflight reported hardware backends with at least 4 qubits.
- Lowest-pending backend at preflight time: `ibm_fez`.

Recommended hardware verification command after rotating/reconfirming the IBM API key:

```bash
BUS=30 DEVICE=ibm IBM_BACKEND=ibm_fez ENV_NAME=synthgrad \
  sbatch scripts/run_ibm_verification_hellbender.sbatch
```

Use a small first run if queue time or credits matter:

```bash
BUS=30 DEVICE=ibm IBM_BACKEND=ibm_fez ENV_NAME=synthgrad \
  SHOTS=1024 N_POINTS=4 \
  sbatch scripts/run_ibm_verification_hellbender.sbatch
```

Before submission, generate the token-free manifest:

```bash
python scripts/ibm_verification_manifest.py \
  --preflight runs/quantum_architectures/ibm_preflight_synthgrad.json \
  --env-name synthgrad \
  --backend ibm_fez \
  --shots 1024 \
  --n-points 4
```

This records the existing simulator/Aer/noisy-Aer verification, planned backend,
exact Slurm command, and expected hardware JSON path in `IBM_VERIFICATION_MANIFEST.md`.

Important: an IBM API token was accidentally printed during an earlier preflight run. Rotate/recreate that IBM API key before submitting real hardware jobs. The current preflight script redacts token material and only writes `token: <hidden>`.
