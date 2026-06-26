# IBM Hardware Smoke-Test Plan

This is a token-free plan for the first real-QPU verification of the Q-NPG VQC actor.

## Scope

- Bus: IEEE 30-bus
- Qubits: 4
- Backend: `ibm_fez`
- Operating points: 4
- Shots per circuit: 1024
- Estimated hardware circuit evaluations: 4
- Estimated total shots: 4096
- Policy: `runs/policies/qnpg_30_policy.npz`

## Existing Verification Status

| Device | Status |
| --- | --- |
| aer | complete |
| aer_noisy | complete |
| ibm | planned_after_token_rotation |
| sim | complete |

## Slurm Command

Run from `/home/mkfqm/qfdia_rl_ondemand` or another clean code checkout on Hellbender:

```bash
BUS=30 DEVICE=ibm IBM_BACKEND=ibm_fez ENV_NAME=synthgrad SHOTS=1024 N_POINTS=4 POLICY=runs/policies/qnpg_30_policy.npz RESULT_TAG=ibm_smoke_30 sbatch scripts/run_ibm_verification_hellbender.sbatch
```

Expected output JSON:

- `runs/quantum_architectures/verify_ibm_30_ibm_smoke_30.json`
- latest alias: `runs/quantum_architectures/verify_ibm_30.json`

## Safety Gate

- Rotate/recreate the IBM API token if the old exposed token has not been rotated.
- Run scripts/ibm_quantum_preflight.py from Slurm/OOD and confirm token is redacted as <hidden>.
- Submit only this small first run before increasing shots or operating points.
- Archive stdout/stderr and the resulting verify_ibm JSON with paper artifacts.
