# IBM Verification Manifest

- Generated: `1782424574.241656`
- Bus: IEEE 30-bus
- Policy: `runs/policies/qnpg_30_policy.npz`
- Planned backend: `ibm_fez`
- Shots / points: 1024 / 4
- Token status: required before real hardware submission if the old exposed token has not been rotated

## Existing Non-Hardware Verification

| Device | Available | Qubits | Points | Shots | Device stealth | Device SDS | Flagged | Verdict | JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| sim | True | 4 | 2 | 1024 | 0.9931 | 0.0743 | 0.0000 | survives_device | `runs/quantum_architectures/verify_sim_30.json` |
| aer | True | 4 | 2 | 1024 | 0.9926 | 0.0780 | 0.0000 | survives_device | `runs/quantum_architectures/verify_aer_30.json` |
| aer_noisy | True | 4 | 2 | 1024 | 0.9939 | 0.0711 | 0.0000 | survives_device | `runs/quantum_architectures/verify_aer_noisy_30.json` |

## Planned IBM Command

```bash
BUS=30 DEVICE=ibm IBM_BACKEND=ibm_fez ENV_NAME=synthgrad SHOTS=1024 N_POINTS=4 POLICY=runs/policies/qnpg_30_policy.npz RESULT_TAG=ibm_smoke_30 sbatch scripts/run_ibm_verification_hellbender.sbatch
```

## Safety Gate

- Rotate/recreate the IBM API key if not already done after the earlier accidental token print.
- Rerun scripts/ibm_quantum_preflight.py and confirm token is redacted as <hidden>.
- Use a small first hardware smoke run: 4 operating points and 1024 shots.
- Archive the Slurm stdout/stderr and resulting verify_ibm JSON with the paper artifacts.
