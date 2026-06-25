# Quantum Verification Results

Regenerate this file with:

```bash
python scripts/export_quantum_verification_results.py
```

## Q-NPG Policy Verification

| Status | Device | Backend | Bus | Qubits | Points | Shots | Device stealth | Device SDS | Flagged | Attack delta | Verdict | JSON |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| complete | sim | - | 30 | 4 | 2 | 1024 | 0.9931 | 0.0743 | 0.0000 | 0.0000 | survives_device | `runs/quantum_architectures/verify_sim_30.json` |
| complete | aer | - | 30 | 4 | 2 | 1024 | 0.9926 | 0.0780 | 0.0000 | 0.0002 | survives_device | `runs/quantum_architectures/verify_aer_30.json` |
| complete | aer_noisy | FakeKolkataV2 | 30 | 4 | 2 | 1024 | 0.9939 | 0.0711 | 0.0000 | 0.0020 | survives_device | `runs/quantum_architectures/verify_aer_noisy_30.json` |
| planned_after_token_rotation | ibm | ibm_fez | 30 | 4 | 4 | 1024 | - | - | - | - | pending_token_rotation_and_qpu_submission | `runs/quantum_architectures/verify_ibm_30_ibm_smoke_30.json` |

## Notes

- `sim`, `aer`, and `aer_noisy` rows are completed non-hardware checks of the trained Q-NPG VQC actor.
- The `ibm` row remains planned until the IBM API key has been rotated/reconfirmed and the hardware smoke job has completed.
- The planned hardware smoke run is intentionally small: one circuit evaluation per operating point, with the listed shot count.
- Hardware safety gate and exact command are recorded in `runs/quantum_architectures/ibm_verification_manifest.json`.
