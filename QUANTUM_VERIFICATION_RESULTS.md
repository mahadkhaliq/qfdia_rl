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
| complete | ibm | ibm_miami | 30 | 4 | 4 | 1024 | 0.9942 | 0.0691 | 0.0000 | 0.0018 | survives_device | `runs/quantum_architectures/verify_ibm_30_ibm_smoke_30.json` |
| complete | ibm | ibm_miami | 57 | 6 | 2 | 1024 | 0.9984 | 0.0307 | 0.0000 | 0.0018 | survives_device | `runs/quantum_architectures/verify_ibm_57_ibm_smoke_57_2pt.json` |
| complete | ibm | ibm_miami | 118 | 8 | 1 | 1024 | 0.9987 | 0.0110 | 0.0000 | 0.0005 | survives_device | `runs/quantum_architectures/verify_ibm_118_ibm_smoke_118_1pt.json` |

## Notes

- `sim`, `aer`, and `aer_noisy` rows are completed non-hardware checks of the trained Q-NPG VQC actor.
- The `ibm` rows are completed real-hardware smoke verifications of the trained Q-NPG VQC actor.
- The hardware smoke run is intentionally small: one circuit evaluation per operating point, with the listed shot count.
- Hardware smoke plot: `runs/plots/paper/ibm_hardware_smoke.png`.
- Hardware safety gate and exact command are recorded in `runs/quantum_architectures/ibm_verification_manifest.json`.
