# Attacker Paper Results: Q-NPG-FDIA

This file intentionally excludes the detector/CNN benchmark tables. Use it for
the Q-NPG attacker paper only.

Regenerate the source tables with the dedicated scripts listed beside each
section. The shared `PAPER_RESULTS.md` file is a detector-comparison artifact
with Q-NPG appendices and should not be treated as attacker-paper-only.

## Analytical Stealth Ceiling

Source table: `paper_tables/sds_ceiling_ratios.csv`

| Bus | a_max | SDS Ceiling | Learned SDS | Learned/Ceiling | Stealth Residual | BDD Tau |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.3000 | 2.3238 | 0.4701 | 0.2023 | 2.62e-23 | 79.082 |
| 57 | 0.1800 | 1.9043 | 0.2167 | 0.1138 | 1.12e-23 | 139.921 |
| 118 | 0.1200 | 1.7362 | 0.0722 | 0.0416 | 1.18e-22 | 272.836 |

## Approximate-H Diagnostic

Source tables: `paper_tables/approx_h_diagnostic_30_bus.csv`,
`paper_tables/approx_h_diagnostic_57_bus.csv`,
`paper_tables/approx_h_diagnostic_118_bus.csv`

These rows evaluate saved Q-NPG policies under a 2% detector-Jacobian
perturbation.

| Bus | relH | Mean SDS | Fixed-H Evasion | Approx-H Evasion | Drop Points | Approx Median Chi2/Tau |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.020 | 0.4680 | 1.0000 | 0.0000 | 100.0 | 5.689 |
| 57 | 0.020 | 0.2160 | 1.0000 | 0.9344 | 6.6 | 0.816 |
| 118 | 0.020 | 0.0729 | 1.0000 | 1.0000 | 0.0 | 0.202 |

## Approximate-H Training Ablation

Source table: `paper_tables/approx_h_ablation_30_bus.csv`

This was run with the wired Q-NPG attacker stack:
`environments.grid_env.FDIAGridEnv`, `models.vqc_policy.VQCPolicy`, and
`training.qnpg_trainer.QNPGTrainer`.

| Bus | relH | mu | Seed | Updates | Mean SDS | Approx-H Evasion | Drop Points | Phys Trace |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.020 | 0.000 | 0 | 80 | 0.0211 | 1.0000 | 0.0 | 0.0087 |
| 30 | 0.020 | 1.000 | 0 | 80 | 0.0268 | 1.0000 | 0.0 | 0.0061 |

Current interpretation: this ablation is not paper-ready evidence for a
robust-direction claim. The trained SDS values are much smaller than the saved
learned-policy SDS for the 30-bus case, so the run is best treated as a harness
smoke test plus a weak trace-reduction signal.

## Scaled Ablation Check

Source table: `paper_tables/approx_h_ablation_30_bus_scaled.csv`

At roughly comparable SDS near 0.08, the scaled check is suggestive but not
strictly matched:

| mu | Scale | Mean SDS | Fixed-H Evasion | Approx-H Evasion | Approx Median Chi2/Tau |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 4.0 | 0.0846 | 0.0156 | 0.0031 | 1.5034 |
| 1.000 | 3.0 | 0.0805 | 0.5312 | 0.1594 | 1.1584 |

This should not be written as a final C1 win until a matched-SDS evaluation is
run at fixed target SDS levels.

## Provenance Checks

- Local and Hellbender `scripts/approx_h_ablation.py` SHA-256:
  `7f19ddd62b6ffdea55c178c06220ec37abc3076d139f4f2aa287a05ada6b9194`
- Local and Hellbender ablation CSV SHA-256:
  `bb9584cd3125ce282a3ba77704da3d5ee055e31e32844505910b6ef75b1c9023`
- Hellbender ablation job `14687036`: completed with exit code `0:0`.

## Related Hellbender Directories

- `/home/mkfqm/qfdia_rl`: older original working directory and data/output source.
- `/home/mkfqm/qfdia_rl_ablation`: isolated ablation run directory copied from the older work.
- `/home/mkfqm/qfdia_rl_git`: older dirty git checkout, left untouched.
- `/home/mkfqm/qfdia_rl_ondemand`: clean code-only upload for OnDemand use.
