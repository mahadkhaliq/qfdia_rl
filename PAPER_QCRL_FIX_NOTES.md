# QCRL Paper Fix Notes

This file records values extracted from the current code/results so the paper can be edited without inventing numbers.

## Reward And Training Hyperparameters

Reward terms in `environments/grid_env.py` use:

`R = w_impact * impact_scale * SDS - w_stealth * residual_distance - w_physics * residual_L1 - w_adv * detector_flag + stealth_bonus`

| Bus | w_impact | w_stealth | w_physics | w_adv | w_time | stealth_bonus | impact_scale | phys_scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 1.0 | 3.0 | 1.0 | 0.0 | 0.05 | 2.0 | 10.0 | 1.0 |
| 57 | 1.0 | 3.5 | 1.5 | 0.0 | 0.05 | 2.0 | 10.0 | 1.0 |
| 118 | 1.0 | 4.0 | 2.0 | 0.0 | 0.05 | 2.0 | 10.0 | 1.0 |

ASR success criterion: `stealthy == True` and `state_dev > 0.02`.

Policy noise: `sigma_pi` is learned, state-independent, and per-action. It is initialized with `init_log_std = -3.4` and clipped during training to `[-6.0, -2.8]`; the actual action standard deviation is `a_max * exp(log_std)` in the current training instantiation because `main.py` passes `env.a_max` as the policy epsilon.

Other training values: `gamma = 0.99`, `GAE lambda = 0.95`, `nat_lr = 0.30`, `KL trust = 0.02`, `damping = 1e-2`, `mu_phys = 1.0`, `Adam lr = 3e-3`, `value_coef = 0.5`, `entropy_coef = 1e-3`.

## Q-NPG Architecture And Parameter Counts

Observation dimension is `m + 3`, where `m = 4n` meter channels and the extra three entries are residual-history features. Action dimension is `m`.

The critic is a one-hidden-layer MLP: `obs_dim -> 64 -> 1` with tanh activation.

| Bus | obs_dim | action_dim | qubits | VQC layers | variational params | critic params | encoder params | action-head params | log_std params | total trainable params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 123 | 120 | 4 | 3 | 36 | 8001 | 492 | 600 | 120 | 9249 |
| 57 | 231 | 228 | 6 | 4 | 72 | 14913 | 1386 | 1596 | 228 | 18195 |
| 118 | 475 | 472 | 8 | 4 | 96 | 30529 | 3800 | 4248 | 472 | 39145 |

## IBM Quantum Hardware Smoke Verification

Generated from `verify_ibm.py` hardware jobs on Hellbender/IBM Runtime and summarized in `QUANTUM_VERIFICATION_RESULTS.md` and `paper_tables/quantum_verification_results.csv`.

| Bus | Backend | Qubits | VQC layers | Points | Shots | Device stealth | Device SDS | Flagged | Mean attack delta | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | ibm_miami | 4 | 3 | 4 | 1024 | 0.9942 | 0.0691 | 0.0000 | 0.0018 | survives_device |
| 57 | ibm_miami | 6 | 4 | 2 | 1024 | 0.9984 | 0.0307 | 0.0000 | 0.0018 | survives_device |
| 118 | ibm_miami | 8 | 4 | 1 | 1024 | 0.9987 | 0.0110 | 0.0000 | 0.0005 | survives_device |

Interpretation: these are small real-hardware smoke checks, not full statistical hardware validation. They support the limited claim that the trained Q-NPG VQC actor can be executed through IBM hardware on the 30-, 57-, and 118-bus policies while preserving BDD stealth and nonzero attack impact at the sampled operating points.

## QGrid-Synth Attack Labels

There are five non-learned baseline attack families in the current QGrid-Synth generator:

`ramp/step`, `random`, `multiplicative`, `coordinated_sparse`, and `liu_stealthy`.

When a trained Q-NPG policy is provided, `rl_learned` is appended as the learned attack class. So the paper can say five baseline families, or six attack labels if the learned Q-NPG class is counted.

## Analytical SDS Ceiling

Generated with:

```bash
/opt/anaconda3/envs/qfdia/bin/python scripts/max_sds_ceiling.py --all --restarts 40 --iters 25 --noise-check --noise-trials 200 --csv-out paper_tables/sds_ceiling_ratios.csv --robustness-csv-out paper_tables/sds_ceiling_approx_h_robustness.csv
```

| Bus | a_max | SDS ceiling | learned SDS | learned/ceiling | stealth residual | tau_bdd |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.3000 | 2.3238 | 0.4701 | 0.2023 | 2.62e-23 | 79.082 |
| 57 | 0.1800 | 1.9043 | 0.2167 | 0.1138 | 1.12e-23 | 139.921 |
| 118 | 0.1200 | 1.7362 | 0.0722 | 0.0416 | 1.18e-22 | 272.836 |

Paper wording for the linearized-model limitation:

> In the linearized fixed-H regime, the stealthy maximum-SDS problem admits a direct analytical ceiling under the same box constraint. The learned Q-NPG policy reaches 20.2%, 11.4%, and 4.2% of this ceiling on the IEEE 30-, 57-, and 118-bus systems, respectively, while maintaining near-zero stealth residuals. Extending the policy to time-varying or approximate-H regimes, where this closed-form stealth ceiling is unavailable and residual-sensitivity terms become active, is the next step.

Paper wording for the quantum-advantage sentence:

> We treat the quantum policy as a parameter-efficient quantum natural-gradient architecture rather than as a demonstrated quantum advantage claim. A parameter-matched classical natural-policy-gradient policy remains the cleanest follow-up ablation for isolating the role of the variational quantum core.

## C1 Status And Approximate-H Diagnostics

The shared `paper_qcrl2026 (2).tex` already contains the deadline-safe C1 tag in contribution #2:

> We propose and analyze this metric; in the regime studied it remained near-inactive, so its benefit is motivated by construction and left for empirical validation.

The strong C1 diagnostic is implemented in `scripts/approx_h_ablation.py`. It rolls out saved trained policies and evaluates the same attacks under perturbed detector Jacobians. At `relH = 0.02`, with 32 rollouts and 10 perturbations per rollout:

| Bus | mean SDS | fixed-H evasion | approximate-H evasion | evasion drop | approx median chi2/tau |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.4680 | 1.000 | 0.000 | 100.0 points | 5.689 |
| 57 | 0.2160 | 1.000 | 0.934 | 6.6 points | 0.816 |
| 118 | 0.0729 | 1.000 | 1.000 | 0.0 points | 0.202 |

Interpretation: the approximate-H mechanism is clearly active on 30-bus and mildly active on 57-bus at 2% model error, but not active on 118-bus at the learned policy's lower SDS. This supports the honest limitation: the fixed-H reported runs do not validate the residual-sensitivity term, but approximate-H regimes can create the off-manifold pressure that term is designed to handle.

The ceiling-direction robustness sweep is logged in `paper_tables/sds_ceiling_approx_h_robustness.csv`. It shows that full analytical-ceiling attacks become non-evasive at 0.5% H drift on all grids, while learned-magnitude scaled attacks are more robust, especially on 57/118-bus.

## Pending Strong C1/C2 Ablation

The diagnostic tables do not yet prove that the physics term finds a more robust direction; they show that approximate-H pressure exists. The decisive experiment is now wired in `scripts/approx_h_ablation.py --ablate` and should be run through Slurm, not on the Hellbender login node:

```bash
sbatch scripts/run_approx_h_ablation_hellbender.sbatch
```

Default wrapper settings: IEEE 30-bus, `relH=0.02`, `mu in {0,1}`, seeds `0 1 2`, 80 updates, 256 transitions/update, and evaluation scales `0.5 0.75 1.0`. To run a quick single-seed pass:

```bash
SEEDS="0" UPDATES=40 sbatch scripts/run_approx_h_ablation_hellbender.sbatch
```

The result to look for: at matched or very similar SDS under H drift, `mu=1` should retain higher approximate-H evasion than `mu=0`. If it does, write a robustness-result paragraph. If it does not, keep the current contribution tag and report this as a limitation/tradeoff characterization.
