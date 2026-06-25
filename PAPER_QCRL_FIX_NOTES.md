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

## QGrid-Synth Attack Labels

There are five non-learned baseline attack families in the current QGrid-Synth generator:

`ramp/step`, `random`, `multiplicative`, `coordinated_sparse`, and `liu_stealthy`.

When a trained Q-NPG policy is provided, `rl_learned` is appended as the learned attack class. So the paper can say five baseline families, or six attack labels if the learned Q-NPG class is counted.

## Analytical SDS Ceiling

Generated with:

```bash
/opt/anaconda3/envs/qfdia/bin/python scripts/max_sds_ceiling.py --all --restarts 40 --iters 25 --csv-out paper_tables/sds_ceiling_ratios.csv
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
