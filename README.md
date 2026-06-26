# Q-NPG-FDIA — Quantum Natural Policy Gradient for FDIA generation


This package implements **Algorithm 4.1 (Q-NPG-FDIA)** from the Theoretical
Framework: a variational quantum policy that *learns to generate stealthy,
physics-consistent False Data Injection Attacks* on AC state estimation, trained
with the **physics-augmented Quantum Natural Gradient**.

It is the foundation the rest of the team builds on:
* **PC-QPO** (Srikar) wraps this natural-gradient step inside a Lagrangian primal-dual loop.
* **Q-ASP** (Varshith) drops this attacker into a self-play game against a detector.

So the deliverable here is the **attacker / generator** and the shared grid
environment. The detector (QB-DETECT / QLSTM) and the other two algorithms are
separate tasks; clean hooks are left for them (the env accepts a `detector_fn`
and the reward already has the `R_adv` term, switched off in Phase 1).

---

## 1. What the code actually does (the physics)

We model **AC state estimation** on an IEEE system (PandaPower):

```
state         x = [theta_1..theta_n, |V|_1..|V|_n]          (2n unknowns)
measurement   z = [P_inj, Q_inj, |V|, theta]                 (m = 4n)
model         z = h(x),   H = dh/dx   (measurement Jacobian, from Ybus)
```

**Bad-Data Detection (Liu–Ning–Reiter, CCS 2009).** The linearised BDD residual
of an attack `a` is `r(a) = S a`, where `S = I − H Hᵀ⁺` projects onto the space
*orthogonal* to `col(H)`. So:

* an attack `a = H c` (the column space of `H`) gives `r = 0` → **perfectly stealthy**;
* the chi-squared detector fires when `‖r‖² / σ² > τ = chi2.ppf(0.95, m − 2n)`.

The induced **state-estimate deviation** is `Δx̂ = H⁺ a` (the attack's *impact*).
Because a stealthy attack `a = H c` yields `Δx̂ ≈ c`, maximising impact while
staying stealthy pushes the policy toward attacks living in `col(H)` — exactly
the structure the RL agent has to discover.

**Compound reward** (Section 12 of the framework): impact `+ ‖Δx̂‖`, a smooth
distance-linear stealth penalty `− ‖r‖/σ`, a power-flow consistency term, and a
stealth bonus that makes the BDD-passing region a clear reward peak.

---

## 2. The algorithm (your contribution)

Standard policy gradient treats parameter space as Euclidean. **Q-NPG-FDIA**
preconditions the update by the **physics-augmented Quantum Fisher Information
Matrix** (Definition 4.1):

```
F_FDIA(θ) = F_Q(θ)  +  μ · J_physᵀ S J_phys  +  δ I
θ ← θ − α · F_FDIA⁻¹ ∇J          (α set by a KL trust region)
```

* `F_Q` — the **block-diagonal Fubini–Study metric** (the QFIM) of the variational
  circuit, averaged over visited states. This is the genuine *Quantum Natural
  Gradient* (Stokes et al. 2020). Computed with `qml.metric_tensor`.
* `J_phys = ∂(attack)/∂θ` — the physics term **shrinks steps that push the attack
  off the `col(H)` manifold**, i.e. a built-in barrier against physics-inconsistent
  (detectable) attacks.
* Only the variational ansatz parameters `θ_q` receive the natural gradient; the
  classical encoder / action head / critic use Adam.

`F_Q`, `J_phys`, and the policy gradient are all computed by automatic
differentiation through the PennyLane circuit (no parameter-shift loop needed on
the simulator; `diff_method="backprop"`).

---

## 3. Install & run

No GPU and **no PyTorch** required — the whole thing is PennyLane + NumPy.

```bash
conda create -n qfdia python=3.11 && conda activate qfdia
pip install -r requirements.txt
# optional, big pandapower speedup on HPC:  pip install numba

# ~10-minute smoke test on a laptop / login node:
python main.py --bus 30 --quick

# full runs:
python main.py --bus 30  --updates 200
python main.py --bus 57  --updates 200
python main.py --bus 118 --updates 150 --device lightning.qubit

# classical MLP policy baseline (same pipeline, for ablation):
python main.py --bus 30 --classical
```

### On Hellbender

```bash
sbatch run_hellbender.sh
```

`run_hellbender.sh` creates a dedicated `qfdia` conda env (kept separate from your
Qiskit `synthgrad` env), installs the stack, and runs all three bus systems. Edit
the `--partition` line to match your allocation. CPU is fine for 30/57-bus; the
118-bus case is heavier (8-qubit circuits) and benefits from more cores or the
`lightning.gpu` backend.

### Outputs (in `outputs/`)

| file | contents |
|---|---|
| `qnpg_<bus>_history.csv` | per-update training curve (reward, ASR, stealth, SDS, QFIM trace, KL step) |
| `qnpg_<bus>_results.csv` | final attack-generation table (Q-NPG vs baselines) — **paper Table I** |
| `qnpg_<bus>_policy.npz`  | trained policy parameters |

---

## 4. What you get (illustrative 30-bus, short run)

```
method                 n      ASR   evasion   mean_SDS   stealth
----------------------------------------------------------------
Q-NPG-FDIA            24    0.875     1.000     0.0274     0.868
liu_stealthy          24    0.292     1.000     0.0171     1.000
random                24    0.542     1.000     0.0223     0.474
step                  24    0.000     0.000     0.0475     0.000
```

Read this honestly: **the learned quantum policy keeps 100% BDD evasion while
producing higher-impact, higher-success stealthy attacks than the random
Liu–Ning–Reiter construction.** That is the claim — *parameter-efficient,
physics-consistent, topology-aware stealthy attack generation* — **not**
"quantum beats classical." (SDS was still climbing at the end of the short run;
the full 200-update run pushes it toward the ~0.054 stealthy ceiling.)

---

## 5. Honest design notes (read before the meeting)

* **Linearised measurement model.** `H` is built once at the base operating point
  from `Ybus` and reused (the env drifts the operating point only for observation
  variety). This is the standard DC/linearised-AC simplification. Swapping in a
  per-step AC re-solve of `H` is the obvious next step for full AC fidelity.
* **Non-accumulating episodes.** Each step is a full attack on the current
  snapshot (not an increment). With accumulation, per-step exploration noise
  random-walks off the (thin) stealthy manifold within an episode and the agent
  cannot learn. This was the single biggest thing to get right.
* **Exploration is sized to the stealthy slab** (`init_log_std`), so most
  exploration stays near-stealthy and the natural gradient has a usable signal.
* **The physics term `J_physᵀ S J_phys` is small once the policy is near `col(H)`**
  (little off-manifold component left to penalise). It does the most work early /
  off-manifold and acts as a consistency regulariser. It would matter much more in
  a transfer setting where the attacker's `H` is approximate.
* **Reference check.** The "GenAI-FDIA, arXiv:2605.18873" citation that appears in
  the framework docs **does not resolve** to a real arXiv paper — treat it as
  unverified. A real, citable physics-informed AC-FDIA substitute is Zhao et al.,
  *Controllable Blind AC FDIA via Physics-Informed Extrapolative AVAE*, **Sensors
  25(3):943, 2025** (already in the doc's reference list as [N8]). GridSTAGE,
  Morris/UAH, and Liu–Ning–Reiter all check out.

---

## 6. File map

```
qfdia_rl/
├── config.py                      BusConfig (30/57/118) + TrainingConfig
├── main.py                        pipeline: train -> evaluate -> save
├── requirements.txt
├── run_hellbender.sh              SLURM batch script
├── environments/
│   ├── grid_env.py                AC SE env: H-matrix, BDD, col(H) stealth, reward
│   └── attack_types.py            rule-based + Liu-stealthy baselines
├── models/
│   └── vqc_policy.py              VQC actor-critic (+ classical MLP baseline)
├── training/
│   └── qnpg_trainer.py            Q-NPG: GAE, QFIM, physics term, NG step, Adam
└── evaluation/
    └── metrics.py                 ASR / evasion / SDS / stealth + baseline table
```
