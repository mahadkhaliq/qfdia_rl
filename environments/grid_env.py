"""
environments/grid_env.py
========================
FDIAGridEnv -- the shared TA-QMDP environment for the Q-NPG-FDIA attacker.

Physics
-------
We model AC state estimation on a standard IEEE bus system (PandaPower):

  state         x  = [theta_1..theta_n, |V|_1..|V|_n]            (2n)
  measurement   z  = [P_inj, Q_inj, |V|, theta]                   (m = 4n)
  measurement model  z = h(x),  with  H = dh/dx  (the measurement Jacobian)

The clean operating point x* is obtained from a PandaPower power-flow solve.
h(x) and H are computed from the bus admittance matrix Ybus, so no extra
power-flow solves are needed once Ybus and x* are known (fast, exact to fp).

Bad-data detection (Liu-Ning-Reiter, CCS 2009)
----------------------------------------------
The linearised BDD residual of an attack a is  r(a) = S a,  where
  S = I - H H^+        (residual-sensitivity / hat-matrix complement)
S is an orthogonal projector onto the space *orthogonal* to col(H). Hence an
attack a = H c (column space of H) gives r = 0 -> perfectly stealthy. This is
exactly the Liu-Ning-Reiter stealthiness condition. The chi-squared detector
flags the attack when ||r(a)||^2 > tau = chi2.ppf(0.95, df = m - 2n).

The induced state-estimate deviation is  dx_hat = H^+ a,  so for a stealthy
attack a = H c the operator's estimate shifts by ~c. Maximising impact while
staying stealthy therefore pushes the policy toward attacks living in col(H).

Reward (Section 12 of the Theoretical Framework)
------------------------------------------------
  R = w_impact * ||dx_hat||           (state-estimation deviation)
    - w_stealth * max(0, ||r||^2 - tau)   (BDD evasion)
    - w_physics * ||r||_1               (power-flow / manifold consistency)
    - w_adv * phi * 1[detector fires]    (adversarial; phi=0 in Phase 1)
    + w_time * beta * t                  (survival bonus; longer => stealthier)
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.stats import chi2

warnings.filterwarnings("ignore")  # silence pandapower/numba chatter


def _load_case(bus_size: int):
    import pandapower as pp
    import pandapower.networks as pn
    loader = {30: pn.case30, 57: pn.case57, 118: pn.case118}.get(bus_size)
    if loader is None:
        raise ValueError(f"Unsupported bus size {bus_size}; use 30, 57, or 118.")
    net = loader()
    pp.runpp(net, calculate_voltage_angles=True, init="dc", numba=False)
    if not net.converged:
        raise RuntimeError(f"Base-case power flow did not converge for case{bus_size}.")
    return net


class FDIAGridEnv:
    """A lightweight Gym-style environment (reset/step) for stealthy FDIA generation."""

    def __init__(self, bus_size: int, cfg=None, detector_fn=None,
                 noise_std: float = 1e-2, seed: int = 0):
        from config import BUS_CONFIGS
        self.cfg = cfg or BUS_CONFIGS[bus_size]
        self.bus_size = bus_size
        self.detector_fn = detector_fn          # optional: z_corrupted -> bool (Phase 3)
        self.noise_std = noise_std
        self.sigma = float(noise_std)            # measurement std for BDD normalisation
        self.rng = np.random.default_rng(seed)

        # ---- build the measurement model from the converged base case ----
        net = _load_case(bus_size)
        ppc = net._ppc
        self.Ybus = np.asarray(ppc["internal"]["Ybus"].todense())
        Vm = ppc["bus"][:, 7].astype(float)
        Va = np.deg2rad(ppc["bus"][:, 8].astype(float))
        self.n = self.Ybus.shape[0]
        self.m = 4 * self.n
        self.x_star = np.concatenate([Va, Vm])           # [theta(n), |V|(n)]
        self.z_clean = self._h(self.x_star)
        self.H = self._build_H(self.x_star)              # (m, 2n)
        self.Hpinv = np.linalg.pinv(self.H)              # (2n, m)
        self.S = np.eye(self.m) - self.H @ self.Hpinv    # residual sensitivity (m, m)
        self.S = 0.5 * (self.S + self.S.T)               # symmetrise (numerical)

        # chi-squared BDD threshold; df = #measurements - #states
        self.chi2_df = self.m - 2 * self.n
        self.tau_bdd = float(chi2.ppf(0.95, df=self.chi2_df))

        # spaces
        self.obs_dim = self.m + 3                         # z_current + 3 residual-history
        self.action_dim = self.m                          # attack increment per meter
        self.a_max = self.cfg.a_max_mult * self.cfg.epsilon

        # episode state
        self._a = np.zeros(self.m)
        self._z0 = self.z_clean.copy()
        self._t = 0
        self._res_hist = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ physics
    def _h(self, x: np.ndarray) -> np.ndarray:
        th, vm = x[:self.n], x[self.n:]
        V = vm * np.exp(1j * th)
        S = V * np.conj(self.Ybus @ V)            # complex power injection (p.u.)
        return np.concatenate([S.real, S.imag, vm, th])

    def _build_H(self, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        twon = 2 * self.n
        H = np.zeros((self.m, twon))
        for k in range(twon):
            xp = x0.copy(); xm = x0.copy()
            xp[k] += eps; xm[k] -= eps
            H[:, k] = (self._h(xp) - self._h(xm)) / (2.0 * eps)
        return H

    def residual_vec(self, a: np.ndarray) -> np.ndarray:
        return self.S @ a

    def bdd_residual(self, a: np.ndarray) -> float:
        """Noise-normalised chi-squared statistic  ||S a||^2 / sigma^2.

        The chi-squared threshold tau = chi2.ppf(0.95, df) is calibrated for
        residuals normalised by the measurement standard deviation sigma, so the
        attack residual must be divided by sigma^2 to live on the same scale.
        An attack a = H c (col(H)) gives S a = 0 -> statistic 0 -> always passes.
        """
        r = self.S @ a
        return float(r @ r) / (self.sigma ** 2)

    def state_deviation(self, a: np.ndarray) -> float:
        return float(np.linalg.norm(self.Hpinv @ a))

    def liu_stealthy_attack(self, c: np.ndarray) -> np.ndarray:
        """Exact Liu-Ning-Reiter stealthy attack a = H c (residual == 0 by design)."""
        return self.H @ c

    # ---------------------------------------------------------------- encoding
    def piqe_encode(self, z: np.ndarray):
        """Physics-Informed Quantum Encoding (Def. 3.1): amplitude + residual register.

        Returns (amp_enc, phys_enc, alpha, stealth_score). alpha is the mixing
        angle arctan(||r||/||z||); small alpha => clean, large => corrupt.
        """
        a = z - self._z0
        r = self.S @ a
        nz = np.linalg.norm(z) + 1e-12
        nr = np.linalg.norm(r)
        amp_enc = z / nz
        phys_enc = (self.H @ (self.Hpinv @ z)); phys_enc /= (np.linalg.norm(phys_enc) + 1e-12)
        alpha = float(np.arctan2(nr, nz))
        stealth_score = float(max(0.0, 1.0 - (r @ r) / (self.sigma ** 2) / self.tau_bdd))
        return amp_enc, phys_enc, alpha, stealth_score

    # ------------------------------------------------------------------- gym API
    def _obs(self) -> np.ndarray:
        z_obs = self._z0 + self.rng.normal(0, self.sigma, self.m)
        return np.concatenate([z_obs, np.asarray(self._res_hist)])

    def reset(self):
        # sample an operating point (state jitter); H stays the fixed linearisation point
        self._x = self.x_star + self.rng.normal(0, 0.01, self.x_star.shape)
        self._z0 = self._h(self._x)
        self._a = np.zeros(self.m)
        self._t = 0
        self._res_hist = [0.0, 0.0, 0.0]
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # the action IS the full attack on the current snapshot (NON-accumulating):
        # accumulating would let per-step exploration noise random-walk off the
        # thin stealthy manifold within a single episode.
        a = np.clip(np.asarray(action, dtype=float), -self.a_max, self.a_max)
        self._a = a
        self._t += 1

        r_vec = self.S @ a
        res_norm = float(r_vec @ r_vec) / (self.sigma ** 2)   # chi-squared statistic
        sdev = self.state_deviation(a)                        # ||dx_hat|| (impact)
        dist = float(np.sqrt(r_vec @ r_vec)) / self.sigma     # off-manifold distance (||r||/sigma)
        off_l1 = float(np.sum(np.abs(r_vec))) / self.sigma    # normalised off-manifold L1

        detected = bool(self.detector_fn(self._z0 + a)) if self.detector_fn else False
        stealthy = res_norm < self.tau_bdd

        c = self.cfg
        # smooth, distance-linear penalties: ~ -w at the chi-squared threshold,
        # gradient is constant (never saturates) so the policy is always pushed
        # toward col(H) (the Liu-Ning-Reiter stealthy subspace where dist -> 0).
        # A stealth bonus makes the stealthy region a clear reward peak (reward cliff).
        sqrt_tau = np.sqrt(self.tau_bdd)
        r_impact  = c.w_impact * c.impact_scale * sdev
        r_stealth = -c.w_stealth * (dist / sqrt_tau)
        r_physics = -c.w_physics * c.phys_scale * (off_l1 / self.m)
        r_bonus   = c.stealth_bonus if stealthy else 0.0
        r_adv     = -c.w_adv * (1.0 if detected else 0.0)
        reward = r_impact + r_stealth + r_physics + r_bonus + r_adv

        # drift the operating point for the next step (load fluctuation -> obs variety)
        self._x = self._x + self.rng.normal(0, 0.003, self._x.shape)
        self._z0 = self._h(self._x)
        done = self._t >= self.cfg.horizon

        prox = min(res_norm / self.tau_bdd, 10.0)
        self._res_hist = [self._res_hist[1], self._res_hist[2], float(prox)]

        info = {
            "residual2": res_norm, "state_dev": sdev, "phys_l1": off_l1,
            "stealthy": bool(stealthy), "detected": detected,
            "stealth_score": float(max(0.0, 1.0 - res_norm / self.tau_bdd)),
            "attack": a.copy(),
        }
        return self._obs(), float(reward), bool(done), False, info

    # convenience for evaluation / dataset generation
    def rollout_attack(self, policy, deterministic: bool = True):
        obs, _ = self.reset()
        done = False; last = {}
        while not done:
            action = policy.act(obs, deterministic=deterministic)[0]
            obs, _, done, _, last = self.step(action)
        return last
