"""
training/qnpg_trainer.py
=======================
Q-NPG-FDIA: Quantum Natural Policy Gradient for FDIA  (Algorithm 4.1).

This is the core of Mahad's task. One update cycle:

  [1] ROLLOUT      collect transitions under the current policy
  [2] GRADIENT     GAE advantages -> policy-gradient estimate (autograd)
  [3] QFIM         F_FDIA = F_Q + mu * J_phys^T S J_phys + delta I     (Def. 4.1)
                     F_Q   = block-diagonal Fubini-Study metric (the QFIM),
                             averaged over visited states
                     J_phys= d(attack mean)/d(theta_q) -> physics term shrinks
                             steps that push the attack OFF the col(H) manifold
  [4] NAT. STEP    solve F_FDIA . dtheta = grad ; KL-trust-region scaling
  [5] CLASSICAL    Adam step for encoder / heads / critic

Only the variational ansatz parameters theta_q receive the natural gradient;
this is the genuine Quantum Natural Gradient (Stokes et al. 2020), here
augmented with the power-flow residual geometry.
"""
from __future__ import annotations
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from autograd import grad as ag_grad   # autograd traverses dict params; qml.grad does not
from autograd import jacobian as ag_jacobian  # version-independent (qml.jacobian kwarg differs across versions)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.act, self.logp = [], [], []
        self.rew, self.val, self.done = [], [], []

    def add(self, o, a, lp, r, v, d):
        self.obs.append(o); self.act.append(a); self.logp.append(lp)
        self.rew.append(r); self.val.append(v); self.done.append(d)

    def __len__(self):
        return len(self.obs)

    def compute_gae(self, last_value, gamma, lam):
        n = len(self)
        rew = np.asarray(self.rew); val = np.asarray(self.val + [last_value]); done = np.asarray(self.done)
        adv = np.zeros(n); gae = 0.0
        for t in reversed(range(n)):
            mask = 1.0 - float(done[t])
            delta = rew[t] + gamma * val[t + 1] * mask - val[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
        ret = adv + val[:n]
        return adv, ret


class QNPGTrainer:
    def __init__(self, env, policy, tc):
        self.env, self.policy, self.tc = env, policy, tc
        self.S = env.S                      # residual-sensitivity (m, m), PSD projector
        self._adam = {}                     # Adam moments for classical params
        self._t_adam = 0

    # ----------------------------------------------------------- rollout
    def collect(self):
        buf = RolloutBuffer()
        obs, _ = self.env.reset()
        ep_returns, ep_stealth, ep_sdev, ep_success, ep_len = [], [], [], [], []
        running = 0.0; last_info = {}
        while len(buf) < self.tc.steps_per_update:
            a, lp, v = self.policy.act(obs, deterministic=False)
            nobs, r, done, _, info = self.env.step(a)
            buf.add(obs, a, lp, r, v, done)
            running += r; last_info = info; obs = nobs
            if done:
                ep_returns.append(running)
                ep_stealth.append(info["stealth_score"]); ep_sdev.append(info["state_dev"])
                ep_success.append(1.0 if (info["stealthy"] and info["state_dev"] > self.tc.min_impact) else 0.0)
                running = 0.0
                obs, _ = self.env.reset()
        last_value = self.policy.get_value(obs)
        adv, ret = buf.compute_gae(last_value, self.tc.gamma, self.tc.gae_lambda)
        stats = {
            "ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "stealth": float(np.mean(ep_stealth)) if ep_stealth else 0.0,
            "state_dev": float(np.mean(ep_sdev)) if ep_sdev else 0.0,
            "asr": float(np.mean(ep_success)) if ep_success else 0.0,
            "n_eps": len(ep_returns),
        }
        return buf, adv, ret, stats

    # ----------------------------------------------------------- loss (autograd)
    def _loss_fn(self, obs_b, act_b, adv_b, ret_b):
        pol, tc = self.policy, self.tc
        eps = pol.epsilon

        def loss(params):
            mu = pol.mean_batch(params, obs_b)                             # (B, A)
            log_std = pnp.clip(params["log_std"], tc.min_log_std, tc.max_log_std)
            std = eps * pnp.exp(log_std)                                    # fraction of epsilon
            logp = pnp.sum(-0.5 * ((act_b - mu) / std) ** 2 - pnp.log(std)
                           - 0.5 * np.log(2 * np.pi), axis=1)               # (B,)
            policy_loss = -pnp.mean(logp * adv_b)
            v = pol.value_batch(params, obs_b)                             # (B,)
            value_loss = pnp.mean((v - ret_b) ** 2)
            entropy = pnp.sum(pnp.log(std)) + 0.5 * pol.action_dim * np.log(2 * np.pi * np.e)
            return policy_loss + tc.value_coef * value_loss - tc.entropy_coef * entropy
        return loss

    # ----------------------------------------------------------- QFIM + physics
    def _quantum_fisher(self, params, obs_sub):
        """Block-diagonal Fubini-Study metric (QFIM) averaged over states."""
        d = self.policy.d_theta
        F = np.zeros((d, d))
        try:
            mt_fn = qml.metric_tensor(self.policy.circuit, argnums=1, approx="block-diag")
        except TypeError:                       # older/newer PennyLane uses 'argnum'
            mt_fn = qml.metric_tensor(self.policy.circuit, argnum=1, approx="block-diag")
        cnt = 0
        for o in obs_sub:
            ang = pnp.array(np.asarray(pnp.tanh(params["W_enc"] @ o) * np.pi), requires_grad=False)
            mt = np.asarray(mt_fn(ang, params["theta_q"])).reshape(d, d)
            F += mt; cnt += 1
        return F / max(cnt, 1)

    def _physics_fisher(self, params, obs_sub):
        """J_phys^T S J_phys with J_phys = d(mu)/d(theta_q) at the mean state."""
        pol = self.policy; d = pol.d_theta
        ang_mean = np.mean([np.asarray(pnp.tanh(params["W_enc"] @ o) * np.pi) for o in obs_sub], axis=0)
        ang_mean = pnp.array(ang_mean, requires_grad=False)

        def circ_vec(a, t):
            return pnp.stack(pol.circuit(a, t))
        dE = np.asarray(ag_jacobian(circ_vec, argnum=1)(ang_mean, params["theta_q"])).reshape(pol.n_qubits, d)
        e = np.asarray(circ_vec(ang_mean, params["theta_q"]))
        W_act = np.asarray(params["W_act"]); b_act = np.asarray(params["b_act"])
        raw = W_act @ e + b_act
        sech2 = 1.0 - np.tanh(raw) ** 2                          # (A,)
        dmu = (pol.epsilon * sech2)[:, None] * (W_act @ dE)      # (A, d)
        return dmu.T @ self.S @ dmu                              # (d, d), PSD

    # ----------------------------------------------------------- one update
    def update(self, buf, adv, ret):
        tc, pol = self.tc, self.policy
        obs = np.asarray(buf.obs); act = np.asarray(buf.act)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # subsample transitions for the (autograd) gradient
        n = len(buf); gb = min(tc.grad_batch, n)
        gi = np.random.default_rng(self._t_adam).choice(n, gb, replace=False)
        obs_b = pnp.array(obs[gi], requires_grad=False)
        act_b = pnp.array(act[gi], requires_grad=False)
        adv_b = pnp.array(adv[gi], requires_grad=False)
        ret_b = pnp.array(ret[gi], requires_grad=False)

        loss = self._loss_fn(obs_b, act_b, adv_b, ret_b)
        grads = ag_grad(loss)(pol.params)                        # dict of grads (same keys)

        info = {}
        if pol.d_theta > 0:                                      # quantum ansatz present
            qsub = obs[np.random.default_rng(self._t_adam + 1).choice(n, min(tc.qfim_batch, n), replace=False)]
            F_q = self._quantum_fisher(pol.params, qsub)
            F_p = self._physics_fisher(pol.params, qsub)
            F = F_q + tc.mu_phys * F_p + tc.damping * np.eye(pol.d_theta)
            g = np.asarray(grads["theta_q"]).reshape(-1)
            nat = np.linalg.solve(F, g)                          # F^{-1} grad
            gFg = float(g @ nat)
            alpha = np.sqrt(2.0 * tc.kl_trust / (abs(gFg) + 1e-10))
            step = tc.nat_lr * min(alpha, 1.0 / (tc.nat_lr + 1e-9))
            pol.params["theta_q"] = pol.params["theta_q"] - pnp.array(
                (step * nat).reshape(pol.theta_shape))
            info.update({"natgrad_norm": float(np.linalg.norm(nat)), "kl_step": float(step),
                         "qfim_trace": float(np.trace(F_q)), "phys_trace": float(np.trace(F_p))})

        # ---- Adam for classical params (encoder / heads / critic) ----
        self._t_adam += 1; t = self._t_adam
        b1, b2, eps = 0.9, 0.999, 1e-8
        flat = []
        for k in pol.classical_keys:
            flat.append(np.asarray(grads[k]).reshape(-1))
        gnorm = np.linalg.norm(np.concatenate(flat)) if flat else 0.0
        clip = min(1.0, tc.max_grad_norm / (gnorm + 1e-8))
        for k in pol.classical_keys:
            gk = np.asarray(grads[k]) * clip
            m = self._adam.setdefault(k + "_m", np.zeros_like(gk))
            v = self._adam.setdefault(k + "_v", np.zeros_like(gk))
            m = b1 * m + (1 - b1) * gk
            v = b2 * v + (1 - b2) * gk ** 2
            self._adam[k + "_m"], self._adam[k + "_v"] = m, v
            mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
            upd = tc.adam_lr * mhat / (np.sqrt(vhat) + eps)
            pol.params[k] = pol.params[k] - pnp.array(upd)
        info["grad_norm"] = float(gnorm)
        return info

    # ----------------------------------------------------------- full training
    def train(self, callback=None):
        history = []
        for u in range(self.tc.total_updates):
            buf, adv, ret, stats = self.collect()
            uinfo = self.update(buf, adv, ret)
            rec = {"update": u + 1, **stats, **uinfo}
            history.append(rec)
            if callback:
                callback(rec)
        return history
