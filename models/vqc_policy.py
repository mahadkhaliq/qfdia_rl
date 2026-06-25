"""
models/vqc_policy.py
===================
The quantum actor-critic policy for Q-NPG-FDIA.

Architecture (kept deliberately simple so the natural-gradient math is clean):

  obs (m+3) --[W_enc]--> angles (n_qubits)            classical data encoder
            --AngleEmbedding + StronglyEntanglingLayers(theta_q)-->
            --[expval(Z_i)]--> e (n_qubits)            the quantum policy core
            --[W_act, b_act]--> mu = eps * tanh(...)    Gaussian action mean (m)
   log_std : learnable, state-independent vector (m)
   critic  : small MLP on obs -> scalar value (for GAE)

Only `theta_q` (the variational ansatz parameters) receive the *quantum natural
gradient* (QFIM preconditioning) -- this is the genuinely quantum part and the
basis of the Quantum Natural Gradient of Stokes et al. (2020). The data-encoder,
heads and critic are trained with Adam.

All trainable tensors are pennylane.numpy arrays so that qml.grad / qml.jacobian
/ qml.metric_tensor differentiate through the circuit automatically. The default
simulator uses diff_method="backprop" (fast); lightning.qubit is used if present.
"""
from __future__ import annotations
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def _make_device(n_qubits: int, device_name: str):
    for name in ([device_name] if device_name else []) + ["lightning.qubit", "default.qubit"]:
        try:
            return qml.device(name, wires=n_qubits), name
        except Exception:
            continue
    return qml.device("default.qubit", wires=n_qubits), "default.qubit"


class VQCPolicy:
    def __init__(self, obs_dim, action_dim, n_qubits, n_layers, epsilon,
                 init_log_std=-1.0, device_name="default.qubit", seed=0):
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.epsilon = float(epsilon)
        rng = np.random.default_rng(seed)

        self.dev, self.device_name = _make_device(n_qubits, device_name)
        # backprop is exact + fast on simulators; lightning falls back to adjoint
        diff = "backprop" if self.device_name == "default.qubit" else "adjoint"

        @qml.qnode(self.dev, interface="autograd", diff_method=diff)
        def circuit(angles, theta):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(theta, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        th_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.theta_shape = th_shape
        self.d_theta = int(np.prod(th_shape))

        hidden = 64
        self.params = {
            "W_enc": pnp.array(rng.normal(0, 1 / np.sqrt(obs_dim), (n_qubits, obs_dim)),
                               requires_grad=True),
            "theta_q": pnp.array(rng.uniform(-np.pi / 4, np.pi / 4, th_shape),
                                 requires_grad=True),
            "W_act": pnp.array(rng.normal(0, 0.02, (action_dim, n_qubits)), requires_grad=True),
            "b_act": pnp.array(np.zeros(action_dim), requires_grad=True),
            "log_std": pnp.array(np.full(action_dim, init_log_std), requires_grad=True),
            # critic MLP
            "Wv1": pnp.array(rng.normal(0, 1 / np.sqrt(obs_dim), (hidden, obs_dim)),
                             requires_grad=True),
            "bv1": pnp.array(np.zeros(hidden), requires_grad=True),
            "Wv2": pnp.array(rng.normal(0, 1 / np.sqrt(hidden), (1, hidden)), requires_grad=True),
            "bv2": pnp.array(np.zeros(1), requires_grad=True),
        }
        # which params get Adam (everything except the quantum ansatz)
        self.classical_keys = ["W_enc", "W_act", "b_act", "log_std", "Wv1", "bv1", "Wv2", "bv2"]

    # ----------------------------------------------------- differentiable cores
    def angles(self, params, obs):
        return pnp.tanh(params["W_enc"] @ obs) * np.pi

    def mean(self, params, obs):
        e = pnp.stack(self.circuit(self.angles(params, obs), params["theta_q"]))
        raw = params["W_act"] @ e + params["b_act"]
        return self.epsilon * pnp.tanh(raw)

    def value(self, params, obs):
        h = pnp.tanh(params["Wv1"] @ obs + params["bv1"])
        return (params["Wv2"] @ h + params["bv2"])[0]

    # ---- vectorised forms used by the trainer's batched loss (architecture-specific) ----
    def mean_batch(self, params, obs_b):
        angles = pnp.tanh(obs_b @ params["W_enc"].T) * np.pi                  # (B, q)
        e = pnp.stack(self.circuit(angles, params["theta_q"]), axis=-1)       # (B, q)
        return self.epsilon * pnp.tanh(e @ params["W_act"].T + params["b_act"])  # (B, A)

    def value_batch(self, params, obs_b):
        hv = pnp.tanh(obs_b @ params["Wv1"].T + params["bv1"])
        return (hv @ params["Wv2"].T + params["bv2"])[:, 0]

    def std(self, params):
        # std is a fraction of the action bound epsilon (prevents saturation/bang-bang)
        return self.epsilon * pnp.exp(params["log_std"])

    # ------------------------------------------------------------- sampling API
    def act(self, obs, deterministic=False):
        obs = pnp.array(obs, requires_grad=False)
        mu = np.asarray(self.mean(self.params, obs))
        if deterministic:
            return mu, 0.0, float(np.asarray(self.value(self.params, obs)))
        std = np.asarray(self.std(self.params))
        a = mu + std * np.random.default_rng().normal(size=mu.shape)
        logp = float(np.sum(-0.5 * ((a - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
        return a, logp, float(np.asarray(self.value(self.params, obs)))

    def get_value(self, obs):
        return float(np.asarray(self.value(self.params, pnp.array(obs, requires_grad=False))))

    # --------------------------------------------------------- save / load
    def state_dict(self):
        return {k: np.asarray(v) for k, v in self.params.items()}

    def load_state_dict(self, sd):
        for k in self.params:
            if k in sd:
                self.params[k] = pnp.array(sd[k], requires_grad=True)


class ClassicalGaussianPolicy(VQCPolicy):
    """Drop-in classical MLP baseline (same interface) for ablations / no-PennyLane runs."""
    def __init__(self, obs_dim, action_dim, n_qubits, n_layers, epsilon,
                 init_log_std=-1.0, device_name="cpu", seed=0, hidden=128):
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.epsilon = float(epsilon)
        self.n_qubits, self.n_layers, self.d_theta = n_qubits, n_layers, 0
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": pnp.array(rng.normal(0, 1 / np.sqrt(obs_dim), (hidden, obs_dim)), requires_grad=True),
            "b1": pnp.array(np.zeros(hidden), requires_grad=True),
            # head init scaled so the INITIAL attack magnitude matches the VQC policy
            # (both start near a=0 / stealthy); otherwise the wider MLP fan-in starts detected
            "W2": pnp.array(rng.normal(0, 0.004, (action_dim, hidden)), requires_grad=True),
            "b2": pnp.array(np.zeros(action_dim), requires_grad=True),
            "log_std": pnp.array(np.full(action_dim, init_log_std), requires_grad=True),
            "Wv1": pnp.array(rng.normal(0, 1 / np.sqrt(obs_dim), (hidden, obs_dim)), requires_grad=True),
            "bv1": pnp.array(np.zeros(hidden), requires_grad=True),
            "Wv2": pnp.array(rng.normal(0, 1 / np.sqrt(hidden), (1, hidden)), requires_grad=True),
            "bv2": pnp.array(np.zeros(1), requires_grad=True),
        }
        self.classical_keys = list(self.params.keys())  # everything is Adam; no natural grad
        self.theta_shape = (0,)

    def mean(self, params, obs):
        h = pnp.tanh(params["W1"] @ obs + params["b1"])
        return self.epsilon * pnp.tanh(params["W2"] @ h + params["b2"])

    def mean_batch(self, params, obs_b):
        h = pnp.tanh(obs_b @ params["W1"].T + params["b1"])
        return self.epsilon * pnp.tanh(h @ params["W2"].T + params["b2"])
