#!/usr/bin/env python3
"""
verify_ibm.py
=============
Run a *trained* Q-NPG-FDIA policy's variational circuit on IBM hardware (or a
noisy simulator) and check that the learned attack stays stealthy and
high-impact under real device noise.

This is INFERENCE only. Training stays on the statevector simulator; the
quantum natural gradient needs the QFIM (many circuit evaluations per step over
thousands of updates) and is impractical on a queued backend. Here we load the
trained weights and evaluate the policy circuit on the chosen device.

The circuit and the environment are imported from this package, so the device
run uses the *exact* trained circuit -- there is no hand-ported re-derivation to
get wrong. Only the PennyLane device is swapped.

Workflow (recommended order):
  1. Validate the port is faithful (no convention bugs), noiseless Qiskit vs default.qubit:
       python verify_ibm.py --bus 30 --load outputs/qnpg_30_policy.npz --device aer
     The reported "faithfulness max|e_sim - e_dev|" should be ~1e-3 or less.
  2. See the effect of device noise locally (fake backend):
       python verify_ibm.py --bus 30 --load outputs/qnpg_30_policy.npz --device aer_noisy
  3. Run on real IBM hardware (uses your saved QiskitRuntimeService account):
       python verify_ibm.py --bus 30 --load outputs/qnpg_30_policy.npz --device ibm \
           --ibm-backend ibm_pittsburgh --n-points 16 --shots 4096

Dependencies for the qiskit paths (install in your synthgrad env):
    pip install pennylane-qiskit qiskit-aer qiskit-ibm-runtime
"""
from __future__ import annotations
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pennylane as qml
from config import BUS_CONFIGS
from environments.grid_env import FDIAGridEnv
from models.vqc_policy import VQCPolicy


def build_circuit(dev, n_qubits):
    """The exact policy circuit: AngleEmbedding(Y) + StronglyEntanglingLayers -> <Z_i>."""
    @qml.qnode(dev)
    def circ(angles, theta):
        qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(theta, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circ


def make_device(mode, n_qubits, shots, ibm_backend, fake_backend):
    if mode == "sim":
        return qml.device("default.qubit", wires=n_qubits)

    if mode == "aer":                       # noiseless Qiskit (port-faithfulness check)
        return qml.device("qiskit.aer", wires=n_qubits, shots=shots)

    if mode == "aer_noisy":                 # local emulation of a real backend's noise
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider as fp
        Fake = getattr(fp, fake_backend)
        sim = AerSimulator.from_backend(Fake())
        return qml.device("qiskit.aer", wires=n_qubits, backend=sim, shots=shots)

    if mode == "ibm":                       # real hardware via Qiskit Runtime
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()    # uses your saved account
        backend = (service.backend(ibm_backend) if ibm_backend
                   else service.least_busy(operational=True, simulator=False,
                                           min_num_qubits=n_qubits))
        print(f"  IBM backend: {backend.name}")
        return qml.device("qiskit.remote", wires=n_qubits, backend=backend, shots=shots)

    raise ValueError(f"unknown device mode {mode}")


def attack_from(circ, params, obs, epsilon):
    """Deterministic policy forward: encoder -> circuit -> head -> attack vector."""
    angles = np.tanh(params["W_enc"] @ obs) * np.pi          # classical encoder
    e = np.array(circ(angles, params["theta_q"]), dtype=float)  # quantum readout
    return epsilon * np.tanh(params["W_act"] @ e + params["b_act"])  # classical head


def main():
    ap = argparse.ArgumentParser(description="Verify a trained Q-NPG policy on IBM hardware")
    ap.add_argument("--bus", type=int, default=30, choices=[30, 57, 118])
    ap.add_argument("--load", required=True, help="trained policy .npz")
    ap.add_argument("--device", default="sim", choices=["sim", "aer", "aer_noisy", "ibm"])
    ap.add_argument("--n-points", type=int, default=16, help="operating points to evaluate")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--ibm-backend", default=None, help="e.g. ibm_pittsburgh; else least busy")
    ap.add_argument("--fake-backend", default="FakeKolkataV2", help="for aer_noisy (>= n_qubits)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = BUS_CONFIGS[args.bus]
    env = FDIAGridEnv(args.bus, cfg=cfg, seed=args.seed)
    env.reset()

    # load trained weights into a default.qubit policy (gives encoder/head/theta_q)
    policy = VQCPolicy(env.obs_dim, env.action_dim, cfg.n_qubits, cfg.vqc_layers,
                       cfg.epsilon, device_name="default.qubit", seed=args.seed)
    sd = np.load(args.load)
    policy.load_state_dict({k: sd[k] for k in sd.files})
    params = {k: np.asarray(v, dtype=float) for k, v in policy.params.items()}
    print(f"  loaded {args.load} | bus {args.bus} | {cfg.n_qubits} qubits, "
          f"{cfg.vqc_layers} layers, {policy.d_theta} VQC params")

    sim_circ = build_circuit(qml.device("default.qubit", wires=cfg.n_qubits), cfg.n_qubits)
    dev = make_device(args.device, cfg.n_qubits, args.shots, args.ibm_backend, args.fake_backend)
    dev_circ = build_circuit(dev, cfg.n_qubits)

    # ---- faithfulness check (cheap, skip on real hardware to save shots) ----
    if args.device in ("aer", "aer_noisy"):
        o = env.reset()
        ang = np.tanh(params["W_enc"] @ o) * np.pi
        e_sim = np.array(sim_circ(ang, params["theta_q"]), dtype=float)
        e_dev = np.array(dev_circ(ang, params["theta_q"]), dtype=float)
        print(f"  faithfulness max|e_sim - e_{args.device}| = {np.max(np.abs(e_sim - e_dev)):.4f}"
              + ("   (noiseless: should be ~0)" if args.device == "aer" else "   (noise expected)"))

    # ---- paired evaluation: same operating points on simulator and on the device ----
    res_sim, sds_sim, res_dev, sds_dev, dabs = [], [], [], [], []
    for k in range(args.n_points):
        obs = env.reset()
        a_sim = attack_from(sim_circ, params, obs, cfg.epsilon)
        a_dev = attack_from(dev_circ, params, obs, cfg.epsilon)
        res_sim.append(env.bdd_residual(a_sim)); sds_sim.append(env.state_deviation(a_sim))
        res_dev.append(env.bdd_residual(a_dev)); sds_dev.append(env.state_deviation(a_dev))
        dabs.append(float(np.max(np.abs(a_sim - a_dev))))
        print(f"    point {k+1:>2}/{args.n_points} done", end="\r")

    tau = env.tau_bdd
    def summarise(res, sds):
        res, sds = np.array(res), np.array(sds)
        stealth = np.clip(1 - res / tau, 0, None)
        return stealth.mean(), sds.mean(), float((res >= tau).mean())

    st_s, sd_s, fl_s = summarise(res_sim, sds_sim)
    st_d, sd_d, fl_d = summarise(res_dev, sds_dev)

    print(f"\n\n  IEEE {args.bus}-bus | {args.n_points} operating points | tau = {tau:.1f} | device = {args.device}")
    print(f"  {'':<12}{'stealth':>10}{'SDS':>10}{'flagged':>10}")
    print(f"  {'simulator':<12}{st_s:>10.3f}{sd_s:>10.3f}{fl_s:>10.3f}")
    print(f"  {args.device:<12}{st_d:>10.3f}{sd_d:>10.3f}{fl_d:>10.3f}")
    print(f"  mean |a_sim - a_device| per attack: {np.mean(dabs):.4f}")

    # ---- verdict ----
    impact_kept = sd_d > 0.5 * sd_s
    still_stealthy = fl_d < 0.10
    if still_stealthy and impact_kept:
        print("\n  VERDICT: the learned attack survives this device -- stealthy and impactful on hardware.")
    elif still_stealthy:
        print("\n  VERDICT: still stealthy, but device noise reduced impact. Consider more shots or error mitigation.")
    else:
        print("\n  VERDICT: device noise pushes attacks above the BDD threshold. Apply error mitigation (DD+TREX) or increase shots.")


if __name__ == "__main__":
    main()
