"""Run the agent loop against a mock TrainJob end-to-end.

This is the fastest way to see the whole pipeline work (Metrics Watcher ->
Loss-Curve Analyst/Cost Estimator -> Supervisor -> KAN Gate -> Executor)
without a real cluster or a trained KAN gate checkpoint. Once Minikube +
Kubeflow Trainer are set up (task 7), swap MockTrainJobClient for the real
Kubeflow client - agents/graph.py does not change.
"""
from __future__ import annotations

import argparse

from agents.graph import build_graph
from agents.trainjob_client import MockTrainJobClient
from kan_gate.gate import DEFAULT_CKPT_PATH, KANGate


def load_or_build_gate() -> KANGate:
    try:
        return KANGate.load(DEFAULT_CKPT_PATH)
    except Exception:
        print("[run_loop] No trained KAN gate checkpoint found - run "
              "`python -m kan_gate.train --synthetic` first for a trained gate. "
              "Falling back to an untrained gate for this demo run.")
        return KANGate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default="finetune-demo-01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--plateau-at-epoch", type=int, default=12)
    parser.add_argument("--check-every", type=int, default=5,
                         help="Run the agent loop every N epochs.")
    args = parser.parse_args()

    client = MockTrainJobClient()
    client.create(args.job_name, total_epochs=args.epochs,
                   plateau_at_epoch=args.plateau_at_epoch, seed=7)
    gate = load_or_build_gate()
    graph = build_graph(client, gate)

    state = {"trainjob_name": args.job_name, "audit_log": []}
    running = True
    while running:
        running = client.step(args.job_name)
        status = client.get_status(args.job_name)
        if status.epoch % args.check_every != 0 and running:
            continue

        state = graph.invoke(state)
        entry = state["audit_log"][-1]
        print(
            f"[epoch {status.epoch:>2}] decision={entry['decision']:<11} "
            f"score={entry['gate_score']:5.1f} outcome={entry['outcome']}\n"
            f"           formula: {entry['gate_formula']}\n"
            f"           llm proposed: {entry['llm_proposal']} - {entry['llm_rationale']}"
        )

        if status.phase != "Running":
            print(f"[run_loop] Job '{args.job_name}' ended: {status.phase}")
            break

    print(f"\n[run_loop] Full audit log has {len(state['audit_log'])} entries.")


if __name__ == "__main__":
    main()
