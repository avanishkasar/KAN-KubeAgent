# agents/ — LangGraph Agent Layer

Implements the Metrics Watcher / Loss-Curve Analyst / Cost Estimator /
Supervisor agents from `research/proposal/methodology_draft.md` Section 3,
wired through `kan_gate.KANGate` per Section 4.

## Key rule: the KAN gate is authoritative

The Supervisor's LLM call (`agents/llm.py::propose_action`) produces a
*proposal* only. `agents/graph.py`'s `kan_gate_node` always scores the same
features independently, and `execute_node` acts on **the gate's decision**,
not the LLM's — even when they disagree. This is deliberate: see
`LEARNING_GUIDE.md` Section 3C for why the LLM never gets the final say.

## Running it

```bash
pip install -r kan_gate/requirements.txt -r agents/requirements.txt

# Train a gate checkpoint first (see kan_gate/README notes in train.py)
python -m kan_gate.train --synthetic --steps 250

# Run the full agent loop against a mock TrainJob
python -m agents.run_loop --epochs 30 --plateau-at-epoch 12 --check-every 5
```

Set `ANTHROPIC_API_KEY` to use real LLM reasoning for the Supervisor's
proposal; without it, `agents/llm.py` falls back to the same rule-based
reference policy used to label the gate's training data (clearly marked
`[heuristic fallback]` in the audit log) so the loop still runs end-to-end
for local development.

## Swapping in a real TrainJob

`agents/trainjob_client.py` defines the `TrainJobClient` interface that
`agents/graph.py` depends on. `agents/kubeflow_client.py::KubeflowTrainJobClient`
implements it against a real Kubeflow `TrainJob` (see `k8s/README.md` for
cluster setup) - `agents/graph.py` does not change when you swap it in for
`MockTrainJobClient`.

**Status:** `KubeflowTrainJobClient` is written and passes a structural
protocol check (same method signatures as `MockTrainJobClient`), but has
**not** been exercised against a live cluster - the sandbox this repo was
built in blocks `registry.k8s.io`, so Minikube cannot start there (see
`k8s/README.md`). Validate it against your own cluster before relying on
it; the metric-parsing (`_parse_metrics`) and LR-override ConfigMap
handshake with `training/fashion_mnist_cnn.py` are the most likely spots
to need small fixes against a real Trainer install.
