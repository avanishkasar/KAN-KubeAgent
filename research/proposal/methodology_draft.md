# Methodology Draft — KAN-KubeAgent

**Version:** 0.2 (Working Draft — Fine-Tuning Job Optimizer direction)
**Last Updated:** September 2026

---

## 1. System Overview

KAN-KubeAgent consists of four integrated components:

```
[1] Observation Layer     →  [2] LLM Agent Core   →  [3] KAN Gating Layer   →  [4] Executor
(Kubeflow TrainJob state)    (LangGraph FSM)          (Continue/Adjust/Stop)    (Patch TrainJob CR)
```

---

## 2. Component 1: Observation & Feature Extraction

### 2.1 Data Sources

- **Kubeflow `TrainJob` status** — current epoch, phase, conditions (via the Trainer Python SDK / K8s API)
- **Training logs / metrics** — loss, gradient norm, learning rate, emitted per step or epoch
- **Cluster resource state** — GPU/CPU utilization of the training pod, elapsed wall-clock time
- **Budget config** — total GPU-hour budget allotted to the job, set at job submission

### 2.2 Feature Engineering for the KAN Gate

For each decision point (e.g. every N epochs), extract:

| Feature | Source | Description |
|---------|--------|-------------|
| `loss_plateau_score` | Loss history (last K epochs) | 0-1: how flat has the loss curve been recently |
| `gradient_trend` | Gradient norm history | Direction/magnitude of gradient norm change |
| `lr_decay_benefit` | LR schedule + loss history | Estimated improvement from reducing LR now |
| `gpu_hours_remaining_vs_budget` | Budget config + elapsed time | 0-1: how much budget is left |
| `epochs_since_improvement` | Loss history | Epochs since the best-seen validation loss |

Real data comes directly from the running `TrainJob`. A **synthetic loss-curve generator** (exponential decay + noise + injectable plateaus) is kept behind a `--synthetic` flag / dashboard toggle for fast KAN-gate testing and edge-case reproduction — it is a debugging aid, not the primary data path.

---

## 3. Component 2: LLM Agent Core

### 3.1 Supervisor Agent

```python
# Pseudocode for supervisor
class SupervisorAgent:
    def run(self, trainjob_state: TrainJobState):
        # 1. Gather observations from domain agents
        metrics = self.metrics_watcher.observe(trainjob_state)
        curve_analysis = self.loss_analyst.analyze(metrics)
        cost = self.cost_estimator.estimate(trainjob_state)

        # 2. Propose a control action
        proposed_action = self.llm.plan(curve_analysis, cost)

        # 3. Every proposed action must pass the KAN gate
        gate_result = self.kan_gate.decide(curve_analysis, cost)

        if gate_result.decision != proposed_action.kind:
            # KAN gate overrides the agent's proposal — its verdict is final
            proposed_action = gate_result.to_action()

        return self.executor.apply(proposed_action, gate_result.formula)
```

### 3.2 Domain Agents (3-4 + Supervisor)

| Agent | Tools | Signal Produced |
|-------|-------|------------------|
| **Metrics Watcher** | `get_trainjob_status`, `stream_logs` | Raw loss/gradient/LR time series |
| **Loss-Curve Analyst** | `compute_plateau_score`, `compute_gradient_trend` | Plateau score, gradient trend |
| **Cost Estimator** | `get_elapsed_gpu_hours`, `get_budget` | GPU-hours remaining vs. budget |
| **Supervisor** | routes to KAN gate, calls `TrainJob` patch/delete | Final action + formula |

*(A fourth agent — e.g. a **Hyperparameter Advisor** suggesting candidate LR values when the gate signals "adjust" — can be added once the 3-agent core loop is validated end-to-end.)*

---

## 4. Component 3: KAN Gating Layer (Core Contribution)

### 4.1 Architecture

```python
import torch
from kan import KAN  # pykan library

class KANGate:
    def __init__(self):
        self.model = KAN(
            width=[5, 4, 3, 1],  # 5 features → 4 → 3 → 1 decision score
            grid=10,
            k=3,
            seed=42
        )

    def decide(self, features: dict) -> GateResult:
        x = self.encode_features(features)

        # Trained directly against 0-100 score labels, so the raw output
        # IS the score - clamped only to guard extrapolation, not squashed.
        raw_output = self.model(x)
        stop_score = torch.clamp(raw_output, 0, 100)

        formula = self.model.symbolic_formula()

        decision = self.route(stop_score)  # continue / adjust_lr / early_stop

        return GateResult(decision=decision, score=stop_score.item(), formula=formula)

    def extract_formula(self) -> str:
        """
        After training, KAN identifies symbolic forms for each edge.
        Example output:
        'stop_score = 0.91·plateau(loss_slope) + 0.12·lr_decay_benefit
                     - 0.40·remaining_gpu_hours'
        """
        self.model.auto_symbolic()
        return self.model.symbolic_formula()[0][0]
```

### 4.2 Training Procedure

**Dataset:** labeled decision points from real training runs (see Section 6) — each labeled with the action a practitioner (or a strong scheduler baseline) would take: continue / adjust LR / early-stop.

```python
dataset = {
    'train_input': X_train,  # [N, 5] feature matrix
    'train_label': y_train,  # [N, 1] decision score
    'test_input': X_test,
    'test_label': y_test
}

model.train(
    dataset,
    opt='Adam',
    steps=500,
    lamb=0.001,       # L1 regularisation for sparsity
    lamb_entropy=2.0   # entropy regularisation for interpretability
)

model.refine(grid=20)  # grid refinement after initial training
```

### 4.3 Symbolic Formula Extraction

```python
model.auto_symbolic(lib=['x', 'x^2', 'x^3', 'sin', 'exp', 'log', 'sqrt'])
model.prune()

formula = model.symbolic_formula()
# Example: stop_score = 0.9*x_0 - 0.4*x_3 + 0.12*x_2
# Where x_0=loss_plateau_score, x_2=lr_decay_benefit, x_3=gpu_hours_remaining
```

---

## 5. Component 4: Executor & Audit Trail

```python
class KANGatedExecutor:

    def execute(self, action, gate_result):
        self.audit_log.record(action, gate_result)

        if gate_result.decision == "continue":
            pass  # no-op, resume training

        elif gate_result.decision == "adjust_lr":
            self.trainer_client.patch_trainjob_lr(action.trainjob_name, action.new_lr)
            self.audit_log.record_outcome(action, "LR_ADJUSTED", gate_result.formula)

        elif gate_result.decision == "early_stop":
            self.trainer_client.delete_trainjob(action.trainjob_name)
            self.audit_log.record_outcome(action, "EARLY_STOPPED", gate_result.formula)
```

Every action — including "continue" — is logged with the KAN's score and formula, so the audit trail shows *why* the job was left running just as clearly as why it was stopped.

---

## 6. Dataset & Training Signal

### 6.1 Primary: Real Training Runs

- A small CPU-friendly model (e.g. a small CNN on Fashion-MNIST, or fine-tuning a small transformer like DistilBERT on a small text classification subset) is trained via a real Kubeflow `TrainJob` on the local cluster
- Loss/gradient/LR are logged at each epoch and become the real feature stream the KAN gate consumes
- Multiple runs (with different seeds, LR schedules, and injected "bad" runs that should plateau early) build up a labeled dataset of decision points

### 6.2 Secondary: Synthetic Generator (toggle, not default)

- Generates loss curves as exponential decay + noise, with an option to inject a plateau at a chosen epoch
- Used for (a) fast iteration on the KAN gate before real training infra is ready, and (b) reproducing specific edge cases (e.g. "what does the gate do on a curve that plateaus then recovers?") that are hard to control for in real runs
- Exposed via a `--synthetic` CLI flag and a toggle button in the dashboard — never the default data source

### 6.3 Labeling Methodology

Decision-point labels come from:
1. A simple rule-based reference policy (e.g. "stop if no improvement in last K epochs and remaining budget < threshold") as an initial weak label source
2. Manual review/correction of a sample of labeled points by the research team
3. Cross-checking against what Hyperband/ASHA would have decided at the same point, for baseline comparison

---

## 7. Evaluation Plan

### 7.1 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Decision Accuracy** | Agreement with expert/reference-policy labels | > 0.85 |
| **GPU-Hours Saved** | vs. manual babysitting baseline, same final accuracy | > 30% reduction |
| **False Early-Stop Rate** | Jobs stopped that would have kept improving | < 10% |
| **Wasted-Compute Rate** | Jobs kept running well past plateau | < 15% |
| **Formula Faithfulness** | % of formula's dominant term matching the feature that actually drove the decision | > 70% |
| **User Study Score** | ML practitioners rate formula readability 1-5 | > 3.5/5 |

### 7.2 Baselines

| Baseline | Description |
|----------|--------------|
| **Manual** | Fixed-epoch training, no early stopping |
| **Hyperband** | Standard bandit-based early stopping |
| **ASHA** | Asynchronous successive halving |
| **MLP Gate** | Same 5 features, standard neural network gate (no formula) |
| **KAN-KubeAgent (Ours)** | KAN-gated agentic control |

### 7.3 Experimental Environment

```yaml
cluster:
  tool: minikube (local dev)
  k8s_version: 1.30+
  crds: kubeflow-trainer (TrainJob)

training_job:
  compute: CPU-only (no GPU cluster dependency)
  model: small CNN (Fashion-MNIST) or small transformer fine-tune
  runs: multiple seeds + injected bad-run scenarios

agent_llm: Claude API (via LangGraph)
```
