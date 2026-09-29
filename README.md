# KAN-KubeAgent

Interpretable Agentic Control of Kubeflow Fine-Tuning Jobs via Kolmogorov-Arnold Gating

## Overview

KAN-KubeAgent sits at the intersection of three technologies:

| Domain | Technology | Role in this project |
|--------|-----------|----------------------|
| KAN | Kolmogorov-Arnold Networks | Interpretable gating layer that scores every training-control decision and outputs a human-readable formula |
| Kubernetes | Kubeflow TrainJob CRD | Real, live-managed fine-tuning job that the agents observe and control |
| Agentic AI | LLM-based multi-agent system | Watches training in real time, proposes continue/adjust/stop actions |

### Core research question

Can a Kolmogorov-Arnold Network act as a transparent, formula-producing gate inside an agentic fine-tuning controller, deciding when to continue, adjust the learning rate, or stop and reallocate a Kubeflow TrainJob, in a way a human can audit at a glance?

## Problem statement

Fine-tuning jobs on Kubernetes today are babysat in one of two ways:

- Manually: an engineer watches a loss curve in a dashboard and decides by eye when to stop or adjust, wasting GPU-hours while not looking.
- By a black-box scheduler (Hyperband, ASHA, PBT): these do stop bad runs early, but the decision is a bandit-algorithm score with no human-readable justification, and none of them are agentic (they don't reason over multiple signals or explain themselves).

Neither approach gives you the exact formula that decided to stop a job.

### Our solution: KAN as a gating layer

A team of agents watches a live Kubeflow TrainJob and proposes an action. Before that action is ever applied to the cluster, it must pass through a KAN (Kolmogorov-Arnold Network) gate, which:

1. Takes in features of the current training state (loss plateau score, LR decay benefit, GPU-hours remaining vs. budget, gradient trend).
2. Outputs a decision: continue, adjust learning rate, or early-stop and reallocate.
3. Exposes the learned symbolic formula behind that decision, for example:

```
stop_score = 0.91 * plateau(loss_slope) + 0.12 * lr_decay_benefit - 0.40 * remaining_gpu_hours
```

This formula is readable by a human reviewer, something no black-box scheduler or raw LLM judgement can provide.

## System flow

```
Kubeflow TrainJob (live)
        |
        v
Supervisor LLM agent  <-->  Domain agents (Metrics Watcher / Loss-Curve Analyst / Cost Estimator)
        |
        v
Proposed control action (continue / adjust LR / early-stop)
        |
        v
KAN gating layer
  inputs:  loss_plateau_score, gradient_trend, lr_decay_benefit,
           gpu_hours_remaining_vs_budget, epochs_since_improvement
  output:  decision + symbolic formula
        |
   -----------------------------
   |            |               |
CONTINUE    ADJUST LR      EARLY-STOP
(no-op)   (patch TrainJob) (stop + free GPU)
```

Two run modes are available from the dashboard: Live, which drives a real training subprocess with real CPU/GPU telemetry, and Mock/Synthetic, which scores an instantly generated synthetic curve for fast iteration on the gate itself.

## Repository structure

```
KAN-KubeAgent/
  README.md

  research/
    literature/                annotated base papers
    proposal/
      research_gap.md          formal gap statement
      methodology_draft.md     experimental design
    notes/
      weekly_log.md            research progress log

  kan_gate/                    KAN gating module (pykan-based)
  agents/                      LangGraph agent layer
  k8s/                         Minikube + Kubeflow Trainer setup
  training/                    the real training workload the agents control
  dashboard/                   live decision dashboard (Live + Mock/Synthetic modes)
  datasets/                    dataset documentation

  references/
    bibliography.bib
    reading_list.md
```

## Key novelty claims

1. First use of a Kolmogorov-Arnold Network as an interpretable gating layer inside an agentic fine-tuning controller, not a standalone predictor.
2. First agentic system to control a real Kubeflow TrainJob lifecycle (continue/adjust/stop) with a formula-justified decision at every step.
3. A novel evaluation angle: comparing KAN-gated decisions against Hyperband/ASHA on both efficiency (GPU-hours saved) and explainability (can a human verify the stop reason?).
4. Fully reproducible on local infrastructure: no GPU cluster dependency; runs on a laptop-scale Minikube cluster with a small CPU/GPU-optional training job.

## Base papers

| Paper | Role | Reference |
|-------|------|-----------|
| KAN: Kolmogorov-Arnold Networks | Architecture foundation | arXiv:2404.19756 |
| Hyperband | Closest non-interpretable prior art (early stopping) | arXiv:1603.06560 |
| ASHA | Asynchronous successive halving baseline | arXiv:1810.05934 |
| Kubeflow Trainer / TrainJob | Kubernetes-native training CRD this project controls | kubeflow.org/docs/components/trainer |
| MOYA Framework (multi-agent CloudOps) | Multi-agent architecture reference | arXiv:2501.08243 |

See `research/literature/` for full annotated notes.

## Research timeline

| Phase | Goal |
|-------|------|
| 1: Literature and design | Finalise architecture, KAN feature set, agent roles |
| 2: KAN gate | Build and test the gating module standalone on synthetic loss curves |
| 3: Environment | Minikube + Kubeflow Trainer install, run a real tiny fine-tuning job |
| 4: Agent layer | LangGraph agents observing the real TrainJob |
| 5: Integration | Wire agents, KAN gate, and real TrainJob control actions together |
| 6: Dashboard | Live decision curve and formula audit log |
| 7: Paper | Write and submit to Avishkar Research Convention |

No fixed hard deadline is set yet; phases are sequenced by dependency, not calendar weeks.

## Authors

- Avanish Kasar, Research Lead
- Rupali Biradar, Contributor
- Viverun, Contributor

## Citation

```bibtex
@article{kasar2026kankubeagent,
  title   = {KAN-KubeAgent: Interpretable Agentic Control of Kubeflow
             Fine-Tuning Jobs via Kolmogorov-Arnold Gating},
  author  = {Kasar, Avanish and Biradar, Rupali and Viverun},
  journal = {Under Review},
  year    = {2026}
}
```

This repository is the active research workspace for an ongoing paper. All content reflects work in progress.
