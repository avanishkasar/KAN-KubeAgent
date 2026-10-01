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

## Getting started

Requirements: Python 3.10 or newer and internet access for the first run (Fashion-MNIST is downloaded once). An NVIDIA GPU is optional.

1. Clone and enter the repository:

```
git clone https://github.com/avanishkasar/KAN-KubeAgent.git
cd KAN-KubeAgent
```

2. Start everything with one command. It creates a virtual environment, installs dependencies, trains the gate from the bundled real learning curves if no checkpoint exists, and launches the dashboard in the background:

```
.\start.ps1        # Windows (PowerShell)
./start.sh         # Linux / macOS
```

3. Open http://localhost:8000 and click Start real training on the Live tab. Use the KAN Network tab to inspect the gate and the Research tab for the benchmark.

4. Stop the server with `Stop-Process -Id (Get-Content dashboard.pid)` on Windows, or `kill $(cat dashboard.pid)` on Linux.

To use a GPU, install the CUDA build of PyTorch from the index described in training/requirements.txt, then restart. To enable real LLM reasoning for the Supervisor agent, set the ANTHROPIC_API_KEY environment variable before starting; without it the Supervisor uses a rule-based fallback and the gate's decisions are unaffected.

Training modes on the Live tab: Background (capped CPU, low priority, safe to use while working), Normal, and Turbo (maximum throughput). Auto-continuous keeps starting fresh runs. On a machine with several GPUs, pick the one to use from the GPU dropdown.

Run the tests with `python -m pytest agents/tests kan_gate/tests`.

## Results

On 36 real Fashion-MNIST runs (6 learning rates, 2 batch sizes, 3 seeds, 40-epoch budget) with seed-grouped cross-validation, the hindsight-trained gate saves 31.0% of the epoch budget at 0.20% mean regret in best validation loss. The same gate trained on rule labels saves 4.4%. A hindsight oracle reaches 0.18% regret. In closed-loop control of 12 held-out live runs the system saved 37.1% of epochs with no significant change in best validation loss. A linear gate trained on the same labels performs comparably, so the KAN's contribution is an exact, readable, non-linear decision formula rather than higher accuracy. Limitations (one small model, intervention cooldown missing, Kubeflow client untested on a live cluster) are listed in the paper.

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

The dashboard has four views: Live (a real training subprocess with real CPU/GPU telemetry, the agent pipeline lighting up node by node, and every gate decision traced through the network), KAN Network (the gate's learned edge functions, what-if sliders, and an exact per-feature breakdown of each score), Research (the benchmark results), and Mock/Synthetic (an instantly generated synthetic curve for fast iteration).

## Hindsight-supervised gate

The gate is trained on hindsight labels: for each decision point of a completed real run, the target stop score is computed from how much validation-loss improvement was actually still to come (kan_gate/hindsight.py). This replaces the original setup, in which the gate imitated a hand-written three-rule policy and harvested live runs were labelled with the gate's own decisions, a circular loop that could only teach the gate to copy itself.

After training, every edge of the KAN is fixed to a symbolic function, so the formula shown in the dashboard is the function that makes the decision, not an approximation of it.

## Experiments and paper

```
python -m experiments.collect_curves      # 36 real Fashion-MNIST runs, full learning curves
python -m experiments.benchmark           # offline replay of every stopping policy, 3-fold CV by seed
python -m experiments.closed_loop         # the full agent + gate system controlling real runs
python -m experiments.figures             # paper figures from the results above
python -m kan_gate.train --curves experiments/data/curves.jsonl   # train the deployed gate
```

Results land in experiments/results/. The paper (IEEE conference format) is in paper/, with its PDF at paper/kan_kubeagent.pdf.

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

  kan_gate/                    KAN gating module (pykan-based), hindsight labels, introspection
  agents/                      LangGraph agent layer
  k8s/                         Minikube + Kubeflow Trainer setup
  training/                    the real training workload the agents control
  dashboard/                   control-room dashboard (Live, KAN Network, Research, Mock)
  experiments/                 curve collection, benchmark, closed-loop runs, figures
  paper/                       research paper (LaTeX, IEEE format) and PDF
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
| 6: Dashboard | Done: live control room, KAN network view, research view |
| 7: Paper | Draft complete (paper/); next: cluster validation and larger models |

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
