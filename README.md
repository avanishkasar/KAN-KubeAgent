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

## Try it in the browser

A static preview is published on GitHub Pages at https://avanishkasar.github.io/KAN-KubeAgent/ . It runs the real trained gate in your browser (KAN Network tab), shows the stored benchmark results (Research tab) and the Guide. Live training needs the app on your own machine, see below. The preview is rebuilt with `python tools/build_pages.py`.

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

All results come from real training runs; no synthetic data enters any of them.

- Data: 144 complete, uncontrolled runs on four workloads (Fashion-MNIST CNN, MNIST CNN, CIFAR-10 CNN, all with Adam; Fashion-MNIST MLP with SGD), each 6 learning rates x 2 batch sizes x 3 seeds, 40 epochs.
- Unseen workloads (gate trained on three workloads, tested on the fourth): the KAN gate saves 25.0% of epochs at 0.21% mean regret in best validation loss, with 1.9% of runs stopped prematurely. That is the lowest regret of every learned gate. The same KAN trained on hand-written rule labels saves only 15.4%.
- A patience rule of 8 epochs performs comparably (29.3% saved, 0.22% regret), and a linear gate on the same labels sits close to the same trade-off curve. The KAN's contribution is an exact, readable decision formula that transfers without per-workload tuning and also controls the learning rate. It does not stop runs better than these simpler rules.
- Live closed-loop control of 48 runs on unseen workloads: 1,502 of 1,920 epochs used (21.8% saved), with no significant change in best validation loss (Wilcoxon p = 0.51).
- Learning-rate guard: without it the controller halved the learning rate up to 12 times in one run. With it, at most 3 halvings per run, and the worst single-run loss fell from +11.1% to +8.1% (Fashion-MNIST) and from +4.2% to +2.8% (CIFAR-10).

The full write-up is the MethodsX article in `paper/methodsx/` (PDF: `paper/methodsx/kan_gate_methodsx.pdf`). Every number, table and figure in it is regenerated from the experiment outputs by `python -m experiments.paper_methodsx`. The earlier IEEE conference draft, which covers only the Fashion-MNIST workload, is in `paper/`.

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

The dashboard has five views: Guide (a plain-language explanation of every number and chart, linked from each one), Live (a real training subprocess with real CPU/GPU telemetry, the agent pipeline lighting up node by node, and every gate decision traced through the network), KAN Network (the gate's learned edge functions, what-if sliders, and an exact per-feature breakdown of each score), Research (the benchmark results), and Mock/Synthetic (an instantly generated synthetic curve for fast iteration).

## Hindsight-supervised gate

The gate is trained on hindsight labels: for each decision point of a completed real run, the target stop score is computed from how much validation-loss improvement was actually still to come (kan_gate/hindsight.py). This replaces the original setup, in which the gate imitated a hand-written three-rule policy and harvested live runs were labelled with the gate's own decisions, a circular loop that could only teach the gate to copy itself.

After training, every edge of the KAN is fixed to a symbolic function, so the formula shown in the dashboard is the function that makes the decision, not an approximation of it.

## Experiments and paper

```
python -m experiments.collect_curves --workload fmnist_cnn   # also mnist_cnn, cifar10_cnn, fmnist_mlp_sgd
python -m experiments.generalization      # all policies, within-workload CV and leave-one-workload-out
python -m experiments.benchmark           # the original single-workload benchmark
python -m experiments.closed_loop --workload cifar10_cnn --gate lowo   # live control; --no-guard to disable the LR guard
python -m kan_gate.train --curves experiments/data/curves*.jsonl       # train the deployed gate on all workloads
python -m experiments.paper_methodsx      # numbers, tables and figures of the MethodsX article
```

Results land in experiments/results/. The MethodsX article is in paper/methodsx/ and the earlier IEEE conference draft in paper/.

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
| 7: Paper | MethodsX article complete (paper/methodsx/); next: cluster validation and larger models |

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
