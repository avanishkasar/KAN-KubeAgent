# KAN-KubeAgent 🧠☸️🤖

> **Interpretable Agentic Control of Kubeflow Fine-Tuning Jobs via Kolmogorov-Arnold Gating**

[![Research Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)](.)
[![Topics](https://img.shields.io/badge/Topics-KAN%20%7C%20Kubernetes%20%7C%20Agentic%20AI-blue)](.)
[![Paper](https://img.shields.io/badge/Paper-In%20Progress-orange)](.)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](.)

---

## 🔬 Research Overview

**KAN-KubeAgent** sits at the intersection of three technologies:

| Domain | Technology | Role in this Project |
|--------|-----------|----------------------|
| **KAN** | Kolmogorov-Arnold Networks | Interpretable gating layer — scores every training-control decision and outputs a human-readable formula |
| **Kubernetes** | Kubeflow `TrainJob` CRD | Real, live-managed fine-tuning job that the agents observe and control |
| **Agentic AI** | LLM-based multi-agent system | Watches training in real time, proposes continue/adjust/stop actions |

### Core Research Question

> *"Can a Kolmogorov-Arnold Network act as a transparent, formula-producing gate inside an agentic fine-tuning controller — deciding when to continue, adjust the learning rate, or stop-and-reallocate a Kubeflow `TrainJob` — in a way a human can audit at a glance?"*

---

## 🎯 Problem Statement

Fine-tuning jobs on Kubernetes today are babysat in one of two ways:

- **Manually** — an engineer watches a loss curve in a dashboard and decides by eye when to stop or adjust, wasting GPU-hours while they're not looking
- **By a black-box scheduler** (Hyperband, ASHA, PBT) — these *do* stop bad runs early, but the decision is a bandit-algorithm score with no human-readable justification, and none of them are agentic (they don't reason over multiple signals or explain themselves)

Neither approach gives you: *"here is the exact formula that decided to stop this job."*

### Our Solution: KAN as a Gating Layer

A team of agents watches a live Kubeflow `TrainJob` and proposes an action. Before that action is ever applied to the cluster, it must pass through a **KAN (Kolmogorov-Arnold Network) gate**, which:

1. Takes in features of the current training state (loss plateau score, LR decay benefit, GPU-hours remaining vs. budget, gradient trend)
2. Outputs a decision: **continue / adjust learning rate / early-stop and reallocate**
3. Exposes the **learned symbolic formula** behind that decision — e.g.:

```
stop_score = 0.91·plateau(loss_slope) + 0.12·lr_decay_benefit − 0.40·remaining_gpu_hours
```

This formula is readable by a human reviewer — something no black-box scheduler or raw LLM judgement can provide.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    KAN-KubeAgent System                            │
│                                                                     │
│  ┌───────────────┐   ┌──────────────┐   ┌───────────────────┐      │
│  │ Kubeflow       │──▶│  Supervisor  │──▶│  Domain Agents     │      │
│  │ TrainJob (live)│   │  LLM Agent   │   │  (Watcher / Loss-  │      │
│  │                │   │  (LangGraph) │   │  Curve / Cost)     │      │
│  └───────────────┘   └──────┬───────┘   └─────────┬──────────┘      │
│                              │                     │                 │
│                              ▼                     ▼                 │
│                    ┌──────────────────────────────────────┐          │
│                    │   Proposed Control Action             │          │
│                    │  (continue / adjust LR / early-stop) │          │
│                    └────────────────┬─────────────────────┘          │
│                                     │                                 │
│                                     ▼                                 │
│                    ┌──────────────────────────────────────┐          │
│                    │   🧠 KAN GATING LAYER                │          │
│                    │                                        │          │
│                    │  Input Features:                      │          │
│                    │  • loss_plateau_score                 │          │
│                    │  • gradient_trend                     │          │
│                    │  • lr_decay_benefit                   │          │
│                    │  • gpu_hours_remaining_vs_budget       │          │
│                    │  • epochs_since_improvement            │          │
│                    │                                        │          │
│                    │  Output: Decision + Symbolic Formula  │          │
│                    └────────────────┬─────────────────────┘          │
│                                     │                                 │
│              ┌──────────────────────┼──────────────────────┐         │
│              ▼                      ▼                      ▼         │
│         CONTINUE              ADJUST LR              EARLY-STOP      │
│      (no-op, resume)      (patch TrainJob)      (stop + free GPU)    │
│                                                                       │
└───────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
KAN-KubeAgent/
│
├── README.md                          ← You are here
│
├── research/                          ← All research documentation
│   ├── literature/                    ← Annotated base papers
│   ├── proposal/
│   │   ├── research_gap.md            ← Formal gap statement
│   │   └── methodology_draft.md       ← Experimental design
│   └── notes/
│       └── weekly_log.md              ← Research progress log
│
├── kan_gate/                          ← KAN gating module (pykan-based)
├── agents/                            ← LangGraph agent layer
├── k8s/                               ← Minikube + Kubeflow Trainer setup
├── dashboard/                         ← Live decision dashboard (later stage)
│
├── datasets/                          ← Dataset documentation
│
└── references/
    ├── bibliography.bib               ← BibTeX references
    └── reading_list.md                ← Prioritised reading list
```

---

## 🔑 Key Novelty Claims

1. **First** use of a Kolmogorov-Arnold Network as an *interpretable gating layer inside an agentic fine-tuning controller* — not a standalone predictor
2. **First** agentic system to control a real Kubeflow `TrainJob` lifecycle (continue/adjust/stop) with a formula-justified decision at every step
3. **Novel evaluation angle**: comparing KAN-gated decisions against Hyperband/ASHA on both efficiency (GPU-hours saved) *and* explainability (can a human verify the stop reason?)
4. **Fully reproducible on local infra** — no GPU cluster dependency; runs on a laptop-scale Minikube cluster with a small CPU/GPU-optional training job

---

## 📚 Base Papers (Starting Points)

| Paper | Role | Reference |
|-------|------|-----------|
| KAN: Kolmogorov-Arnold Networks | Architecture foundation | [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| Hyperband | Closest non-interpretable prior art (early stopping) | [arXiv:1603.06560](https://arxiv.org/abs/1603.06560) |
| ASHA | Asynchronous successive halving baseline | [arXiv:1810.05934](https://arxiv.org/abs/1810.05934) |
| Kubeflow Trainer / TrainJob | K8s-native training CRD this project controls | [kubeflow.org/docs/components/trainer](https://www.kubeflow.org/docs/components/trainer/) |
| MOYA Framework (multi-agent CloudOps) | Multi-agent architecture reference | [arXiv:2501.08243](https://arxiv.org/abs/2501.08243) |

*(See `research/literature/` for full annotated notes; `02_KubeIntellect.md` is retained as a reference for agentic-K8s system design patterns even though this project's domain has moved away from security remediation.)*

---

## 🗓️ Research Timeline

| Phase | Goal |
|-------|------|
| **Phase 1: Literature & Design** | Finalise architecture, KAN feature set, agent roles |
| **Phase 2: KAN Gate** | Build & test the gating module standalone on synthetic loss curves |
| **Phase 3: Environment** | Minikube + Kubeflow Trainer install, run a real tiny fine-tuning job |
| **Phase 4: Agent Layer** | LangGraph agents (Claude-powered) observing the real TrainJob |
| **Phase 5: Integration** | Wire agents → KAN gate → real `TrainJob` control actions |
| **Phase 6: Dashboard** | Live decision curve + formula audit log (built once the core pipeline works) |
| **Phase 7: Paper** | Write and submit to Avishkar Research Convention |

*(No fixed hard deadline is set yet — phases are sequenced by dependency, not calendar weeks.)*

---

## 👥 Authors

- **Avanish Kasar** - Research Lead
- **Rupali Biradar** - Contributor
- **Viverun** - Contributor

---

## 📖 Citation

```bibtex
@article{kasar2026kankubeagent,
  title   = {KAN-KubeAgent: Interpretable Agentic Control of Kubeflow
             Fine-Tuning Jobs via Kolmogorov-Arnold Gating},
  author  = {Kasar, Avanish and Biradar, Rupali and Viverun},
  journal = {Under Review},
  year    = {2026}
}
```

---

*This repository is the active research workspace for an ongoing paper. All content reflects work-in-progress.*
