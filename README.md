# KAN-KubeAgent 🧠☸️🤖

> **Trustworthy Autonomous Remediation in Kubernetes using Kolmogorov-Arnold Verification Networks**

[![Research Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)](.)
[![Topics](https://img.shields.io/badge/Topics-KAN%20%7C%20Kubernetes%20%7C%20Agentic%20AI-blue)](.)
[![Paper](https://img.shields.io/badge/Paper-In%20Progress-orange)](.)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](.)

---

## 🔬 Research Overview

**KAN-KubeAgent** is an active research project at the intersection of three cutting-edge AI/ML domains:

| Domain | Technology | Role in this Project |
|--------|-----------|----------------------|
| **KAN** | Kolmogorov-Arnold Networks | Interpretable trust/safety verification layer |
| **Kubernetes** | Container orchestration | Target environment for autonomous operations |
| **Agentic AI** | LLM-based multi-agent systems | Autonomous remediation workflow engine |

### Core Research Question

> *"Can a Kolmogorov-Arnold Network serve as a transparent, mathematically verifiable trust layer inside an LLM-driven Kubernetes agent — making autonomous infrastructure changes explainable and auditable to security engineers?"*

---

## 🎯 Problem Statement

Modern Kubernetes clusters face a fundamental trust problem with autonomous AI agents:

- **LLM agents** (like KubeIntellect) can detect and remediate security misconfigurations automatically
- But **security engineers refuse to let agents touch production RBAC** because there is no auditable explanation of *why* the agent decided an action was safe
- Existing systems use static rule-gates (deny-by-default) which require manual human approval for every action — defeating the purpose of automation

### Our Solution: KAN as a Verification Layer

Instead of a black-box classifier or static rules deciding "is this action safe?", we insert a **KAN (Kolmogorov-Arnold Network)** as the trust-scoring layer. After the agent proposes a remediation action, the KAN:

1. Takes in features of the proposed change (resource type, namespace risk, blast radius, change frequency, actor identity)
2. Outputs a **continuous trust score [0-100]**
3. Exposes the **learned symbolic formula** for that decision — e.g.:

```
trust_score = 0.8 · f(namespace_risk) + 0.6 · sin(change_frequency) − 0.4 · g(blast_radius)
```

This formula is auditable by a human security engineer — something no MLP or LLM can provide.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KAN-KubeAgent System                          │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  K8s Cluster │───▶│  Supervisor  │───▶│  Domain Agents   │   │
│  │  (Live State)│    │  LLM Agent   │    │  (Logs/RBAC/     │   │
│  │              │    │  (LangGraph) │    │   Metrics/Net)   │   │
│  └──────────────┘    └──────┬───────┘    └────────┬─────────┘   │
│                             │                     │              │
│                             ▼                     ▼              │
│                    ┌────────────────────────────────────┐        │
│                    │     Remediation Action Proposal     │        │
│                    │  (YAML patch / RBAC change / etc.) │        │
│                    └────────────────┬───────────────────┘        │
│                                     │                            │
│                                     ▼                            │
│                    ┌────────────────────────────────────┐        │
│                    │   🧠 KAN TRUST VERIFICATION LAYER  │        │
│                    │                                    │        │
│                    │  Input Features:                   │        │
│                    │  • namespace_risk_score            │        │
│                    │  • blast_radius                    │        │
│                    │  • change_frequency                │        │
│                    │  • actor_privilege_level           │        │
│                    │  • resource_criticality            │        │
│                    │                                    │        │
│                    │  Output: Trust Score [0-100]       │        │
│                    │  + Symbolic Formula (explainable)  │        │
│                    └────────────────┬───────────────────┘        │
│                                     │                            │
│              ┌──────────────────────┼──────────────────────┐     │
│              │                      │                      │     │
│              ▼                      ▼                      ▼     │
│        Score > 80            Score 40-80             Score < 40  │
│        AUTO-APPLY            HUMAN REVIEW            AUTO-REJECT  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
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
│   │   ├── 01_KAN_original.md         ← Liu et al. 2024 (arXiv:2404.19756)
│   │   ├── 02_KubeIntellect.md        ← arXiv:2509.02449 analysis
│   │   ├── 03_KAN_security.md         ← KAN-MID, GloroKAN papers
│   │   └── 04_related_work.md         ← All related papers with gaps
│   │
│   ├── proposal/
│   │   ├── research_gap.md            ← Formal gap statement
│   │   ├── novelty_claims.md          ← What we contribute
│   │   └── methodology_draft.md       ← Experimental design
│   │
│   └── notes/
│       ├── weekly_log.md              ← Research progress log
│       └── ideas_scratchpad.md        ← Ongoing ideas
│
├── architecture/                      ← System design documents
│   ├── system_overview.md
│   ├── kan_trust_layer_design.md
│   └── agent_workflow_design.md
│
├── datasets/                          ← Dataset documentation
│   ├── README.md                      ← Available datasets guide
│   ├── cicids2017_notes.md
│   └── synthetic_k8s_logs_spec.md
│
├── experiments/                       ← Experiment tracking
│   ├── baseline_results/
│   └── kan_results/
│
└── references/
    ├── bibliography.bib               ← BibTeX references
    └── reading_list.md                ← Prioritised reading list
```

---

## 🔑 Key Novelty Claims

1. **First paper** using KAN as a *trust verification layer inside an LLM-agent loop* — not just as a standalone predictor
2. **First explainable autonomous remediation system** for Kubernetes — produces a human-readable formula for every auto-applied action
3. **Novel evaluation metric**: Trust Formula Faithfulness (TFF) — measuring if the symbolic formula correctly explains what drove the trust score
4. **Multi-cluster evaluation** — tested across heterogeneous cluster configurations, not just one lab setup

---

## 📚 Base Papers (Starting Points)

| Paper | Role | arXiv |
|-------|------|-------|
| KAN: Kolmogorov-Arnold Networks | Architecture foundation | [2404.19756](https://arxiv.org/abs/2404.19756) |
| KubeIntellect | Agentic K8s system baseline | [2509.02449](https://arxiv.org/abs/2509.02449) |
| KAN for Intrusion Detection (KAN-MID) | KAN security application | ResearchGate 2025 |
| GloroKAN (trust/Lipschitz) | KAN verification methods | OpenReview 2025 |
| MOYA Framework (multi-agent CloudOps) | Multi-agent architecture | [2501.08243](https://arxiv.org/abs/2501.08243) |
| KAN RL Policy (network control) | Closest prior art | [2506.16392](https://arxiv.org/abs/2506.16392) |

---

## 🗓️ Research Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| **Phase 1: Literature** | Weeks 1-3 | Deep-read all 6 base papers, document gaps |
| **Phase 2: Design** | Weeks 4-5 | Finalise architecture, KAN feature set |
| **Phase 3: Environment** | Week 6 | Set up minikube cluster, simulate attack/normal traffic |
| **Phase 4: Baseline** | Weeks 7-8 | Implement & evaluate KubeIntellect-style agent (no KAN) |
| **Phase 5: KAN Layer** | Weeks 9-11 | Build & train KAN trust layer on K8s audit log features |
| **Phase 6: Integration** | Weeks 12-13 | Plug KAN into agent loop, end-to-end evaluation |
| **Phase 7: Paper** | Weeks 14-16 | Write and submit |

---

## 👥 Authors

- **Avanish Kasar** - Research Lead
- **Rupali Biradar** - Contributor 

---

## 📖 Citation

```bibtex
@article{kasar2026tankubeagent,
  title   = {TrustK8s: Kolmogorov-Arnold Verification Layers for
             Explainable Autonomous Remediation in Kubernetes Security Agents},
  author  = {Kasar, Avanish},
  journal = {Under Review},
  year    = {2026}
}
```

---

*This repository is the active research workspace for an ongoing paper. All content reflects work-in-progress.*
