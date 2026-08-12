# Reading List — Prioritised

## 🔴 Priority 1: Must Read Before Starting (Base Papers)

| # | Paper | arXiv / Link | Why Critical | Status |
|---|-------|-------------|-------------|--------|
| 1 | KAN: Kolmogorov-Arnold Networks | [2404.19756](https://arxiv.org/abs/2404.19756) | Core architecture — understand B-splines and symbolic regression | ⬜ |
| 2 | KubeIntellect | [2509.02449](https://arxiv.org/abs/2509.02449) | Main baseline — we extend this | ⬜ |
| 3 | MOYA: Multi-Agent CloudOps | [2501.08243](https://arxiv.org/abs/2501.08243) | Multi-agent design patterns | ⬜ |
| 4 | KAN RL Policy (network control) | [2506.16392](https://arxiv.org/abs/2506.16392) | Closest prior art — KAN as policy | ⬜ |

---

## 🟠 Priority 2: Read in Week 2-3

| # | Paper | Link | Why Important | Status |
|---|-------|------|--------------|--------|
| 5 | KAN 2.0 | [2408.10205](https://arxiv.org/abs/2408.10205) | MultKAN, more efficient, better symbolic regression | ⬜ |
| 6 | GloroKAN | [OpenReview 2025](https://openreview.net) | KAN Lipschitz bounds for robustness certification | ⬜ |
| 7 | KAN-MID (intrusion detection) | ResearchGate 2025 | KAN applied to cloud security — closest data domain | ⬜ |
| 8 | CogDevSecOps | ResearchGate 2026 | ML anomaly + LLM remediation loop | ⬜ |
| 9 | Graph-based K8s anomaly | [2503.14114](https://arxiv.org/abs/2503.14114) | Graph representation of cluster state | ⬜ |
| 10 | LLM-assisted SRE | [2501.16744](https://arxiv.org/abs/2501.16744) | LLM for K8s failure prediction | ⬜ |

---

## 🟡 Priority 3: Background / Reference

| # | Paper | Why | Status |
|---|-------|-----|--------|
| 11 | MITRE ATT&CK for Containers | Attack taxonomy for dataset labeling | ⬜ |
| 12 | CIS Kubernetes Benchmark v1.9 | Security standards for K8s risk scoring | ⬜ |
| 13 | LangGraph documentation | Implementation framework | ⬜ |
| 14 | pykan documentation | KAN implementation library | ⬜ |
| 15 | KAN-AD (anomaly detection) | [ICML 2025](https://icml.cc) | KAN for continuous data anomaly detection | ⬜ |

---

## 📖 Papers That Cite Our Research Gap (Use for Related Work Section)

| Paper | What They Found | Our Improvement |
|-------|----------------|-----------------|
| KubeIntellect (2509.02449) | 93% query success but all mutations need HITL | We replace HITL with dynamic KAN trust gate |
| KAN RL Policy (2506.16392) | KAN for network control policy | We extend to K8s-specific agentic context |
| CogDevSecOps (2026) | ML anomaly + LLM remediation | No trust verification between ML and LLM layers |
| MOYA (2501.08243) | Multi-agent CloudOps | No explainability for agent decisions |

---

## 🗒️ Notes Template (Use for Each Paper Read)

```markdown
## Paper: [Title]
**Read Date:** 
**Time Spent:** 

### What they did (2 sentences):

### What they didn't do (our gap):

### Methodology I want to borrow:

### Key numbers/results to cite:

### Datasets they used:
```
