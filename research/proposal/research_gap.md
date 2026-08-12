# Research Gap & Novelty Claims

## 1. Formal Gap Statement

### Gap 1 — The "Explainability Void" in Autonomous Kubernetes Agents

**Evidence from literature:**
- KubeIntellect (arXiv:2509.02449) achieves 93% query resolution but uses a binary HITL gate for ALL mutations
- The paper explicitly states: *"all mutating operations require human approval"* — acknowledging that the system cannot self-certify safety
- No existing paper provides a mathematical justification for *why* an autonomous K8s action is safe to apply

**Why this matters:**  
A 2026 CNCF survey found that the #1 blocker for production deployment of autonomous K8s agents is "lack of auditable decision trails for security-sensitive operations." Engineers won't trust an agent that can't explain itself.

**Our contribution:**  
A KAN-based trust layer that produces an inspectable symbolic formula for every trust decision — turning "the agent said it's safe" into "the formula `0.8·f(ns_risk) - 0.4·g(blast_radius) = 72` says it's safe."

---

### Gap 2 — KAN Has Never Been Used Inside an Agent Loop

**Evidence from literature:**
- All 50+ KAN papers (Liu et al. 2024; KAN-MID; GloroKAN; TKAN; etc.) use KAN as a **standalone predictor**
- KAN RL Policy paper (arXiv:2506.16392) uses KAN for network load-balancing *policy* — but still as a standalone model, not as a sub-component in an LLM-orchestrated multi-agent system
- Zero papers place a KAN *inside* an agent's decision pipeline as a gatekeeper/verifier

**Why this matters:**  
Using KAN as a verification layer rather than a primary predictor is a fundamentally different architectural role that unlocks new capabilities (real-time auditing, formula-based safety certificates) that standalone KAN models can't provide.

**Our contribution:**  
First paper to use KAN as a **verification sub-component** inside an LLM-agent loop, establishing a new pattern for trustworthy agentic AI.

---

### Gap 3 — No Kubernetes-Specific Security Dataset with Labeled Risk Ground Truth

**Evidence from literature:**
- Kubernetes anomaly detection papers (arXiv:2503.14114, etc.) generate synthetic datasets but don't label individual *actions* with risk scores
- CICIDS2017 and UNSW-NB15 cover general network intrusion but are not K8s-native
- No paper has published a labeled dataset of K8s audit log events with expert-assigned risk scores

**Why this matters:**  
You can't train a KAN trust layer without ground-truth risk labels for K8s operations.

**Our contribution:**  
We synthesise a **K8s Risk Action Dataset (K-RAD)** — 10,000 simulated K8s API events labeled as [low/medium/high/critical] risk by mapping to RBAC privilege levels, CIS Kubernetes Benchmark controls, and MITRE ATT&CK for Containers.

---

## 2. Novelty Claims (In Priority Order)

### Claim 1 (Primary — Architectural)
> **"We are the first to use a Kolmogorov-Arnold Network as a trust verification layer inside an LLM-orchestrated multi-agent system."**

Strength: Very strong. Zero prior art.

### Claim 2 (Secondary — Applied)
> **"We are the first to apply KAN to Kubernetes-native security audit log data for autonomous action trust scoring."**

Strength: Strong. KAN has been applied to cloud/network security in general, but never to K8s-specific audit logs or to action trust scoring.

### Claim 3 (Tertiary — Dataset)
> **"We introduce K-RAD, the first labeled dataset of Kubernetes API audit events annotated with risk scores derived from CIS Benchmark and MITRE ATT&CK for Containers."**

Strength: Moderate. Dataset papers are valued but the labeling methodology must be rigorous.

### Claim 4 (Evaluation — Metric)
> **"We introduce Trust Formula Faithfulness (TFF), a novel metric for evaluating whether the KAN's symbolic formula accurately reflects its numerical trust decision."**

Strength: Moderate. Evaluation metric contributions are secondary but add value.

---

## 3. Research Questions

**RQ1 (Primary):** Can a KAN trust layer accurately distinguish between safe and unsafe Kubernetes remediation actions, and does its symbolic formula output match human expert assessments?

**RQ2:** How does a KAN-gated autonomous agent compare to (a) fully manual operations and (b) a static rule-gate in Mean Time to Remediation (MTTR)?

**RQ3:** What is the false-positive rate (safe actions incorrectly blocked) and false-negative rate (unsafe actions incorrectly approved) of the KAN trust layer?

**RQ4:** Are the symbolic formulas learned by the KAN interpretable to domain experts (DevOps/Security engineers), as measured by a user study?

---

## 4. Comparison to Prior Art

| Criterion | KubeIntellect | KAN-MID | KAN RL Policy | **Ours (KAN-KubeAgent)** |
|-----------|--------------|---------|---------------|-------------------------|
| Kubernetes-specific | ✅ | ❌ | ❌ | ✅ |
| Agentic AI | ✅ | ❌ | Partial | ✅ |
| KAN used | ❌ | ✅ | ✅ | ✅ |
| Explainable decisions | ❌ | Partial | Partial | ✅ Full formula |
| Autonomous mutation | Blocked (HITL) | N/A | N/A | ✅ Risk-gated |
| Formal safety cert | ❌ | ❌ | ❌ | ✅ Lipschitz bound |
| K8s-native dataset | ❌ | ❌ | ❌ | ✅ K-RAD |
