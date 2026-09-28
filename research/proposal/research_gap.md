# Research Gap & Novelty Claims

## 1. Formal Gap Statement

### Gap 1 — No Interpretable Justification for Autonomous Training-Lifecycle Decisions

**Evidence from literature:**
- Hyperband (arXiv:1603.06560) and ASHA (arXiv:1810.05934) both stop underperforming training runs early, but the stopping rule is a bandit-style resource allocation score — there is no human-readable justification for any individual stop decision
- Population Based Training (PBT) and similar schedulers optimize hyperparameters online but likewise produce no inspectable reasoning trail
- No existing fine-tuning controller — bandit-based, RL-based, or LLM-based — outputs a symbolic formula explaining *why* a specific stop/adjust/continue decision was made

**Why this matters:**
Practitioners routinely override or ignore automated early-stopping because they cannot verify the reasoning behind it, wasting the GPU-hours the automation was meant to save. A decision that can be read and checked in one line ("stopped because loss_plateau_score=0.91 and remaining budget=8 GPU-hours") is far more trustworthy than a black-box bandit score.

**Our contribution:**
A KAN-based gating layer that produces an inspectable symbolic formula for every continue/adjust/stop decision — turning "the scheduler decided to stop" into "the formula `0.91·plateau(loss_slope) + 0.12·lr_decay_benefit − 0.40·remaining_gpu_hours = stop` says to stop."

---

### Gap 2 — KAN Has Never Been Used Inside an Agentic Training-Control Loop

**Evidence from literature:**
- All KAN papers to date (Liu et al. 2024; KAN-MID; GloroKAN; TKAN; the KAN RL-policy paper at arXiv:2506.16392) use KAN as a **standalone predictor or policy network**
- The KAN RL-policy paper applies KAN to network load-balancing decisions, but as a single model — not as a gating sub-component inside a multi-agent, LLM-orchestrated system
- Zero papers place a KAN *inside* an agent's decision pipeline as a gate that agents must pass every proposed action through

**Why this matters:**
Using KAN as a gate rather than the primary decision-maker is a different architectural role: agents handle observation and reasoning (what's happening, what might help), while KAN handles verification (is this specific action justified by the numbers). This separation is what makes the system both agentic *and* auditable.

**Our contribution:**
First system to use KAN as a **gating sub-component** inside an LLM-agent loop that controls a real Kubernetes-native training job, establishing a reusable pattern for trustworthy agentic MLOps.

---

### Gap 3 — No Agentic System Operates a Real Kubernetes-Native Training CRD End-to-End

**Evidence from literature:**
- Hyperband/ASHA implementations (Ray Tune, Optuna) typically run as a standalone Python process managing trials directly — they are not built around a Kubernetes-native training abstraction like Kubeflow's `TrainJob` CRD
- Multi-agent CloudOps frameworks (e.g. MOYA, arXiv:2501.08243) demonstrate general cloud-operations agents but do not target the ML fine-tuning lifecycle specifically, nor gate their actions through an interpretable verifier
- No paper combines: (a) a real, actively maintained K8s training CRD, (b) a multi-agent observation/decision loop, and (c) an interpretable gate between proposal and execution

**Why this matters:**
Building on `TrainJob` (rather than a bespoke script) means the system's actions are real Kubernetes operations — patches and deletes against a live CRD — not a simulation, and the "why Kubernetes" justification is concrete rather than incidental.

**Our contribution:**
An agentic controller that observes and patches a real Kubeflow `TrainJob`, with every mutating action passing through the KAN gate before it touches the cluster.

---

## 2. Novelty Claims (In Priority Order)

### Claim 1 (Primary — Architectural)
> **"We are the first to use a Kolmogorov-Arnold Network as an interpretable gating layer inside an LLM-orchestrated multi-agent training-lifecycle controller."**

Strength: Very strong. Zero prior art combines these.

### Claim 2 (Secondary — Applied)
> **"We are the first to apply KAN-gated agentic control to a real Kubeflow `TrainJob`, producing a formula-justified decision for every continue/adjust-LR/early-stop action."**

Strength: Strong. KAN has been applied to policy and control tasks in general, but never to K8s-native fine-tuning lifecycle management.

### Claim 3 (Tertiary — Evaluation)
> **"We compare KAN-gated decisions against Hyperband/ASHA on both GPU-hours saved and formula faithfulness (does the symbolic formula's dominant term match the feature that actually drove the decision)."**

Strength: Moderate-to-strong. Extends explainability evaluation methodology to the fine-tuning scheduling domain, where it hasn't previously been applied.

---

## 3. Research Questions

**RQ1 (Primary):** Can a KAN gate accurately decide when to continue, adjust learning rate, or early-stop a real fine-tuning job, and does its symbolic formula match what a human expert would point to as the deciding factor?

**RQ2:** How does a KAN-gated agentic controller compare to (a) manual babysitting and (b) Hyperband/ASHA in GPU-hours saved for a given final-accuracy target?

**RQ3:** What is the false-early-stop rate (jobs stopped that would have kept improving) and the wasted-compute rate (jobs kept running well past their plateau) for the KAN gate versus the baselines?

**RQ4:** Are the symbolic formulas learned by the KAN interpretable to ML practitioners, as measured by a small user study?

---

## 4. Comparison to Prior Art

| Criterion | Hyperband/ASHA | PBT | KAN RL-Policy | **Ours (KAN-KubeAgent)** |
|-----------|-----------------|-----|----------------|---------------------------|
| Kubernetes-native (real CRD) | ❌ | ❌ | ❌ | ✅ Kubeflow `TrainJob` |
| Agentic (multi-step reasoning) | ❌ | ❌ | Partial | ✅ |
| KAN used | ❌ | ❌ | ✅ | ✅ |
| Explainable decisions | ❌ | ❌ | Partial | ✅ Full formula per decision |
| Autonomous control action | ✅ (stop only) | ✅ | N/A | ✅ Continue / adjust LR / stop |
| Runs on local infra, no GPU cluster required | ✅ | ✅ | N/A | ✅ |
