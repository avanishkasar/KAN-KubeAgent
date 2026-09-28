# 📚 KAN-KubeAgent — Complete Learning Guide

> This guide teaches you everything you need to understand and build this project.
> Start from the top. Read in order. Each section builds on the previous one.

---

## 🗺️ Learning Map

```
LEVEL 1 — Foundations (Read First)
    ├── 1A: How Neural Networks Work (MLP)
    ├── 1B: How Kubernetes Works
    └── 1C: What is an AI Agent?

LEVEL 2 — The Three Technologies in This Project
    ├── 2A: KAN Networks — How They Work & Make Decisions
    ├── 2B: Kubernetes Deep Dive — Kubeflow TrainJob
    └── 2C: Agentic AI — LLM Agents + LangGraph

LEVEL 3 — How They Connect in KAN-KubeAgent
    ├── 3A: The Full System Walkthrough
    ├── 3B: How the KAN Makes a Gating Decision (Step by Step)
    └── 3C: How the Agent and KAN Work Together
```

**Time estimate:** 2-3 weeks reading, 2-3 weeks building

---

# LEVEL 1 — Foundations

---

## 1A: How Neural Networks Work (MLP)

*You need to understand this before KANs will make sense.*

### The basic idea

A neural network is a mathematical function that learns from examples.
It takes in some numbers → does math → outputs a prediction.

```
Input numbers ──▶ [Hidden Layers] ──▶ Output prediction
```

### What a single neuron does

```
inputs:  x₁ = 0.3,  x₂ = 0.7,  x₃ = 0.1
weights: w₁ = 2.0,  w₂ = -1.5, w₃ = 0.8

Step 1 — Weighted sum:
  z = (x₁ × w₁) + (x₂ × w₂) + (x₃ × w₃)
  z = (0.3×2.0) + (0.7×-1.5) + (0.1×0.8)
  z = 0.6 - 1.05 + 0.08 = -0.37

Step 2 — Activation function (makes it non-linear):
  output = ReLU(z) = max(0, -0.37) = 0
```

### What "training" means

The network starts with random weights. You show it 1000 examples.
For each wrong answer, you adjust the weights slightly in the right direction.
Do this 10,000 times → network gets accurate.

This adjustment process = **Backpropagation + Gradient Descent**.

### The problem with MLPs (Why KAN was invented)

After training an MLP, you can't look inside and see *why* it made a decision.
- It uses hundreds of weights
- No single weight means anything interpretable
- You can only observe: "input X → output Y"
- You cannot extract a human-readable formula

This is called the **black-box problem**.

**Resources to learn MLP:**
- [3Blue1Brown: Neural Networks playlist (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — 4 videos, 1 hour total. Best explanation ever made.
- [fast.ai Practical Deep Learning](https://course.fast.ai/) — free, practical

---

## 1B: How Kubernetes Works

### The core problem Kubernetes solves

Imagine you have an app. It gets popular. You need 100 copies of it running.
You need to:
- Start/stop copies automatically
- Restart crashed copies
- Update to new version without downtime
- Route traffic between copies

Doing this manually across many servers = nightmare.
Kubernetes does all of this **automatically**.

### Key concepts (the vocabulary)

```
CLUSTER
│
├── NODE (a physical/virtual machine, like a server)
│   └── POD (smallest unit — one or more containers running together)
│       └── CONTAINER (your actual app, packaged with Docker)
│
├── DEPLOYMENT (says: "I always want 3 copies of this app running")
│
├── SERVICE (gives pods a stable network address, load balances traffic)
│
├── NAMESPACE (like a folder — groups related resources together)
│
├── CRD (Custom Resource Definition — lets Kubernetes understand new
│        kinds of objects, e.g. a "TrainJob" for ML training)
│
└── CONFIGMAP (stores configuration — like a config file in the cluster)
```

### How Kubernetes makes decisions

There's a central brain called the **Control Plane**:

```
kube-apiserver     ← Everything talks to this. It's the "front door".

etcd               ← The database. Stores ALL cluster state.

kube-scheduler     ← Decides which Node to run a new Pod on.

controller-manager ← Watches the cluster. If a TrainJob says
                     "run until done" and the pod crashes, it
                     restarts it automatically.
```

### Kubeflow `TrainJob` — the CRD this project controls

Kubeflow Trainer adds a `TrainJob` object to Kubernetes — you describe a
training run declaratively, and Kubernetes manages the pod(s) that run it:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: finetune-run-01
spec:
  trainer:
    image: my-training-image:latest
    numNodes: 1
    resourcesPerNode:
      cpu: "4"
      memory: "8Gi"
```

When our agents want to change a running job — e.g. lower the learning
rate or stop it — that's a **patch** or **delete** on this `TrainJob`
object, exactly the same kind of API call as any other Kubernetes action:

```
PATCH /apis/trainer.kubeflow.org/v1alpha1/namespaces/default/trainjobs/finetune-run-01
verb:      PATCH
resource:  trainjobs
namespace: default
name:      finetune-run-01
```

**Resources to learn Kubernetes:**
- [Kubernetes official tutorials](https://kubernetes.io/docs/tutorials/) — start with "Hello Minikube"
- [TechWorld with Nana: Kubernetes Full Course (YouTube)](https://www.youtube.com/watch?v=X48VuDVv0do) — 4 hours, best free K8s course
- [Kubeflow Trainer docs](https://www.kubeflow.org/docs/components/trainer/) — the `TrainJob` CRD this project uses

---

## 1C: What is an AI Agent?

### The difference between a chatbot and an agent

| Chatbot | AI Agent |
|---------|---------|
| You ask → It answers | It has a **goal** |
| One exchange at a time | Plans and executes **multiple steps** |
| Only talks | Can **use tools** (search web, run code, call APIs) |
| Passive | Proactive — keeps going until goal is done |

### How an agent works (the loop)

```
┌─────────────────────────────────────────────┐
│              AGENT LOOP                      │
│                                              │
│  1. OBSERVE ──▶ What is the current state?   │
│                                              │
│  2. THINK ────▶ What should I do next?       │
│                 (LLM reasons here)           │
│                                              │
│  3. ACT ──────▶ Call a tool / execute code   │
│                                              │
│  4. OBSERVE ──▶ What happened? Did it work?  │
│                                              │
│  5. REPEAT until goal is achieved            │
└─────────────────────────────────────────────┘
```

### What is LangGraph?

LangGraph is a Python library for building agents as **state machines**.

```python
# Very simplified LangGraph structure
from langgraph.graph import StateGraph

graph = StateGraph()
graph.add_node("observe", observe_trainjob)      # what's the loss doing?
graph.add_node("plan", plan_action)              # what to do?
graph.add_node("verify", kan_gate_check)         # is it justified? ← KAN goes here
graph.add_node("execute", apply_to_trainjob)     # do it
graph.add_node("report", generate_report)        # tell the user

graph.add_edge("observe", "plan")
graph.add_conditional_edge("verify", route_by_gate_decision, {
    "continue":   "report",
    "adjust_lr":  "execute",
    "early_stop": "execute"
})
```

**Resources to learn Agentic AI:**
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — official, has tutorials
- [Andrew Ng: AI Agents course (DeepLearning.AI)](https://learn.deeplearning.ai/) — free short courses

---

# LEVEL 2 — The Three Technologies in Depth

---

## 2A: KAN Networks — How They Work & Make Decisions

### Recall the MLP problem

MLP edge = scalar weight (just a number)
MLP node = fixed activation function (ReLU, Sigmoid)

### The KAN solution: put learnable functions ON the edges

```
MLP:
  x ──[w=0.7]──▶ node ──[ReLU(·)]──▶ output

KAN:
  x ──[φ(x) = learned curve]──▶ node ──[sum]──▶ output
```

In a KAN, each edge learns its own custom function.
The node just adds up what comes in.

### What does "B-spline" mean?

A B-spline is a smooth, flexible curve made of polynomial pieces joined together.

```
           ┃
    1.0  ──┼───────────╮
           ┃           │  ← each segment is a polynomial
    0.5  ──┼──────╮    │
           ┃      │    ╰──────╮
    0.0  ──┼──────┴───────────┴────────
           0    0.25  0.5   0.75   1.0
                    input x
```

After training, you can plot this curve and see: "the network learned a
sine wave for this feature" or "it learned a quadratic."

### How KAN makes a gating decision (step by step)

Suppose we have 3 features: `loss_plateau_score`, `gradient_trend`, `gpu_hours_remaining`

```
Step 1 — Each feature goes through its learned edge function:

  loss_plateau_score = 0.9  ──▶  φ₁(0.9) = 0.72   (learned curve)
  gradient_trend      = 0.1  ──▶  φ₂(0.1) = -0.15  (learned curve)
  gpu_hours_remaining = 0.2  ──▶  φ₃(0.2) = -0.30  (learned curve)

Step 2 — Hidden node sums the outputs:

  hidden = 0.72 + (-0.15) + (-0.30) = 0.27

Step 3 — Hidden node goes through next layer's edge functions:

  0.27  ──▶  φ₄(0.27) = 0.55  (another learned curve)

Step 4 — Final node sums → stop score:

  stop_score = sigmoid(0.55) × 100 = 63.4
```

**Result:** Stop score is 63.4 → falls in the "adjust learning rate" band, not full stop.

### The symbolic regression step (the magic)

After training, KAN tries to identify the mathematical names of the learned curves:

```python
model.auto_symbolic(lib=['x', 'x^2', 'sin', 'exp', 'log'])

# KAN scans each edge and finds the closest known function:
# φ₁ looks like:  0.8 * x         (linear relationship with loss_plateau_score)
# φ₂ looks like: -0.3 * x^2       (quadratic — gradient still moving matters a lot)
# φ₃ looks like: -0.5 * x         (linear penalty as budget runs out)

# Final human-readable formula:
# stop_score ≈ 0.8*(loss_plateau_score) - 0.3*(gradient_trend²) - 0.5*(gpu_hours_remaining)
```

A practitioner can read this formula and say:
- "Yes, plateau score is weighted most — makes sense, that's the main stop signal"
- "Yes, a moving gradient pulls the score down hard — don't stop if it's still learning"
- "Running low on GPU budget pushes toward stopping — that's the cost tradeoff we wanted"

### How to install and run KAN

```python
# Install
pip install pykan

# Basic example
import torch
from kan import KAN

# Create KAN: 3 inputs → 4 hidden → 1 output
model = KAN(width=[3, 4, 1], grid=5, k=3)

# Sample data: 3 features, 1 stop-score label
X = torch.tensor([
    [0.9, 0.1, 0.2],  # plateaued, gradient nearly flat, low budget left — stop
    [0.1, 0.8, 0.9],  # still improving fast, plenty of budget — keep going
    [0.6, 0.4, 0.5],  # borderline — could go either way
])
y = torch.tensor([[85.0], [10.0], [50.0]])  # stop scores

dataset = {'train_input': X, 'train_label': y,
           'test_input': X,  'test_label': y}

model.train(dataset, opt='Adam', steps=300, lamb=0.01)

model.plot()            # visualise each edge's learned function
model.auto_symbolic()   # identify symbolic names
print(model.symbolic_formula())
```

---

## 2B: Kubernetes Deep Dive — Kubeflow TrainJob

### What we interact with in Kubernetes

Our agent layer needs to:
1. **Read** the live status of a `TrainJob` → loss, gradients, LR, elapsed time
2. **Propose** a control action (e.g. "reduce LR to 1e-5")
3. **Gate** that action with KAN → is it justified?
4. **Apply** the action to the cluster (patch or delete the `TrainJob`)

### Kubeflow Trainer Python client (how to talk to a TrainJob from code)

```python
from kubeflow.trainer import TrainerClient

client = TrainerClient()

# List running TrainJobs
jobs = client.list_jobs()
for job in jobs:
    print(f"{job.name}: {job.status}")

# Get the current status of a specific job (loss, step, etc. come from
# the training container's own metrics logging, read via job logs/events)
status = client.get_job(name="finetune-run-01")

# Update the job (needs KAN gate approval first!)
client.update_job(name="finetune-run-01", train_func_parameters={"lr": 1e-5})

# Stop the job (needs KAN gate approval first!)
client.delete_job(name="finetune-run-01")
```

### How we extract the 5 KAN features from a TrainJob

```python
def extract_features(loss_history: list, lr_schedule: dict, budget: dict) -> dict:
    """
    Given the recent training history of a TrainJob, compute the 5
    features that the KAN gate uses to decide continue/adjust/stop.
    """

    return {
        # 1. How flat has the loss curve been in the last K epochs?
        "loss_plateau_score": compute_plateau_score(loss_history, window=5),

        # 2. Is the gradient norm still shrinking, or has it flattened?
        "gradient_trend": compute_gradient_trend(loss_history),

        # 3. Estimated improvement from halving the learning rate now
        "lr_decay_benefit": estimate_lr_decay_benefit(loss_history, lr_schedule),

        # 4. Fraction of GPU-hour budget left
        "gpu_hours_remaining_vs_budget": budget["remaining_hours"] / budget["total_hours"],

        # 5. Epochs since the best validation loss seen so far
        "epochs_since_improvement": epochs_since_best(loss_history) / 30  # normalised
    }
```

---

## 2C: Agentic AI — LLM Agents + LangGraph

### How the LLM agent "thinks"

The LLM (Claude, in this project) acts as the brain.
It receives a **prompt** describing the current training state and available tools.
It outputs either a thought (reasoning step) or a tool call (action to take).

```
Prompt sent to LLM:
"You are a fine-tuning supervisor agent. Current TrainJob state:
- Job 'finetune-run-01', epoch 12 of 30
- Loss has been flat for the last 5 epochs (plateau_score=0.91)
- 8 GPU-hours remaining out of a 20-hour budget
- Learning rate has not been reduced since epoch 0

Available tools:
- get_trainjob_status(name) → returns loss/gradient/LR history
- kan_gate_check(features) → returns decision + formula [ALWAYS call before mutating]
- patch_learning_rate(name, new_lr) → [MUTATING - needs KAN gate approval]
- stop_trainjob(name) → [MUTATING - needs KAN gate approval]

What should you do?"

LLM response:
"Thought: Loss has plateaued and LR has never been decayed — worth
checking if a lower LR helps before considering a full stop.
Action: kan_gate_check(features={...})
..."
```

### LangGraph state machine for our project

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    loss_history: list
    proposed_action: dict
    gate_decision: str       # continue / adjust_lr / early_stop
    gate_formula: str
    executed: bool
    audit_log: list

def observe_node(state: AgentState) -> AgentState:
    """Metrics Watcher: pull latest TrainJob status"""
    history = get_trainjob_status("finetune-run-01")
    return {**state, "loss_history": history}

def plan_node(state: AgentState) -> AgentState:
    """Loss-Curve Analyst + Cost Estimator feed the supervisor's proposal"""
    action = llm_agent.plan(state["loss_history"])
    return {**state, "proposed_action": action}

def kan_gate_node(state: AgentState) -> AgentState:
    """Every proposed action is gated by the KAN before it can execute"""
    features = extract_features(state["loss_history"], ..., ...)
    result = kan_gate.decide(features)
    return {**state, "gate_decision": result.decision, "gate_formula": result.formula}

def route_by_gate(state: AgentState) -> str:
    return state["gate_decision"]  # "continue" | "adjust_lr" | "early_stop"

def execute_node(state: AgentState) -> AgentState:
    apply_action(state["proposed_action"])
    return {**state, "executed": True}

workflow = StateGraph(AgentState)
workflow.add_node("observe",   observe_node)
workflow.add_node("plan",      plan_node)
workflow.add_node("kan_gate",  kan_gate_node)
workflow.add_node("execute",   execute_node)
workflow.add_node("report",    log_and_report_node)

workflow.set_entry_point("observe")
workflow.add_edge("observe",  "plan")
workflow.add_edge("plan",     "kan_gate")
workflow.add_conditional_edges("kan_gate", route_by_gate, {
    "continue":   "report",
    "adjust_lr":  "execute",
    "early_stop": "execute"
})
workflow.add_edge("execute", "report")
workflow.add_edge("report",  END)

app = workflow.compile()
```

**Resources to learn Agentic AI:**
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [Claude API docs](https://docs.claude.com/) — tool use / agent loops

---

# LEVEL 3 — How Everything Connects in KAN-KubeAgent

---

## 3A: The Full System Walkthrough

**Scenario:** A fine-tuning run has plateaued and is burning through its GPU budget.

### Step-by-step what happens

```
[Epoch 12] TrainJob 'finetune-run-01' status:
  loss:  2.31, 2.30, 2.29, 2.29, 2.28   (last 5 epochs — barely moving)
  lr:    2e-4 (unchanged since start)
  budget: 8 GPU-hours remaining of 20

[Epoch 12] Metrics Watcher agent pulls this from the Trainer client.

[Epoch 12] Loss-Curve Analyst computes:
  loss_plateau_score = 0.91  (very flat)
  gradient_trend      = 0.08  (nearly zero — barely moving)

[Epoch 12] Cost Estimator computes:
  gpu_hours_remaining_vs_budget = 0.40
  epochs_since_improvement      = 0.30 (normalised)

[Epoch 12] Supervisor proposes: "consider reducing LR before stopping"

[Epoch 12] KAN Gate scores the situation:

  stop_score = 91.0  ← ABOVE the early-stop threshold

  formula: stop_score = 0.91·(loss_plateau_score)
                        + 0.12·(lr_decay_benefit)
                        - 0.40·(gpu_hours_remaining_vs_budget)

  "Loss has plateaued hard and LR was never decayed — but the dominant
   term is still the plateau score, and budget is under half gone, so:
   EARLY-STOP and reallocate the remaining 8 GPU-hours."

[Epoch 12] Executor stops the job.
  → Kubeflow API: DELETE trainjobs/finetune-run-01 ✓

[Epoch 12] Audit log entry created:
  action:        DELETE trainjobs/finetune-run-01
  triggered_by:  KAN-KubeAgent loss-plateau detection
  stop_score:    91.0
  formula:       [formula above]
  decision:      EARLY_STOPPED
  result:        SUCCESS, freed 8 GPU-hours

[Epoch 12] Notification sent:
  "⏹ EARLY-STOPPED: finetune-run-01 at epoch 12/30
   KAN Stop Score: 91.0/100
   Formula: stop = 0.91·plateau - 0.40·gpu_budget_remaining + 0.12·lr_benefit
   Predicted accuracy gain from continuing: <2% for 8 more GPU-hours
   View full audit: dashboard.link/audit/finetune-run-01"
```

**Zero human intervention required for the decision. Full audit trail with a readable formula.**

---

## 3B: How the KAN Makes a Gating Decision (Step by Step)

Let's trace through the plateau scenario above with the actual math:

```
Input feature vector:
x = [0.91, 0.08, 0.65, 0.40, 0.30]
     plateau  grad  lr_benefit  budget_left  since_improve

KAN Layer 1 (each of the 5 features goes through its learned edge function):

  x₁=0.91 → φ₁(0.91) →  0.78   (plateau: strong linear weighting)
  x₂=0.08 → φ₂(0.08) →  0.02   (gradient nearly flat — barely contributes)
  x₃=0.65 → φ₃(0.65) →  0.31   (moderate benefit from decaying LR)
  x₄=0.40 → φ₄(0.40) → -0.24   (some budget left — pulls slightly against stopping)
  x₅=0.30 → φ₅(0.30) →  0.09   (mild — not that long since improvement)

KAN Hidden Layer (3 nodes, each summing subsets):

  h₁ = φ₁+φ₂ = 0.78+0.02 = 0.80 → ψ₁(0.80) = 0.71  (plateau-signal node)
  h₂ = φ₃+φ₄ = 0.31-0.24 = 0.07 → ψ₂(0.07) = 0.06
  h₃ = φ₅ (alone)                → ψ₃(0.09) = 0.08

KAN Output Layer (1 node):

  raw_output = ψ_out(h₁ + h₂ + h₃)
             = ψ_out(0.71 + 0.06 + 0.08)
             = ψ_out(0.85)
             = 1.45

Stop Score = sigmoid(1.45) × 100 = 81.0 → "EARLY-STOP" band
```

### After training, symbolic regression finds:
```
stop_score ≈ 0.91·(loss_plateau_score)
           + 0.12·(lr_decay_benefit)
           - 0.40·(gpu_hours_remaining_vs_budget)
```

A practitioner looks at this and says: *"Good — the model correctly
weighted the plateau signal highest, gave some credit to a possible LR
decay helping, and appropriately discounted the decision by how much
budget was left. This makes engineering sense."*

---

## 3C: How the Agent and KAN Work Together

The key relationship: **the agents propose, the KAN gates.**

```
AGENTS' JOB:
  - Observe the TrainJob (loss, gradients, LR, budget)
  - Understand what's happening (using LLM reasoning)
  - Propose WHAT action might help (continue / adjust LR / stop)
  - Do NOT decide IF it's justified → that's the KAN gate's job

KAN GATE'S JOB:
  - Receive the proposed action's context features
  - Output a decision with a formula
  - Never interpret context, never reason — just score
  - The score is deterministic for the same inputs
```

**Why not just let the LLM decide when to stop?**

The LLM's judgement:
- Changes with temperature (non-deterministic)
- Is not auditable — "the LLM thought it had plateaued" is not a verifiable record
- Can hallucinate — "loss looks flat to me" without checking the actual numbers rigorously
- Cannot produce a mathematical formula

The KAN gate's judgement:
- Fully deterministic for the same inputs
- Produces an auditable symbolic formula
- Trained on real historical training-run data
- Can be checked against what a human expert would have decided

**This is the core research contribution.**

---

# 📋 Your Learning Checklist

Use this to track what you've learned:

## Foundations

> 🏆 **Certifications already completed — items marked [CERT] below are already done.**
> LFS158 (Intro to Kubernetes) + LFS147 (AI/ML Toolkits with Kubeflow) — Linux Foundation, August 2026

- [ ] Watched 3Blue1Brown neural network series (4 videos)
- [ ] Understand what weights, activations, and backprop are
- [CERT] Ran `minikube start` and deployed a test pod ✅
- [CERT] Understand pods, namespaces, deployments, services ✅
- [CERT] Know what Kubeflow Pipelines are and how they run on K8s ✅
- [ ] Installed Kubeflow Trainer and created a `TrainJob` on Minikube
- [ ] Understand what an AI agent loop is (observe → think → act)

## KAN
- [ ] Installed `pykan` and ran the hello-world example in `weekly_log.md`
- [ ] Can explain what a B-spline is (even roughly)
- [ ] Understand the difference between MLP and KAN architectures
- [ ] Ran `model.auto_symbolic()` and read the output
- [ ] Can explain why symbolic regression is useful for a gating decision

## Kubernetes / Kubeflow
- [ ] Can run `kubectl get pods --all-namespaces`
- [ ] Can create a `TrainJob` with `kubectl apply -f trainjob.yaml`
- [ ] Can read a `TrainJob`'s status and understand its phases/conditions
- [ ] Set up the Kubeflow Trainer Python client and queried a running job

## Agentic AI
- [ ] Read LangGraph "Getting Started" tutorial
- [ ] Built a simple 3-node LangGraph (observe → plan → act)
- [ ] Understand what a state machine is
- [ ] Can add a conditional edge (branch based on a value)

## Full System
- [ ] Can trace the full flow: TrainJob metrics → agents → KAN gate → execute
- [ ] Understand why the KAN gate is in the middle (not the LLM)
- [ ] Can explain the 5 features and why each one matters
- [ ] Can explain the 3 decision bands (continue / adjust LR / early-stop)
- [ ] Can draw the system architecture from memory

---

# 🔗 All Learning Resources

| Topic | Resource | Format | Time |
|-------|----------|--------|------|
| Neural networks | [3Blue1Brown YouTube playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Video | 1 hour |
| Deep learning | [fast.ai course](https://course.fast.ai) | Course | 7 hours |
| KAN paper | [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) | Paper | 3 hours |
| pykan library | [github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) | Code | ongoing |
| Kubernetes | [TechWorld with Nana - K8s Full Course](https://www.youtube.com/watch?v=X48VuDVv0do) | Video | 4 hours |
| K8s official | [kubernetes.io/docs/tutorials](https://kubernetes.io/docs/tutorials/) | Docs | 2 hours |
| Kubeflow Trainer | [kubeflow.org/docs/components/trainer](https://www.kubeflow.org/docs/components/trainer/) | Docs | 2 hours |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | Docs | 2 hours |
| Agentic AI | [DeepLearning.AI short courses](https://learn.deeplearning.ai/) | Course | 2 hours |
| Hyperband | [arXiv:1603.06560](https://arxiv.org/abs/1603.06560) | Paper | 1 hour |

**Suggested order:** 3Blue1Brown → K8s Nana video → pykan hello world → Kubeflow Trainer docs → LangGraph tutorial → KAN paper → Hyperband paper
