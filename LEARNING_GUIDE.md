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
    ├── 2B: Kubernetes Deep Dive — Security, RBAC, Audit Logs
    └── 2C: Agentic AI — LLM Agents + LangGraph

LEVEL 3 — How They Connect in KAN-KubeAgent
    ├── 3A: The Full System Walkthrough
    ├── 3B: How the KAN Makes a Trust Decision (Step by Step)
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
│   Examples: namespace "prod", namespace "staging", namespace "monitoring"
│
├── CONFIGMAP (stores configuration — like a config file in the cluster)
│
└── SECRET (stores sensitive data — passwords, API keys)
```

### How Kubernetes makes decisions

There's a central brain called the **Control Plane**:

```
kube-apiserver   ← Everything talks to this. It's the "front door".
                   Every action goes through here and gets logged.

etcd             ← The database. Stores ALL cluster state.

kube-scheduler   ← Decides which Node to run a new Pod on.

controller-manager ← Watches the cluster. If you said "3 replicas"
                     and one crashes, it starts a new one automatically.
```

### What an "API call" to Kubernetes looks like

When you (or an agent) does something to the cluster, it's an HTTP request:

```
DELETE /api/v1/namespaces/production/pods/my-database-pod-xyz
verb:      DELETE
resource:  pods
namespace: production
name:      my-database-pod-xyz
```

This gets **logged in the Audit Log** — which is our data source for the KAN.

### RBAC — Role Based Access Control

RBAC controls *who* can do *what* to *which* resources.

```
User "deploy-bot"
  └── has RoleBinding → Role "deployer"
        └── Role allows: [GET, LIST, CREATE, PATCH] on [pods, deployments]
        └── Role denies: [DELETE] on [secrets, clusterroles]
```

**Why this matters for our project:**
- A ClusterAdmin can do everything → HIGH risk actor
- A read-only service account can only GET → LOW risk actor
- The KAN uses `actor_privilege_level` as one of its 8 features

### The Audit Log (our data source)

Every API call is recorded:

```json
{
  "verb": "delete",
  "resource": "pods",
  "namespace": "production",
  "user": {"username": "system:serviceaccount:monitoring:agent"},
  "responseStatus": {"code": 200},
  "requestReceivedTimestamp": "2026-08-12T14:22:01Z",
  "objectRef": {"name": "payment-service-7d9f4b-xk2p"}
}
```

**Resources to learn Kubernetes:**
- [Kubernetes official tutorials](https://kubernetes.io/docs/tutorials/) — start with "Hello Minikube"
- [TechWorld with Nana: Kubernetes Full Course (YouTube)](https://www.youtube.com/watch?v=X48VuDVv0do) — 4 hours, best free K8s course

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

A state machine is a system that:
- Has defined **states** (e.g., "observing", "planning", "executing")
- Has defined **transitions** between states
- Is deterministic — you know exactly what it will do next

```python
# Very simplified LangGraph structure
from langgraph.graph import StateGraph

graph = StateGraph()
graph.add_node("observe", observe_cluster)      # what's happening?
graph.add_node("plan", plan_action)             # what to do?
graph.add_node("verify", kan_trust_check)       # is it safe? ← KAN goes here
graph.add_node("execute", apply_to_cluster)     # do it
graph.add_node("report", generate_report)       # tell the user

graph.add_edge("observe", "plan")
graph.add_conditional_edge("verify", check_trust_score, {
    "auto_apply": "execute",
    "human_review": "wait_for_human",
    "reject": "report"
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

Think of it like this:
- Take the range of your input [0, 1]
- Divide it into 5 equal intervals (the "grid")
- In each interval, fit a smooth polynomial curve
- Join them together smoothly

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

The **shape** of this curve is what gets learned during training.
After training, you can plot the curve and see: "the network learned a sine wave for this feature" or "it learned a quadratic."

### How KAN makes a decision (step by step)

Suppose we have 3 features: `namespace_risk`, `blast_radius`, `change_frequency`

```
Step 1 — Each feature goes through its learned edge function:

  namespace_risk = 0.9  ──▶  φ₁(0.9) = 0.72   (learned curve)
  blast_radius   = 0.3  ──▶  φ₂(0.3) = -0.15  (learned curve)
  change_freq    = 0.1  ──▶  φ₃(0.1) = 0.08   (learned curve)

Step 2 — Hidden node sums the outputs:

  hidden = 0.72 + (-0.15) + 0.08 = 0.65

Step 3 — Hidden node goes through next layer's edge functions:

  0.65  ──▶  φ₄(0.65) = 0.81  (another learned curve)

Step 4 — Final node sums → trust score:

  trust_score = sigmoid(0.81) × 100 = 69.2
```

**Result:** Trust score is 69.2 → Human review required (40-80 range)

### The symbolic regression step (the magic)

After training, KAN tries to identify the mathematical names of the learned curves:

```python
model.auto_symbolic(lib=['x', 'x^2', 'sin', 'exp', 'log'])

# KAN scans each edge and finds the closest known function:
# φ₁ looks like:  0.8 * x         (linear relationship with namespace_risk)
# φ₂ looks like: -0.4 * x^2      (quadratic penalty for blast_radius)
# φ₃ looks like:  0.6 * sin(x)   (periodic relationship with change_frequency)

# Final human-readable formula:
# trust_score ≈ 0.8*(namespace_risk) - 0.4*(blast_radius²) + 0.6*sin(change_freq)
```

Now a security engineer can read this formula and say:
- "Yes, namespace risk is weighted most — makes sense"
- "Yes, blast radius is penalised quadratically — aggressive changes are very risky"
- "The sine of change frequency is interesting — suggests cyclic patterns in safe/unsafe changes"

### How to install and run KAN

```python
# Install
pip install pykan

# Basic example
import torch
from kan import KAN

# Create KAN: 3 inputs → 4 hidden → 1 output
model = KAN(width=[3, 4, 1], grid=5, k=3)

# Sample data: 3 features, 1 trust score label
X = torch.tensor([
    [0.9, 0.3, 0.1],  # high namespace risk, low blast radius, low frequency
    [0.1, 0.1, 0.5],  # low risk, low blast, medium frequency
    [0.8, 0.9, 0.8],  # high risk, high blast, high frequency — very dangerous
])
y = torch.tensor([[25.0], [90.0], [5.0]])  # trust scores

# Train
dataset = {'train_input': X, 'train_label': y,
           'test_input': X,  'test_label': y}

model.train(dataset, opt='Adam', steps=300, lamb=0.01)

# Check learned formulas
model.plot()            # visualise each edge's learned function
model.auto_symbolic()   # identify symbolic names
print(model.symbolic_formula())
```

---

## 2B: Kubernetes Deep Dive — For This Project

### What we interact with in Kubernetes

Our agent needs to:
1. **Read** audit logs → detect security problems
2. **Propose** a fix (e.g., "delete this misconfigured ClusterRoleBinding")
3. **Score** that fix with KAN → is it safe?
4. **Apply** or escalate to human

### Kubernetes Python Client (how to talk to K8s from code)

```python
from kubernetes import client, config

# Load credentials (from local ~/.kube/config or in-cluster service account)
config.load_kube_config()

v1 = client.CoreV1Api()
rbac = client.RbacAuthorizationV1Api()

# List all pods in the production namespace
pods = v1.list_namespaced_pod(namespace="production")
for pod in pods.items:
    print(f"{pod.metadata.name}: {pod.status.phase}")

# List all ClusterRoleBindings (RBAC audit)
bindings = rbac.list_cluster_role_binding()
for b in bindings.items:
    role = b.role_ref.name
    subjects = [s.name for s in (b.subjects or [])]
    if role == "cluster-admin":
        print(f"WARNING: cluster-admin bound to: {subjects}")

# Delete a dangerous ClusterRoleBinding (needs KAN approval first!)
rbac.delete_cluster_role_binding(name="suspicious-admin-binding")
```

### How we extract the 8 KAN features from Kubernetes

```python
def extract_features(proposed_action: dict, cluster_state: dict) -> dict:
    """
    Given a proposed Kubernetes action, compute the 8 risk features
    that the KAN trust layer will use to score safety.
    """
    
    namespace = proposed_action["namespace"]
    verb      = proposed_action["verb"]       # GET, CREATE, PATCH, DELETE
    resource  = proposed_action["resource"]   # pods, clusterrolebindings, secrets
    actor     = proposed_action["actor"]      # service account name
    
    return {
        # 1. How critical is this namespace?
        "namespace_risk_score": {
            "production": 1.0,
            "staging":    0.6,
            "dev":        0.2
        }.get(namespace, 0.5),
        
        # 2. How many downstream services depend on this resource?
        "blast_radius": compute_blast_radius(resource, namespace, cluster_state),
        
        # 3. How often has this resource been changed recently?
        "change_frequency": get_change_rate(resource, namespace, last_24h=True),
        
        # 4. What privilege level does the actor have?
        "actor_privilege_level": get_rbac_privilege_level(actor),
        
        # 5. How critical is this resource type?
        "resource_criticality": {
            "secret":             1.0,
            "clusterrole":        0.9,
            "networkpolicy":      0.9,
            "clusterrolebinding": 0.85,
            "deployment":         0.6,
            "configmap":          0.3,
            "pod":                0.5
        }.get(resource, 0.5),
        
        # 6. How risky is this action verb?
        "action_verb_risk": {
            "delete": 0.9,
            "patch":  0.6,
            "create": 0.5,
            "update": 0.5,
            "get":    0.05,
            "list":   0.05
        }.get(verb, 0.5),
        
        # 7. Historical failure rate for this action type
        "historical_failure_rate": get_failure_rate(verb, resource),
        
        # 8. Time since last deployment (fresh deployments = higher risk)
        "time_since_last_deploy": get_hours_since_last_deploy(namespace) / 720  # normalise to 30 days
    }
```

---

## 2C: Agentic AI — LLM Agents + LangGraph

### How the LLM agent "thinks"

The LLM (like GPT-4o, Claude, or Gemini) acts as the brain.
It receives a **prompt** describing the cluster situation and available tools.
It outputs either:
- A thought (reasoning step)
- A tool call (action to take)

```
Prompt sent to LLM:
"You are a Kubernetes security agent. Current cluster state:
- Pod 'payment-service' in namespace 'production' has been crashlooping for 10 min
- Audit logs show: 3 DELETE attempts on NetworkPolicy 'deny-all' in last hour
- ClusterRoleBinding 'temp-admin' grants cluster-admin to service account 'unknown-bot'

Available tools:
- get_pod_logs(pod_name, namespace) → returns recent logs
- list_rbac_bindings(namespace) → returns role bindings
- delete_rbac_binding(name) → deletes a ClusterRoleBinding [MUTATING - needs KAN approval]
- patch_network_policy(name, namespace, spec) → [MUTATING - needs KAN approval]

What should you investigate and fix?"

LLM response:
"Thought: The ClusterRoleBinding 'temp-admin' to an unknown service account is suspicious.
I should investigate this first, then consider removing it.
Action: list_rbac_bindings(namespace='kube-system')
..."
```

### LangGraph state machine for our project

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    cluster_alerts: List[str]      # what anomalies were detected
    proposed_action: dict           # what the agent wants to do
    trust_score: float              # KAN's trust score for the action
    trust_formula: str              # KAN's explanation
    human_approved: bool            # did a human approve it?
    executed: bool                  # was the action applied?
    audit_log: List[dict]           # full trace of decisions

def observe_node(state: AgentState) -> AgentState:
    """Scan cluster for security issues"""
    alerts = detect_anomalies_in_audit_logs()
    return {**state, "cluster_alerts": alerts}

def plan_node(state: AgentState) -> AgentState:
    """LLM decides what action to take"""
    action = llm_agent.plan(state["cluster_alerts"])
    return {**state, "proposed_action": action}

def kan_verify_node(state: AgentState) -> AgentState:
    """KAN scores the proposed action"""
    features = extract_features(state["proposed_action"])
    result   = kan_model.score(features)
    return {**state,
            "trust_score":   result.score,
            "trust_formula": result.formula}

def route_by_trust(state: AgentState) -> str:
    """Routing logic based on KAN score"""
    score = state["trust_score"]
    if score >= 80:   return "auto_execute"
    elif score >= 40: return "human_review"
    else:             return "auto_reject"

def execute_node(state: AgentState) -> AgentState:
    """Apply the action to the cluster"""
    apply_action(state["proposed_action"])
    return {**state, "executed": True}

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("observe",      observe_node)
workflow.add_node("plan",         plan_node)
workflow.add_node("kan_verify",   kan_verify_node)
workflow.add_node("execute",      execute_node)
workflow.add_node("human_review", request_human_approval_node)
workflow.add_node("reject",       log_rejection_node)

workflow.set_entry_point("observe")
workflow.add_edge("observe",    "plan")
workflow.add_edge("plan",       "kan_verify")
workflow.add_conditional_edges("kan_verify", route_by_trust, {
    "auto_execute": "execute",
    "human_review": "human_review",
    "auto_reject":  "reject"
})
workflow.add_edge("execute",      END)
workflow.add_edge("human_review", "execute")  # if approved
workflow.add_edge("reject",       END)

app = workflow.compile()
```

---

# LEVEL 3 — How Everything Connects in KAN-KubeAgent

---

## 3A: The Full System Walkthrough

**Scenario:** An attacker compromises a CI/CD service account and creates a rogue `ClusterRoleBinding` that gives `cluster-admin` to `serviceaccount:default`.

### Step-by-step what happens

```
[10:00:01] Kubernetes logs:
  CREATE clusterrolebindings "rogue-admin" by serviceaccount:ci-bot
  subjects: [serviceaccount:default]
  roleRef: ClusterRole/cluster-admin

[10:00:02] Our Observation Agent detects this in audit log.
  → Alert: "Privilege escalation attempt detected"

[10:00:03] Supervisor LLM receives alert.
  → Plans action: DELETE clusterrolebindings "rogue-admin"

[10:00:04] Feature extraction for KAN:
  namespace_risk_score  = 1.0  (cluster-scoped, not namespaced)
  blast_radius          = 0.95 (cluster-admin affects EVERYTHING)
  change_frequency      = 0.8  (recent spike in admin-level changes)
  actor_privilege_level = 0.7  (ci-bot has moderate privileges)
  resource_criticality  = 0.85 (ClusterRoleBinding is very sensitive)
  action_verb_risk      = 0.9  (DELETE is highest risk verb)
  historical_failure_rate= 0.05 (deletions usually succeed)
  time_since_last_deploy= 0.1  (recent deployment window)

[10:00:04] KAN Trust Layer scores the proposed DELETE action:
  trust_score = 82.4  ← ABOVE 80 → AUTO-APPLY

  formula: trust = 0.78·f(blast_radius) + 0.65·sin(detection_confidence)
                   - 0.42·(change_frequency²) + 0.31·actor_privilege
  
  "High confidence this is a true positive attack — auto-apply approved"

[10:00:05] Executor deletes the rogue ClusterRoleBinding.
  → Kubernetes API: DELETE clusterrolebindings/rogue-admin ✓

[10:00:05] Audit log entry created:
  action:        DELETE clusterrolebindings/rogue-admin
  triggered_by:  KAN-KubeAgent anomaly detection
  trust_score:   82.4
  trust_formula: [formula above]
  decision:      AUTO_APPLIED
  result:        SUCCESS

[10:00:06] Security engineer gets Slack notification:
  "⚡ AUTO-REMEDIATED: Deleted rogue cluster-admin binding
   KAN Trust Score: 82.4/100
   Formula: trust = 0.78·f(blast) + 0.65·sin(confidence) - 0.42·freq²
   View full audit: dashboard.link/audit/1234"
```

**Total time: 5 seconds. Zero human intervention. Full audit trail.**

---

## 3B: How the KAN Makes a Trust Decision (Step by Step)

Let's trace through the DELETE action above with the actual math:

```
Input feature vector:
x = [1.0, 0.95, 0.8, 0.7, 0.85, 0.9, 0.05, 0.1]
     ns    blast  freq  actor  res   verb  fail  deploy

KAN Layer 1 (each of the 8 features goes through its learned edge function):

  x₁=1.0  → φ₁(1.0) →  0.78   (namespace_risk: linear high-risk weighting)
  x₂=0.95 → φ₂(0.95)→  0.71   (blast_radius: this is HIGH but attack-like)
  x₃=0.8  → φ₃(0.8) → -0.12   (change_frequency: penalises spike — suspicious!)
  x₄=0.7  → φ₄(0.7) →  0.45   (actor privilege: moderate weight)
  x₅=0.85 → φ₅(0.85)→  0.63   (resource criticality)
  x₆=0.9  → φ₆(0.9) →  0.58   (verb risk: DELETE is high risk)
  x₇=0.05 → φ₇(0.05)→  0.02   (low failure rate: action will succeed)
  x₈=0.1  → φ₈(0.1) →  0.04   (recent deploy — slightly elevated risk)

KAN Hidden Layer (5 nodes, each summing subsets):

  h₁ = φ₁+φ₂ = 0.78+0.71 = 1.49 → ψ₁(1.49) = 0.82  (attack signature node)
  h₂ = φ₃+φ₄ = -0.12+0.45 = 0.33 → ψ₂(0.33) = 0.31
  h₃ = φ₅+φ₆ = 0.63+0.58 = 1.21 → ψ₃(1.21) = 0.75
  h₄ = φ₇+φ₈ = 0.02+0.04 = 0.06 → ψ₄(0.06) = 0.06
  h₅ = all features sum → ψ₅(sum) = 0.68

KAN Output Layer (1 node):

  raw_output = ψ_out(h₁ + h₂ + h₃ + h₄ + h₅)
             = ψ_out(0.82 + 0.31 + 0.75 + 0.06 + 0.68)
             = ψ_out(2.62)
             = 0.646

Trust Score = sigmoid(0.646) × 100 = 65.6 → "Human Review"

Wait — but above we said 82.4?

The difference: the KAN also takes in a derived feature:
  "detection_confidence" = how sure the anomaly detector was this is an attack
  In this case: 0.97 (very sure — cluster-admin binding with no prior history)

With this feature added:
  Trust Score = 82.4 → AUTO-APPLY ✓

This shows why feature engineering matters as much as the KAN itself.
```

### After training, symbolic regression finds:
```
trust ≈ 0.78·(namespace_risk) 
      + 0.65·sin(detection_confidence)  ← sin because confidence wraps around
      - 0.42·(change_frequency²)        ← squared to punish rapid changes hard
      + 0.31·(actor_privilege)
```

A security engineer looks at this and says: *"Good. The model correctly identified that attack detection confidence drives the trust score up, while rapid suspicious changes drive it down. This makes security sense."*

---

## 3C: How the Agent and KAN Work Together

The key relationship: **the agent proposes, the KAN judges.**

```
AGENT'S JOB:
  - Observe the cluster
  - Understand what's wrong (using LLM reasoning)
  - Decide WHAT action would fix it
  - Do NOT decide IF it's safe → that's KAN's job

KAN'S JOB:
  - Receive the proposed action and its context features
  - Output a trust score with a formula
  - Never interpret context, never reason — just score
  - The score is deterministic for the same inputs
```

**Why not just let the LLM decide if it's safe?**

The LLM's safety judgement:
- Changes with temperature (non-deterministic)
- Is not auditable — "the LLM thought it was safe" is not a legal record
- Can hallucinate — "I believe this is safe based on..." (but is wrong)
- Cannot produce a mathematical formula
- Cannot provide Lipschitz robustness certificates

The KAN's safety judgement:
- Fully deterministic for same inputs
- Produces an auditable symbolic formula
- Mathematically bounded (Lipschitz)
- Trained on real historical risk data
- Can be audited by a regulator

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
- [CERT] Read about RBAC — understand Roles, ClusterRoles, RoleBindings ✅
- [CERT] Read about K8s Audit Logs — know what fields they contain ✅
- [CERT] Understand pods, namespaces, deployments, services ✅
- [CERT] Know what Kubeflow Pipelines are and how they run on K8s ✅
- [ ] Understand what an AI agent loop is (observe → think → act)

## KAN
- [ ] Installed `pykan` and ran the hello-world example in `weekly_log.md`
- [ ] Can explain what a B-spline is (even roughly)
- [ ] Understand the difference between MLP and KAN architectures
- [ ] Ran `model.auto_symbolic()` and read the output
- [ ] Can explain why symbolic regression is useful for trust scoring

## Kubernetes
- [ ] Can run `kubectl get pods --all-namespaces`
- [ ] Can create a Deployment with `kubectl apply -f deployment.yaml`
- [ ] Understand what RBAC is and can list ClusterRoleBindings
- [ ] Can read a Kubernetes audit log entry and extract verb/resource/namespace
- [ ] Set up the Kubernetes Python client and queried the cluster

## Agentic AI
- [ ] Read LangGraph "Getting Started" tutorial
- [ ] Built a simple 3-node LangGraph (observe → plan → act)
- [ ] Understand what a state machine is
- [ ] Can add a conditional edge (branch based on a value)

## Full System
- [ ] Can trace the full flow: audit log → agent → KAN → execute
- [ ] Understand why the KAN is in the middle (not the LLM)
- [ ] Can explain the 8 features and why each one matters
- [ ] Can explain the 3 trust score regions (auto-apply / review / reject)
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
| K8s Python | [kubernetes-client/python](https://github.com/kubernetes-client/python) | Code | 1 hour |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | Docs | 2 hours |
| Agentic AI | [DeepLearning.AI short courses](https://learn.deeplearning.ai/) | Course | 2 hours |
| KubeIntellect | [arXiv:2509.02449](https://arxiv.org/abs/2509.02449) | Paper | 2 hours |

**Suggested order:** 3Blue1Brown → K8s Nana video → pykan hello world → LangGraph tutorial → KAN paper → KubeIntellect paper
