# Methodology Draft — KAN-KubeAgent

**Version:** 0.1 (Working Draft)  
**Last Updated:** August 2026

---

## 1. System Overview

KAN-KubeAgent consists of four integrated components:

```
[1] Observation Layer    →  [2] LLM Agent Core  →  [3] KAN Trust Layer  →  [4] Executor
(K8s API + Audit Logs)     (LangGraph FSM)         (Verification Gate)      (Apply / Reject)
```

---

## 2. Component 1: Observation & Feature Extraction

### 2.1 Data Sources
- **Kubernetes Audit Log:** Every API request to kube-apiserver logged with verb, resource, user, namespace, response code
- **Prometheus Metrics:** CPU/memory per pod, namespace-level quotas, error rates
- **RBAC State Snapshot:** Current role-binding graph at time of proposed action
- **Cluster Event Stream:** Pod restarts, OOMKills, failed deployments

### 2.2 Feature Engineering for KAN Trust Layer

For each proposed remediation action, extract:

| Feature | Source | Description |
|---------|--------|-------------|
| `namespace_risk_score` | RBAC + criticality label | 0-1: is this namespace prod/staging/dev? |
| `blast_radius` | Resource graph | How many downstream services affected? |
| `change_frequency` | Audit log history | How often has this resource been changed? |
| `actor_privilege_level` | RBAC | What permissions does the service account have? |
| `resource_criticality` | Manually labeled | 0-1: StatefulSet > Deployment > ConfigMap |
| `action_verb_risk` | CIS Benchmark | DELETE > PATCH > CREATE > GET |
| `historical_failure_rate` | Audit log | Past error rate for this action type |
| `time_since_last_deploy` | Deployment history | Recent changes increase risk |

---

## 3. Component 2: LLM Agent Core

### 3.1 Supervisor Agent

```python
# Pseudocode for supervisor
class SupervisorAgent:
    def run(self, user_query: str, cluster_state: ClusterState):
        # 1. Parse intent
        intent = self.llm.parse(user_query)
        
        # 2. Plan action sequence
        plan = self.llm.plan(intent, cluster_state)
        
        # 3. Execute plan via domain agents
        for step in plan.steps:
            agent = self.route(step)
            result = agent.execute(step)
            
            # 4. If action is mutating → go through KAN gate
            if step.is_mutating:
                trust = self.kan_gate.score(step, result)
                if trust.score < 40:
                    return self.reject(step, trust.formula)
                elif trust.score < 80:
                    return self.request_human_approval(step, trust.formula)
                else:
                    self.execute(step)
```

### 3.2 Domain Agents (Subset for Security Focus)

| Agent | Tools | K8s APIs Used |
|-------|-------|--------------|
| **RBAC Agent** | list_roles, list_bindings, check_permissions | `/apis/rbac.authorization.k8s.io/` |
| **Audit Log Agent** | query_audit_logs, detect_anomalies | `/logs/kube-apiserver-audit.log` |
| **Network Agent** | list_network_policies, check_connectivity | `/apis/networking.k8s.io/` |
| **Pod Security Agent** | check_psp, check_privileged | `/api/v1/pods` |

---

## 4. Component 3: KAN Trust Layer (Core Contribution)

### 4.1 Architecture

```python
import torch
from kan import KAN  # pykan library

class KANTrustLayer:
    def __init__(self):
        self.model = KAN(
            width=[8, 5, 3, 1],  # 8 features → 5 → 3 → 1 trust score
            grid=10,              # spline grid resolution
            k=3,                  # cubic splines
            seed=42
        )
    
    def score(self, action_features: dict) -> TrustResult:
        x = self.encode_features(action_features)
        
        # Forward pass → trust score [0, 100]
        raw_score = self.model(x)
        trust_score = torch.sigmoid(raw_score) * 100
        
        # Extract symbolic formula
        formula = self.model.symbolic_formula()
        
        # Compute Lipschitz bound (robustness certificate)
        lipschitz = self.compute_lipschitz()
        
        return TrustResult(
            score=trust_score.item(),
            formula=formula,
            certificate=lipschitz
        )
    
    def extract_formula(self) -> str:
        """
        After training, KAN can identify symbolic forms.
        Example output:
        'trust = 0.82·relu(namespace_risk) + 0.61·sin(change_freq) 
                - 0.43·blast_radius^2 + 0.11·actor_privilege'
        """
        self.model.auto_symbolic()
        return self.model.symbolic_formula()[0][0]
```

### 4.2 Training Procedure

**Dataset:** K-RAD (K8s Risk Action Dataset)
- 10,000 K8s API audit events
- Labels: [0=safe, 1=low_risk, 2=medium_risk, 3=high_risk, 4=critical]
- Converted to continuous trust score: safe=95, low=75, medium=50, high=20, critical=5

**Training:**
```python
# KAN training (pykan API)
dataset = {
    'train_input': X_train,  # [N, 8] feature matrix
    'train_label': y_train,  # [N, 1] trust scores
    'test_input': X_test,
    'test_label': y_test
}

model.train(
    dataset,
    opt='Adam',
    steps=500,
    lamb=0.001,      # L1 regularisation for sparsity
    lamb_entropy=2.0  # entropy regularisation for interpretability
)

# Grid refinement (coarse → fine)
model.refine(grid=20)  # increase resolution after initial training
```

### 4.3 Symbolic Formula Extraction

After training:
```python
# Try to identify known symbolic functions for each edge
model.auto_symbolic(lib=['x', 'x^2', 'x^3', 'sin', 'exp', 'log', 'sqrt'])

# Prune low-importance edges
model.prune()

# Get human-readable formula
formula = model.symbolic_formula()
# Example: trust = 0.8*x_0 + 0.6*sin(x_2) - 0.4*x_1^2 + 0.1*x_3
# Where x_0=namespace_risk, x_1=blast_radius, x_2=change_freq, x_3=actor_privilege
```

---

## 5. Component 4: Executor & Audit Trail

```python
class TrustGatedExecutor:
    
    AUTO_APPLY_THRESHOLD = 80
    HUMAN_REVIEW_THRESHOLD = 40
    
    def execute(self, action, trust_result):
        # Log everything to audit DB
        self.audit_log.record(action, trust_result)
        
        if trust_result.score >= self.AUTO_APPLY_THRESHOLD:
            # Auto-apply with formula logged
            result = self.k8s_client.apply(action)
            self.audit_log.record_outcome(action, "AUTO_APPLIED", trust_result.formula)
            
        elif trust_result.score >= self.HUMAN_REVIEW_THRESHOLD:
            # Show human the formula and ask for approval
            approval = self.notify_human(
                action=action,
                score=trust_result.score,
                formula=trust_result.formula,
                certificate=trust_result.certificate
            )
            if approval:
                self.k8s_client.apply(action)
                
        else:
            # Auto-reject with explanation
            self.audit_log.record_outcome(action, "AUTO_REJECTED", trust_result.formula)
```

---

## 6. Dataset: K-RAD (K8s Risk Action Dataset)

### 6.1 Data Generation

Using `minikube` cluster + automated attack/benign scenario simulation:

**Benign scenarios (70% of dataset):**
- Normal rolling deployments
- ConfigMap updates
- HPA scaling events
- Certificate rotations

**Attack / anomaly scenarios (30% of dataset):**
- RBAC privilege escalation (binding ClusterAdmin to service account)
- Deleting network policies silently
- Launching privileged containers
- Accessing secrets outside service scope
- API server audit log tampering

### 6.2 Labeling Methodology

Risk labels assigned using:
1. **CIS Kubernetes Benchmark v1.9** — defines security levels for each K8s operation
2. **MITRE ATT&CK for Containers** — maps K8s actions to attack techniques
3. **RBAC Blast Radius Calculator** — measures downstream impact of each change

### 6.3 Dataset Statistics (Target)

| Risk Level | Count | % |
|------------|-------|---|
| Safe (95) | 3,500 | 35% |
| Low (75) | 2,500 | 25% |
| Medium (50) | 2,000 | 20% |
| High (20) | 1,500 | 15% |
| Critical (5) | 500 | 5% |
| **Total** | **10,000** | **100%** |

---

## 7. Evaluation Plan

### 7.1 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Trust Score Accuracy** | Spearman correlation with expert labels | > 0.85 |
| **MTTR Improvement** | Time to remediate vs manual ops | > 40% reduction |
| **False Negative Rate** | Unsafe actions auto-approved | < 2% |
| **False Positive Rate** | Safe actions blocked | < 15% |
| **TFF (Trust Formula Faithfulness)** | % of formula terms that match SHAP feature importance | > 70% |
| **User Study Score** | DevOps engineers rate formula readability 1-5 | > 3.5/5 |

### 7.2 Baselines

| Baseline | Description |
|----------|-------------|
| **Static Rules** | CIS Benchmark ruleset only (no ML) |
| **MLP Classifier** | Same 8 features, standard neural network |
| **Random Forest** | Ensemble baseline |
| **KubeIntellect HITL** | Binary approve/reject gate |
| **KAN-KubeAgent (Ours)** | KAN trust scoring |

### 7.3 Experimental Environment

```yaml
# Cluster setup
clusters:
  - name: dev-cluster
    nodes: 3
    workload: web-microservices
    
  - name: ml-cluster  
    nodes: 4
    workload: pytorch-training-jobs
    
  - name: prod-sim-cluster
    nodes: 5
    workload: mixed (stateful + stateless)
    
tool: minikube (local) / kind (CI)
k8s_version: 1.30+
```
