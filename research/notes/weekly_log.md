# Weekly Research Log

## 🏆 Certifications Completed

| Cert | Provider | Completed | Relevance to Project |
|------|----------|-----------|---------------------|
| Introduction to Kubernetes (LFS158) | Linux Foundation | Aug 2026 | Core infrastructure — K8s concepts, kubectl, pods, deployments, RBAC |
| Introduction to AI/ML Toolkits with Kubeflow (LFS147) | Linux Foundation | Aug 2026 | Kubeflow runs ON K8s — ML pipelines, training jobs, model serving inside clusters |

### Why These Certs Matter for KAN-KubeAgent

- **LFS158 (K8s):** You now have certified knowledge of everything the agent observes and manipulates — pods, namespaces, RBAC, audit logs, deployments. The K8s basics in `LEARNING_GUIDE.md` Level 1 can be skipped — you're already past that.
- **LFS147 (Kubeflow):** Kubeflow is a real-world example of AI workloads running inside Kubernetes. This gives you direct insight into the kinds of ML-specific RBAC patterns, resource quotas, and security configurations that our agent would encounter in a production AI cluster. This is bonus context that most Kubernetes security papers don't have.

---

## Template (Copy for each week)
```
## Week [N] — [Date Range]
### What I read:
### What I understood:
### What confused me:
### Next week plan:
```

---

## Week 1 — 12–14 August 2026

### Goal: Project setup + foundation certifications

**Completed ✅:**
- [x] Linux Foundation — Introduction to Kubernetes (LFS158)
- [x] Linux Foundation — Introduction to AI/ML Toolkits with Kubeflow (LFS147)
- [x] Repository created: KAN-KubeAgent (github.com/avanishkasar/KAN-KubeAgent)
- [x] Core architecture diagram drafted
- [x] Novelty claims documented (4 formal claims)
- [x] Methodology draft written (KAN trust layer design + K-RAD dataset spec)
- [x] Prioritised reading list (15 papers across 3 tiers)
- [x] Full learning guide written (LEARNING_GUIDE.md)

**Papers targeted for next week:**
- [ ] arXiv:2404.19756 (KAN original — Liu et al.)
- [ ] arXiv:2509.02449 (KubeIntellect)

### Kubeflow Connection to Project
Since you now know Kubeflow (LFS147), note that:
- Kubeflow Pipelines use K8s CRDs (Custom Resource Definitions) — a perfect use case for our agent's anomaly detection
- Kubeflow uses ServiceAccounts with specific RBAC — our K-RAD dataset should include Kubeflow-style RBAC patterns
- A future extension of KAN-KubeAgent could focus specifically on **securing AI/ML workloads in Kubeflow clusters** — a very niche and publishable angle

### Next Week Plan (Week 2)
- Read KAN original paper (arXiv:2404.19756) — focus on Section 2 (architecture) and Section 3 (symbolic regression)
- Install pykan and run the hello-world in this file
- Read KubeIntellect paper — focus on the HITL gate weakness (our research gap)
- Set up minikube locally
- Write notes in this log after each paper

---

## Setup Commands (Run Once)

```bash
# Install pykan
pip install pykan

# Install Kubernetes Python client
pip install kubernetes

# Install LangGraph
pip install langgraph langchain

# Install minikube (for local K8s cluster)
# Windows: winget install minikube

# Verify
python -c "import kan; print('KAN ready')"
python -c "import kubernetes; print('K8s client ready')"
python -c "import langgraph; print('LangGraph ready')"
```

## Quick KAN Hello World

```python
import torch
from kan import KAN

# Create a simple 2-layer KAN
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)

# Generate sample data
x = torch.rand(100, 2)
y = torch.sin(x[:, 0]) + x[:, 1]**2

# Train
dataset = {'train_input': x, 'train_label': y.unsqueeze(1),
           'test_input': x, 'test_label': y.unsqueeze(1)}

model.train(dataset, opt='Adam', steps=200)

# Visualise (shows the learned functions on each edge)
model.plot()

# Try to identify symbolic formulas
model.auto_symbolic(lib=['sin', 'x', 'x^2'])
print(model.symbolic_formula())
# Should output something like: sin(x_0) + x_1^2
```
