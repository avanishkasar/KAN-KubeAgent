# Weekly Research Log

## Template (Copy for each week)
```
## Week [N] — [Date Range]
### What I read:
### What I understood:
### What confused me:
### Next week plan:
```

---

## Week 1 — August 2026

### Goal: Read KAN original paper + KubeIntellect paper in depth

**Papers targeted:**
- [ ] arXiv:2404.19756 (KAN original)
- [ ] arXiv:2509.02449 (KubeIntellect)

### Notes:
- Research project initiated
- Repository created: KAN-KubeAgent
- Core architecture diagram drafted
- Novelty claims documented

### Next week plan:
- Finish both papers above
- Set up minikube environment
- Install pykan library and run hello-world example
- Install kubernetes Python client

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
