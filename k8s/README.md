# k8s/ — Minikube + Kubeflow Trainer Setup

Run this on your own machine (or the college GPU box, as a bonus — not a
requirement). This could not be run inside the Claude Code cloud sandbox
used to build this repo: that sandbox's network policy blocks
`registry.k8s.io`, the registry Minikube pulls its own control-plane images
from, so `minikube start` fails there with a 403 before it ever reaches the
code in this repo. That's an environment restriction, not a bug in these
scripts - on a normal machine (or a Claude Code environment with broader
network access) this should just work.

## 1. Install prerequisites

- [Docker](https://docs.docker.com/get-docker/) (or another Minikube driver)
- [`minikube`](https://minikube.sigs.k8s.io/docs/start/)
- [`kubectl`](https://kubernetes.io/docs/tasks/tools/)

## 2. Start the cluster and install Kubeflow Trainer

```bash
./k8s/setup_minikube.sh
```

This script (see `setup_minikube.sh`):
1. Starts Minikube with the Docker driver (`--cpus=4 --memory=8g` - CPU-only,
   no GPU dependency, per the project's constraints)
2. Installs the Kubeflow Trainer CRDs (`TrainJob`, `TrainingRuntime`)
3. Verifies the install with `kubectl get trainjobs -A`

## 3. Submit a test TrainJob

```bash
kubectl apply -f k8s/trainjob-example.yaml
kubectl get trainjobs
kubectl describe trainjob fashion-mnist-demo
```

## 4. Point the agent layer at the real cluster

Once the cluster is up, swap `MockTrainJobClient` for
`KubeflowTrainJobClient` (`agents/kubeflow_client.py`) - same
`TrainJobClient` interface, so `agents/graph.py` does not change:

```python
from agents.kubeflow_client import KubeflowTrainJobClient
from agents.graph import build_graph
from kan_gate.gate import KANGate

client = KubeflowTrainJobClient(namespace="default")
gate = KANGate.load()
graph = build_graph(client, gate)
```

`agents/kubeflow_client.py` was written against the Kubeflow Trainer Python
SDK's documented interface but has **not** been exercised against a live
cluster from this sandbox (see the network limitation above) - validate it
against your own cluster and expect to need small fixes if the installed
Trainer version's API differs.

## 5. Training workload

The `trainjob-example.yaml` job fine-tunes a small CNN on a Fashion-MNIST
subset - CPU-only, a few minutes per run, no GPU cluster dependency, per
the project's hard constraints. See `training/fashion_mnist_cnn.py` for the
training script the job runs.
