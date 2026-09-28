#!/usr/bin/env bash
# Start Minikube and install Kubeflow Trainer (TrainJob CRDs).
# See k8s/README.md for prerequisites and why this can't run in the
# Claude Code cloud sandbox this repo was built in.
set -euo pipefail

CPUS="${MINIKUBE_CPUS:-4}"
MEMORY="${MINIKUBE_MEMORY:-8g}"
TRAINER_VERSION="${KUBEFLOW_TRAINER_VERSION:-v2.1.0}"

echo "==> Starting Minikube (cpus=${CPUS}, memory=${MEMORY}, CPU-only)"
minikube start --cpus="${CPUS}" --memory="${MEMORY}" --driver=docker

echo "==> Installing Kubeflow Trainer ${TRAINER_VERSION} (TrainJob CRDs)"
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${TRAINER_VERSION}"

echo "==> Waiting for the Trainer controller to be ready"
kubectl wait --for=condition=available --timeout=180s \
  deployment/kubeflow-trainer-controller-manager -n kubeflow-system

echo "==> Verifying: listing TrainJob CRD"
kubectl get crd trainjobs.trainer.kubeflow.org

echo "==> Done. Submit a job with: kubectl apply -f k8s/trainjob-example.yaml"
