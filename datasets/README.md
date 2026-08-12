# Datasets Guide — KAN-KubeAgent

## Overview

This project requires two types of data:
1. **Training data for KAN** — labeled K8s action risk scores
2. **Agent evaluation data** — realistic K8s cluster scenarios for end-to-end testing

---

## Dataset 1: K-RAD (K8s Risk Action Dataset) — To Be Created

**Status:** Will be synthesised as part of this research

### Generation Method

```bash
# Step 1: Spin up a local cluster
minikube start --nodes=3 --kubernetes-version=v1.30.0

# Step 2: Enable audit logging
# Add to kube-apiserver config:
# --audit-log-path=/var/log/k8s-audit.log
# --audit-policy-file=audit-policy.yaml
# --audit-log-maxage=30

# Step 3: Run attack simulation scripts
python scripts/simulate_benign_traffic.py
python scripts/simulate_attack_scenarios.py

# Step 4: Label with risk scores
python scripts/label_with_cis_benchmark.py
```

### Risk Score Mapping

| K8s Action | CIS Control | MITRE Technique | Risk Score |
|-----------|-------------|-----------------|-----------|
| GET pods | N/A | Discovery (TA0007) | 95 (safe) |
| PATCH deployment (replicas) | 5.2.1 | Impact | 70 |
| CREATE ClusterRoleBinding (admin) | 5.1.1 | Privilege Escalation | 10 |
| DELETE NetworkPolicy | 5.3.2 | Defense Evasion | 5 (critical) |
| EXEC into pod | 5.4.1 | Execution | 15 |
| PATCH configmap | N/A | N/A | 80 |
| DELETE secret | 5.4.1 | Credential Access | 8 |
| CREATE privileged pod | 5.2.1 | Privilege Escalation | 5 |

---

## Dataset 2: CICIDS2017 (Public — Network Intrusion)

**Status:** Publicly available, use for KAN pre-training

**URL:** https://www.unb.ca/cic/datasets/ids-2017.html

**Why use this:**
- Well-established benchmark for intrusion detection
- Provides initial KAN training before K8s-specific fine-tuning
- Allows comparison with KAN-MID paper results

**Relevant attack types:**
- DoS/DDoS → maps to K8s resource exhaustion
- PortScan → maps to K8s cluster reconnaissance
- Web Attacks → maps to K8s API server probing

---

## Dataset 3: UNSW-NB15 (Public — Network Traffic)

**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Why use this:** Secondary benchmark, used in KAN security papers

---

## Dataset 4: Kubernetes Audit Log Samples (Public GitHub)

Several repositories contain real anonymised K8s audit logs:
- `falcosecurity/falco` — contains sample audit events
- `kubernetes/kubernetes` — test fixtures with realistic audit log formats
- `open-policy-agent/gatekeeper` — policy violation audit samples

---

## Feature Extraction Schema

```python
# Feature vector for each K8s API event
FEATURE_SCHEMA = {
    "namespace_risk_score": float,    # 0.0 (dev) - 1.0 (prod)
    "blast_radius": float,            # 0.0 - 1.0 (normalised downstream impact)
    "change_frequency": float,        # events/hour for this resource (normalised)
    "actor_privilege_level": float,   # 0.0 (viewer) - 1.0 (cluster-admin)
    "resource_criticality": float,    # 0.0 (configmap) - 1.0 (secret/clusterrole)
    "action_verb_risk": float,        # GET=0.1, LIST=0.1, CREATE=0.5, PATCH=0.6, DELETE=0.9
    "historical_failure_rate": float, # failures/total for this action type (0-1)
    "time_since_last_deploy": float   # hours (normalised, 0=just deployed, 1=stable)
}

# Label
LABEL_SCHEMA = {
    "trust_score": float              # 0 (critical risk) - 100 (completely safe)
}
```
