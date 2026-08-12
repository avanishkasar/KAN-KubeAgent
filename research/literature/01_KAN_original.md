# Literature Review: KAN — Kolmogorov-Arnold Networks

**Paper:** "KAN: Kolmogorov-Arnold Networks"  
**Authors:** Ziming Liu, Yixuan Wang, Sachin Vaidya, et al. (MIT CSAIL)  
**arXiv:** [2404.19756](https://arxiv.org/abs/2404.19756)  
**Published:** April 2024 | Accepted: ICLR 2025  
**GitHub:** [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)

---

## 1. Core Theorem (The "Protocol")

KAN is grounded in the **Kolmogorov-Arnold Representation Theorem (1957)**:

> Any continuous multivariate function `f: [0,1]^n → R` can be written as:
>
> `f(x₁,...,xₙ) = Σ_{q=1}^{2n+1} Φ_q( Σ_{p=1}^{n} φ_{q,p}(x_p) )`
>
> where `Φ_q` and `φ_{q,p}` are continuous univariate functions.

**What this means in practice:** Any function of multiple variables can be decomposed into sums and compositions of simpler *one-variable* functions. KAN makes those univariate functions learnable.

---

## 2. Architecture vs MLP

### MLP (Traditional)
```
Node i → [weight w_{ij}] → Node j → σ(·) [fixed activation on node]
```
- Weights are **scalars** on edges
- Activation functions are **fixed** (ReLU, Sigmoid, etc.) on nodes
- Formula: `y = σ(W·x)`

### KAN
```
Node i → [φ_{i,j}(·)] → Node j → Σ [sum at node]
```
- Each edge carries a **learnable univariate function** φ
- Nodes just **sum** incoming values (no activation)
- Formula: `y = Σ φ_{i,j}(x_i)` where each φ is a B-spline

---

## 3. B-Spline Implementation (The Engine)

Since the theorem guarantees existence but doesn't define the functions, KAN parametrises each φ as a **B-spline**:

```
φ(x) = w_b · b(x) + w_s · spline(x)

where:
  b(x) = SiLU(x) = x / (1 + e^{-x})   [base function]
  spline(x) = Σ_i c_i · B_i(x)         [B-spline sum]
  c_i = learnable coefficients
  B_i = B-spline basis functions
```

**B-spline properties that matter for security applications:**
- **Local support:** Each basis function B_i is non-zero only in a small interval → changing one coefficient only affects local behaviour
- **Smooth:** Infinitely differentiable → Lipschitz constants are computable
- **Grid refinement:** Start coarse, refine → allows progressive precision
- **Visualisable:** You can literally plot φ_{i,j}(x) and see what the network learned

---

## 4. Training Process

```python
# Simplified KAN forward pass
class KANLayer:
    def forward(self, x):
        # x shape: [batch, in_features]
        y = torch.zeros(batch, out_features)
        for i in range(in_features):
            for j in range(out_features):
                # Each edge applies its own learned spline
                y[:, j] += self.phi[i][j](x[:, i])
        return y
```

Key hyperparameters:
- `grid` — number of grid intervals for B-splines (default: 5, more = more expressive)
- `k` — polynomial order of splines (default: 3 = cubic)
- `grid_range` — input domain [a, b] for the splines
- `lamb` — L1 regularisation on spline coefficients (encourages sparsity / pruning)

---

## 5. Symbolic Regression (The "Explainability Superpower")

After training, KAN can attempt to **identify** the mathematical form of each learned edge function:

```python
# After training, try to identify symbolic formulas
kan.auto_symbolic()
# Output: "phi[0][1] ≈ sin(x)", "phi[1][0] ≈ x^2", etc.

# Fix a symbolic formula for an edge
kan.fix_symbolic(0, 1, 'sin')
```

This is what we exploit for **trust score explainability** in our paper.

---

## 6. Benchmark Performance (from paper)

| Task | KAN params | MLP params | Winner |
|------|-----------|-----------|--------|
| Fitting `f = J₀(20x + J₀(20y))` | 200 | 100K | KAN (accuracy) |
| Special relativity formula | 2-layer | 5-layer MLP | KAN (fewer params) |
| Symbolic formula recovery | ✅ Often | ❌ Never | KAN |

**Important caveat:** These are small-scale scientific problems. KAN has NOT been extensively tested on large-scale, noisy real-world data. **This is our opportunity.**

---

## 7. Weaknesses / Open Problems (Our Research Gaps)

From the paper itself and subsequent critiques:

1. **Training speed:** 10x slower than comparable MLPs on same hardware (no optimised CUDA kernels for splines)
2. **Noisy data robustness:** B-splines can overfit to noise; limited study of robustness
3. **Scalability:** Not yet demonstrated on datasets with millions of samples
4. **No deployment in agentic systems:** All existing KAN papers use it as a standalone predictor, never as a sub-component in a larger agentic decision pipeline
5. **No formal trust/safety certification:** GloroKAN provides Lipschitz bounds but this hasn't been applied in a real system

---

## 8. Relevance to KAN-KubeAgent

| KAN Feature | How We Use It |
|-------------|--------------|
| Learnable edge functions | Each feature (namespace_risk, blast_radius, etc.) gets its own learned curve |
| Symbolic regression | After training, extract the trust formula: `score = f(risk) + g(frequency) - h(blast)` |
| Local Lipschitz bounds | Certify robustness: "if inputs change by ε, trust score changes by at most δ" |
| Grid refinement | Start with coarse risk model, refine as more audit log data accumulates |
| Sparsity/pruning | Remove irrelevant features automatically → find the truly important K8s risk factors |

---

## 9. Key Citations from This Paper

```bibtex
@article{liu2024kan,
  title   = {KAN: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and 
             Ruehle, Fabian and Halverson, James and Soljacic, Marin 
             and Hou, Thomas Y and Tegmark, Max},
  journal = {arXiv preprint arXiv:2404.19756},
  year    = {2024}
}
```

---

## 10. Follow-up Papers to Read (KAN Extensions)

- **KAN 2.0** (arXiv:2408.10205) — MultKAN, more efficient, better symbolic regression
- **GloroKAN** (OpenReview 2025) — Lipschitz-bounded KANs for robustness certification  
- **KAN-MID** (ResearchGate 2025) — KAN for intrusion detection in cloud environments
- **TKAN** (remigenet/TKAN) — Temporal KAN for sequential data
- **KAN RL Policy** (arXiv:2506.16392) — KAN for interpretable reinforcement learning policies (closest to our work)
