"""Read a trained KANGate's internals for the dashboard's network view and
for per-decision attribution.

Everything here is computed from the model itself, not re-derived:
- each edge's learned univariate function phi_{l,j,i} is sampled by
  calling that layer directly on a sweep of its input node's values;
- a decision's edge values and node values are pykan's own cached
  activations (`spline_postacts`, `acts`) from one real forward pass;
- feature contributions are occlusion against an all-zero baseline,
  f(x) - f(x with feature i set to 0). For an additive model (every
  layer after the first linear, which is what the pruned symbolic gate
  ends up as) these contributions sum exactly to f(x) - f(0), and
  `additive` reports whether that check held.
"""
from __future__ import annotations

import sympy
import torch

from kan_gate.features import FEATURE_NAMES
from kan_gate.gate import KANGate, _score_to_decision

CURVE_POINTS = 48
_REFERENCE_BATCH = 256


def _layer_sizes(model) -> list[int]:
    return [w[0] if isinstance(w, list) else w for w in model.width]


def _edge_outputs(model, layer: int, x: torch.Tensor) -> torch.Tensor:
    """Per-edge outputs of one layer, shape (batch, out, in)."""
    _, _, numeric, _ = model.act_fun[layer](x)
    _, symbolic = model.symbolic_fun[layer](x)
    return numeric + symbolic


def _edge_expression(model, layer: int, i: int, j: int) -> tuple[str, bool]:
    """Readable c*f(a*x+b)+d for a symbolic edge; 'spline' for a numeric one."""
    sym = model.symbolic_fun[layer]
    if float(sym.mask[j][i]) > 0:
        name = sym.funs_name[j][i]
        if name == "0":
            return "0", False
        from kan.utils import SYMBOLIC_LIB

        a, b, c, d = (float(v) for v in sym.affine[j, i].detach())
        x = sympy.Symbol("x")
        expr = sympy.expand(c * SYMBOLIC_LIB[name][1](a * x + b) + d)
        expr = expr.xreplace({n: sympy.Float(float(sympy.N(n, 3)), 3)
                              for n in expr.atoms(sympy.Float)})
        return str(expr), True
    if float(model.act_fun[layer].mask[i][j]) > 0:
        return "spline", True
    return "0", False


def network(gate: KANGate) -> dict:
    """Static structure: layers, active edges with their learned function
    sampled over the input range each edge actually sees."""
    model = gate.model
    sizes = _layer_sizes(model)
    g = torch.Generator().manual_seed(0)
    reference = torch.rand(_REFERENCE_BATCH, sizes[0], generator=g)
    with torch.no_grad():
        model(reference)
        acts = [a.detach() for a in model.acts]
        postacts = [p.detach() for p in model.spline_postacts]

    edges = []
    for layer in range(len(sizes) - 1):
        lo = acts[layer].min(dim=0).values
        hi = acts[layer].max(dim=0).values
        sweep = torch.stack([torch.linspace(float(lo[i]), float(hi[i]), CURVE_POINTS)
                             for i in range(sizes[layer])], dim=1)
        with torch.no_grad():
            curves = _edge_outputs(model, layer, sweep)
        for j in range(sizes[layer + 1]):
            for i in range(sizes[layer]):
                expr, active = _edge_expression(model, layer, i, j)
                ys = curves[:, j, i]
                active = active and float(ys.max() - ys.min()) > 1e-4  # a constant edge carries no signal
                edges.append({
                    "layer": layer, "i": i, "j": j, "active": active, "expr": expr,
                    "importance": float(postacts[layer][:, j, i].std()),
                    "curve": {"x": sweep[:, i].tolist(), "y": curves[:, j, i].tolist()},
                })
    return {
        "layers": sizes,
        "feature_names": FEATURE_NAMES,
        "edges": edges,
        "formula": gate.formula(),
    }


def explain(gate: KANGate, features: dict[str, float]) -> dict:
    """One decision, traced through the network."""
    model = gate.model
    x = gate.encode(features)
    with torch.no_grad():
        out = model(x)
        node_values = [a[0].tolist() for a in model.acts]
        edge_values = [
            {"layer": layer, "i": i, "j": j, "value": float(p[0, j, i])}
            for layer, p in enumerate(model.spline_postacts)
            for j in range(p.shape[1]) for i in range(p.shape[2])
        ]
        baseline = float(model(torch.zeros_like(x)).item())
        contributions = {}
        for k, name in enumerate(FEATURE_NAMES):
            occluded = x.clone()
            occluded[0, k] = 0.0
            contributions[name] = float(out.item() - model(occluded).item())

    raw = float(out.item())
    score = max(0.0, min(100.0, raw))
    return {
        "features": features,
        "raw_output": raw,
        "score": score,
        "decision": _score_to_decision(score),
        "baseline": baseline,
        "contributions": contributions,
        "additive": abs(baseline + sum(contributions.values()) - raw) < 1e-3,
        "node_values": node_values,
        "edge_values": edge_values,
    }
