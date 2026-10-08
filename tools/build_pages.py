"""Builds the static GitHub Pages preview into docs/.

The preview serves the Guide, the KAN Network view (the real trained gate,
evaluated in the browser from its exact closed-form edge functions) and the
Research results. Live training needs a backend and a local machine, so the
Live and Mock tabs explain how to run it locally instead.

    python tools/build_pages.py
"""
from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

import sympy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kan.utils import SYMBOLIC_LIB

from kan_gate.gate import KANGate
from kan_gate.introspect import network

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def exact_expr(model, layer: int, i: int, j: int) -> str:
    sym = model.symbolic_fun[layer]
    if float(sym.mask[j][i]) <= 0 or sym.funs_name[j][i] == "0":
        return "0"
    a, b, c, d = (float(v) for v in sym.affine[j, i].detach())
    x = sympy.Symbol("x")
    return str(c * SYMBOLIC_LIB[sym.funs_name[j][i]][1](a * x + b) + d)


SHIM = r"""/* Static preview shim: answers the dashboard's /api calls from files in data/,
   and evaluates the real gate in the browser from its exact edge functions. */
(function () {
  const realFetch = window.fetch.bind(window);
  let net = null, results = null;
  const FEATURES = ['loss_plateau_score', 'gradient_trend', 'lr_decay_benefit', 'gpu_hours_remaining_vs_budget', 'epochs_since_improvement'];
  const json = (o) => new Response(JSON.stringify(o), { status: 200, headers: { 'Content-Type': 'application/json' } });
  const no = () => new Response('{}', { status: 503 });
  const compile = (s) => new Function('x', `const sin=Math.sin; return ${s.replace(/\bsin\(/g, 'sin(')};`);
  async function loadNet() {
    if (net) return net;
    net = await (await realFetch('data/network.json')).json();
    net.f = {}; net.edges.forEach(e => { net.f[`${e.layer}-${e.i}-${e.j}`] = compile(e.exact); });
    return net;
  }
  function run(x) {
    const h = [0, 1, 2].map(j => [0, 1, 2, 3, 4].reduce((a, i) => a + net.f[`0-${i}-${j}`](x[i]), 0));
    const ev = [];
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) ev.push({ layer: 0, i, j, value: net.f[`0-${i}-${j}`](x[i]) });
    let out = 0;
    for (let j = 0; j < 3; j++) { const v = net.f[`1-${j}-0`](h[j]); out += v; ev.push({ layer: 1, i: j, j: 0, value: v }); }
    return { out, h, ev };
  }
  async function explain(features) {
    await loadNet();
    const x = FEATURES.map(k => features[k]);
    const r = run(x), base = run([0, 0, 0, 0, 0]).out, contributions = {};
    FEATURES.forEach((k, idx) => { const y = x.slice(); y[idx] = 0; contributions[k] = r.out - run(y).out; });
    const score = Math.max(0, Math.min(100, r.out));
    const sum = Object.values(contributions).reduce((a, b) => a + b, 0);
    return { features, raw_output: r.out, score, decision: score < 40 ? 'continue' : score < 75 ? 'adjust_lr' : 'early_stop',
      baseline: base, contributions, additive: Math.abs(base + sum - r.out) < 1e-3,
      node_values: [x, r.h, [r.out]], edge_values: r.ev };
  }
  window.fetch = async (url, opts) => {
    const u = String(url);
    if (!u.includes('/api/')) return realFetch(url, opts);
    if (u.endsWith('/api/gate/network')) { const n = await loadNet(); return json({ layers: n.layers, feature_names: n.feature_names, edges: n.edges, formula: n.formula }); }
    if (u.endsWith('/api/gate/explain')) return json(await explain(JSON.parse(opts.body).features));
    if (u.endsWith('/api/gate')) { const n = await loadNet(); return json({ formula: n.formula, feature_names: n.feature_names }); }
    if (u.endsWith('/api/research/results')) { results = results || await (await realFetch('data/results.json')).json(); return json({ available: true, ...results }); }
    return no();
  };
  window.WebSocket = function () { return { close() {}, send() {} }; };
  document.addEventListener('DOMContentLoaded', () => {
    const live = document.getElementById('live-start-btn'); if (live) live.title = 'Run locally to start real training';
  });
})();
"""

BANNER = """<div class="panel" id="static-banner" style="border-color: var(--text-muted);">
  <strong style="font-size:13px;">Static preview</strong>
  <p class="note">This page is hosted on GitHub Pages, so it has no training backend. The <b>KAN Network</b> tab runs the real trained gate in your browser, the <b>Research</b> tab shows the stored benchmark results, and the <b>Guide</b> explains every number. Live training, hardware telemetry and the Mock tab need the app running on your own machine:
  clone the <a href="https://github.com/avanishkasar/KAN-KubeAgent" style="color:inherit">repository</a> and run <code>start.ps1</code> (Windows) or <code>start.sh</code> (Linux, macOS). The research paper is <a href="paper.pdf" style="color:inherit">here</a>.</p>
</div>"""


def main() -> None:
    gate = KANGate.load()
    net = network(gate)
    for e in net["edges"]:
        e["exact"] = exact_expr(gate.model, e["layer"], e["i"], e["j"])
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data").mkdir(exist_ok=True)
    (DOCS / "data" / "network.json").write_text(json.dumps(net))
    shutil.copy(ROOT / "experiments" / "results" / "results.json", DOCS / "data" / "results.json")
    shutil.copy(ROOT / "paper" / "kan_kubeagent.pdf", DOCS / "paper.pdf")
    html = (ROOT / "dashboard" / "frontend" / "index.html").read_text()
    html = html.replace("<script>\nconst API_BASE", '<script src="static-shim.js"></script>\n<script>\nconst API_BASE', 1)
    html = html.replace('<div class="error-banner" id="error-banner"></div>', '<div class="error-banner" id="error-banner"></div>\n  ' + BANNER, 1)
    (DOCS / "index.html").write_text(html)
    (DOCS / "static-shim.js").write_text(SHIM)
    (DOCS / ".nojekyll").write_text("")
    print("built", DOCS)


if __name__ == "__main__":
    main()
