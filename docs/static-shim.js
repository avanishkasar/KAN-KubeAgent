/* Static preview shim: answers the dashboard's /api calls from files in data/,
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
