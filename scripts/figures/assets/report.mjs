/** Validate report CSVs and pass selected comparisons to the figure renderer. */
const methods = { greq: 'GReQ', ql: 'QL', zap: 'Zap', 'zap-batched': 'Zap-batched' };
const metrics = [
  { key: 'value_linf', label: '‖Q − Q*‖∞' },
  { key: 'bellman_linf', label: 'Bellman error' },
  { key: 'policy_gap', label: 'Policy gap' },
  { key: 'policy_return', label: 'Policy return' },
];
const numberFields = ['gamma', 'eta', 'rho', 'k0', 'batch_size', 'seed', 'step', 'n_steps', 'eval_every', 'n_states', 'n_actions', 'branching', 'env_seed', 'train_seconds', ...metrics.map(m => m.key)];

export function parseReport(rows) {
  if (!rows.length) throw Error('CSV contains no observations.');
  return rows.map(row => {
    const r = { ...row };
    for (const key of numberFields) {
      if (!(key in r)) throw Error(`CSV column missing: ${key}`);
      r[key] = r[key] === '' ? null : Number(r[key]);
      if (r[key] !== null && !Number.isFinite(r[key])) throw Error(`Invalid CSV value: ${key}`);
    }
    if (!['graph', 'garnet'].includes(r.environment) || !Object.hasOwn(methods, r.method)) throw Error('Unknown environment or algorithm.');
    if (!Number.isInteger(r.seed) || r.seed < 0 || r.seed >= 2 ** 32 || !Number.isInteger(r.step) || r.step < 0 || !(r.gamma > 0 && r.gamma < 1)) throw Error('Invalid run identity.');
    if (!['completed', 'failed'].includes(r.status)) throw Error('Invalid run status.');
    if (!['scalar', 'vmap'].includes(r.execution) || (r.train_seconds !== null && r.execution !== 'scalar')) throw Error('Training times require explicitly scalar execution.');
    if (r.status === 'completed' && metrics.some(m => r[m.key] === null)) throw Error('Completed run has missing metrics.');
    return r;
  });
}

export function groupReport(rows, kind) {
  const groups = new Map(), settings = new Map(), seedSets = new Map(), environments = new Map();
  for (const row of rows) {
    if (row.status !== 'completed') throw Error('The CSV contains failed runs. Each comparison requires successful observations for every seed.');
    const cell = JSON.stringify([row.environment, row.method, row.gamma]);
    const setting = JSON.stringify([row.eta, row.rho, row.k0, row.batch_size, row.execution]);
    if (settings.has(cell) && settings.get(cell) !== setting) throw Error('Evaluation mixes hyperparameter settings within a comparison cell.');
    settings.set(cell, setting);
    const identity = JSON.stringify([row.n_states, row.n_actions, row.branching, row.env_seed, row.n_steps, row.eval_every]);
    if (environments.has(row.environment) && environments.get(row.environment) !== identity) throw Error('CSV mixes environment instances or sample budgets.');
    environments.set(row.environment, identity);
    if (!seedSets.has(cell)) seedSets.set(cell, new Set());
    seedSets.get(cell).add(row.seed);
    const key = JSON.stringify([row.environment, row.method, row.gamma, row.seed]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  let seeds;
  for (const values of seedSets.values()) {
    const current = JSON.stringify([...values].sort((a, b) => a - b));
    if (values.size !== 11 || (seeds && current !== seeds)) throw Error('Every comparison cell must contain the same eleven evaluation seeds.');
    seeds = current;
  }
  for (const values of groups.values()) {
    values.sort((a, b) => a.step - b.step);
    const first = values[0], last = values.at(-1);
    if (last.step !== last.n_steps) throw Error('Run does not reach its sample budget.');
    if (kind === 'runtime') {
      if (values.length !== 1 || !(last.train_seconds > 0) || last.eval_every !== last.n_steps) throw Error('Runtime requires one independently measured whole-run observation per seed.');
    } else {
      if (first.step !== 0 || values.length !== last.n_steps / last.eval_every + 1 || values.some((r, i) => r.step !== i * last.eval_every)) throw Error('Learning curve has missing or duplicate observations.');
      if (values.some(r => r.train_seconds !== null)) throw Error('Vmapped learning reports must not masquerade as scalar timings.');
    }
  }
  return [...groups.values()];
}

async function mount() {
  const notice = document.querySelector('#notice'), frame = document.querySelector('#figure-page');
  const environment = document.querySelector('#environment'), discount = document.querySelector('#discount'), view = document.querySelector('#view');
  const params = new URLSearchParams(location.search);
  const data = new URL(params.get('data') ?? '../../runs/results/', location.href);
  if (!data.pathname.endsWith('/')) data.pathname += '/';
  const quote = value => `'${String(value).replaceAll("'", "'\\''")}'`;
  const output = decodeURIComponent(data.pathname).replace(/^.*?\/runs\//, 'runs/').replace(/\/$/, '');
  const command = 'uv run --locked python -m scripts.report';
  if (location.protocol === 'file:') {
    notice.textContent = 'CSV loading requires a local server. From the repository root run:\nnode scripts/figures/serve.mjs 8002 .\nThen open http://localhost:8002/scripts/figures/';
    document.documentElement.dataset.ready = 'true';
    return;
  }
  if (data.origin !== location.origin) throw Error('CSV reports must be on the same origin.');
  let loaded = {}, warnings = [], template, observer, protocol, checksums = {};
  frame.addEventListener('load', () => {
    observer?.disconnect();
    const content = frame.contentDocument?.querySelector('main');
    if (!content) return;
    const resize = () => { frame.style.height = `${Math.ceil(content.getBoundingClientRect().height + 32)}px`; };
    observer = new ResizeObserver(resize); observer.observe(content); resize();
  });
  const options = (select, values, label) => {
    const before = select.value; select.replaceChildren();
    for (const value of values) { const o = document.createElement('option'); o.value = value; o.textContent = label(value); select.append(o); }
    if (values.includes(before)) select.value = before;
  };
  const show = () => {
    frame.hidden = true;
    environment.disabled = discount.disabled = view.value === 'runtime';
    environment.closest('label').hidden = discount.closest('label').hidden = view.value === 'runtime';
    notice.textContent = warnings.join('\n\n');
    if (!loaded[view.value]) return;
    try {
      const rows = loaded[view.value].filter(r => view.value === 'runtime' || (r.environment === environment.value && r.gamma === Number(discount.value)));
      if (!rows.length) throw Error('No observations for this environment and discount.');
      const groups = groupReport(rows, view.value), csvs = {}, runs = [];
      for (const [index, members] of groups.entries()) {
        const r = members[0], file = `run-${index}.csv`;
        csvs[file] = window.d3.csvFormat(members.map(row => Object.fromEntries(['step', ...(view.value === 'runtime' ? ['train_seconds'] : []), ...metrics.map(m => m.key)].map(k => [k, row[k]]))));
        runs.push({ file, label: methods[r.method], seed: r.seed, status: r.status, environment: r.environment === 'graph' ? 'Graph (6×6 padded)' : `Garnet (${r.n_states}×${r.n_actions})`, gamma: r.gamma });
      }
      const runtime = view.value === 'runtime';
      const manifest = { source_table: new URL(`${view.value}.csv`, data).pathname, source_sha256: checksums[view.value], kind: runtime ? 'runtime' : 'learning', step_column: 'step', time_column: runtime ? 'train_seconds' : null, metrics, runs, title: runtime ? 'Median JIT time — 11 seeds' : `${environment.selectedOptions[0].textContent} — tuned — 11 seeds`, time_label: 'run time (s)', time_description: 'Median of individually measured scalar sampling-and-learning times. Compilation, warmup, evaluation and I/O excluded.' };
      const doc = new DOMParser().parseFromString(template, 'text/html');
      const base = doc.createElement('base'); base.href = new URL('assets/', location.href).href; doc.head.prepend(base);
      const embedded = doc.createElement('script'); embedded.id = 'figure-data'; embedded.type = 'application/json'; embedded.textContent = JSON.stringify({ manifest, csvs }).replaceAll('<', '\\u003c'); doc.body.prepend(embedded);
      frame.srcdoc = '<!doctype html>' + doc.documentElement.outerHTML;
      frame.hidden = false; frame.title = manifest.title;
    } catch (error) { notice.textContent += `\n\n${error.message}\nInspect the CSV and its failure records; regenerate into a fresh output directory with:\n${command}${view.value === 'runtime' ? ' --runtime' : ''}`; }
  };
  const reload = async () => {
    loaded = {}; warnings = []; checksums = {}; protocol = null; frame.hidden = true;
    const missing = [];
    for (const name of ['tuning', 'evaluation', 'runtime']) {
      const url = new URL(`${name}.csv`, data);
      try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw Error(`HTTP ${response.status}`);
        const metadataResponse = await fetch(new URL(`${name}.json`, data), { cache: 'no-store' });
        if (!metadataResponse.ok) throw Error(`missing ${name}.json cache metadata`);
        const metadata = await metadataResponse.json();
        if (metadata.request?.phase !== name) throw Error('CSV phase does not match its requested report');
        protocol = metadata.request.protocol;
        if (!crypto.subtle) throw Error('checksum validation requires localhost or HTTPS');
        const bytes = await response.arrayBuffer();
        const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
        if (digest !== metadata.sha256) throw Error('CSV checksum mismatch');
        checksums[name] = digest;
        loaded[name] = parseReport(window.d3.csvParse(new TextDecoder().decode(bytes)));
      } catch (error) { missing.push({ name, url, message: error.message }); }
    }
    const flags = ['environments', 'methods', 'gammas', 'selection_seeds', 'evaluation_seeds', 'etas', 'rhos', 'ql_etas', 'ql_k0s', 'n_states', 'n_actions', 'branching', 'env_seed', 'n_steps', 'eval_every'];
    const args = ` --output ${quote(output)}` + flags.filter(k => protocol?.[k] !== undefined).map(k => ` --${k.replaceAll('_', '-')} ${[].concat(protocol[k]).map(quote).join(' ')}`).join('');
    warnings = missing.map(({ name, url, message }) => `${name}.csv is missing or unreadable (${message}).\nExpected: ${url.pathname}\nRun from the repository root:\n${name === 'tuning' ? 'uv run --locked python -m scripts.finetune' : command}${args}${name === 'runtime' ? ' --runtime' : ''}\nFor damaged or mismatched existing files, choose a fresh --output directory.`);
    if (!template) {
      const response = await fetch('assets/index.html');
      if (!response.ok) throw Error('figure template is missing');
      template = await response.text();
    }
    const rows = loaded.evaluation ?? loaded.runtime ?? [];
    options(environment, [...new Set(rows.map(r => r.environment))].sort(), e => {
      const row = rows.find(r => r.environment === e);
      return e === 'graph' ? 'Graph' : `Garnet (${row.n_states}×${row.n_actions})`;
    });
    options(discount, [...new Set(rows.map(r => String(r.gamma)))].sort((a,b) => +a - +b), g => `γ = ${g}`);
    show(); document.documentElement.dataset.ready = 'true';
  };
  environment.addEventListener('change', show); discount.addEventListener('change', show); view.addEventListener('change', show);
  document.querySelector('#reload').addEventListener('click', reload);
  await reload();
}

if (typeof document !== 'undefined' && document.querySelector('#notice')) mount().catch(error => {
  document.querySelector('#notice').textContent = `Cannot load reports: ${error.message}\nRun uv run --locked python -m scripts.report from the repository root.`;
  document.documentElement.dataset.ready = 'true';
});
