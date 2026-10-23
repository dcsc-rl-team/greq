/**
 * Capture responsive and fixed-size screenshots and report browser errors. Exits 1 on errors.
 *   node tests/browser/check.mjs <url> [--out shots] [--query palette=gray] [--steps steps.js]
 * Paper mode comes from ?mode=paper (themePanel reads it). Needs Node 22+ and Chrome (override with CHROME=...).
 * --offline: disable network access while checking a standalone file.
 * --frame-steps: expose a JS test body to gallery steps for each embedded figure.
 * --steps: a JS file whose body runs in the page after the web load, e.g. to click controls. It may `await`;
 * a throw is reported as an error. A "controls" screenshot is saved after it. Clicks on figures work in web mode only.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const argv = process.argv.slice(2), url = argv.find(a => !a.startsWith('--') && !argv[argv.indexOf(a) - 1]?.startsWith('--'));
const flag = (name, fallback) => (argv.includes(`--${name}`) ? argv[argv.indexOf(`--${name}`) + 1] : fallback);
if (!url) { console.error('usage: node check.mjs <url> [--out shots] [--query k=v] [--steps steps.js]'); process.exit(2); }
const out = flag('out', 'shots'), query = flag('query', ''), steps = flag('steps', null), frameSteps = flag('frame-steps', null);
const chromePath = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(chromePath, ['--headless=new', '--hide-scrollbars', `--remote-debugging-port=${port}`,
  `--user-data-dir=${mkdtempSync(join(tmpdir(), 'viz-check-'))}`, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page'); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0, errors = [];
const pending = new Map();
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    const entry = m.params.entry;
    const expected = argv.includes('--allow-missing-csv') && /404/.test(entry.text) && /\/(tuning|evaluation|runtime)\.csv$/.test(entry.url ?? '');
    if (!expected) errors.push(`${entry.text} ${entry.url ?? ''}`);
  }
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
/** Evaluate JavaScript in the page and collect exceptions in the error report. */
async function run(js) {
  const r = await send('Runtime.evaluate', { expression: js, awaitPromise: true, returnByValue: true });
  if (r?.exceptionDetails) { errors.push(`script: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`); return undefined; }
  return r?.result?.value;
}
await send('Runtime.enable'); await send('Log.enable'); await send('Page.enable');
if (argv.includes('--offline')) {
  await send('Network.enable');
  await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 });
}
mkdirSync(out, { recursive: true });

const figures = () => run("document.querySelectorAll('.fig > svg').length");
async function load(width, params) {
  const u = new URL(url);
  for (const [k, v] of new URLSearchParams([query, params].filter(Boolean).join('&'))) u.searchParams.set(k, v);
  await send('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 2, mobile: width < 600 });
  for (let attempt = 0; attempt < 2; attempt++) {                 // one retry: dev servers occasionally drop a module request
    const before = errors.length;
    await send('Page.navigate', { url: u.href });
    await sleep(2500);
    if ((await figures()) > 0 || attempt) return;
    errors.splice(before);
  }
}
async function shot(name, width) {
  await sleep(300);                                               // let resize-driven redraws settle before measuring
  const height = await run('Math.ceil(document.querySelector("main")?.getBoundingClientRect().bottom ?? document.documentElement.scrollHeight) + 24');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 2, mobile: width < 600 });
  await sleep(300);
  const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  writeFileSync(join(out, `${name}.png`), Buffer.from(data, 'base64'));
  console.log(`${name}: ${await figures()} figures → ${join(out, name)}.png`);
}

await load(1100, ''); await shot('web', 1100);
if (frameSteps) await run(`window.figureChecks = ${JSON.stringify(readFileSync(frameSteps, 'utf8'))}`);
if (steps) { await run(`(async () => { ${readFileSync(steps, 'utf8')} })()`); await shot('controls', 1100); }
await load(1100, 'mode=paper'); await shot('paper', 1100);
await load(390, ''); await shot('phone', 390);
console.log(errors.length ? `errors:\n${[...new Set(errors)].join('\n')}` : 'console: clean');
ws.close(); chrome.kill();
process.exit(errors.length ? 1 : 0);
