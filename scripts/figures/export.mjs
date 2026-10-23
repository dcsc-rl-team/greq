/** Export vector SVG/PDF figures through the page's shared controls into runs/. */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';

const [address, destination] = process.argv.slice(2);
if (!address || !destination) throw Error('Usage: node scripts/figures/export.mjs URL runs/PAGE/exports');
const output = resolve(destination), within = relative(resolve('runs'), output);
if (!within || within.startsWith('..') || within.startsWith('/')) throw Error('Exports must be under runs/');
if (existsSync(output)) throw Error('Export directory already exists');
mkdirSync(output, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), 'greq-export-'));
const port = 9800 + Math.floor(Math.random() * 500);
const executable = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const chrome = spawn(executable, ['--headless=new', '--hide-scrollbars', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
let spawnError, ws;
chrome.on('error', error => { spawnError = error; });
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  let target;
  for (let attempt = 0; attempt < 50 && !target; attempt++) {
    await sleep(200);
    if (spawnError) throw spawnError;
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(item => item.type === 'page'); } catch {}
  }
  if (!target) throw Error('Chrome debugging endpoint did not start; set CHROME to the executable');
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.addEventListener('open', resolve); ws.addEventListener('error', reject); });
  let nextId = 0;
  const pending = new Map(), errors = [];
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (pending.has(message.id)) {
      const request = pending.get(message.id);
      clearTimeout(request.timer);
      pending.delete(message.id);
      if (message.error) request.reject(Error(message.error.message)); else request.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => { pending.delete(id); reject(Error(`CDP timeout: ${method}`)); }, 30_000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
    return result.result?.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: output });
  const url = new URL(address);
  url.searchParams.delete('mode');
  await send('Page.navigate', { url: url.href });
  let ready = false;
  for (let attempt = 0; attempt < 100 && !ready; attempt++) {
    await sleep(100);
    if (errors.length) throw Error(errors.join('\n'));
    ready = await evaluate("document.documentElement.dataset.ready === 'true'");
  }
  if (!ready) throw Error('Figure page did not become ready');
  await evaluate('document.fonts.ready.then(() => true)');
  const names = await evaluate(`(async () => {
    const buttons = [...document.querySelectorAll('.exports button')];
    if (!buttons.length) throw Error('No export controls');
    const names = [];
    for (const button of buttons) {
      await button.onclick();
      if (button.dataset.exportStatus !== 'completed') throw Error('Export failed');
      names.push(button.dataset.figure + '.' + button.dataset.format);
    }
    return names;
  })()`);
  for (let attempt = 0; attempt < 100; attempt++) {
    if (names.every(name => existsSync(join(output, name)))) break;
    await sleep(100);
  }
  for (const name of names) {
    const path = join(output, name);
    if (!existsSync(path)) throw Error(`Export did not finish: ${name}`);
    const bytes = readFileSync(path);
    if (name.endsWith('.svg')) {
      const svg = bytes.toString('utf8');
      if (!/\bwidth="(?:234|487)pt"/.test(svg) || /NaN|Infinity/.test(svg)) throw Error(`Invalid SVG: ${name}`);
    } else if (!bytes.subarray(0, 5).equals(Buffer.from('%PDF-')) || bytes.length < 1000) {
      throw Error(`Invalid PDF: ${name}`);
    }
    console.log(path);
  }
} finally {
  ws?.close();
  chrome.kill();
  await sleep(200);
  rmSync(profile, { recursive: true, force: true });
}
