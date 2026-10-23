const check = (condition, message) => { if (!condition) throw Error(message); };
check(document.documentElement.dataset.ready === 'true', 'Page did not load');
const embedded = document.querySelector('#figure-data');
const manifest = embedded ? JSON.parse(embedded.textContent).manifest : await (await fetch('data/manifest.json')).json();
if (manifest.kind === 'runtime') {
  check(document.querySelectorAll('.fig > svg').length === 1, 'Runtime needs one combined figure');
  const environments = new Set(manifest.runs.map(r => r.environment));
  const gammas = new Set(manifest.runs.map(r => r.gamma));
  const discount = document.querySelector('#controls select');
  check(discount.options.length === gammas.size, 'Discount choices missing');
  check(document.querySelectorAll('.x-axis').length === environments.size && !document.querySelector('.y-axis'), 'Runtime needs one X axis per MDP');
  check(!document.querySelector('.percentile-band'), 'Runtime has misleading replicate bands');
  for (const gamma of gammas) {
    discount.value = String(gamma); discount.dispatchEvent(new Event('change', { bubbles: true }));
    check(+document.querySelector('#runtime').dataset.gamma === gamma, 'Discount selector did not redraw');
    check(![...document.querySelectorAll('#runtime svg text')].some(t => t.textContent.includes('γ')), 'Unexpected discount annotation in the plot area');
    check(document.querySelectorAll('.runtime-panel').length === environments.size, 'Wrong runtime row count');
    const selected = manifest.runs.filter(r => r.gamma === gamma);
    const groups = window.d3.group(selected, r => JSON.stringify([r.environment, r.label]));
    check(document.querySelectorAll('.runtime-point').length === groups.size, 'Runtime needs one median per method and MDP');
    for (const replicates of groups.values()) {
      const run = replicates[0];
      const times = await Promise.all(replicates.map(async r => {
        const rows = embedded ? window.d3.csvParse(JSON.parse(embedded.textContent).csvs[r.file]) : await window.d3.csv(r.file);
        return +rows.at(-1)[manifest.time_column];
      }));
      const point = [...document.querySelectorAll('.runtime-panel')].find(p => p.dataset.environment === run.environment)
        .querySelector(`[data-method="${run.label}"]`);
      check(+point.dataset.seconds === window.d3.median(times), 'Runtime is not the median of actual individual measurements');
      check(+point.dataset.count === replicates.length, 'Runtime replicate count differs');
      check(+point.getAttribute('y1') + +point.getAttribute('y2') === 0, 'Method positions were jittered vertically');
      if (run.label === 'GReQ') check(point.getAttribute('stroke') === '#e5470c', 'Wrong primary color');
      const label = point.closest('.runtime-panel').querySelector(`.runtime-label[data-method="${run.label}"]`);
      check(label.querySelector('.runtime-value').textContent === `${window.d3.format('.3~g')(+point.dataset.seconds)} s`, 'Runtime label does not show measured seconds');
      check(label.getBBox().y + label.getBBox().height < 0, 'Algorithm/runtime label is not above its axis');
    }
    for (const row of document.querySelectorAll('.runtime-panel')) {
      const title = row.querySelector('.runtime-environment');
      check(title.getBBox().x + title.getBBox().width < 0, 'MDP title is not left of its axis');
      const nameBox = title.querySelector('.runtime-name').getBBox();
      check(Math.abs(nameBox.y + nameBox.height / 2) < 0.1, 'MDP name is not vertically centered on its X axis');
      if (row.dataset.environment.startsWith('Graph')) check(title.textContent === 'Graph', 'Unexpected Graph label');
      check(row.querySelectorAll('.x-axis').length === 1, 'MDP is missing its X axis');
      for (const side of ['above', 'below']) {
        const boxes = [...row.querySelectorAll(`.runtime-label[data-side="${side}"]`)].map(t => t.getBoundingClientRect()).sort((a, b) => a.left - b.left);
        for (let i = 1; i < boxes.length; i++) check(boxes[i].left >= boxes[i - 1].right - 1, 'Runtime labels overlap');
      }
    }
  }
  for (const axis of document.querySelectorAll('.x-axis')) {
    const ticks = [...axis.querySelectorAll('.tick')];
    check(ticks.every(t => t.__data__ > 0 && Math.abs(Math.log10(t.__data__) - Math.round(Math.log10(t.__data__))) < 1e-8), 'Runtime axis is not logarithmic with decade ticks');
    const boxes = ticks.map(t => t.querySelector('text').getBoundingClientRect()).sort((a, b) => a.left - b.left);
    for (let i = 1; i < boxes.length; i++) check(boxes[i].left >= boxes[i - 1].right - 1, 'Overlapping runtime ticks');
  }
  const point = document.querySelector('.runtime-point'), before = point.getAttribute('stroke');
  document.querySelector('#theme input[value="gray"]').click();
  check(document.querySelector('.runtime-point').getAttribute('stroke') !== before, 'Runtime palette did not change');
  document.querySelector('#theme input[value="focus"]').click();
  const click = HTMLAnchorElement.prototype.click, downloads = [];
  HTMLAnchorElement.prototype.click = function () { downloads.push({name: this.download, url: this.href}); };
  const width = document.querySelector('#runtime svg').getAttribute('width');
  try {
    for (const format of ['svg', 'pdf']) {
      await document.querySelector(`button[data-figure="runtime"][data-format="${format}"]`).onclick();
      const content = await (await fetch(downloads.at(-1).url)).text();
      check(downloads.at(-1).name === `runtime.${format}`, 'Wrong runtime filename');
      check(format === 'svg' ? content.includes('487pt') && content.includes(`γ = ${discount.value}`) : content.startsWith('%PDF-'), 'Invalid runtime export');
    }
  } finally { HTMLAnchorElement.prototype.click = click; }
  check(document.querySelector('#runtime svg').getAttribute('width') === width, 'Runtime export changed visible layout');
  return;
}
check(document.querySelectorAll('.fig > svg').length === (manifest.time_column ? 2 : 1), 'Wrong panel count');
check(document.querySelectorAll('h2, h3, p').length === 0, 'Unexpected figure prose');
check(!document.querySelector('.panel-letter'), 'Unexpected panel lettering');
const heading = document.querySelector('#figure-title');
check(heading.textContent === (manifest.title ?? ''), 'Wrong figure title');
check(heading.hidden === !manifest.title, 'Incorrect title visibility');
check(!document.querySelector('.seed'), 'Expected aggregated curves only');
check(document.querySelectorAll('#controls input[role="switch"]').length === 2, 'Missing axis switches');
check(document.querySelectorAll('#controls select').length === 1, 'Unexpected dropdown');
check(JSON.stringify([...document.querySelectorAll('#theme input')].map(input => input.value)) === JSON.stringify(['focus', 'gray', 'colorblind', 'raspberry', 'cobalt']), 'Unexpected palette choices');
check(document.querySelector('#theme input[value="focus"]').checked, 'Focus is not the default');
const primary = document.querySelector('[data-series="GReQ"] .median');
if (primary) check(primary.getAttribute('stroke') === '#e5470c', 'GReQ is not the primary color');
if (manifest.time_column) check([...document.querySelectorAll('#time text')].some(node => node.textContent === 'run time (s)'), 'Wrong time-axis title');
const validPaths = () => {
  for (const path of document.querySelectorAll('svg path')) check(!/NaN|Infinity/.test(path.getAttribute('d')), 'Invalid plot path');
};
const checkTickSpacing = () => {
  for (const axis of document.querySelectorAll('.x-axis')) {
    const boxes = [...axis.querySelectorAll('.tick text')].filter(text => text.textContent).map(text => text.getBoundingClientRect()).sort((a, b) => a.left - b.left);
    for (let i = 1; i < boxes.length; i++) check(boxes[i].left >= boxes[i - 1].right - 1, 'Overlapping x tick labels');
  }
};
const checkMajorTicks = () => {
  for (const axis of document.querySelectorAll('.x-axis, .y-axis')) {
    const ticks = [...axis.querySelectorAll('.tick')];
    check(ticks.every(tick => tick.querySelector('text').textContent.trim()), 'Expected labelled major ticks');
    check(axis.querySelectorAll('line').length === ticks.length + 1, 'Tick-line count differs from labelled ticks');
  }
};
const sample = document.querySelector('#samples');
check(sample.dataset.metric === manifest.metrics[0].key, 'Initial metric ignores manifest order');
const domains = [sample.dataset.xDomain, sample.dataset.yDomain];
const chip = document.querySelector('#methods button');
chip.click();
check(chip.getAttribute('aria-pressed') === 'false', 'Method filter did not toggle');
check(JSON.stringify([sample.dataset.xDomain, sample.dataset.yDomain]) === JSON.stringify(domains), 'Hiding a method rescaled axes');
chip.click();
const metric = document.querySelector('#controls select');
const original = metric.value;
if (metric.options.length > 1) {
  metric.value = metric.options[1].value;
  metric.dispatchEvent(new Event('change', { bubbles: true }));
  check(sample.dataset.metric === metric.value, 'Metric control did not redraw');
  metric.value = original;
  metric.dispatchEvent(new Event('change', { bubbles: true }));
}
const yAxis = document.querySelector('#controls input[data-axis="y"]');
check(!yAxis.disabled, 'Positive metric should allow log y');
const initialY = sample.dataset.yScale;
yAxis.click();
check(sample.dataset.yScale !== initialY, 'Log Y did not toggle');
yAxis.click();
const xAxis = document.querySelector('#controls input[data-axis="x"]');
xAxis.click();
for (const plot of document.querySelectorAll('.fig')) {
  check(plot.dataset.xScale === 'log', 'Log X did not apply to every panel');
  check(JSON.parse(plot.dataset.xDomain)[0] > 0, 'Log X includes zero');
}
validPaths(); checkTickSpacing(); checkMajorTicks();
xAxis.click();
check(sample.dataset.xDomain === domains[0], 'Linear X did not restore original domain');
xAxis.click();
const lower = document.querySelector('.percentiles .lower');
const upper = document.querySelector('.percentiles .upper');
check(lower.min === '0' && lower.max === '50' && upper.min === '50' && upper.max === '100', 'Wrong percentile constraints');
check(lower.value === '25' && upper.value === '75', 'Wrong default percentiles');
const fieldBox = document.querySelector('.percentiles').getBoundingClientRect();
const readoutBox = document.querySelector('.percentiles output').getBoundingClientRect();
check(readoutBox.top >= fieldBox.top && readoutBox.right <= fieldBox.right + 1, 'Percentile readout escaped its control');
const setBand = (lo, hi) => {
  lower.value = lo; upper.value = hi;
  lower.dispatchEvent(new Event('input', { bubbles: true }));
  for (const plot of document.querySelectorAll('.fig')) {
    check(+plot.dataset.lowerPercentile === +lower.value && +plot.dataset.upperPercentile === +upper.value, 'Band did not redraw');
  }
  validPaths();
};
setBand(0, 100);
check(document.querySelectorAll('.percentile-band').length === document.querySelectorAll('.median').length, 'Missing bands');
setBand(75, 25);
check(lower.value === '50' && upper.value === '50', 'Handles can cross the median');
check(document.querySelector('.percentiles').classList.contains('collapsed'), 'Coincident handles are not separated');
for (const band of document.querySelectorAll('.percentile-band')) check(band.__data__.every(p => p.lo === p.y && p.hi === p.y), '50–50 band is not the median');
lower.focus(); check(document.activeElement === lower, 'Lower handle cannot receive keyboard focus');
upper.focus(); check(document.activeElement === upper, 'Upper handle cannot receive keyboard focus');
setBand(10, 90);
const before = document.querySelector('.median').getAttribute('stroke');
document.querySelector('#theme input[value="gray"]').click();
check(document.querySelector('.median').getAttribute('stroke') !== before, 'Palette did not change');
document.querySelector('#theme input[value="focus"]').click();
const anchorClick = HTMLAnchorElement.prototype.click;
const downloads = [];
HTMLAnchorElement.prototype.click = function () { downloads.push({ name: this.download, url: this.href }); };
const width = sample.querySelector('svg').getAttribute('width');
try {
  await document.querySelector('#export-samples').onclick();
  const svg = await (await fetch(downloads.at(-1).url)).text();
  check(downloads.at(-1).name === 'samples.svg' && svg.includes('234pt'), 'Invalid SVG download');
  check(svg.includes('10–90%') && !svg.includes('panel-letter'), 'SVG does not reflect current controls');
  await document.querySelector('#export-samples-pdf').onclick();
  const pdf = await (await fetch(downloads.at(-1).url)).arrayBuffer();
  check(downloads.at(-1).name === 'samples.pdf' && new TextDecoder().decode(pdf.slice(0, 5)) === '%PDF-', 'Invalid PDF download');
  check(pdf.byteLength > 1000, 'Empty PDF');
  const pdfText = new TextDecoder('latin1').decode(pdf);
  check(pdfText.includes('/Helvetica-Bold') && !pdfText.includes('/Times-Roman'), 'PDF font substitution changed the figure');
} finally { HTMLAnchorElement.prototype.click = anchorClick; }
check(sample.querySelector('svg').getAttribute('width') === width, 'Export changed the visible layout');
check(!document.querySelector('.export-error').textContent, 'Export failed');
setBand(25, 75);
checkTickSpacing(); checkMajorTicks();
