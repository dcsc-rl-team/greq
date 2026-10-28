/** Render CSV curves and runtimes from a fetched or embedded data manifest. */

export function quantile(values, probability) {
  if (!values.length || !Number.isFinite(probability) || probability < 0 || probability > 1) throw Error('Invalid quantile');
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index), upper = Math.ceil(index);
  return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
}

export function parseRows(rows, manifest) {
  if (!rows.length) throw Error('Empty CSV');
  const axes = [manifest.step_column, manifest.time_column].filter(Boolean);
  const columns = [...new Set([...axes, ...manifest.metrics.map(m => m.key)])];
  const result = rows.map(row => Object.fromEntries(columns.map(key => {
    if (row[key] == null || String(row[key]).trim() === '') throw Error(`Missing ${key}`);
    const value = Number(row[key]);
    if (!Number.isFinite(value)) throw Error(`Nonfinite ${key}`);
    return [key, value];
  })));
  for (let i = 0; i < result.length; i++) {
    const step = result[i][manifest.step_column];
    if (!Number.isInteger(step) || step < 0 || (i && step <= result[i - 1][manifest.step_column])) {
      throw Error('Samples must be strictly increasing nonnegative integers');
    }
    if (manifest.time_column) {
      const time = result[i][manifest.time_column];
      if (time < 0 || (i && time < result[i - 1][manifest.time_column])) throw Error('Time must be nondecreasing');
    }
  }
  return result;
}

function interpolate(points, x) {
  if (x < points[0].x || x > points.at(-1).x) throw Error('Extrapolation is not permitted');
  let left = 0, right = points.length - 1;
  while (left < right) {
    const middle = Math.floor((left + right + 1) / 2);
    if (points[middle].x <= x) left = middle; else right = middle - 1;
  }
  const a = points[left], b = points[Math.min(left + 1, points.length - 1)];
  return a.x === x || a.x === b.x ? a.y : a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x);
}

function paletteSlots(labels) {
  const slots = new Map([['GReQ', 1], ['QL', 0], ['Zap', 2], ['Zap-batched', 3], ['NeSA', 5]].filter(([name]) => labels.includes(name)));
  for (const name of [...labels].sort()) {
    if (!slots.has(name)) slots.set(name, [0, 1, 2, 3, 4, 5].find(slot => ![...slots.values()].includes(slot)));
  }
  return slots;
}

/** Aggregate at equal x values on the runs' shared support. */
export function summarize(runs, metric, axis, [lower, upper] = [0.25, 0.75]) {
  if (![lower, upper].every(Number.isFinite) || lower < 0 || lower > 0.5 || upper < 0.5 || upper > 1) throw Error('Percentiles must bracket the median');
  const groups = new Map();
  for (const run of runs) {
    if (!groups.has(run.label)) groups.set(run.label, []);
    groups.get(run.label).push(run);
  }
  if (groups.size > 6) throw Error('Use separate pages for more than six groups');
  const slots = paletteSlots([...groups.keys()]);
  return [...groups].map(([label, members]) => {
    const slot = slots.get(label);
    const traces = members.map(run => {
      const unique = new Map(run.rows.map(row => [row[axis], row[metric]]));
      return [...unique].map(([x, y]) => ({ x, y })).sort((a, b) => a.x - b.x);
    });
    const low = Math.max(...traces.map(points => points[0].x));
    const high = Math.min(...traces.map(points => points.at(-1).x));
    if (high < low) throw Error(`Runs for ${label} have no shared support`);
    const grid = [...new Set(traces.flatMap(points => points.map(p => p.x)))].filter(x => x >= low && x <= high).sort((a, b) => a - b);
    const points = grid.map(x => {
      const values = traces.map(trace => interpolate(trace, x));
      return { x, y: quantile(values, 0.5), lo: quantile(values, lower), hi: quantile(values, upper) };
    });
    const partial = members.some(run => !['completed', 'unknown'].includes(run.status));
    return { key: label, label, slot, points, traces, count: members.length, partial };
  });
}

function domain(values, log = false) {
  let low = Infinity, high = -Infinity;
  for (const value of values) { low = Math.min(low, value); high = Math.max(high, value); }
  if (log && low <= 0) throw Error('Log axes require positive values');
  if (low !== high) return [low, high];
  return log ? [low / 2, high * 2] : [low - (Math.abs(low) * 0.1 || 1), high + (Math.abs(high) * 0.1 || 1)];
}

const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

/** Compute the median of independently measured scalar runtimes per cell. */
export function runtimePoints(runs, timeColumn) {
  const groups = new Map();
  for (const run of runs) {
    const seconds = run.rows.at(-1)?.[timeColumn];
    if (!run.environment || !Number.isFinite(run.gamma) || !Number.isFinite(seconds) || seconds <= 0) throw Error('Invalid scalar runtime');
    const key = JSON.stringify([run.environment, run.gamma, run.label]);
    if (!groups.has(key)) groups.set(key, { environment: run.environment, gamma: run.gamma, method: run.label, values: [] });
    groups.get(key).values.push(seconds);
  }
  return [...groups.values()].map(({ values, ...point }) => ({ ...point, seconds: quantile(values, 0.5), count: values.length }));
}

async function mountRuntime(manifest, runs) {
  const d3 = window.d3;
  const { figure, xAxis, exportSVG, exportPDF } = await import('./viz/fig.mjs');
  const { panel, themePanel } = await import('./viz/controls.mjs');
  const { setPalette, colorOf } = await import('./viz/palette.mjs');
  const points = runtimePoints(runs, manifest.time_column);
  const environments = [...new Set(points.map(p => p.environment))];
  const gammas = [...new Set(points.map(p => p.gamma))].sort((a, b) => a - b);
  const methods = [...new Set(points.map(p => p.method))].sort((a, b) => {
    const order = ['GReQ', 'QL', 'NeSA', 'Zap', 'Zap-batched'];
    return order.indexOf(a) - order.indexOf(b);
  });
  const slots = paletteSlots(methods);
  const values = points.map(p => p.seconds);
  const low = Math.floor(Math.log10(Math.min(...values))), high = Math.ceil(Math.log10(Math.max(...values)));
  const extent = [10 ** low, 10 ** Math.max(low + 1, high)];
  let selectedGamma = gammas[0];
  for (const selector of ['#methods', '#time-panel']) document.querySelector(selector).remove();
  document.querySelector('main').classList.add('runtime-page');
  const container = document.querySelector('#samples');
  container.id = 'runtime';
  setPalette('focus');
  const plot = figure(container, { width: 'full', aspect: 0.34, minHeight: 300,
    margin: { left: 7.6, right: 0.8, top: 5, bottom: 3.6 },
    draw(ctx) {
      const { svg, g, iw, ih, t } = ctx;
      container.dataset.gamma = selectedGamma;
      svg.append('desc').text(`Discount γ = ${selectedGamma}. ${manifest.time_description ?? 'Single-run synchronized JIT training time; no vmap.'}`);
      const x = d3.scaleLog().domain(extent).range([0, iw]);
      const y = d3.scalePoint().domain(environments).range([0, ih]);
      for (const environment of environments) {
        const group = g.append('g').attr('class', 'runtime-panel').attr('data-environment', environment).attr('data-gamma', selectedGamma)
          .attr('transform', `translate(0,${y(environment)})`);
        const name = environment.split(' (')[0], detail = environment.slice(name.length).trim();
        const title = group.append('text').attr('class', 'runtime-environment').attr('x', -ctx.m.left + t.size * 0.3)
          .attr('y', 0).attr('font-weight', 600);
        const nameText = title.append('tspan').attr('class', 'runtime-name').text(name);
        const nameBox = nameText.node().getBBox();
        title.attr('y', -nameBox.y - nameBox.height / 2);
        if (detail && name !== 'Graph') title.append('tspan').attr('x', -ctx.m.left + t.size * 0.3).attr('dy', '1.5em').attr('font-size', '0.8em').text(detail);
        xAxis({ ...ctx, g: group, ih: 0 }, x, { label: 'run time (s)', grid: false, offset: 0 }).attr('class', 'x-axis');
        const selected = points.filter(p => p.environment === environment && p.gamma === selectedGamma);
        for (const p of selected) {
          const height = p.method === 'GReQ' ? t.dot * 1.8 : t.dot;
          group.append('line').attr('class', 'runtime-point').attr('data-method', p.method).attr('data-seconds', p.seconds).attr('data-count', p.count)
            .attr('x1', x(p.seconds)).attr('x2', x(p.seconds)).attr('y1', -height).attr('y2', height)
            .attr('stroke-width', t.line).attr('stroke', colorOf(slots.get(p.method)))
            .append('title').text(`${p.method}: median ${d3.format('.4g')(p.seconds)} s across ${p.count} independently timed run${p.count === 1 ? '' : 's'}`);
        }
        const entries = [...selected].sort((a, b) => a.seconds - b.seconds);
        const labels = entries.map(p => {
          const label = group.append('text').attr('class', 'runtime-label').attr('data-method', p.method).attr('data-side', 'above')
            .attr('y', -2.8 * t.size).attr('fill', colorOf(slots.get(p.method)));
          label.append('tspan').attr('x', 0).text(p.method);
          label.append('tspan').attr('class', 'runtime-value').attr('x', 0).attr('dy', '1.2em').text(`${d3.format('.3~g')(p.seconds)} s`);
          return label;
        });
        const measure = label => Math.max(...label.selectAll('tspan').nodes().map(node => node.getComputedTextLength()));
        let widths = labels.map(measure);
        const gap = t.size * 0.5;
        const scale = Math.min(1, (iw - gap * (labels.length - 1)) / d3.sum(widths));
        labels.forEach(label => label.attr('font-size', t.size * scale));
        widths = labels.map(measure);
        const lefts = [];
        for (let i = 0; i < entries.length; i++) lefts.push(Math.max(0, x(entries[i].seconds) - widths[i] / 2, i ? lefts[i - 1] + widths[i - 1] + gap : 0));
        for (let i = entries.length - 1; i >= 0; i--) lefts[i] = Math.min(lefts[i], (i === entries.length - 1 ? iw : lefts[i + 1] - gap) - widths[i]);
        entries.forEach((p, i) => {
          const left = lefts[i];
          labels[i].attr('x', left).selectAll('tspan').attr('x', left);
          labels[i].append('title').text(`${d3.format('.6g')(p.seconds)} s`);
          group.append('line').attr('class', 'runtime-leader').attr('x1', x(p.seconds)).attr('x2', left + widths[i] / 2)
            .attr('y1', -t.dot * 1.8).attr('y2', -t.size * 1.2)
            .attr('stroke', colorOf(slots.get(p.method))).attr('stroke-width', t.hair);
        });
      }
    },
  });
  panel(document.querySelector('#controls'), [{ type: 'select', key: 'gamma', label: 'Discount',
    options: gammas.map(gamma => ({ value: String(gamma), label: `γ = ${gamma}` })), value: String(selectedGamma),
  }], next => { selectedGamma = Number(next.gamma); plot.render(); });
  for (const button of document.querySelectorAll('.exports button')) {
    button.dataset.figure = 'runtime';
    button.id = `export-runtime${button.dataset.format === 'pdf' ? '-pdf' : ''}`;
    button.onclick = async () => {
      const errorOutput = document.querySelector('.export-error');
      errorOutput.textContent = ''; button.disabled = true; button.dataset.exportStatus = 'running';
      try {
        await (button.dataset.format === 'pdf' ? exportPDF : exportSVG)(plot);
        button.dataset.exportStatus = 'completed';
      } catch (error) {
        button.dataset.exportStatus = 'failed'; errorOutput.textContent = error.message; throw error;
      } finally { button.disabled = false; }
    };
  }
  themePanel(document.querySelector('#theme'), { palette: 'focus' });
  document.documentElement.dataset.ready = 'true';
}

async function mount() {
  const d3 = window.d3;
  const { figure, xAxis, yAxis, endLabels, tooltip, exportSVG, exportPDF } = await import('./viz/fig.mjs');
  const { percentiles } = await import('./viz/percentiles.mjs');
  const { panel, chips, themePanel } = await import('./viz/controls.mjs');
  const { setPalette, recolor, cssColor } = await import('./viz/palette.mjs');
  const embedded = document.querySelector('#figure-data');
  const data = embedded ? JSON.parse(embedded.textContent) : null;
  const manifest = data ? data.manifest : await (async () => {
    const response = await fetch('data/manifest.json');
    if (!response.ok) throw Error(`Manifest HTTP ${response.status}`);
    return response.json();
  })();
  const heading = document.querySelector('#figure-title');
  heading.textContent = manifest.title ?? '';
  heading.hidden = !manifest.title;
  if (manifest.title) document.title = manifest.title;
  const runs = await Promise.all(manifest.runs.map(async run => {
    const rows = data ? d3.csvParse(data.csvs[run.file]) : await d3.csv(run.file);
    return { ...run, rows: parseRows(rows, manifest) };
  }));
  if (manifest.kind === 'runtime') return mountRuntime(manifest, runs);
  const firstMetric = manifest.metrics[0];
  const positive = key => runs.every(run => run.rows.every(row => row[key] > 0));
  const state = { metric: firstMetric.key, scale: positive(firstMetric.key) ? 'log' : 'linear', xscale: 'linear', lower: 25, upper: 75, hidden: new Set(), focus: null };
  const figures = [];
  setPalette('focus');

  /** Draw median curves and percentile bands with selectable axis scales. */
  function learningCurve(container, axis, xLabel) {
    let cachedKey, cachedSeries;
    const getSeries = () => {
      const key = `${state.metric}:${state.lower}:${state.upper}`;
      if (key !== cachedKey) {
        cachedKey = key;
        cachedSeries = summarize(runs, state.metric, axis, [state.lower / 100, state.upper / 100]);
      }
      return cachedSeries;
    };
    let series = getSeries();
    recolor(series);
    const tip = tooltip(container);
    const fig = figure(container, { width: 'column', aspect: 0.9,
      margin: { left: 4.2, right: 7.8, bottom: 3.9, top: 2.2 },
      draw(ctx) {
        const { g, iw, ih, t, web } = ctx;
        const displayed = series.map(s => ({ ...s, points: s.points.filter(p => state.xscale !== 'log' || p.x > 0) })).filter(s => s.points.length);
        const points = displayed.flatMap(s => s.points);
        const x = (state.xscale === 'log' ? d3.scaleLog() : d3.scaleLinear()).domain(domain(points.map(p => p.x), state.xscale === 'log')).range([0, iw]);
        const y = (state.scale === 'log' ? d3.scaleLog() : d3.scaleLinear()).domain(domain(points.flatMap(p => [p.lo, p.hi]), state.scale === 'log')).range([ih, 0]);
        container.dataset.xDomain = JSON.stringify(x.domain());
        container.dataset.yDomain = JSON.stringify(y.domain());
        container.dataset.metric = state.metric;
        container.dataset.xScale = state.xscale;
        container.dataset.yScale = state.scale;
        container.dataset.lowerPercentile = state.lower;
        container.dataset.upperPercentile = state.upper;
        ctx.svg.append('desc').text(`Median and empirical ${state.lower}–${state.upper}% percentile band across seeds. Primary algorithm: GReQ.${axis === manifest.time_column ? ` Timing: ${manifest.time_description ?? manifest.time_label ?? 'recorded training time'}.` : ''}`);
        const tickCount = Math.max(2, Math.floor(iw / (t.size * 5)));
        const numberFormat = axis === manifest.time_column ? '.3~g' : '~s';
        const tickFormat = state.xscale === 'log' ? x.tickFormat(tickCount, numberFormat) : d3.format(numberFormat);
        xAxis(ctx, x, { label: xLabel, ticks: [tickCount], format: tickFormat }).attr('class', 'x-axis');
        yAxis(ctx, y, { label: manifest.metrics.find(metric => metric.key === state.metric).label, format: d3.format('.2~g') }).attr('class', 'y-axis');
        const line = d3.line().x(p => x(p.x)).y(p => y(p.y));
        const area = d3.area().x(p => x(p.x)).y0(p => y(p.lo)).y1(p => y(p.hi));
        const marks = ctx.clip(), named = displayed.filter(s => !state.hidden.has(s.key));
        for (const s of named) {
          const group = marks.append('g').attr('data-series', s.key).attr('opacity', state.focus && state.focus !== s.key ? 0.18 : 1);
          group.append('path').datum(s.points).attr('class', 'percentile-band').attr('d', area(s.points)).attr('fill', s.color).attr('fill-opacity', 0.18);
          group.append('path').attr('class', 'median').attr('d', line(s.points)).attr('fill', 'none').attr('stroke', s.color)
            .attr('stroke-width', t.line).attr('stroke-dasharray', s.partial ? '4,3' : null);
          group.selectAll('.point').data(s.points).join('circle').attr('class', 'point').attr('cx', p => x(p.x)).attr('cy', p => y(p.y))
            .attr('r', s.points.length <= 20 ? t.dot * 0.65 : 0).attr('fill', s.color);
          const end = s.points.at(-1);
          g.append('circle').attr('cx', x(end.x)).attr('cy', y(end.y)).attr('r', t.dot).attr('fill', s.color);
        }
        endLabels(ctx, named.map(s => ({ label: s.label, color: s.color, x: x(s.points.at(-1).x), y: y(s.points.at(-1).y) })));
        if (!web) return;
        g.append('rect').attr('width', iw).attr('height', ih).attr('fill', 'transparent')
          .on('pointermove', event => {
            const [px, py] = d3.pointer(event, g.node());
            const rows = named.map(s => {
              const p = s.points[d3.bisector(p => p.x).center(s.points, x.invert(px))];
              const band = `; P${state.lower}–P${state.upper} ${d3.format('.3g')(p.lo)}–${d3.format('.3g')(p.hi)}`;
              return `<div style="--c:${s.color}"><i></i>${escape(s.label)} (n=${s.count}${s.partial ? ', partial' : ''})<span>${d3.format('.3g')(p.y)}${band}</span></div>`;
            });
            tip.show(rows.join(''), ctx.m.left + px, ctx.m.top + py);
          }).on('pointerleave', () => tip.hide());
      },
    });
    return { fig, refresh() { series = getSeries(); recolor(series); fig.render(); }, recolor() { recolor(series); } };
  }

  function exports(plot) {
    for (const button of document.querySelectorAll(`button[data-figure="${plot.fig.name}"]`)) {
      button.onclick = async () => {
        const errorOutput = button.closest('.plot').querySelector('.export-error');
        errorOutput.textContent = '';
        button.disabled = true;
        button.dataset.exportStatus = 'running';
        try {
          await (button.dataset.format === 'pdf' ? exportPDF : exportSVG)(plot.fig);
          button.dataset.exportStatus = 'completed';
        } catch (error) {
          button.dataset.exportStatus = 'failed';
          errorOutput.textContent = error.message;
          throw error;
        } finally { button.disabled = false; }
      };
    }
  }
  const samples = learningCurve(document.querySelector('#samples'), manifest.step_column, 'Samples');
  figures.push(samples);
  exports(samples);
  if (manifest.time_column) {
    const time = learningCurve(document.querySelector('#time'), manifest.time_column, manifest.time_label ?? 'run time (s)');
    figures.push(time);
    exports(time);
  } else {
    document.querySelector('#time-panel').remove();
    document.querySelector('main').classList.add('single-panel');
  }
  const refresh = () => figures.forEach(plot => plot.refresh());
  const controls = panel(document.querySelector('#controls'), [
    { type: 'select', key: 'metric', label: 'Metric', options: manifest.metrics.map(metric => ({ value: metric.key, label: metric.label })), value: state.metric },
    { type: 'switches', items: [
      { key: 'logx', label: 'Log X', value: state.xscale === 'log' },
      { key: 'logy', label: 'Log Y', value: state.scale === 'log' },
    ] },
  ], next => {
    state.metric = next.metric;
    state.xscale = next.logx ? 'log' : 'linear';
    state.scale = next.logy ? 'log' : 'linear';
    document.querySelector('#controls input[data-axis="y"]').disabled = !positive(state.metric);
    if (next.logy && !positive(state.metric)) { controls.set('logy', false); return; }
    refresh();
  });
  const axisControls = document.querySelectorAll('#controls input[role="switch"]');
  axisControls[0].dataset.axis = 'x';
  axisControls[1].dataset.axis = 'y';
  axisControls[1].disabled = !positive(state.metric);
  axisControls[0].title = 'Log x omits zero-valued sample counts and times.';
  axisControls[0].disabled = [manifest.step_column, manifest.time_column].filter(Boolean).some(axis =>
    !summarize(runs, state.metric, axis).some(group => group.points.some(p => p.x > 0)));
  percentiles(document.querySelector('#controls'), { onChange: (lower, upper) => {
    state.lower = lower;
    state.upper = upper;
    refresh();
  } });
  const methods = summarize(runs, state.metric, manifest.step_column);
  chips(document.querySelector('#methods'), methods.map(s => ({ ...s, label: escape(s.key + (s.partial ? ' (partial)' : '')), color: cssColor(s.slot) })), {
    onToggle: hidden => { state.hidden = hidden; refresh(); },
    onFocus: focus => { state.focus = focus; refresh(); },
  });
  themePanel(document.querySelector('#theme'), { palette: 'focus', onPalette: () => {
    figures.forEach(plot => plot.recolor());
  } });
  document.documentElement.dataset.ready = 'true';
}

if (typeof window !== 'undefined') await mount();
