/**
 * Figure layout for interactive pixel-width views and fixed-point-width exports.
 * SVG attributes carry the styling into exported files.
 */
const d3 = window.d3;

export const THEMES = {
  // High-contrast labels, axes, and data strokes for screen and print.
  // axisW = axis line and tick width; hair = thin secondary strokes (outlines, leaders, isolines).
  web: { name: 'web', font: 'system-ui, -apple-system, sans-serif', size: 16, weight: 500, ink: '#0f172a', text: '#1e293b', axis: '#1e293b', axisW: 1.5, grid: '#e5e7eb', gridW: 1, line: 3, hair: 1.3, dot: 4.4 },
  paper: { name: 'paper', font: 'Helvetica, Arial, sans-serif', size: 9, weight: 400, ink: '#000000', text: '#111111', axis: '#111111', axisW: 0.8, grid: '#d9d9d9', gridW: 0.6, line: 1.6, hair: 0.7, dot: 2.4 },
};
/** Paper widths in pt: one column of a two-column paper, and full text width. */
export const WIDTHS = { column: 234, full: 487 };

let mode = 'web';
const figures = new Set();
export const getMode = () => mode;
export function setMode(next) { mode = next; for (const f of figures) f.render(); }
/** Redraw every figure, e.g. after setPalette(). */
export function rerender() { for (const f of figures) f.render(); }

/**
 * margin is in units of the font size (an object, or a function for state-dependent margins).
 * draw(ctx) receives {svg, g, w, h, iw, ih, m, t, web, clip}.
 * ctx.clip() appends a group clipped to the plot area: put data marks in it so nothing spills onto the axes.
 * Keep labels, end dots and annotations outside it.
 */
let clipCount = 0;
export function figure(container, { width = 'column', aspect = 0.85, minHeight = 260, margin = {}, draw }) {
  const root = d3.select(container).classed('fig', true);
  const svg = root.append('svg').attr('xmlns', 'http://www.w3.org/2000/svg');
  let pixels = 0;
  const fig = { node: svg.node(), render, snapshot, name: container.id || 'figure' };
  function snapshot() {
    try {
      render('paper');
      const clone = svg.node().cloneNode(true);
      clone.removeAttribute('style');
      return clone;
    } finally { render(); }
  }
  function render(renderMode = mode) {
    const t = THEMES[renderMode], paper = renderMode === 'paper';
    const w = paper ? WIDTHS[width] : pixels;
    if (!w) return;
    const h = paper ? Math.round(w * aspect) : Math.max(minHeight, Math.round(w * aspect));   // phones keep a usable height
    const em = { top: 2, right: 1, bottom: 3, left: 3.4, ...(typeof margin === 'function' ? margin() : margin) };
    const m = Object.fromEntries(Object.entries(em).map(([k, v]) => [k, v * t.size]));
    svg.selectAll('*').remove();
    root.selectAll('.tip').attr('hidden', true);          // a redraw invalidates any open tooltip
    svg.attr('viewBox', `0 0 ${w} ${h}`)
      .attr('width', paper ? `${w}pt` : w).attr('height', paper ? `${h}pt` : h)
      .style('width', paper ? `${w * 4 / 3}px` : null)
      .attr('font-family', t.font).attr('font-size', t.size).attr('fill', t.text);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    const iw = w - m.left - m.right, ih = h - m.top - m.bottom, id = `plot-clip-${++clipCount}`, pad = t.line / 2;
    // Half-stroke padding preserves the full width of lines along the plot edge.
    svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('x', -pad).attr('y', -pad).attr('width', iw + 2 * pad).attr('height', ih + 2 * pad);
    const clip = () => g.append('g').attr('clip-path', `url(#${id})`);
    draw({ svg, g, w, h, iw, ih, m, t, web: !paper, clip });
  }
  new ResizeObserver(() => {
    const next = Math.floor(container.clientWidth);
    if (next !== pixels) { pixels = next; if (mode === 'web') render(); }
  }).observe(container);
  figures.add(fig);
  return fig;
}

/**
 * Range-frame axes: the axis line spans only the data extent and is offset from the plot.
 * A grey grid is drawn behind data at major labelled ticks; grid: false disables it.
 * ticks are d3 arguments, e.g. [5].
 * Log scales covering at least 1.5 decades get labels at powers of ten (10⁻¹, 10⁰, 10¹).
 * Only major log ticks are drawn, including for narrower domains.
 * Band and point scales work too: ticks, grid and frame sit at category centres.
 */
export function xAxis({ g, ih, iw, t }, x, { ticks, format, label, extent, offset = 0.6, grid = true } = {}) {
  ticks ??= [Math.max(3, Math.round(iw / (t.size * 7)))];
  extent ??= extentOf(x);
  const p = at(x);
  if (grid) gridLines(g, t, major(x, ticks, extent).map(v => [p(v), 0, p(v), ih]));
  const a = g.append('g').attr('transform', `translate(0,${ih + offset * t.size})`)
    .call(ticked(d3.axisBottom(x), x, ticks, format, t, extent));
  frame(a, t, [p(extent[0]), p(extent[1])], 'x', x, extent);
  if (label) g.append('text').attr('x', iw).attr('y', ih + (offset + 2.8) * t.size).attr('text-anchor', 'end').attr('fill', t.ink).attr('font-weight', 600).text(label);
  return a;
}
export function yAxis({ g, m, iw, t }, y, { ticks = [4], format, label, extent, offset = 0.6, grid = true } = {}) {
  extent ??= extentOf(y);
  const p = at(y);
  if (grid) gridLines(g, t, major(y, ticks, extent).map(v => [0, p(v), iw, p(v)]));
  const a = g.append('g').attr('transform', `translate(${-offset * t.size},0)`)
    .call(ticked(d3.axisLeft(y), y, ticks, format, t, extent));
  frame(a, t, [p(extent[0]), p(extent[1])], 'y', y, extent);
  if (label) g.append('text').attr('x', -m.left).attr('y', -0.9 * t.size).attr('fill', t.ink).attr('font-weight', 600).text(label);
  return a;
}
const categorical = s => typeof s.bandwidth === 'function';
const at = s => (categorical(s) ? v => s(v) + s.bandwidth() / 2 : s);          // pixel position, band centre for categories
const extentOf = s => (categorical(s) ? [s.domain()[0], s.domain().at(-1)] : s.domain());
const powers = s => s.base && Math.log10(s.domain()[1] / s.domain()[0]) >= 1.5;
/** Major tick values: categories, decades for power-of-ten axes, otherwise only ticks that carry a label. */
function major(s, ticks, extent) {
  if (categorical(s)) return s.domain();
  if (powers(s)) {
    const [lo, hi] = extent.map(Math.log10);
    return d3.range(Math.ceil(Math.min(lo, hi) - 1e-9), Math.floor(Math.max(lo, hi) + 1e-9) + 1).map(k => 10 ** k);
  }
  const label = s.tickFormat(...ticks);
  return s.ticks(...ticks).filter(v => label(v) !== '');
}
function gridLines(g, t, segments) {
  g.insert('g', ':first-child').attr('class', 'grid').attr('pointer-events', 'none').selectAll('line').data(segments).join('line')
    .attr('x1', d => d[0]).attr('y1', d => d[1]).attr('x2', d => d[2]).attr('y2', d => d[3])
    .attr('stroke', t.grid).attr('stroke-width', t.gridW);
}
function ticked(axis, s, ticks, format, t, extent) {
  axis.tickSize(t.size * 0.45).tickPadding(t.size * 0.35).tickSizeOuter(0);
  if (categorical(s)) return axis.tickFormat(format ?? null);
  if (powers(s)) return axis.tickValues(major(s, ticks, extent)).tickFormat(() => '');
  if (s.base) {
    const label = format ?? s.tickFormat(...ticks);
    const values = major(s, ticks, extent).filter(value => label(value) !== '');
    return axis.tickValues(values).tickFormat(label);
  }
  return axis.ticks(...ticks).tickFormat(format ?? null);
}
function frame(a, t, [p0, p1], dir, s, extent) {
  a.attr('font-family', null).attr('font-size', null).attr('color', t.axis);
  a.selectAll('.tick line').attr('stroke-width', t.axisW);
  a.selectAll('text').attr('fill', t.text).attr('font-weight', t.weight);
  a.select('.domain').remove();
  a.insert('line', ':first-child').attr('stroke', t.axis).attr('stroke-width', t.axisW).attr('stroke-linecap', 'square')
    .attr(dir === 'x' ? 'x1' : 'y1', p0).attr(dir === 'x' ? 'x2' : 'y2', p1);
  if (!powers(s)) return;
  // "10" with a raised exponent, built from a tspan so it exports cleanly.
  a.selectAll('.tick text').each(function (d) {
    const k = Math.round(Math.log10(d));
    d3.select(this).text('10').append('tspan').attr('dy', '-0.55em').attr('font-size', '0.72em').text(String(k).replace('-', '−'));
  });
}

/** Reference lines at meaningful values (0, chance, a target): darker than the grid. */
export function rules({ g, iw, t }, y, values, { dashed = false } = {}) {
  g.append('g').selectAll('line').data(values).join('line').attr('x2', iw).attr('y1', y).attr('y2', y)
    .attr('stroke', t.axis).attr('stroke-opacity', dashed ? 1 : 0.35).attr('stroke-width', dashed ? t.hair : t.axisW).attr('stroke-dasharray', dashed ? '3,3' : null);
}

/**
 * End labels: names at line ends, spread apart, with a short leader line when a label was moved.
 * items: [{label, color, x, y}] in pixels.
 */
export function endLabels({ g, ih, t }, items, { dx = 0.6 } = {}) {
  const gap = t.size * 1.2, placed = [...items].sort((a, b) => a.y - b.y).map(d => ({ ...d, ly: d.y }));
  for (let i = 1; i < placed.length; i++) placed[i].ly = Math.max(placed[i].ly, placed[i - 1].ly + gap);
  const over = (placed.at(-1)?.ly ?? 0) - ih;
  if (over > 0) for (const d of placed) d.ly -= over;
  for (let i = placed.length - 2; i >= 0; i--) placed[i].ly = Math.min(placed[i].ly, placed[i + 1].ly - gap);
  const lab = g.append('g').selectAll('g').data(placed).join('g').attr('opacity', d => d.opacity ?? 1);
  lab.filter(d => Math.abs(d.ly - d.y) > 2).append('path').attr('fill', 'none').attr('stroke', d => d.color).attr('stroke-width', t.hair)
    .attr('d', d => `M${d.x + 2},${d.y}C${d.x + dx * t.size},${d.y} ${d.x + 0.2 * t.size},${d.ly} ${d.x + dx * t.size},${d.ly}`);
  lab.append('text').attr('x', d => d.x + (dx + 0.3) * t.size).attr('y', d => d.ly).attr('dy', '0.35em')
    .attr('fill', d => d.color).attr('font-weight', 600).text(d => d.label);
}

/** Stepped colorbar matching contour thresholds. (x, y) is its top right corner. */
export function colorbar({ g, t }, thresholds, color, { x, y, width = 9, label, format = d3.format('~g') }) {
  const n = thresholds.length - 1, cw = (width * t.size) / n, bar = g.append('g').attr('transform', `translate(${x - n * cw},${y})`);
  bar.selectAll('rect').data(d3.range(n)).join('rect').attr('x', i => i * cw).attr('width', cw + 0.3).attr('height', t.size * 0.6)
    .attr('fill', i => color((thresholds[i] + thresholds[i + 1]) / 2));
  const show = [0, Math.floor(n / 2), n];
  bar.selectAll('text').data(show).join('text').attr('x', i => i * cw).attr('y', t.size * 1.75).attr('text-anchor', 'middle')
    .attr('font-size', t.size * 0.9).attr('font-weight', t.weight).attr('fill', t.text).text(i => format(thresholds[i]));
  if (label) bar.append('text').attr('x', -0.5 * t.size).attr('y', t.size * 0.3).attr('dy', '0.35em').attr('text-anchor', 'end').attr('fill', t.ink).attr('font-weight', 600).text(label);
}

/** Horizontal key of dots or lines, inside the SVG so it survives export. items: [{label, color, dashed?}] */
export function inlineKey({ g, t }, items, { x = 0, y = 0, shape = 'dot' } = {}) {
  const key = g.append('g').attr('transform', `translate(${x},${y})`);
  let cx = 0;
  for (const d of items) {
    const item = key.append('g').attr('transform', `translate(${cx},0)`);
    if (shape === 'dot') item.append('circle').attr('cx', t.size * 0.35).attr('r', t.size * 0.35).attr('fill', d.color);
    else item.append('line').attr('x2', t.size * 1.4).attr('stroke', d.color).attr('stroke-width', t.line).attr('stroke-dasharray', d.dashed ? '3,2' : null);
    const text = item.append('text').attr('x', t.size * (shape === 'dot' ? 1 : 1.8)).attr('dy', '0.35em').attr('fill', t.ink).text(d.label);
    cx += text.node().getComputedTextLength() + t.size * (shape === 'dot' ? 2.2 : 3);
  }
  return key;
}

/** Annotation: ring on a point, a curved leader, and haloed text. note: {x, y, text, dx, dy} in data units and ems. */
export function annotate({ g, t }, x, y, note) {
  const px = x(note.x), py = y(note.y), tx = px + (note.dx ?? 2) * t.size, ty = py + (note.dy ?? -2) * t.size;
  const a = g.append('g').attr('pointer-events', 'none');
  a.append('circle').attr('cx', px).attr('cy', py).attr('r', t.dot * 1.9).attr('fill', 'none').attr('stroke', t.ink).attr('stroke-width', t.hair);
  a.append('path').attr('fill', 'none').attr('stroke', t.ink).attr('stroke-width', t.hair)
    .attr('d', `M${px + Math.sign(tx - px) * t.dot * 2},${py}Q${px + (tx - px) * 0.1},${ty} ${tx - Math.sign(tx - px) * 0.3 * t.size},${ty}`);
  a.append('text').attr('x', tx).attr('y', ty).attr('dy', '0.35em').attr('text-anchor', tx >= px ? 'start' : 'end')
    .attr('fill', t.ink).attr('stroke', 'white').attr('stroke-width', t.size * 0.3).attr('paint-order', 'stroke').text(note.text);
}

/** Shaded x-span with a label at its top, for "this region" annotations. */
export function span({ g, ih, t }, x, [x0, x1], text) {
  const s = g.insert('g', ':first-child').attr('pointer-events', 'none');
  s.append('rect').attr('x', x(x0)).attr('width', x(x1) - x(x0)).attr('height', ih).attr('fill', t.ink).attr('fill-opacity', 0.05);
  if (text) s.append('text').attr('x', (x(x0) + x(x1)) / 2).attr('y', t.size * 0.9).attr('text-anchor', 'middle').attr('fill', t.ink).text(text);
}

/** Hover readout for web mode. A small floating card; never exported. */
export function tooltip(container) {
  const tip = d3.select(container).style('position', 'relative').append('div').attr('class', 'tip').attr('hidden', true);
  return {
    show(html, px, py) {
      tip.html(html).attr('hidden', null);
      const { width: tw, height: th } = tip.node().getBoundingClientRect();
      tip.style('left', `${px + 12 + tw > container.clientWidth ? px - 12 - tw : px + 12}px`).style('top', `${Math.max(0, py - th / 2)}px`);
    },
    hide() { tip.attr('hidden', true); },
  };
}

export { exportSVG, exportPDF } from './export.mjs';

/** Gaussian KDE on a grid, used by marginals and rainclouds. */
export function kde(values, grid, bandwidth) {
  const k = 1 / (Math.sqrt(2 * Math.PI) * bandwidth * values.length);
  return grid.map(x => [x, k * d3.sum(values, v => Math.exp(-0.5 * ((x - v) / bandwidth) ** 2))]);
}
export const silverman = v => 1.06 * (d3.deviation(v) || 1) * v.length ** -0.2;
