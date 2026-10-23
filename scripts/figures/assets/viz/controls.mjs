/**
 * Controllers for explanatory pages. Each one is small, keyboard accessible, and calls back with plain values.
 *   chips       legend that is also a filter: hover focuses a series, click hides it
 *   stepper     numbered steps that walk a figure through annotated states
 *   scrubber    play/pause plus a slider over an ordered sequence (time, training step)
 *   scrubNumber a number in running text that you drag left/right to change
 *   panel       a control panel template built from a spec (select, range, number, segmented, switches, buttons)
 *   themePanel  the palette selector; fixed-size output is handled by the export buttons
 */
import { PALETTES, setPalette } from './palette.mjs';
import { setMode, rerender } from './fig.mjs';
const d3 = window.d3;

export function chips(el, items, { onToggle, onFocus, shape = 'line' } = {}) {
  const hidden = new Set();
  d3.select(el).classed('chips', true).selectAll('button').data(items).join('button')
    .attr('type', 'button').attr('aria-pressed', 'true').attr('style', d => `--c:${d.color}`)
    .html(d => `<span class="sw ${shape}${d.dashed ? ' dashed' : ''}"></span>${d.label}`)
    .on('click', function (event, d) {
      hidden.has(d.key) ? hidden.delete(d.key) : hidden.add(d.key);
      this.setAttribute('aria-pressed', String(!hidden.has(d.key)));
      onToggle?.(new Set(hidden));
    })
    .on('pointerenter focus', (event, d) => !hidden.has(d.key) && onFocus?.(d.key))
    .on('pointerleave blur', () => onFocus?.(null));
}

/** steps: [{label}] — label is a short caption shown beside the dots. */
export function stepper(el, steps, onStep) {
  let i = 0;
  const root = d3.select(el).classed('stepper', true);
  const prev = root.append('button').attr('type', 'button').attr('aria-label', 'Previous').html('&#8249;');
  const dots = root.append('div').attr('class', 'dots').selectAll('button').data(steps).join('button')
    .attr('type', 'button').attr('aria-label', (d, k) => `Step ${k + 1}`).text((d, k) => k + 1);
  const next = root.append('button').attr('type', 'button').attr('aria-label', 'Next').html('&#8250;');
  const caption = root.append('span').attr('class', 'caption').attr('aria-live', 'polite');
  const go = k => {
    i = Math.max(0, Math.min(steps.length - 1, k));
    dots.attr('aria-current', (d, j) => (j === i ? 'step' : null));
    prev.attr('disabled', i === 0 ? true : null); next.attr('disabled', i === steps.length - 1 ? true : null);
    caption.text(steps[i].label);
    onStep(i, steps[i]);
  };
  prev.on('click', () => go(i - 1)); next.on('click', () => go(i + 1)); dots.on('click', (e, d) => go(steps.indexOf(d)));
  root.on('keydown', e => { if (e.key === 'ArrowRight') go(i + 1); if (e.key === 'ArrowLeft') go(i - 1); });
  go(0);
  return { go };
}

export function scrubber(el, { values, format = v => v, delay = 80, loop = true, label = '', onInput }) {
  const root = d3.select(el).classed('scrubber', true);
  const play = root.append('button').attr('type', 'button').attr('class', 'play').attr('aria-label', 'Play');
  if (label) root.append('span').attr('class', 'label').text(label);
  const range = root.append('input').attr('type', 'range').attr('min', 0).attr('max', values.length - 1).attr('step', 1).attr('value', 0);
  const out = root.append('output');
  let timer = null;
  const icon = playing => play.html(playing
    ? '<svg viewBox="0 0 16 16"><rect x="4" y="3" width="3" height="10" rx="1"/><rect x="9" y="3" width="3" height="10" rx="1"/></svg>'
    : '<svg viewBox="0 0 16 16"><path d="M5 3.2v9.6a.6.6 0 0 0 .9.5l7.6-4.8a.6.6 0 0 0 0-1L5.9 2.7a.6.6 0 0 0-.9.5z"/></svg>').attr('aria-label', playing ? 'Pause' : 'Play');
  const set = i => { range.property('value', i); out.text(format(values[i])); onInput(values[i], i); };
  const stop = () => { clearInterval(timer); timer = null; icon(false); };
  play.on('click', () => {
    if (timer) return stop();
    if (+range.property('value') === values.length - 1) set(0);
    icon(true);
    timer = setInterval(() => {
      const i = +range.property('value') + 1;
      if (i >= values.length) { if (loop) set(0); else stop(); } else set(i);
    }, delay);
  });
  range.on('input', () => { stop(); set(+range.property('value')); });
  icon(false); set(0);
  return { set, stop };
}

export function scrubNumber(el, { value, min, max, step = 1, pxPerStep = 4, format = d3.format(','), onInput }) {
  let v = value;
  const span = d3.select(el).classed('scrub', true).attr('tabindex', 0).attr('role', 'slider')
    .attr('aria-valuemin', min).attr('aria-valuemax', max);
  const set = next => {
    v = Math.max(min, Math.min(max, Math.round(next / step) * step));
    span.text(format(v)).attr('aria-valuenow', v);
    onInput(v);
  };
  span.call(d3.drag().on('start', e => { e.subject.v0 = v; e.subject.x0 = e.x; span.classed('active', true); })
    .on('drag', e => set(e.subject.v0 + ((e.x - e.subject.x0) / pxPerStep) * step))
    .on('end', () => span.classed('active', false)));
  span.on('keydown', e => {
    const k = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (k) { e.preventDefault(); set(v + k * step * (e.shiftKey ? 10 : 1)); }
  });
  set(v);
  return { set };
}

/**
 * Build control markup, event handlers, state, and a reset action from a specification.
 *   panel(el, spec, onChange, { vertical })  →  { state, set(key, value), reset() }
 * spec entries:
 *   { type: 'group', label }                                    section label (vertical panels)
 *   { type: 'select', key, label, options: [v | {value, label}], value }
 *   { type: 'range', key, label, min, max, step, value, format }
 *   { type: 'number', key, label, min, max, step, value }        applied only while valid
 *   { type: 'segmented', key, label, options, value }
 *   { type: 'switches', items: [{ key, label, value, color? }] } color makes the switch a legend entry
 *   { type: 'buttons', items: [{ label, action: 'reset' | (state, panel) => void }] }
 * onChange(state, key) runs after every change; key is 'reset' after Reset.
 */
export function panel(el, spec, onChange, { vertical = false } = {}) {
  const state = {}, defaults = {}, setters = {};
  const root = d3.select(el).classed('panel', true).classed('vertical', vertical);
  const emit = key => onChange({ ...state }, key);
  const opt = o => (typeof o === 'object' ? o : { value: o, label: String(o) });
  const field = (label, cls = '') => root.append('label').attr('class', `field ${cls}`).text(label);
  for (const c of spec) {
    if (c.type === 'group') { root.append('div').attr('class', 'group').text(c.label); continue; }
    if (c.type === 'select') {
      const sel = field(c.label, 'wide').append('select');
      sel.selectAll('option').data(c.options.map(opt)).join('option').attr('value', o => o.value).text(o => o.label);
      const cast = v => (typeof opt(c.options[0]).value === 'number' ? +v : v);
      setters[c.key] = v => sel.property('value', v);
      sel.on('change', () => { state[c.key] = cast(sel.property('value')); emit(c.key); });
    } else if (c.type === 'range') {
      const f = c.format ?? (v => v), lab = field(c.label, 'range'), out = lab.append('output');
      const input = lab.append('input').attr('type', 'range').attr('min', c.min).attr('max', c.max).attr('step', c.step ?? 'any');
      setters[c.key] = v => { input.property('value', v); out.text(f(v)); };
      input.on('input', () => { state[c.key] = +input.property('value'); out.text(f(state[c.key])); emit(c.key); });
    } else if (c.type === 'number') {
      const input = field(c.label).append('input').attr('type', 'number').attr('required', true)
        .attr('min', c.min).attr('max', c.max).attr('step', c.step ?? 1);
      setters[c.key] = v => input.property('value', v);
      input.on('input', () => { if (input.node().validity.valid) { state[c.key] = input.property('valueAsNumber'); emit(c.key); } });
    } else if (c.type === 'segmented') {
      const wrap = root.append('div').attr('class', 'field').text(c.label);
      // A div supports grid layout consistently across browsers, including Safari.
      const fs = wrap.append('div').attr('class', 'segmented').attr('role', 'radiogroup').attr('aria-label', c.label);
      const name = `seg-${c.key}-${Math.random().toString(36).slice(2, 7)}`;
      const radios = fs.selectAll('label').data(c.options.map(opt)).join('label').text(o => o.label)
        .insert('input', function () { return this.firstChild; }).attr('type', 'radio').attr('name', name).attr('value', o => o.value);
      setters[c.key] = v => radios.property('checked', o => o.value === v);
      radios.on('change', (e, o) => { state[c.key] = o.value; emit(c.key); });
    } else if (c.type === 'switches') {
      const fs = root.append('fieldset').attr('class', 'choices');
      for (const it of c.items) {
        const lab = fs.append('label').attr('style', it.color ? `--c:${it.color}` : null).text(it.label);
        const input = lab.insert('input', function () { return this.firstChild; }).attr('type', 'checkbox').attr('role', 'switch');
        setters[it.key] = v => input.property('checked', v);
        input.on('change', () => { state[it.key] = input.property('checked'); emit(it.key); });
        state[it.key] = defaults[it.key] = it.value ?? true;
      }
      continue;
    } else if (c.type === 'buttons') {
      const row = root.append('div').attr('class', 'buttons');
      for (const b of c.items) row.append('button').attr('type', 'button').text(b.label)
        .on('click', () => (b.action === 'reset' ? api.reset() : b.action(state, api)));
      continue;
    }
    state[c.key] = defaults[c.key] = c.value;
  }
  for (const [k, v] of Object.entries(state)) setters[k]?.(v);
  const api = {
    state,
    set(key, value) { state[key] = value; setters[key]?.(value); emit(key); },
    reset() { Object.assign(state, defaults); for (const [k, v] of Object.entries(state)) setters[k]?.(v); emit('reset'); },
  };
  return api;
}

/** Palette selector with URL parameters for palette and rendering mode. */
export function themePanel(el, { palette = 'focus', onPalette = () => {} } = {}) {
  const q = new URLSearchParams(location.search);
  const apply = s => { setPalette(s.palette); onPalette(s.palette); rerender(); };
  const names = ['focus', ...Object.keys(PALETTES).filter(name => name !== 'focus')];
  setPalette(palette);
  const api = panel(el, [
    { type: 'segmented', key: 'palette', label: 'Palette', options: names.map(value => ({ value, label: PALETTES[value].label })), value: palette },
  ], apply);
  apply(api.state);
  if (q.get('palette') in PALETTES) api.set('palette', q.get('palette'));
  if (['web', 'paper'].includes(q.get('mode'))) setMode(q.get('mode'));
  return api;
}
