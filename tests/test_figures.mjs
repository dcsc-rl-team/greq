import assert from 'node:assert/strict';
import test from 'node:test';
import { parseRows, quantile, summarize, runtimePoints } from '../scripts/figures/assets/figures.mjs';

const manifest = { step_column: 'step', time_column: 'time', metrics: [{ key: 'loss' }] };
const run = (label, rows, status = 'completed') => ({ label, status, rows: parseRows(rows, manifest) });

test('runtime points use final measured scalar times', () => {
  const runs = [1, 100].map((seconds, i) => ({ label: 'GReQ', environment: i ? 'Graph' : 'Garnet', gamma: .99,
    rows: [{ step: 0, time: 0 }, { step: 10000, time: seconds }] }));
  assert.deepEqual(runtimePoints(runs, 'time').map(p => p.seconds), [1, 100]);
  for (const seconds of [0, -1, NaN, Infinity]) {
    assert.throws(() => runtimePoints([{ ...runs[0], rows: [{ time: seconds }] }], 'time'));
  }
});

test('runtime medians aggregate independent seed times within each cell', () => {
  const run = { environment: 'Graph', gamma: .95, label: 'GReQ' };
  const runs = [100, 1, 3].map((time, seed) => ({ ...run, seed, rows: [{ time: 0 }, { time }] }));
  const points = runtimePoints([...runs, { ...run, gamma: .99, rows: [{ time: 7 }] }], 'time');
  assert.equal(points[0].seconds, 3);
  assert.equal(points[0].count, 3);
  assert.equal(points[1].seconds, 7);
  assert.equal(points[1].count, 1);
});

test('empirical quantiles use linear interpolation', () => {
  assert.equal(quantile([3, 1, 2, 4], 0.5), 2.5);
  assert.equal(quantile([3, 1, 2, 4], 0.25), 1.75);
  assert.equal(quantile([2], 0.75), 2);
});

test('the median is invariant to percentile bounds', () => {
  const runs = [1, 3, 9].map(loss => run('GReQ', [{ step: 0, time: 0, loss }]));
  assert.deepEqual(summarize(runs, 'loss', 'step', [0, 1])[0].points[0], { x: 0, y: 3, lo: 1, hi: 9 });
  assert.deepEqual(summarize(runs, 'loss', 'step', [0.5, 0.5])[0].points[0], { x: 0, y: 3, lo: 3, hi: 3 });
  const point = summarize(runs, 'loss', 'step', [0.1, 0.9])[0].points[0];
  assert.equal(point.y, 3);
  assert.equal(point.lo, 1.4);
  assert.ok(Math.abs(point.hi - 7.8) < 1e-12);
  for (const bounds of [[-.1, 1], [.6, .8], [.1, .4], [0, 1.1], [NaN, .9]]) {
    assert.throws(() => summarize(runs, 'loss', 'step', bounds));
  }
  assert.throws(() => quantile([1, 2], NaN));
});

test('GReQ owns the primary palette slot independently of input order', () => {
  const groups = summarize(['Zap', 'QL', 'GReQ', 'Zap-batched'].map(name => run(name, [{ step: 0, time: 0, loss: 1 }])), 'loss', 'step');
  assert.deepEqual(Object.fromEntries(groups.map(g => [g.key, g.slot])), { Zap: 2, QL: 0, GReQ: 1, 'Zap-batched': 3 });
});

test('NeSA and archived SQL keep stable slots without changing existing colors', () => {
  const names = ['NeSA', 'SQL', 'Zap', 'QL', 'GReQ', 'Zap-batched'];
  const groups = summarize(names.map(name => run(name, [{ step: 0, time: 0, loss: 1 }])), 'loss', 'step');
  assert.deepEqual(Object.fromEntries(groups.map(g => [g.key, g.slot])), { NeSA: 5, SQL: 4, Zap: 2, QL: 0, GReQ: 1, 'Zap-batched': 3 });
});

test('single CSV has a degenerate percentile band', () => {
  const groups = summarize([run('A', [{ step: 0, time: 0, loss: 2 }, { step: 10, time: 1, loss: 1 }])], 'loss', 'step');
  assert.equal(groups.length, 1);
  assert.equal(groups[0].count, 1);
  assert.equal(groups[0].points[1].y, 1);
  assert.equal(groups[0].points[1].lo, 1);
  assert.equal(groups[0].points[1].hi, 1);
});

test('time curves align at equal time on shared support', () => {
  const runs = [
    run('A', [{ step: 0, time: 0, loss: 0 }, { step: 10, time: 10, loss: 10 }]),
    run('A', [{ step: 0, time: 0, loss: 0 }, { step: 10, time: 20, loss: 40 }]),
  ];
  const group = summarize(runs, 'loss', 'time')[0];
  assert.deepEqual(group.points.at(-1), { x: 10, y: 15, lo: 12.5, hi: 17.5 });
  assert.equal(group.traces[1].at(-1).x, 20);
  assert.equal(summarize(runs, 'loss', 'step')[0].points.at(-1).y, 25);
});

test('repeated timestamps use their last recorded value', () => {
  const group = summarize([run('A', [{ step: 0, time: 0, loss: 1 }, { step: 1, time: 0, loss: 2 }])], 'loss', 'time')[0];
  assert.deepEqual(group.points, [{ x: 0, y: 2, lo: 2, hi: 2 }]);
});

test('partial runs remain present and labeled in metadata', () => {
  const group = summarize([run('A', [{ step: 0, time: 0, loss: 1 }], 'failed')], 'loss', 'step')[0];
  assert.equal(group.partial, true);
  assert.equal(group.count, 1);
});

test('invalid numeric values and ordering are rejected', () => {
  for (const loss of ['', null, 'NaN', 'Infinity']) {
    assert.throws(() => parseRows([{ step: 0, time: 0, loss }], manifest));
  }
  assert.throws(() => parseRows([{ step: 0, time: 0, loss: 1 }, { step: 0, time: 1, loss: 2 }], manifest));
  assert.throws(() => parseRows([{ step: 0, time: 2, loss: 1 }, { step: 1, time: 1, loss: 2 }], manifest));
});

test('unrelated CSV groups are not merged', () => {
  const groups = summarize([
    run('A', [{ step: 0, time: 0, loss: 1 }]), run('B', [{ step: 0, time: 0, loss: 7 }]),
  ], 'loss', 'step');
  assert.deepEqual(groups.map(g => g.points[0].y), [1, 7]);
  assert.deepEqual(groups.map(g => g.slot), [0, 1]);
});
