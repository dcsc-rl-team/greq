import test from 'node:test';
import assert from 'node:assert/strict';
import { parseReport, groupReport } from '../scripts/figures/assets/report.mjs';

function record(seed, step, overrides = {}) {
  return { environment: 'garnet', method: 'greq', execution: 'vmap', n_states: '2', n_actions: '2', branching: '2', env_seed: '0', n_steps: '2', eval_every: '2', gamma: '.9', eta: '1', rho: '.7', k0: '', batch_size: '', seed: String(seed), step: String(step), status: 'completed', error: '', train_seconds: '', value_linf: '1', bellman_linf: '1', policy_return: '1', policy_gap: '1', ...overrides };
}
const curves = () => Array.from({ length: 11 }, (_, i) => [record(i + 3, 0), record(i + 3, 2)]).flat();

test('CSV dashboard preserves metrics when grouping eleven complete trajectories', () => {
  const groups = groupReport(parseReport(curves()), 'evaluation');
  assert.equal(groups.length, 11);
  assert.deepEqual(groups[0].map(r => r.step), [0, 2]);
  assert.equal(groups[0][0].train_seconds, null);
});

test('dashboard rejects failed, incomplete, duplicate, or mixed-setting curves', () => {
  const rows = curves();
  for (const bad of [rows.slice(1), [...rows, rows[0]], rows.map((r, i) => i ? r : { ...r, status: 'failed' }), rows.map((r, i) => i ? r : { ...r, eta: '.9' })]) {
    assert.throws(() => groupReport(parseReport(bad), 'evaluation'));
  }
});

test('dashboard runtime is an individually measured final value per seed', () => {
  const rows = Array.from({ length: 11 }, (_, i) => record(i + 3, 2, { execution: 'scalar', train_seconds: String(.1 + i / 100) }));
  assert.equal(groupReport(parseReport(rows), 'runtime').length, 11);
  assert.throws(() => groupReport(parseReport(rows.map(r => ({ ...r, train_seconds: '' }))), 'runtime'));
  assert.throws(() => groupReport(parseReport(curves().map(r => ({ ...r, train_seconds: '.2' }))), 'evaluation'));
});

test('dashboard rejects unknown algorithms and invalid or missing metric fields', () => {
  assert.throws(() => parseReport([record(3, 0, { method: 'unsupported' })]));
  assert.throws(() => parseReport([record(3, 0, { value_linf: 'NaN' })]));
  assert.throws(() => parseReport([record(3, 0, { value_linf: '' })]));
  assert.throws(() => parseReport([]));
});
