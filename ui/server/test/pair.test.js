import './setup.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { planPair } from '../engine.js';

const gib = 1024 ** 3;
const ram = 95.74 * gib;

test('two IQ3 models on two cards share the fixed ram under half of the ram and keep the first one whole', () => {
  const plan = planPair({firstBytes: 71 * gib, secondBytes: 78 * gib, firstVram: 23, secondVram: 7, totalRam: ram});

  assert.equal(plan.pinTotal, 41);
  assert.ok(plan.pinFirst + plan.pinSecond <= plan.pinTotal);
  assert.ok(plan.ramFirst + plan.ramSecond <= plan.ramTotal);
  assert.ok(plan.pinSecond >= 12, 'the second model keeps at least 12 GB fixed');
  assert.ok(plan.pinFirst >= 26, 'the first model keeps most of what it fixes alone (27.4 GB measured)');
  assert.ok(plan.ramFirst >= plan.pinFirst && plan.ramSecond >= plan.pinSecond);
});

test('two IQ3 models on one card split the card and the ram', () => {
  const plan = planPair({firstBytes: 71 * gib, secondBytes: 78 * gib, firstVram: 15, secondVram: 8, totalRam: ram});

  assert.ok(plan.pinFirst + plan.pinSecond <= plan.pinTotal);
  assert.ok(plan.ramFirst + plan.ramSecond <= plan.ramTotal);
  assert.ok(plan.pinSecond >= 12);
});

test('a small second model only takes what it needs', () => {
  const plan = planPair({firstBytes: 71 * gib, secondBytes: 2.5 * gib, firstVram: 15, secondVram: 8, totalRam: ram});

  assert.equal(plan.pinSecond, 2);
  assert.ok(plan.ramSecond <= 4);
  assert.ok(plan.pinFirst >= 33);
});
