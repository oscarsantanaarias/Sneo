import './setup.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { RunQueue } from '../runs.js';

test('runs go one at a time in arrival order and see their position', async () => {
  const queue = new RunQueue(8);
  const order = [];
  const positions = {b: [], c: []};

  const leaveA = await queue.enter(null);
  const waitB = queue.enter(null, position => positions.b.push(position)).then(leave => { order.push('b'); return leave; });
  const waitC = queue.enter(null, position => positions.c.push(position)).then(leave => { order.push('c'); return leave; });

  assert.deepEqual(positions, {b: [1], c: [2]});

  leaveA();
  const leaveB = await waitB;

  assert.deepEqual(order, ['b']);
  assert.deepEqual(positions.c, [2, 1]);

  leaveB();
  leaveB();
  const leaveC = await waitC;

  assert.deepEqual(order, ['b', 'c']);
  leaveC();
  assert.equal(queue.busy, false);
});

test('a full queue answers 429', async () => {
  const queue = new RunQueue(1);
  const leave = await queue.enter(null);
  const waiting = queue.enter(null);

  assert.equal(queue.full, true);
  await assert.rejects(queue.enter(null), error => error.status === 429 && /busy/.test(error.message));

  leave();
  (await waiting)();
  assert.equal(queue.full, false);
});

test('a queued run can be cancelled and the others move up', async () => {
  const queue = new RunQueue(8);
  const leave = await queue.enter(null);
  const stopB = new AbortController();
  const positionsC = [];
  const waitB = queue.enter(stopB.signal);
  const waitC = queue.enter(null, position => positionsC.push(position));

  stopB.abort();
  await assert.rejects(waitB, error => error.name === 'AbortError');
  assert.equal(queue.waiting.length, 1);
  assert.deepEqual(positionsC, [2, 1]);

  leave();
  (await waitC)();
  assert.equal(queue.busy, false);
});

test('an already cancelled request never enters', async () => {
  const queue = new RunQueue(8);
  const stop = new AbortController();
  stop.abort();

  await assert.rejects(queue.enter(stop.signal), error => error.name === 'AbortError');
  assert.equal(queue.busy, false);
});
