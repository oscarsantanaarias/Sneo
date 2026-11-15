import { fixturesDir, isAlive, makeProject, testRoot, waitFor, writeSettings } from './setup.js';
import test, { before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { startData } from '../storage.js';
import { runTool } from '../tools.js';
import { answerApproval, runStore, stopChildren } from '../runs.js';

const spawner = path.join(fixturesDir, 'spawner.cjs');

const newRun = (mode, onEvent = null) => ({id: `run-${Math.random()}`, chatId: '', mode, signal: null, onEvent, children: new Set()});

before(async () => {
  writeSettings();
  await startData();
  makeProject('proc');
});

test('stop kills a terminal started by the run together with its children', async () => {
  const run = newRun('auto');
  const pidFile = path.join(testRoot, 'terminal.pid');

  const session = await runStore.run(run, () => runTool('terminal_start', {project: 'proc', command: `node "${spawner}" "${pidFile}"`}));

  assert.ok(await waitFor(() => fs.existsSync(pidFile) && fs.readFileSync(pidFile, 'utf8').length > 0));

  const grandchild = Number(fs.readFileSync(pidFile, 'utf8'));

  assert.ok(isAlive(session.pid));
  assert.ok(isAlive(grandchild));
  assert.equal(run.children.size, 1);

  await stopChildren(run);

  assert.ok(await waitFor(() => !isAlive(grandchild) && !isAlive(session.pid)), 'the process tree is still running');
});

test('stop kills a running command and its children', async () => {
  const run = newRun('auto');
  const pidFile = path.join(testRoot, 'command.pid');
  const running = runStore.run(run, () => runTool('run_command', {project: 'proc', command: `node "${spawner}" "${pidFile}"`}));
  const outcome = running.then(() => 'finished', () => 'failed');

  assert.ok(await waitFor(() => fs.existsSync(pidFile) && fs.readFileSync(pidFile, 'utf8').length > 0));

  const grandchild = Number(fs.readFileSync(pidFile, 'utf8'));

  assert.ok(isAlive(grandchild));
  await stopChildren(run);

  assert.ok(await waitFor(() => !isAlive(grandchild)), 'the grandchild survived');
  assert.equal(await outcome, 'failed');
});

test('terminals and sandboxes go through the same gate as commands', async () => {
  const asked = [];
  const manual = newRun('manual', (event) => {
    if(event.type === 'approval'){
      asked.push(event.item);
      setTimeout(() => answerApproval(event.item.id, 'denied'), 10);
    }
  });

  await assert.rejects(runStore.run(manual, () => runTool('terminal_start', {project: 'proc', command: 'npm run dev'})), /denied/);
  await assert.rejects(runStore.run(manual, () => runTool('sandbox_start', {project: 'proc', command: 'node server.js'})), /denied/);
  assert.deepEqual(asked.map(item => item.type), ['command', 'command']);
  assert.equal(manual.children.size, 0);

  const plan = newRun('plan');
  await assert.rejects(runStore.run(plan, () => runTool('terminal_start', {project: 'proc', command: 'npm run dev'})), /Plan mode is read-only/);
  await assert.rejects(runStore.run(plan, () => runTool('sandbox_run', {project: 'proc', command: 'npm install'})), /Plan mode is read-only/);

  const looked = await runStore.run(plan, () => runTool('run_command', {project: 'proc', command: 'dir'}));
  assert.ok(typeof looked.stdout === 'string');
});

test('a stop while waiting for approval ends the wait and marks it cancelled', async () => {
  const stop = new AbortController();
  const run = {...newRun('manual'), signal: stop.signal};
  let pendingId = '';
  run.onEvent = (event) => {
    if(event.type === 'approval'){
      pendingId = event.item.id;
      stop.abort();
    }
  };

  await assert.rejects(runStore.run(run, () => runTool('delete_path', {project: 'proc', target: 'x'})), error => error.name === 'AbortError');

  const items = JSON.parse(fs.readFileSync(path.join(testRoot, 'data', 'pending-actions.json'), 'utf8')).items;
  assert.ok(await waitFor(() => JSON.parse(fs.readFileSync(path.join(testRoot, 'data', 'pending-actions.json'), 'utf8')).items.find(item => item.id === pendingId)?.status === 'cancelled'));
  assert.ok(items.some(item => item.id === pendingId));
});
