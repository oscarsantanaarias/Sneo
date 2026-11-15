import { makeProject, projectsDir, writeSettings } from './setup.js';
import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { startModel } from './mockModel.js';
import { startData } from '../storage.js';
import { getAgentAnswer } from '../agents.js';
import { patchPlan, resolveCall } from '../toolAliases.js';
import { swapText } from '../tools.js';

let model = null;

const page = [
  '<!doctype html>',
  '<html>',
  '<body>',
  '  <h1>Old title</h1>',
  '  <p>First line</p>',
  '  <p>Second line</p>',
  '</body>',
  '</html>',
  ''
].join('\n');

const ask = async (message) => await getAgentAnswer({message, history: [], project: 'game', mode: 'auto', onEvent: () => {}});
const pageNow = () => fs.readFileSync(path.join(projectsDir, 'game', 'index.html'), 'utf8');
const resetPage = () => fs.writeFileSync(path.join(projectsDir, 'game', 'index.html'), page);

before(async () => {
  model = await startModel();
  writeSettings({providerBaseUrl: model.url});
  await startData();
  makeProject('game', {'index.html': page, 'win.txt': 'alpha\r\nbeta\r\ngamma\r\n'});
});

after(async () => {
  await model.close();
});

test('names from other agents map to the tools here', () => {
  assert.deepEqual(resolveCall('edit', {file_path: 'a.js', old_string: 'x', new_string: 'y'}), {name: 'replace_in_file', args: {file: 'a.js', search: 'x', replace: 'y', all: false}});
  assert.deepEqual(resolveCall('edit_file', {mode: 'overwrite', content: 'new'}, 'index.html'), {name: 'write_file', args: {file: 'index.html', content: 'new'}});
  assert.deepEqual(resolveCall('edit_file', {context: 'old', replacement: 'new'}, 'index.html'), {name: 'replace_in_file', args: {file: 'index.html', search: 'old', replace: 'new', all: false}});
  assert.deepEqual(resolveCall('write', {file_path: 'b.txt', content: 'hi'}), {name: 'write_file', args: {file: 'b.txt', content: 'hi'}});
  assert.deepEqual(resolveCall('generate_file', {path: 'c.txt', text: 'hi'}), {name: 'write_file', args: {file: 'c.txt', content: 'hi'}});
  assert.deepEqual(resolveCall('shell', {command: ['npm', 'test'], cwd: '.'}), {name: 'run_command', args: {command: 'npm test'}});
  assert.deepEqual(resolveCall('grep_files', {pattern: 'scale', glob: '*.html', i: true}), {name: 'grep', args: {query: 'scale', pattern: '*.html'}});
  assert.deepEqual(resolveCall('read', {file_path: 'a.js', offset: 10, limit: 20}), {name: 'read_range', args: {file: 'a.js', start: 10, end: 29}});
  assert.deepEqual(resolveCall('read_file', {file: 'a.js'}), {name: 'read_file', args: {file: 'a.js'}});
  assert.deepEqual(resolveCall('review_console_logs', {limit: 5}, 'index.html').name, 'visual_check');
  assert.equal(resolveCall('teleport', {}).unknown, true);
  assert.ok(resolveCall('edit', {old_string: 'x', new_string: 'y'}).error);
});

test('a patch with two hunks in one file and a new file becomes real changes', () => {
  const plan = patchPlan([
    '*** Begin Patch',
    '*** Update File: index.html',
    '@@',
    '-  <h1>Old title</h1>',
    '+  <h1>New title</h1>',
    '@@',
    '   <p>First line</p>',
    '-  <p>Second line</p>',
    '+  <p>Second line, fixed</p>',
    '*** Add File: notes.md',
    '+# Notes',
    '+done',
    '*** End Patch'
  ].join('\n'));

  assert.equal(plan.changes.length, 2);
  assert.deepEqual(plan.changes[0], {file: 'index.html', search: '  <h1>Old title</h1>', replace: '  <h1>New title</h1>'});
  assert.deepEqual(plan.changes[1], {file: 'index.html', search: '  <p>First line</p>\n  <p>Second line</p>', replace: '  <p>First line</p>\n  <p>Second line, fixed</p>'});
  assert.deepEqual(plan.writes, [{file: 'notes.md', content: '# Notes\ndone'}]);

  const diff = patchPlan(['--- a/index.html', '+++ b/index.html', '@@ -4,1 +4,1 @@', '-  <h1>Old title</h1>', '+  <h1>Diff title</h1>'].join('\n'));
  assert.deepEqual(diff.changes, [{file: 'index.html', search: '  <h1>Old title</h1>', replace: '  <h1>Diff title</h1>'}]);
});

test('replacing text keeps windows line ends, dollar signs and tolerates trailing spaces', () => {
  assert.equal(swapText('alpha\r\nbeta\r\ngamma\r\n', 'alpha\nbeta', 'one\ntwo').content, 'one\r\ntwo\r\ngamma\r\n');
  assert.equal(swapText('price = 1;', 'price = 1;', 'price = `$&${x}$$`;').content, 'price = `$&${x}$$`;');
  assert.equal(swapText('a  \nb\nc', 'a\nb', 'A\nB').content, 'A\nB\nc');
  assert.equal(swapText('x x x', 'x', 'y', true).content, 'y y y');
  assert.equal(swapText('abc', 'zzz', 'y'), null);
});

test('the agent edits through edit and apply_patch calls named like other agents do', async () => {
  resetPage();
  model.reset([
    {calls: [{id: 'r1', name: 'read_file', args: {file: 'index.html'}}]},
    {calls: [{id: 'e1', name: 'edit', args: {file_path: 'index.html', old_string: '<h1>Old title</h1>', new_string: '<h1>Edited title</h1>'}}]},
    {calls: [{id: 'p1', name: 'apply_patch', args: {input: '*** Begin Patch\n*** Update File: index.html\n@@\n-  <p>First line</p>\n+  <p>First line, patched</p>\n@@\n-  <p>Second line</p>\n+  <p>Second line, patched</p>\n*** End Patch'}}]},
    {content: 'Both changes are in.'}
  ]);

  const result = await ask('Fix the title and both lines');
  const text = pageNow();

  assert.ok(text.includes('<h1>Edited title</h1>'));
  assert.ok(text.includes('<p>First line, patched</p>'));
  assert.ok(text.includes('<p>Second line, patched</p>'));
  assert.ok(result.answer.includes('Both changes are in.'));
});

test('an unknown tool answers with the tools that exist', async () => {
  model.reset([
    {calls: [{id: 'u1', name: 'review_network_logs_v2', args: {}}]},
    {content: 'ok'}
  ]);

  await ask('Check the logs');
  const toolMessage = model.state.requests[1].messages.find(item => item.role === 'tool');

  assert.ok(toolMessage.content.includes('There is no tool named'));
  assert.ok(toolMessage.content.includes('replace_in_file {file, search, replace}'));
});

test('prompt and tool writing notices reach the app, edit steps carry their rows and the model never sees them', async () => {
  resetPage();
  const events = [];
  model.reset([
    {
      notices: [
        {choices: [], prefill: {tokens: 900, cached: 600}},
        {choices: [], prefill: {tokens: 900, cached: 600, ms: 150}},
        {choices: [], writing: {tokens: 42, head: '{"name": "replace_in_file", "arguments": {"file": "index.html", "search": "<h1>Old'}},
        {choices: [], speed: {tokens: 45, ms: 500}}
      ],
      calls: [{id: 'n1', name: 'replace_in_file', args: {file: 'index.html', search: '<h1>Old title</h1>', replace: '<h1>Fresh title</h1>'}}]
    },
    {content: 'Changed.'}
  ]);

  const result = await getAgentAnswer({message: 'Change the title', history: [], project: 'game', mode: 'auto', onEvent: event => events.push(event)});
  const prefills = events.filter(event => event.type === 'prefill');
  const writing = events.find(event => event.type === 'writing');
  const edit = result.steps.find(step => step.tool === 'replace_in_file' && step.status === 'done');
  const toolMessage = model.state.requests[1].messages.find(item => item.role === 'tool');

  assert.deepEqual(prefills.map(event => event.ms), [null, 150]);
  assert.equal(prefills[1].tokens - prefills[1].cached, 300);
  assert.deepEqual([writing.tool, writing.file, writing.tokens], ['replace_in_file', 'index.html', 42]);
  const speed = events.find(event => event.type === 'speed');
  assert.deepEqual([speed.tokens, speed.ms, speed.speed], [45, 500, 90]);
  assert.ok(edit.changes[0].rows.some(row => row.kind === 'del' && row.text.includes('Old title')));
  assert.ok(edit.changes[0].rows.some(row => row.kind === 'add' && row.text.includes('Fresh title')));
  assert.equal(edit.added, 1);
  assert.ok(!toolMessage.content.includes('"rows"'));
  assert.ok(!result.tools[0].result.change);
});

test('crlf files accept edits written with plain line ends', async () => {
  model.reset([
    {calls: [{id: 'c1', name: 'replace_in_file', args: {file: 'win.txt', search: 'alpha\nbeta', replace: 'one\ntwo'}}]},
    {content: 'done'}
  ]);

  await ask('Change the first two lines');
  assert.equal(fs.readFileSync(path.join(projectsDir, 'game', 'win.txt'), 'utf8'), 'one\r\ntwo\r\ngamma\r\n');
});
