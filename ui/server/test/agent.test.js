import { makeProject, projectsDir, testRoot, writeSettings } from './setup.js';
import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { startModel } from './mockModel.js';
import { startData } from '../storage.js';
import { getAgentAnswer } from '../agents.js';
import { answerApproval } from '../runs.js';
import { summaryHeader } from '../fit.js';

let model = null;

const pagePiece = [
  'Here is your page:',
  '```html',
  '<!doctype html>',
  '<html><head><link rel="stylesheet" href="style.css"></head>',
  '<body><h1>Sweet Bakery</h1><p>Fresh bread every day.</p></body></html>',
  '```',
  'And the styles:',
  '```css',
  'body { font-family: sans-serif; background: #fdf6ec; color: #3b2a1a; }',
  'h1 { color: #b5651d; }',
  '```'
].join('\n');

const ask = async (message, mode, extra = {}) => {
  const events = [];
  const result = await getAgentAnswer({
    message,
    history: [],
    project: 'site',
    ...extra,
    mode,
    onEvent: (event) => {
      events.push(event);
      extra.onEvent?.(event);
    }
  });

  return {result, events};
};

const pendingItems = () => JSON.parse(fs.readFileSync(path.join(testRoot, 'data', 'pending-actions.json'), 'utf8')).items;

before(async () => {
  model = await startModel();
  writeSettings({providerBaseUrl: model.url});
  await startData();
  makeProject('site', {'notes.txt': 'hello from the notes'});
});

after(async () => {
  await model.close();
});

test('every native tool call gets its own result, paired by id and in order', async () => {
  model.reset([
    {calls: [{id: 'call_alpha', name: 'read_file', args: {file: 'notes.txt'}}, {id: 'call_beta', name: 'list_files', args: {}}]},
    {content: 'The notes say hello and the project has one file.'}
  ]);

  const {result} = await ask('Look at the project and summarize it', 'auto');
  const second = model.state.requests[1].messages;
  const at = second.findIndex(item => item.role === 'assistant' && item.tool_calls?.length === 2);

  assert.equal(model.state.requests.length, 2);
  assert.ok(at > 0);
  assert.deepEqual(second[at].tool_calls.map(item => item.id), ['call_alpha', 'call_beta']);
  assert.equal(second[at + 1].role, 'tool');
  assert.equal(second[at + 1].tool_call_id, 'call_alpha');
  assert.ok(second[at + 1].content.includes('hello from the notes'));
  assert.equal(second[at + 2].role, 'tool');
  assert.equal(second[at + 2].tool_call_id, 'call_beta');
  assert.ok(result.answer.includes('The notes say hello'));
  assert.equal(result.mode, 'auto');
});

test('plan mode offers no writing tools and refuses a write anyway', async () => {
  model.reset([
    {calls: [{id: 'w1', name: 'write_file', args: {file: 'landing.html', content: '<h1>x</h1>'}}]},
    {content: 'Plan:\n1. Create landing.html with a hero.\n2. Add style.css with the colors.\n3. Open it with visual_check.'}
  ]);

  const {result} = await ask('Make a landing page for this project', 'plan');
  const offered = model.state.requests[0].tools.map(item => item.function.name);
  const toolMessage = model.state.requests[1].messages.find(item => item.role === 'tool');

  assert.ok(!offered.includes('write_file'));
  assert.ok(!offered.includes('delete_path'));
  assert.ok(!offered.includes('terminal_start'));
  assert.ok(offered.includes('read_file'));
  assert.ok(offered.includes('run_command'));
  assert.ok(toolMessage.content.includes('Plan mode is read-only'));
  assert.equal(fs.existsSync(path.join(projectsDir, 'site', 'landing.html')), false);
  assert.ok(result.answer.includes('1. Create landing.html'));
  assert.ok(model.state.requests[0].messages.at(-1).content.includes('PERMISSION MODE: Plan'));
});

test('manual mode waits for approval on changes but not on read-only commands', async () => {
  const approvals = [];

  model.reset([
    {calls: [{id: 'm1', name: 'write_file', args: {file: 'footer.html', content: '<footer>made by hand</footer>'}}]},
    {calls: [{id: 'm2', name: 'run_command', args: {command: 'dir'}}]},
    {calls: [{id: 'm3', name: 'run_command', args: {command: 'npm install left-pad'}}]},
    {content: 'I added footer.html and listed the folder; the install was not allowed.'}
  ]);

  const {result} = await ask('Add a footer file to this project', 'manual', {
    onEvent: (event) => {
      if(event.type !== 'approval'){
        return;
      }

      approvals.push(event.item);
      setTimeout(() => answerApproval(event.item.id, event.item.type === 'patch' ? 'approved' : 'denied'), 20);
    }
  });

  const messages = model.state.requests[3].messages;
  const denied = messages.find(item => item.tool_call_id === 'm3');
  const listed = messages.find(item => item.tool_call_id === 'm2');

  assert.equal(approvals.length, 2);
  assert.equal(approvals[0].type, 'patch');
  assert.equal(approvals[1].type, 'command');
  assert.equal(approvals[1].title, 'Run npm install left-pad');
  assert.equal(fs.readFileSync(path.join(projectsDir, 'site', 'footer.html'), 'utf8'), '<footer>made by hand</footer>');
  assert.ok(listed.content.includes('notes.txt'));
  assert.ok(denied.content.includes('denied'));
  assert.ok(pendingItems().some(item => item.type === 'patch' && item.files?.[0]?.file === 'footer.html'));
  assert.ok(result.answer.length > 0);
});

test('a page pasted in the chat ends as real files in auto mode', async () => {
  makeProject('bakery');
  model.reset([{content: pagePiece}]);

  const {result, events} = await ask('In this project create a page for my bakery', 'auto', {project: 'bakery'});
  const folder = path.join(projectsDir, 'bakery');

  assert.equal(model.state.requests.length, 3);
  assert.equal(model.state.requests[1].messages[0].content, model.state.requests[0].messages[0].content);
  assert.equal(model.state.requests[2].messages[0].content, model.state.requests[0].messages[0].content);
  assert.ok(model.state.requests[1].messages.at(-1).content.includes('did not write any file'));
  assert.ok(fs.readFileSync(path.join(folder, 'index.html'), 'utf8').includes('Sweet Bakery'));
  assert.ok(fs.readFileSync(path.join(folder, 'style.css'), 'utf8').includes('#fdf6ec'));
  assert.ok(result.answer.includes('Files written to "bakery"'));
  assert.ok(result.answer.includes('`index.html`'));
  assert.ok(result.answer.includes('`style.css`'));
  assert.ok(!result.answer.includes('<h1>Sweet Bakery</h1>'));
  assert.ok(events.some(event => event.type === 'files'));
});

test('the same reply in manual mode queues the files for approval', async () => {
  makeProject('bakery-manual');
  model.reset([{content: pagePiece}]);

  const {result} = await ask('In this project create a page for my bakery', 'manual', {project: 'bakery-manual'});
  const folder = path.join(projectsDir, 'bakery-manual');
  const queued = pendingItems().filter(item => item.project === 'bakery-manual' && item.status === 'pending');

  assert.equal(fs.existsSync(path.join(folder, 'index.html')), false);
  assert.equal(fs.existsSync(path.join(folder, 'style.css')), false);
  assert.deepEqual(queued.map(item => item.files[0].file).sort(), ['index.html', 'style.css']);
  assert.ok(queued[0].files[0].newContent.length > 20);
  assert.ok(result.answer.includes('waiting for your approval'));
});

test('the same reply in plan mode writes nothing and says so', async () => {
  makeProject('bakery-plan');
  model.reset([{content: pagePiece}]);

  const {result} = await ask('In this project create a page for my bakery', 'plan', {project: 'bakery-plan'});

  assert.equal(model.state.requests.length, 1);
  assert.deepEqual(fs.readdirSync(path.join(projectsDir, 'bakery-plan')), []);
  assert.ok(result.answer.includes('Plan mode is read-only, so nothing was written'));
  assert.ok(result.answer.includes('`index.html`'));
});

test('a plain question with code in the answer writes nothing', async () => {
  const before = fs.readdirSync(path.join(projectsDir, 'site')).sort();
  model.reset([{content: 'A grid splits the page into rows and columns:\n```css\n.layout { display: grid; grid-template-columns: 1fr 3fr; gap: 16px; }\n```'}]);

  const {result} = await ask('How does a CSS grid layout work?', 'auto');

  assert.equal(model.state.requests.length, 1);
  assert.deepEqual(fs.readdirSync(path.join(projectsDir, 'site')).sort(), before);
  assert.ok(result.answer.includes('display: grid'));
});

test('searches that find nothing stop the run with a plain sentence', async () => {
  model.reset([(body, index) => body.tools
    ? {calls: [{name: 'grep', args: {query: `missing-${index}`}}]}
    : {content: ''}]);

  const {result} = await ask('Find where the payment gateway is configured', 'auto');
  const greps = result.tools.filter(item => item.tool === 'grep');
  const redirected = model.state.requests.some(request => request.messages.some(item => item.role === 'tool' && item.content.includes('[SNEO]')));

  assert.equal(greps.length, 4);
  assert.ok(redirected);
  assert.ok(result.answer.includes('searches found nothing'));
  assert.equal(result.guard, 'searches-failed');
});

test('an empty reply never ends the run silently', async () => {
  model.reset([{content: ''}]);

  const {result} = await ask('hello there', 'auto');

  assert.ok(result.answer.trim().length > 20);
});

test('a window smaller than the system prompt still gets an answer', async () => {
  writeSettings({providerBaseUrl: model.url, numCtx: 8192});
  model.reset([{content: 'Small window answer.'}]);

  const {result, events} = await ask('hello there, small window', 'auto');

  assert.equal(model.state.requests.length, 1);
  assert.ok(result.answer.includes('Small window answer.'));
  assert.ok(events.some(event => event.type === 'tool' && /do not fit the context window/.test(event.tool)));
  writeSettings({providerBaseUrl: model.url});
});

test('a long history is summarized once and the summary is reused next turn', async () => {
  writeSettings({providerBaseUrl: model.url, numCtx: 32768});
  model.reset([(body) => body.messages[0].content.startsWith('You condense')
    ? {content: 'Earlier the user and the agent discussed eight big files.'}
    : {content: 'Here is the answer after the summary.'}]);

  const history = [];

  for(let i = 0; i < 10; i++){
    history.push({role: 'user', content: `question ${i} `.repeat(1200)});
    history.push({role: 'assistant', content: `answer ${i} `.repeat(800)});
  }

  const first = await ask('What did we decide about the cache?', 'auto', {history});
  const summarizer = model.state.requests[0];
  const main = model.state.requests.find(request => !request.messages[0].content.startsWith('You condense'));

  assert.ok(summarizer.messages[0].content.startsWith('You condense'));
  assert.equal(summarizer.tools, undefined);
  assert.equal(model.state.requests.at(-1), main);
  assert.ok(main.messages[1].content.startsWith(summaryHeader));
  assert.ok(first.result.memo.count > 0);
  assert.ok(first.result.memo.text.includes('eight big files'));

  model.reset([{content: 'Plain answer.'}]);
  await ask('What did we decide about the cache?', 'auto', {history: []});
  const freshSystem = model.state.requests[0].messages[0].content;

  assert.equal(main.messages[0].content, freshSystem);

  model.reset([{content: 'Answer that reuses the summary.'}]);
  await ask('And what about the build?', 'auto', {history, memo: first.result.memo, ratio: first.result.ratio});

  assert.equal(model.state.requests.length, 1);
  assert.equal(model.state.requests[0].messages[0].content, freshSystem);
  assert.ok(model.state.requests[0].messages[1].content.startsWith(summaryHeader));
  writeSettings({providerBaseUrl: model.url});
});
