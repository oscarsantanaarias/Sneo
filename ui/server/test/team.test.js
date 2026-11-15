import { makeProject, writeSettings } from './setup.js';
import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import { startModel } from './mockModel.js';
import { startData } from '../storage.js';
import { runSubAgent } from '../agents.js';
import { chatAgent, maskRoutes, readAgents, readRoutes, runTeam, saveAgents, saveRoutes, slotOf } from '../teams.js';

let model = null;

before(async () => {
  model = await startModel();
  writeSettings({providerBaseUrl: model.url});
  await startData();
  makeProject('team', {'main.js': 'export const add = (a, b) => a + b;\n'});
});

after(async () => {
  await model.close();
});

test('routes keep their keys on the server and only show *** to the page', async () => {
  const saved = await saveRoutes([
    {id: 'local', kind: 'local', name: 'This PC', baseUrl: model.url},
    {id: 'api', kind: 'openai', name: 'Other API', baseUrl: model.url, apiKey: 'secret-key', model: 'some-model'}
  ]);
  assert.equal(saved.find(item => item.id === 'api').apiKey, 'secret-key');
  assert.equal(maskRoutes(saved).find(item => item.id === 'api').apiKey, '***');
  await saveRoutes(maskRoutes(saved));
  assert.equal((await readRoutes()).find(item => item.id === 'api').apiKey, 'secret-key');
  assert.equal((await readRoutes()).find(item => item.id === 'local').apiKey, '');
});

test('a team plans, gives each agent a route, runs them and merges their answers', async () => {
  model.reset([
    {content: '[{"role": "Reader", "goal": "Read main.js"}, {"role": "Tester", "goal": "Say how to test add"}]'},
    {content: 'main.js exports add.'},
    {content: 'Test add(2, 3) === 5.'},
    {content: 'Final: add sums two numbers, test it with add(2, 3).'}
  ]);
  const events = [];
  const result = await runTeam({
    task: 'Explain main.js and how to test it',
    project: 'team',
    mode: 'auto',
    planner: 'local',
    workers: ['local', 'api'],
    lead: 'local',
    maxAgents: 4,
    onEvent: event => events.push(event),
    runWorker: runSubAgent
  });
  const plan = events.find(event => event.type === 'team-plan');
  const sent = events.filter(event => event.type === 'team-message' && event.from === 'planner');
  const back = events.filter(event => event.type === 'team-message' && event.to === 'lead');

  assert.equal(plan.agents.length, 2);
  assert.deepEqual(plan.agents.map(item => item.kind), ['local', 'openai']);
  assert.equal(sent.length, 2);
  assert.equal(back.length, 2);
  assert.ok(result.answer.startsWith('Final:'));
  assert.ok(events.some(event => event.type === 'team-final'));
  const apiRequest = model.state.requests.find(item => item.model === 'some-model');
  assert.ok(apiRequest, 'the api route got its own model');
  assert.equal(apiRequest.sneo_side, undefined);
  assert.equal(apiRequest.thinking_budget, undefined);
});

test('the engines of this pc take turns and the apis answer several at once', async () => {
  assert.deepEqual(slotOf({id: 'a', kind: 'local', baseUrl: 'http://127.0.0.1:8000/v1', parallel: 3}), {key: 'this-pc', most: 1});
  assert.deepEqual(slotOf({id: 'b', kind: 'local', baseUrl: 'http://localhost:8001/v1', parallel: 1}), {key: 'this-pc', most: 1});
  assert.deepEqual(slotOf({id: 'c', kind: 'local', baseUrl: 'http://192.168.1.40:8000/v1', parallel: 1}), {key: 'c', most: 1});
  assert.deepEqual(slotOf({id: 'd', kind: 'claude', baseUrl: '', parallel: 4}), {key: 'd', most: 4});

  model.reset([{content: 'one'}, {content: 'two'}]);
  const route = (await readRoutes()).find(item => item.id === 'local');
  const second = {...route, id: 'local-two', baseUrl: route.baseUrl.replace('127.0.0.1', 'localhost')};
  let open = 0;
  let most = 0;
  model.reset([
    async () => { open++; most = Math.max(most, open); await new Promise(done => setTimeout(done, 80)); open--; return {content: 'one'}; },
    async () => { open++; most = Math.max(most, open); await new Promise(done => setTimeout(done, 80)); open--; return {content: 'two'}; }
  ]);
  const replies = await Promise.all([
    chatAgent({agent: {name: 'A', route: 'local'}, route, history: [{role: 'user', content: 'hi'}], run: {signal: null, usage: null}}),
    chatAgent({agent: {name: 'B', route: 'local-two'}, route: second, history: [{role: 'user', content: 'hi'}], run: {signal: null, usage: null}})
  ]);

  assert.deepEqual(replies.map(item => item.content).sort(), ['one', 'two']);
  assert.equal(most, 1, 'two local engines never write at the same time');
});

test('agents are saved with their names, jobs and routes', async () => {
  const saved = await saveAgents([{id: 'planner', name: 'Planner', focus: 'Plans the work', route: 'local', tools: true}, {name: ''}]);
  assert.equal(saved.length, 2);
  assert.equal(saved[1].name, 'Agent 2');
  assert.equal(saved[1].tools, false);
  assert.deepEqual((await readAgents()).map(item => item.name), ['Planner', 'Agent 2']);
});

test('each agent answers its own conversation on its own route', async () => {
  model.reset([{content: '<think>who am i</think>I write the code.'}]);
  const route = (await readRoutes()).find(item => item.id === 'api');
  const reply = await chatAgent({
    agent: {id: 'coder', name: 'Coder', focus: 'Code generation', route: 'api', tools: false},
    route,
    history: [{role: 'user', content: 'Who are you?'}, {role: 'assistant', content: 'The coder.'}, {role: 'user', content: 'And what do you do?'}],
    peers: ['Reviewer'],
    run: {signal: null, usage: null}
  });
  const request = model.state.requests.at(-1);

  assert.equal(reply.content, 'I write the code.');
  assert.equal(request.model, 'some-model');
  assert.ok(request.messages[0].content.includes('Coder') && request.messages[0].content.includes('Reviewer'));
  assert.deepEqual(request.messages.slice(1).map(item => item.role), ['user', 'assistant', 'user']);
  await assert.rejects(chatAgent({agent: {name: 'Coder'}, route, history: [{role: 'assistant', content: 'hi'}], run: {signal: null, usage: null}}));
});

test('an agent that can read the project uses its tools with the conversation', async () => {
  model.reset([
    {calls: [{name: 'read_file', args: {path: 'main.js'}}]},
    {content: 'main.js adds two numbers.'}
  ]);
  const route = (await readRoutes()).find(item => item.id === 'local');
  const steps = [];
  const reply = await chatAgent({
    agent: {id: 'reader', name: 'Reader', focus: 'Reads code', route: 'local', tools: true},
    route,
    history: [{role: 'user', content: 'Hi'}, {role: 'assistant', content: 'Hello.'}, {role: 'user', content: 'What does main.js do?'}],
    project: 'team',
    run: {signal: null, usage: null},
    onStep: step => steps.push(step.tool),
    runWorker: runSubAgent
  });

  assert.equal(reply.content, 'main.js adds two numbers.');
  assert.deepEqual(steps, ['read_file']);
  assert.deepEqual(reply.tools, ['read_file']);
  assert.ok(model.state.requests[0].messages.some(item => item.role === 'assistant' && item.content === 'Hello.'));
  assert.ok(JSON.stringify(model.state.requests[1].messages).includes('export const add'));
});
