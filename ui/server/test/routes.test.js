import { makeProject, waitFor, writeSettings } from './setup.js';
import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import http from 'http';
import express from 'express';
import { startModel } from './mockModel.js';
import { readSettings, startData } from '../storage.js';
import router from '../routes.js';
import { queueFor } from '../runs.js';

let model = null;
let server = null;
let base = '';

const openStream = (body) => {
  const stop = new AbortController();
  const events = [];
  const waiters = [];
  const response = fetch(`${base}/api/chat/stream`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({project: 'site', accessMode: 'auto', history: [], ...body}),
    signal: stop.signal
  });

  const reading = response.then(async (res) => {
    if(!res.ok){
      return {status: res.status, body: await res.json()};
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      for await (const chunk of res.body){
        buffer += decoder.decode(chunk, {stream: true});
        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for(const part of parts){
          const line = part.split('\n').find(item => item.startsWith('data: '));

          if(line){
            events.push(JSON.parse(line.slice(6)));
            waiters.forEach(check => check());
          }
        }
      }
    } catch(e) {
      return {status: res.status, aborted: true};
    }

    return {status: res.status};
  }).catch(() => ({aborted: true}));

  return {
    events,
    reading,
    stop: () => stop.abort(),
    until: (type) => new Promise(resolve => {
      const check = () => {
        const found = events.find(item => item.type === type);

        if(found){
          resolve(found);
        }
      };

      waiters.push(check);
      check();
    })
  };
};

before(async () => {
  model = await startModel();
  writeSettings({providerBaseUrl: model.url});
  await startData();
  makeProject('site', {'notes.txt': 'hello'});

  const app = express();
  app.use(express.json({limit: '5mb'}));
  app.use('/api', router);
  server = http.createServer(app);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  server.closeAllConnections?.();
  await new Promise(resolve => server.close(resolve));
  await model.close();
});

test('settings accept the new modes, migrate old ones and keep allowNetwork', async () => {
  const put = async (body) => {
    const res = await fetch(`${base}/api/settings`, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    return {status: res.status, body: await res.json()};
  };

  assert.equal((await put({accessMode: 'plan', allowNetwork: true})).body.settings.accessMode, 'plan');
  assert.equal((await readSettings()).allowNetwork, true);
  assert.equal((await put({accessMode: 'full'})).body.settings.accessMode, 'auto');
  assert.equal((await put({accessMode: 'ask'})).body.settings.accessMode, 'manual');
  assert.equal((await readSettings()).allowNetwork, true);
  assert.equal((await put({accessMode: 'yolo'})).status, 400);
  assert.equal((await put({allowNetwork: 'yes'})).status, 400);
  assert.equal((await put({allowNetwork: false})).body.settings.allowNetwork, false);

  writeSettings({providerBaseUrl: model.url});
});

test('one run at a time: the next waits with its position, the queue cap answers 429', async () => {
  let release = null;
  const hold = new Promise(resolve => {
    release = resolve;
  });

  model.reset([(body, index) => index === 0 ? {content: 'first run finished', wait: hold} : {content: 'second run finished'}]);

  const first = openStream({chatId: 'queue-a', message: 'hello one'});
  assert.ok(await waitFor(() => model.state.requests.length === 1));

  const second = openStream({chatId: 'queue-b', message: 'hello two'});
  const queued = await second.until('queued');
  assert.equal(queued.position, 1);

  const third = await fetch(`${base}/api/chat/stream`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({chatId: 'queue-c', message: 'hello three', project: 'site'})
  });
  const refused = await third.json();

  assert.equal(third.status, 429);
  assert.equal(third.headers.get('retry-after'), '30');
  assert.ok(/busy/.test(refused.message));

  release();
  const firstDone = await first.until('done');
  assert.ok(firstDone.message.includes('first run finished'));

  const secondDone = await second.until('done');
  assert.ok(secondDone.message.includes('second run finished'));
  assert.equal(model.state.requests.length, 2);
});

test('a queued run can be stopped before it starts', async () => {
  let release = null;
  const hold = new Promise(resolve => {
    release = resolve;
  });

  model.reset([(body, index) => index === 0 ? {content: 'holder done', wait: hold} : {content: 'should not run'}]);

  const holder = openStream({chatId: 'stop-a', message: 'hold the engine'});
  assert.ok(await waitFor(() => model.state.requests.length === 1));

  const waiting = openStream({chatId: 'stop-b', message: 'wait in line'});
  await waiting.until('queued');

  const stopRes = await fetch(`${base}/api/chat/stop`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({chatId: 'stop-b'})});
  const stopped = await waiting.until('done');

  assert.equal((await stopRes.json()).stopped, true);
  assert.equal(stopped.stopped, true);
  assert.ok(/before it started/.test(stopped.message));

  release();
  await holder.until('done');
  assert.equal(model.state.requests.length, 1);
});

test('closing the connection cancels the run and frees the engine', async () => {
  model.reset([(body, index) => index === 0 ? {content: 'never seen', wait: new Promise(() => {})} : {content: 'next one runs'}]);

  const gone = openStream({chatId: 'drop-a', message: 'start and leave'});
  assert.ok(await waitFor(() => model.state.requests.length === 1));

  gone.stop();
  assert.ok(await waitFor(() => model.state.closed === 1));

  const settings = await readSettings();
  assert.ok(await waitFor(() => !queueFor(settings.providerBaseUrl).busy));

  const next = openStream({chatId: 'drop-b', message: 'are you free'});
  const done = await next.until('done');

  assert.ok(done.message.includes('next one runs'));
});

test('a second request for a chat that is already working is refused', async () => {
  let release = null;
  const hold = new Promise(resolve => {
    release = resolve;
  });

  model.reset([(body, index) => index === 0 ? {content: 'busy chat done', wait: hold} : {content: 'other'}]);

  const running = openStream({chatId: 'twice', message: 'first'});
  assert.ok(await waitFor(() => model.state.requests.length === 1));

  const again = await fetch(`${base}/api/chat/stream`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({chatId: 'twice', message: 'second', project: 'site'})
  });

  assert.equal(again.status, 409);
  release();
  await running.until('done');
});
