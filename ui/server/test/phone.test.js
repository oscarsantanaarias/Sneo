import { testRoot } from './setup.js';
process.env.SNEO_PHONE_HOST = '127.0.0.1';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import http from 'http';
import express from 'express';
import { applyPhone, phoneAdmin, phoneAllows, privateAddress } from '../phone.js';

const port = 3400 + Math.floor(Math.random() * 400);
const apiRoutes = express.Router();
apiRoutes.get('/sneo', (req, res) => res.json({sneo: true}));
apiRoutes.get('/chats', (req, res) => res.json({chats: []}));
apiRoutes.get('/terminal', (req, res) => res.json({secret: true}));
const engineRoutes = express.Router();
engineRoutes.get('/monitor', (req, res) => res.json({state: 'idle'}));
const routers = {apiRoutes, engineRoutes};

const desk = express();
desk.use(express.json());
desk.use('/api/phone', phoneAdmin(routers));
const deskServer = http.createServer(desk);
await new Promise(done => deskServer.listen(0, '127.0.0.1', done));
const deskUrl = `http://127.0.0.1:${deskServer.address().port}/api/phone`;
const phoneUrl = `http://127.0.0.1:${port}`;

after(async () => {
  await fetch(deskUrl, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: false})}).catch(() => {});
  deskServer.closeAllConnections?.();
  await new Promise(done => deskServer.close(done));
});

const admin = async (method, route = '', body = null) => (await fetch(`${deskUrl}${route}`, {method, headers: {'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined})).json();

// fetch no deja cambiar el host, asi que esta pide con http a mano
const raw = (route, {method = 'GET', headers = {}, body = null} = {}) => new Promise((resolve, reject) => {
  const data = body ? JSON.stringify(body) : null;
  const req = http.request({host: '127.0.0.1', port, path: route, method, headers: {...(data ? {'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data)} : {}), ...headers}}, res => {
    let text = '';
    res.on('data', chunk => { text += chunk; });
    res.on('end', () => resolve({status: res.statusCode, headers: res.headers, text}));
  });
  req.on('error', reject);
  if(data) req.write(data);
  req.end();
});

test('only home network, vpn and local addresses count as private', () => {
  for(const ip of ['127.0.0.1', '::1', '10.1.2.3', '172.16.0.1', '172.31.255.255', '192.168.1.20', '169.254.35.110', '100.101.102.103', '::ffff:192.168.0.7', 'fd12:3456::1', 'fe80::1']){
    assert.equal(privateAddress(ip), true, ip);
  }

  for(const ip of ['8.8.8.8', '172.32.0.1', '100.128.0.1', '192.169.1.1', '2001:4860::8888', '', 'not-an-ip']){
    assert.equal(privateAddress(ip), false, ip);
  }
});

test('the phone can chat, approve and use agents but not the terminal or the settings', () => {
  assert.equal(phoneAllows('POST', '/api/chat/stream'), true);
  assert.equal(phoneAllows('POST', '/api/pending/abc-1/approve'), true);
  assert.equal(phoneAllows('GET', '/api/team/agents?x=1'), true);
  assert.equal(phoneAllows('POST', '/api/engine/files/load'), true);
  assert.equal(phoneAllows('PUT', '/api/settings'), false);
  assert.equal(phoneAllows('PUT', '/api/team/routes'), false);
  assert.equal(phoneAllows('POST', '/api/terminal/start'), false);
  assert.equal(phoneAllows('POST', '/api/engine/shutdown'), false);
  assert.equal(phoneAllows('GET', '/api/phone'), false);
  assert.equal(phoneAllows('POST', '/api/projects/pick'), false);
});

test('a phone pairs with a one time code and loses access when it is removed', async () => {
  const off = await admin('GET');
  assert.equal(off.enabled, false);
  assert.equal(off.listening, false);

  fs.writeFileSync(path.join(testRoot, 'data', 'phone.json'), JSON.stringify({enabled: false, port, devices: []}));
  const on = await admin('POST', '', {enabled: true});
  assert.equal(on.listening, true, on.problem);
  assert.match(on.code.code, /^[A-Z2-9]{8}$/);

  const host = {Host: `127.0.0.1:${port}`};
  assert.equal((await raw('/phone/me', {headers: host})).text, '{"paired":false,"name":""}');
  assert.equal((await raw('/api/sneo', {headers: host})).status, 401);
  assert.equal((await raw('/api/sneo', {headers: {Host: `evil.example:${port}`}})).status, 403);
  assert.equal((await raw('/phone/pair', {method: 'POST', headers: host, body: {code: 'WRONGCOD'}})).status, 403);

  const paired = await raw('/phone/pair', {method: 'POST', headers: host, body: {code: on.code.code, name: 'Test phone'}});
  assert.equal(paired.status, 200);
  const cookie = String(paired.headers['set-cookie']?.[0] || '').split(';')[0];
  assert.match(cookie, /^sneo_phone=[a-f0-9]{64}$/);
  assert.equal((await raw('/phone/pair', {method: 'POST', headers: host, body: {code: on.code.code}})).status, 403, 'a code works once');

  const withKey = {...host, Cookie: cookie};
  assert.equal((await raw('/api/sneo', {headers: withKey})).text, '{"sneo":true}');
  assert.equal((await raw('/api/engine/monitor', {headers: withKey})).status, 200);
  assert.equal((await raw('/api/terminal', {headers: withKey})).status, 403);
  assert.equal((await raw('/api/sneo', {headers: {...withKey, Origin: 'http://evil.example'}})).status, 403);

  const saved = JSON.parse(fs.readFileSync(path.join(testRoot, 'data', 'phone.json'), 'utf8'));
  assert.equal(saved.devices.length, 1);
  assert.equal(JSON.stringify(saved).includes(cookie.split('=')[1]), false, 'only the hash of the key is saved');

  const listed = await admin('GET');
  assert.deepEqual(listed.devices.map(item => item.name), ['Test phone']);
  assert.equal(listed.devices[0].tokenHash, undefined);
  await admin('DELETE', `/devices/${listed.devices[0].id}`);
  assert.equal((await raw('/api/sneo', {headers: withKey})).status, 401);

  const stopped = await admin('POST', '', {enabled: false});
  assert.equal(stopped.listening, false);
  await assert.rejects(raw('/phone/me', {headers: host}));
});

test('too many wrong codes are refused for a while', async () => {
  await admin('POST', '', {enabled: true});
  const host = {Host: `127.0.0.1:${port}`};
  const codes = [];

  for(let i = 0; i < 12; i++){
    codes.push((await raw('/phone/pair', {method: 'POST', headers: host, body: {code: `BADCODE${i}`}})).status);
  }

  assert.equal(codes.at(-1), 429);
  await admin('POST', '', {enabled: false});
  await applyPhone(routers);
});
