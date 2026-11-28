import express from 'express';
import fs from 'fs';
import net from 'net';
import os from 'os';
import path from 'path';
import { execFile, spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { promisify } from 'util';
import { paths, port } from './config.js';
import { readSettings, saveSettings } from './storage.js';
import { forgetEngine } from './model.js';
import { readRoutes, routePresets, saveRoutes, setLocalTurns } from './teams.js';
import { downloadState, modelCatalog, startDownload, stopDownload } from './models.js';
import { pickFolder } from './picker.js';

const router = express.Router();
const execFileAsync = promisify(execFile);
const rootDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const workerPage = 'http://127.0.0.1:9096';
const poolDoor = 9097;
const poolActions = ['remove', 'apply', 'unload'];
const requestActions = ['accept', 'decline'];

export const readAllowNetwork = async () => {
  const settings = await readSettings().catch(() => ({}));
  return settings.allowNetwork === true;
};

const engineBase = async () => {
  const settings = await readSettings().catch(() => ({}));
  return (settings.providerBaseUrl || 'http://localhost:8000').trim().replace(/\/+$/, '').replace(/\/v1$/, '');
};

const forward = async (res, method, route, body = null) => {
  try {
    const baseUrl = await engineBase();

    const req = await fetch(`${baseUrl}${route}`, {
      method,
      headers: {'Content-Type': 'application/json'},
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(20000)
    });

    const text = await req.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch(e) {
      data = {message: text || `Sneo answered ${req.status}`};
    }

    if(!req.ok && !data.message){
      data.message = data.error || `Sneo answered ${req.status}`;
    }

    return res.status(req.status).json(data);

  } catch(e) {
    console.log('engine forward failed', route, e.message);
    return res.status(502).json({message: `Sneo is not reachable: ${e.message}`});
  }
};

router.get('/pool', async (req, res) => {
  return await forward(res, 'GET', '/api/pool');
});

router.post('/pool', async (req, res) => {
  const {slots, context, mainGpu, secondGpu, helperGpu, helperGb, helperFirst, frontLayers, draftLength, speculate, copyReach, cpuPlan, warmShare, holdRam, ramWhole, pinShare, pinGb, vision, visionGpu} = req.body ?? {};

  if(slots !== undefined && !Array.isArray(slots)){
    return res.status(400).json({message: 'slots must be a list'});
  }

  if(context !== undefined && (!Number.isInteger(Number(context)) || Number(context) < 1)){
    return res.status(400).json({message: 'Invalid context size'});
  }

  if(mainGpu !== undefined && (!Number.isInteger(Number(mainGpu)) || Number(mainGpu) < -1)){
    return res.status(400).json({message: 'Invalid main GPU'});
  }

  const body = {};

  if(mainGpu !== undefined){
    body.mainGpu = Number(mainGpu);
  }

  if(secondGpu !== undefined){
    body.secondGpu = secondGpu === true;
  }

  // la tarjeta que ayuda, su tamano y el orden de sus expertos no llegaban al motor y volvian a 0 al guardar
  if(helperGpu !== undefined && Number.isInteger(Number(helperGpu)) && Number(helperGpu) >= -1){
    body.helperGpu = Number(helperGpu);
  }

  if(helperGb !== undefined && Number.isFinite(Number(helperGb)) && Number(helperGb) >= 0){
    body.helperGb = Number(helperGb);
  }

  if(helperFirst !== undefined){
    body.helperFirst = helperFirst === true;
  }

  if(frontLayers !== undefined && Number.isInteger(Number(frontLayers)) && Number(frontLayers) >= 0){
    body.frontLayers = Number(frontLayers);
  }

  if(draftLength !== undefined && Number.isInteger(Number(draftLength)) && Number(draftLength) >= 1 && Number(draftLength) <= 16){
    body.draftLength = Number(draftLength);
  }

  if(speculate !== undefined){
    body.speculate = speculate === true;
  }

  if(copyReach !== undefined && Number.isInteger(Number(copyReach)) && Number(copyReach) >= 0 && Number(copyReach) <= 32){
    body.copyReach = Number(copyReach);
  }

  if(['auto', 'performance', 'performance-ht', 'all'].includes(cpuPlan)){
    body.cpuPlan = cpuPlan;
  }

  if(warmShare !== undefined && Number.isInteger(Number(warmShare)) && Number(warmShare) >= 0 && Number(warmShare) <= 8){
    body.warmShare = Number(warmShare);
  }

  if(holdRam !== undefined){
    body.holdRam = holdRam === true;
  }

  if(ramWhole !== undefined){
    body.ramWhole = ramWhole === true;
  }

  if(pinShare !== undefined && Number.isInteger(Number(pinShare)) && Number(pinShare) >= 10 && Number(pinShare) <= 95){
    body.pinShare = Number(pinShare);
  }

  if(pinGb !== undefined && Number.isFinite(Number(pinGb)) && Number(pinGb) >= 0){
    body.pinGb = Number(pinGb);
  }

  if(vision !== undefined){
    body.vision = vision === true;
  }

  if(visionGpu !== undefined && Number.isInteger(Number(visionGpu)) && Number(visionGpu) >= -1){
    body.visionGpu = Number(visionGpu);
  }

  if(slots){
    body.slots = slots.map(item => ({
      node: String(item?.node || ''),
      kind: ['gpu', 'ram', 'ssd'].includes(item?.kind) ? item.kind : 'ram',
      gb: Math.max(0, Number(item?.gb) || 0),
      percent: Math.min(100, Math.max(0, Number(item?.percent) || 0)),
      keep: Math.max(0, Number(item?.keep) || 0)
    })).filter(item => item.node);
  }

  if(context !== undefined){
    body.context = Number(context);
  }

  forgetEngine();
  return await forward(res, 'POST', '/api/pool', body);
});

router.post('/pool/request/:action', async (req, res) => {
  const action = req.params.action;
  const id = String(req.body?.id || '').trim();

  if(!requestActions.includes(action)){
    return res.status(404).json({message: 'Unknown request action'});
  }

  if(!/^[a-z0-9]{1,32}$/.test(id)){
    return res.status(400).json({message: 'Request id required'});
  }

  return await forward(res, 'POST', `/api/pool/request/${action}`, {id});
});

router.post('/pool/:action', async (req, res) => {
  const action = req.params.action;

  if(!poolActions.includes(action)){
    return res.status(404).json({message: 'Unknown pool action'});
  }

  if(action === 'remove' && !String(req.body?.name || '').trim()){
    return res.status(400).json({message: 'name required'});
  }

  return await forward(res, 'POST', `/api/pool/${action}`, {name: String(req.body?.name || '')});
});

router.get('/firewall', async (req, res) => {
  const ruleOk = await execFileAsync('netsh', ['advfirewall', 'firewall', 'show', 'rule', 'name=Sneo pool door'], {windowsHide: true}).then(() => true).catch(() => false);
  const publicNetwork = await execFileAsync('powershell.exe', ['-NoProfile', '-Command', "Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -eq 'Public' } | Select-Object -First 1 -ExpandProperty Name"], {windowsHide: true})
    .then(result => String(result.stdout || '').trim())
    .catch(() => '');

  if(!publicNetwork){
    return res.status(200).json({ready: ruleOk, ruleOk, publicNetwork});
  }

  const rules = await execFileAsync('netsh', ['advfirewall', 'firewall', 'show', 'rule', 'name=all', 'verbose'], {windowsHide: true, maxBuffer: 64 * 1024 * 1024})
    .then(result => String(result.stdout || '').split(/\r?\n\s*\r?\n/))
    .catch(() => []);
  const opens = (program, port) => rules.some(rule => /Enabled:\s+Yes/i.test(rule) && /Direction:\s+In/i.test(rule) && /Action:\s+Allow/i.test(rule)
    && /Profiles:\s+[^\r\n]*(Public|Any)/i.test(rule)
    && (rule.toLowerCase().includes(program.toLowerCase()) || new RegExp(`LocalPort:\\s+${port}\\s`, 'i').test(rule)));
  const publicOpen = opens(path.join(rootDir, 'master.exe'), 9097) && opens(path.join(rootDir, 'worker.exe'), 9090);
  return res.status(200).json({ready: ruleOk && publicOpen, ruleOk, publicNetwork, publicOpen});
});

router.post('/firewall', (req, res) => {
  const script = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'tools', 'pool_setup.ps1');
  spawn('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script], {detached: true, stdio: 'ignore'}).unref();
  return res.status(200).json({started: true});
});

const tailnet = ip => {
  const parts = String(ip).split('.').map(Number);
  return parts.length === 4 && parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127;
};

router.get('/pools/scan', async (req, res) => {
  const own = new Set();
  const prefixes = new Set();
  const networks = new Set();
  for(const list of Object.values(os.networkInterfaces())){
    for(const item of list || []){
      if(item.family !== 'IPv4' || item.internal) continue;
      networks.add(item.address.startsWith('169.254.') ? '169.254' : tailnet(item.address) ? 'tailscale' : item.address.split('.').slice(0, 3).join('.'));
      if(item.address.startsWith('169.254.')) continue;
      own.add(item.address);
      if(/^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(item.address)) prefixes.add(item.address.split('.').slice(0, 3).join('.'));
    }
  }
  const targets = [];
  for(const prefix of prefixes){
    for(let i = 1; i < 255; i++){
      if(!own.has(`${prefix}.${i}`)) targets.push(`${prefix}.${i}`);
    }
  }
  const tailscale = await execFileAsync('tailscale', ['status', '--json'], {windowsHide: true, timeout: 4000})
    .catch(() => execFileAsync('C:\\Program Files\\Tailscale\\tailscale.exe', ['status', '--json'], {windowsHide: true, timeout: 4000}))
    .then(result => JSON.parse(String(result.stdout || '{}')))
    .catch(() => ({}));
  for(const peer of Object.values(tailscale.Peer || {})){
    if(peer.Online === false) continue;
    for(const ip of peer.TailscaleIPs || []){
      if(/^\d{1,3}(\.\d{1,3}){3}$/.test(ip) && !own.has(ip) && !targets.includes(ip)) targets.push(ip);
    }
  }
  const found = [];
  for(let at = 0; at < targets.length; at += 64){
    await Promise.all(targets.slice(at, at + 64).map(async ip => {
      try {
        const reply = await fetch(`http://${ip}:${poolDoor}/hello`, {signal: AbortSignal.timeout(800)});
        const data = await reply.json();
        if(data.sneo) found.push({...data, ip});
      } catch(e) {}
    }));
  }
  const handshake = ip => new Promise(done => {
    const started = performance.now();
    const socket = net.connect({host: ip, port: poolDoor});
    const finish = ms => {
      socket.destroy();
      done(ms);
    };
    socket.setTimeout(800, () => finish(null));
    socket.once('connect', () => finish(performance.now() - started));
    socket.once('error', () => finish(null));
  });
  const hosts = [];
  for(const item of found){
    if(hosts.some(host => host.name === item.name)) continue;
    const links = (item.links || []).filter(link => /^\d{1,3}(\.\d{1,3}){3}$/.test(link.ip));
    if(!links.some(link => link.ip === item.ip)) links.push({ip: item.ip, kind: 'other', mbps: 0});
    const routes = [];
    for(const link of links){
      let best = null;
      for(let i = 0; i < 3; i++){
        const ms = await handshake(link.ip);
        if(ms === null) break;
        best = best === null ? ms : Math.min(best, ms);
      }
      const near = networks.has(link.ip.startsWith('169.254.') ? '169.254' : tailnet(link.ip) ? 'tailscale' : link.ip.split('.').slice(0, 3).join('.'));
      if(tailnet(link.ip)) link.kind = 'tailscale';
      if(best !== null) routes.push({...link, ms: Math.round(best * 100) / 100});
      else if(near && link.kind !== 'other') routes.push({...link, ms: null, blocked: true});
    }
    routes.sort((a, b) => (a.ms ?? 1e9) - (b.ms ?? 1e9));
    hosts.push({...item, routes});
  }
  return res.status(200).json({found: hosts, scanned: targets.length});
});

const workerState = async () => {
  try {
    const reply = await fetch(`${workerPage}/state`, {signal: AbortSignal.timeout(1500)});
    return {running: true, ...(await reply.json())};
  } catch(e) {
    return {running: false};
  }
};

router.post('/shutdown', async (req, res) => {
  if(req.get('x-sneo') !== 'shutdown'){
    return res.status(403).json({message: 'Only the Sneo app can stop the engine'});
  }

  const base = await engineBase();
  const local = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/.test(base);
  await fetch(`${workerPage}/quit`, {method: 'POST', signal: AbortSignal.timeout(10000)}).catch(() => {});

  // al cerrar la app con dos modelos: el primero guarda su configuracion de siempre y el segundo motor tambien se apaga
  const pair = await readSecond();

  if(pair.restore && local){
    await fetch(`${base}/api/pool`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({slots: pair.restore.slots, pinGb: pair.restore.pinGb, vision: pair.restore.vision}), signal: AbortSignal.timeout(10000)}).catch(() => {});
    await writeSecond({...pair, restore: null});
  }

  await askSecond('POST', '/api/shutdown', null, 5000).catch(() => {});

  if(local){
    await fetch(`${base}/api/shutdown`, {method: 'POST', signal: AbortSignal.timeout(5000)}).catch(() => {});
    for(let i = 0; i < 120; i++){
      const alive = await fetch(`${base}/api/monitor`, {signal: AbortSignal.timeout(500)}).then(() => true).catch(() => false);
      if(!alive) break;
      await new Promise(done => setTimeout(done, 250));
    }
  }

  res.status(200).json({stopped: true, engine: local});
  setTimeout(() => process.exit(0), 300);
});

router.get('/pool/host', async (req, res) => {
  const state = await workerState();

  if(!state.running || state.status !== 'joined' || !state.host){
    return res.status(200).json({joined: false});
  }

  try {
    const reply = await fetch(`http://${state.host}:${state.doorPort || poolDoor}/pool`, {signal: AbortSignal.timeout(3000)});
    const data = await reply.json();
    if(!reply.ok) return res.status(200).json({joined: true, host: state.host, error: data.error || `the pool host answered ${reply.status}`});
    return res.status(200).json({joined: true, host: state.host, pool: data});
  } catch(e) {
    return res.status(200).json({joined: true, host: state.host, error: 'the pool host did not answer'});
  }
});

router.get('/worker', async (req, res) => {
  return res.status(200).json(await workerState());
});

router.post('/worker/join', async (req, res) => {
  const host = String(req.body?.host || '').trim();

  if(!/^\d{1,3}(\.\d{1,3}){3}$/.test(host)){
    return res.status(400).json({message: 'Pick a pool from the list or type its IPv4 address'});
  }

  await fetch(`${await engineBase()}/api/models/unload`, {method: 'POST', signal: AbortSignal.timeout(20000)}).catch(() => {});
  forgetEngine();
  let state = await workerState();

  if(!state.running){
    const log = fs.openSync(path.join(rootDir, 'worker.log'), 'a');
    spawn(path.join(rootDir, 'worker.exe'), ['9090', 'cache', '--no-browser'], {cwd: rootDir, detached: true, stdio: ['ignore', log, log], windowsHide: true}).unref();
    fs.closeSync(log);
    for(let i = 0; i < 40 && !state.running; i++){
      await new Promise(done => setTimeout(done, 500));
      state = await workerState();
    }
  }

  if(!state.running){
    return res.status(500).json({message: 'worker.exe did not start; run build.bat to build it'});
  }

  if(state.status === 'joined' && state.host === host){
    return res.status(200).json(state);
  }

  if(state.status === 'joined'){
    await fetch(`${workerPage}/leave`, {method: 'POST', signal: AbortSignal.timeout(5000)}).catch(() => {});
  }

  const reply = await fetch(`${workerPage}/join`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({host, doorPort: poolDoor})}).catch(() => null);
  const data = reply ? await reply.json().catch(() => ({})) : {};

  if(!reply || !reply.ok){
    return res.status(reply ? reply.status : 500).json({message: data.error || 'The worker could not send the request'});
  }

  return res.status(200).json({running: true, ...data});
});

router.post('/worker/leave', async (req, res) => {
  await fetch(`${workerPage}/leave`, {method: 'POST', signal: AbortSignal.timeout(15000)}).catch(() => {});
  const exe = path.join(rootDir, 'worker.exe').replace(/'/g, "''");
  await execFileAsync('powershell.exe', ['-NoProfile', '-Command', `Get-Process worker -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq '${exe}' } | Stop-Process -Force`], {windowsHide: true}).catch(() => {});
  return res.status(200).json({running: false});
});

router.get('/network', async (req, res) => {
  const allowNetwork = await readAllowNetwork();

  return res.status(200).json({
    allowNetwork,
    sneoNetwork: req.app.get('sneoHost') === '0.0.0.0',
    sneoPort: Number(port)
  });
});

router.post('/network', async (req, res) => {
  const {acceptRequests, doorPort, apiListen, allowNetwork} = req.body ?? {};

  if(acceptRequests !== undefined && typeof acceptRequests !== 'boolean'){
    return res.status(400).json({message: 'acceptRequests must be true or false'});
  }

  if(doorPort !== undefined && (!Number.isInteger(Number(doorPort)) || Number(doorPort) < 1 || Number(doorPort) > 65535)){
    return res.status(400).json({message: 'The door port must be a number from 1 to 65535'});
  }

  if(apiListen !== undefined && !['local', 'network'].includes(apiListen)){
    return res.status(400).json({message: 'apiListen must be local or network'});
  }

  if(allowNetwork !== undefined && typeof allowNetwork !== 'boolean'){
    return res.status(400).json({message: 'allowNetwork must be true or false'});
  }

  if(allowNetwork !== undefined){
    try {
      const current = await readSettings();
      await saveSettings({...current, allowNetwork, updatedAt: new Date().toISOString()});
    } catch(e) {
      return res.status(500).json({message: `Could not save settings.json: ${e.message}`});
    }
  }

  const body = {};

  if(acceptRequests !== undefined){
    body.acceptRequests = acceptRequests;
  }

  if(doorPort !== undefined){
    body.doorPort = Number(doorPort);
  }

  if(apiListen !== undefined){
    body.apiListen = apiListen;
  }

  if(Object.keys(body).length === 0){
    return res.status(200).json({message: 'Saved'});
  }

  return await forward(res, 'POST', '/api/pool/network', body);
});

router.get('/monitor', async (req, res) => {
  return await forward(res, 'GET', '/api/monitor');
});

router.get('/files', async (req, res) => {
  return await forward(res, 'GET', '/api/models');
});

router.post('/files/scan', async (req, res) => {
  return await forward(res, 'POST', '/api/models/scan', {dir: String(req.body?.dir || '')});
});

router.post('/files/pick', async (req, res) => {
  try {
    const dir = await pickFolder({title: 'Select the folder with your models', start: String(req.body?.dir || ''), button: 'Use this folder'});
    if(!dir) return res.status(200).json({cancelled: true});
    return await forward(res, 'POST', '/api/models/scan', {dir});
  } catch(e) {
    return res.status(500).json({message: e.message});
  }
});

router.post('/files/stop', async (req, res) => {
  forgetEngine();
  return await forward(res, 'POST', '/api/models/stop');
});

router.post('/files/load', async (req, res) => {
  forgetEngine();
  return await forward(res, 'POST', '/api/models/load', {path: String(req.body?.path || '')});
});

router.get('/catalog', async (req, res) => {
  try {
    return res.status(200).json(await modelCatalog(await engineBase()));
  } catch(e) {
    return res.status(502).json({message: `The model list could not be read: ${e.message}`, job: downloadState()});
  }
});

router.get('/download', (req, res) => {
  return res.status(200).json(downloadState());
});

router.post('/download', async (req, res) => {
  try {
    return res.status(200).json(await startDownload(String(req.body?.id || ''), await engineBase()));
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

router.post('/download/stop', (req, res) => {
  return res.status(200).json(stopDownload());
});

const gib = 1024 ** 3;
const profilesDir = path.join(rootDir, 'profiles');
const heatHead = 24;

const askEngine = async (method, route, body = null, wait = 20000) => {
  const reply = await fetch(`${await engineBase()}${route}`, {
    method,
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(wait)
  });
  const data = await reply.json().catch(() => ({}));
  if(!reply.ok) throw new Error(data.message || data.error || `Sneo answered ${reply.status}`);
  return data;
};

const engineLocal = async () => /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/.test(await engineBase());

const fileStem = file => path.win32.basename(String(file || ''), path.win32.extname(String(file || '')));

const cardName = name => String(name || '').replace(/^NVIDIA (GeForce )?/, '');

const sizeText = bytes => `${Math.round(Number(bytes || 0) / gib)} GB`;

const heatShape = buffer => {
  if(!buffer || buffer.length < heatHead || buffer.toString('latin1', 0, 4) !== 'HEAT' || buffer.readUInt32LE(4) !== 1) return null;
  const layers = buffer.readUInt32LE(8);
  const experts = buffer.readUInt32LE(12);
  if(buffer.length !== heatHead + layers * experts * 8) return null;
  return {layers, experts, tokens: Number(buffer.readBigUInt64LE(16))};
};

const pickFile = async (script) => {
  const result = await execFileAsync('powershell.exe', ['-NoProfile', '-STA', '-Command', `Add-Type -AssemblyName System.Windows.Forms; $owner = New-Object System.Windows.Forms.Form -Property @{TopMost = $true}; ${script}; if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.FileName }`], {
    timeout: 300000,
    windowsHide: false,
    maxBuffer: 1024 * 100
  });
  return String(result.stdout || '').trim();
};

const makeRecipe = async () => {
  const pool = await askEngine('GET', '/api/pool');

  if(!pool.model?.path){
    throw new Error('Pick a model first, a recipe is made for one model');
  }

  const stats = (pool.nodes || []).find(node => node.self)?.stats || {};
  const mine = (pool.slots || []).filter(slot => slot.node === pool.self);
  const share = (kind, total) => {
    const slot = mine.find(item => item.kind === kind);
    if(!slot) return null;
    return {
      share: !slot.gb || !total ? 1 : Math.min(1, Math.round(slot.gb * gib / total * 1000) / 1000),
      percent: Number(slot.percent || 0),
      keep: Number(slot.keep || 0)
    };
  };
  const main = pool.mainGpu >= 0 ? (pool.gpus || []).find(gpu => gpu.ordinal === pool.mainGpu) : null;
  const recipe = {
    sneo: 'recipe',
    version: 1,
    name: `${pool.model.name || fileStem(pool.model.path)} on ${stats.gpu?.present ? cardName(stats.gpu.name) : 'CPU'}`,
    created: new Date().toISOString(),
    model: {
      name: pool.model.name || '',
      file: path.win32.basename(pool.model.path),
      arch: pool.model.arch || '',
      bytes: Number(pool.model.bytes || 0),
      layers: Number(pool.model.layers || 0),
      experts: Number(pool.model.experts || 0),
      expertsUsed: Number(pool.model.expertsUsed || 0)
    },
    hardware: {
      cpu: stats.cpu || '',
      threads: Number(stats.threads || 0),
      ram: Number(stats.ramTotal || 0),
      drive: stats.drive || '',
      gpus: (pool.gpus || []).map(gpu => ({name: gpu.name, vram: Number(gpu.vram || 0)})),
      helpers: (pool.nodes || []).filter(node => !node.self && node.connected && node.accepted).map(node => ({
        cpu: node.stats?.cpu || '',
        ram: Number(node.stats?.ramTotal || 0),
        gpu: node.stats?.gpu?.present ? node.stats.gpu.name : '',
        vram: Number(node.stats?.gpu?.vramTotal || 0)
      }))
    },
    settings: {
      backend: pool.backend || 'auto',
      context: Number(pool.context || 0),
      speculate: pool.speculate !== false,
      draftLength: Number(pool.draftLength || 3),
      copyReach: Number(pool.copyReach ?? 8),
      cpuPlan: pool.cpuPlan || 'all',
      warmShare: Number(pool.warmShare ?? 2),
      holdRam: pool.holdRam !== false,
      ramWhole: pool.ramWhole === true,
      pinShare: Number(pool.pinShare ?? 50),
      pinGb: Number(pool.pinGb || 0),
      vision: pool.vision === true,
      draftSure: Number(pool.draftSure ?? 0.6),
      fastRouting: pool.fastRouting !== false,
      mainGpu: main ? main.name : '',
      secondGpu: pool.secondGpu !== false,
      frontLayers: Number(pool.frontLayers || 0)
    },
    memory: {
      gpu: share('gpu', Number(stats.gpu?.vramTotal || 0)),
      ram: share('ram', Number(stats.ramTotal || 0))
    }
  };

  if(await engineLocal()){
    const buffer = await fs.promises.readFile(path.join(profilesDir, `${fileStem(pool.model.path)}.heat`)).catch(() => null);
    const shape = heatShape(buffer);
    if(shape && shape.layers === recipe.model.layers && shape.experts === recipe.model.experts){
      recipe.heat = {...shape, data: buffer.toString('base64')};
    }
  }

  return recipe;
};

const planRecipe = async (recipe) => {
  if(recipe?.sneo !== 'recipe' || recipe.version !== 1 || typeof recipe.model !== 'object' || typeof recipe.settings !== 'object'){
    throw new Error('This file is not a Sneo recipe');
  }

  const worker = await workerState();

  if(worker.running && worker.status === 'joined'){
    throw new Error('This PC is helping another pool. Import the recipe on the pool host.');
  }

  const pool = await askEngine('GET', '/api/pool');
  const files = await askEngine('GET', '/api/models').catch(() => ({models: []}));
  const stats = (pool.nodes || []).find(node => node.self)?.stats || {};
  const wanted = recipe.model;
  const settings = recipe.settings;
  const hardware = recipe.hardware || {};
  const exact = (files.models || []).find(item => String(item.name).toLowerCase() === String(wanted.file || '').toLowerCase());
  const similar = !exact && pool.model?.path && pool.model.arch === wanted.arch && Number(pool.model.layers) === Number(wanted.layers) && Number(pool.model.experts) === Number(wanted.experts);
  const model = exact
    ? {path: exact.path, match: 'same'}
    : similar
      ? {path: pool.model.path, match: 'similar'}
      : {path: '', match: 'missing'};

  const vramTotal = Number(stats.gpu?.present ? stats.gpu.vramTotal : 0);
  const ramTotal = Number(stats.ramTotal || 0);
  const scale = (slot, take, total) => {
    if(!take || !total) return slot;
    const gb = Number(take.share) >= 0.99 ? 0 : Math.round(Number(take.share) * total / gib * 10) / 10;
    const keep = Math.min(Math.max(0, Number(take.keep) || 0), Math.max(0, Math.floor(total / gib) - 1));
    return {...slot, gb, percent: Math.min(100, Math.max(0, Number(take.percent) || 0)), keep};
  };
  const slots = (pool.slots || []).map(slot => slot.node !== pool.self
    ? slot
    : slot.kind === 'gpu'
      ? scale(slot, recipe.memory?.gpu, vramTotal)
      : slot.kind === 'ram'
        ? scale(slot, recipe.memory?.ram, ramTotal)
        : slot).map(slot => ({node: slot.node, kind: slot.kind, gb: slot.gb, percent: slot.percent, keep: slot.keep}));

  const mainCard = settings.mainGpu ? (pool.gpus || []).find(gpu => gpu.name === settings.mainGpu) : null;
  const limit = Number(pool.model?.contextLength || 0);
  const context = Math.max(4096, Math.min(Number(settings.context) || Number(pool.context) || 32768, model.match !== 'missing' && limit ? limit : 1048576));
  const body = {
    slots,
    context,
    backend: ['auto', 'gpu', 'cpu'].includes(settings.backend) ? settings.backend : 'auto',
    speculate: settings.speculate !== false,
    draftLength: Math.max(1, Math.min(16, Math.round(Number(settings.draftLength) || 3))),
    copyReach: Math.max(0, Math.min(32, Math.round(Number(settings.copyReach ?? 8)))),
    cpuPlan: ['auto', 'performance', 'performance-ht', 'all'].includes(settings.cpuPlan) ? settings.cpuPlan : 'all',
    warmShare: Math.max(0, Math.min(8, Math.round(Number(settings.warmShare ?? 2)))),
    holdRam: settings.holdRam !== false,
    ramWhole: settings.ramWhole === true,
    pinShare: Math.max(10, Math.min(95, Math.round(Number(settings.pinShare ?? 50)))),
    pinGb: Math.max(0, Number(settings.pinGb) || 0),
    vision: settings.vision === true,
    draftSure: Math.max(0, Math.min(0.99, Number(settings.draftSure ?? 0.6))),
    fastRouting: settings.fastRouting !== false,
    mainGpu: mainCard ? mainCard.ordinal : -1,
    secondGpu: settings.secondGpu !== false,
    frontLayers: Math.max(0, Math.round(Number(settings.frontLayers) || 0))
  };

  const heatBuffer = recipe.heat?.data ? Buffer.from(String(recipe.heat.data), 'base64') : null;
  const heat = heatShape(heatBuffer);
  const heatFits = !!heat && heat.layers === Number(wanted.layers) && heat.experts === Number(wanted.experts);
  const local = await engineLocal();
  const theirGpus = (hardware.gpus || []).map(gpu => `${cardName(gpu.name)} ${sizeText(gpu.vram)}`).join(' + ') || 'none';
  const ourGpus = (pool.gpus || []).map(gpu => `${cardName(gpu.name)} ${sizeText(gpu.vram)}`).join(' + ') || 'none';
  const theirHelpers = (hardware.helpers || []).map(helper => helper.gpu ? `${cardName(helper.gpu)} ${sizeText(helper.vram)}` : `CPU ${sizeText(helper.ram)}`);
  const ourHelpers = (pool.nodes || []).filter(node => !node.self && node.connected && node.accepted).map(node => node.stats?.gpu?.present ? `${cardName(node.stats.gpu.name)} ${sizeText(node.stats.gpu.vramTotal)}` : `CPU ${sizeText(node.stats?.ramTotal)}`);
  const gpuShare = recipe.memory?.gpu;
  const ramShare = recipe.memory?.ram;
  const shareText = (take, total) => !take
    ? 'kept as it is'
    : Number(take.share) >= 0.99
      ? `all of it (${sizeText(total)})`
      : `${Math.round(Number(take.share) * 100)}% (${(Number(take.share) * total / gib).toFixed(1)} GB)`;

  const rows = [
    {
      label: 'Model',
      recipe: wanted.file || wanted.name || 'unknown',
      here: model.path ? path.win32.basename(model.path) : 'not in your models folder',
      note: model.match === 'same'
        ? 'Same file, it loads with the recipe.'
        : model.match === 'similar'
          ? 'Another file of the same model. The settings and the heat map still fit.'
          : 'The settings are applied now. Download the model and load it when you have it.'
    },
    {
      label: 'Video card',
      recipe: theirGpus,
      here: ourGpus,
      note: !vramTotal
        ? 'No GPU here, the model runs on the CPU.'
        : `${(hardware.gpus || []).length > (pool.gpus || []).length ? `Made for ${hardware.gpus.length} video cards, this PC has ${(pool.gpus || []).length}, so fewer experts fit in video memory. ` : ''}Video memory for the model: ${shareText(gpuShare, vramTotal)}.`
    },
    {
      label: 'RAM',
      recipe: sizeText(hardware.ram),
      here: sizeText(ramTotal),
      note: `RAM for the model: ${shareText(ramShare, ramTotal)}.`
    },
    {
      label: 'Pool helpers',
      recipe: theirHelpers.length ? theirHelpers.join(', ') : 'none',
      here: ourHelpers.length ? ourHelpers.join(', ') : 'none',
      note: !theirHelpers.length
        ? 'Made for a single PC.'
        : body.frontLayers
          ? `Helpers take at most ${body.frontLayers} layers.`
          : 'Helpers take as many layers as fit.'
    },
    {
      label: 'Context',
      recipe: `${Math.round(Number(settings.context || 0) / 1024)}K tokens`,
      here: `${Math.round(Number(pool.context || 0) / 1024)}K tokens`,
      note: context !== Number(settings.context) ? `Set to ${Math.round(context / 1024)}K, the most this model takes.` : 'Copied.'
    },
    {
      label: 'Drafts',
      recipe: settings.speculate === false ? 'off' : `${body.draftLength} per step`,
      here: pool.speculate === false ? 'off' : `${pool.draftLength} per step`,
      note: 'Copied.'
    },
    {
      label: 'Routing',
      recipe: settings.fastRouting === false ? 'exact' : 'fast',
      here: pool.fastRouting === false ? 'exact' : 'fast',
      note: 'Copied.'
    },
    {
      label: 'Heat map',
      recipe: heatFits ? `learned from ${heat.tokens.toLocaleString('en-US')} tokens` : 'not included',
      here: !heatFits ? '' : local ? 'added to what this PC already learned' : 'skipped, the engine runs on another PC',
      note: heatFits && local ? 'The right experts sit in video memory from the first token.' : ''
    }
  ];

  if(settings.mainGpu && !mainCard){
    rows.splice(2, 0, {label: 'Main GPU', recipe: cardName(settings.mainGpu), here: 'the fastest one', note: 'That card is not in this PC.'});
  }

  return {name: String(recipe.name || 'Sneo recipe'), created: recipe.created || '', rows, model, body, heat: heatFits && local ? heatBuffer : null, stem: fileStem(model.path || wanted.file)};
};

const mergeHeat = async (stem, incoming) => {
  await fs.promises.mkdir(profilesDir, {recursive: true});
  const target = path.join(profilesDir, `${stem}.heat`);
  const held = await fs.promises.readFile(target).catch(() => null);
  const theirs = heatShape(incoming);
  const ours = heatShape(held);
  const merged = Buffer.from(incoming);

  if(ours && ours.layers === theirs.layers && ours.experts === theirs.experts){
    for(let at = 16; at < merged.length; at += 8){
      merged.writeBigUInt64LE(held.readBigUInt64LE(at) + incoming.readBigUInt64LE(at), at);
    }
  }

  await fs.promises.writeFile(`${target}.part`, merged);
  await fs.promises.rename(`${target}.part`, target);
  return !!ours;
};

const showPlan = plan => ({name: plan.name, created: plan.created, rows: plan.rows, loads: !!plan.model.path});

router.get('/recipe', async (req, res) => {
  try {
    return res.status(200).json(await makeRecipe());
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

router.post('/recipe/export', async (req, res) => {
  try {
    const recipe = await makeRecipe();
    const name = recipe.name.replace(/[^A-Za-z0-9 ._-]/g, '').trim().slice(0, 80) || 'sneo recipe';
    const file = await pickFile(`$dialog = New-Object System.Windows.Forms.SaveFileDialog; $dialog.Title = 'Save the Sneo recipe'; $dialog.Filter = 'Sneo recipe (*.sneo.json)|*.sneo.json|All files (*.*)|*.*'; $dialog.FileName = '${name.replace(/'/g, "''")}.sneo.json'`);
    if(!file) return res.status(200).json({cancelled: true});
    await fs.promises.writeFile(file, JSON.stringify(recipe, null, 2));
    return res.status(200).json({saved: file, heat: !!recipe.heat});
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

router.post('/recipe/pick', async (req, res) => {
  try {
    const file = await pickFile(`$dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Title = 'Open a Sneo recipe'; $dialog.Filter = 'Sneo recipe (*.sneo.json;*.json)|*.sneo.json;*.json|All files (*.*)|*.*'`);
    if(!file) return res.status(200).json({cancelled: true});
    let recipe;
    try {
      recipe = JSON.parse(await fs.promises.readFile(file, 'utf8'));
    } catch(e) {
      throw new Error('This file is not a Sneo recipe');
    }
    return res.status(200).json({recipe, ...showPlan(await planRecipe(recipe))});
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

router.post('/recipe/check', async (req, res) => {
  try {
    return res.status(200).json(showPlan(await planRecipe(req.body?.recipe)));
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

router.post('/recipe/apply', async (req, res) => {
  try {
    const plan = await planRecipe(req.body?.recipe);
    const notes = [];
    forgetEngine();

    if(plan.model.path){
      await askEngine('POST', '/api/models/unload', null, 180000);
    }

    if(plan.heat){
      const merged = await mergeHeat(plan.stem, plan.heat);
      notes.push(merged ? 'Heat map added to what this PC already learned.' : 'Heat map saved.');
    }

    await askEngine('POST', '/api/pool', plan.body, 30000);
    notes.push('Settings applied.');

    if(plan.model.path){
      await askEngine('POST', '/api/models/load', {path: plan.model.path}, 30000);
      notes.push(`Loading ${path.win32.basename(plan.model.path)}.`);
    }

    return res.status(200).json({applied: true, notes});
  } catch(e) {
    return res.status(400).json({message: e.message});
  }
});

// un segundo motor con otro modelo al mismo tiempo: su propio puerto, su propia configuracion y su propio log
const secondPort = 8001;
const secondDoor = 9098;
const secondPool = 'pool.second.json';
const secondLog = () => path.join(rootDir, 'master.second.log');
const secondFile = () => path.join(paths.dataDir, 'second-engine.json');
const second = {pid: 0, starting: false, step: '', error: ''};

const askSecond = async (method, route, body = null, wait = 3000) => {
  const reply = await fetch(`http://127.0.0.1:${secondPort}${route}`, {
    method,
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(wait)
  });
  const data = await reply.json().catch(() => ({}));
  if(!reply.ok) throw new Error(data.message || data.error || `The second engine answered ${reply.status}`);
  return data;
};

const readSecond = async () => {
  try {
    return JSON.parse(await fs.promises.readFile(secondFile(), 'utf8'));
  } catch(e) {
    return {};
  }
};

const writeSecond = async (data) => {
  await fs.promises.mkdir(paths.dataDir, {recursive: true});
  await fs.promises.writeFile(secondFile(), JSON.stringify(data, null, 2), 'utf8');
};

const waitFor = async (check, wait, step = 1000) => {
  const until = Date.now() + wait;

  while(Date.now() < until){
    if(await check().catch(() => false)){
      return true;
    }

    await new Promise(done => setTimeout(done, step));
  }

  return false;
};

const logTail = async (file, count = 6) => {
  try {
    return (await fs.promises.readFile(file, 'utf8')).split(/\r?\n/).filter(Boolean).slice(-count);
  } catch(e) {
    return [];
  }
};

// el primero se recarga con menos video memoria para que el segundo entre en la misma tarjeta
const roomSlots = (main, roomGb) => main.slots.map(slot => {
  if(slot.node !== main.self || slot.kind !== 'gpu'){
    return slot;
  }

  const card = (main.gpus || []).find(item => item.ordinal === (main.mainGpu >= 0 ? main.mainGpu : 0));
  const nowGb = Number(slot.limit || 0) > 0 ? Number(slot.limit) / gib : Number(card?.vram || 0) / gib;
  return {node: slot.node, kind: 'gpu', gb: Math.max(2, Math.floor(nowGb - roomGb)), percent: 0, keep: 0};
});

const plainSlots = slots => (slots || []).map(slot => ({node: slot.node, kind: slot.kind, gb: slot.gb, percent: slot.percent, keep: slot.keep}));

// una respuesta corta para que el motor reserve ya lo que pide en su primera respuesta;
// si la tarjeta se llena el motor se carga de nuevo con mas margen y se prueba otra vez
const warmUp = async (base) => {
  for(let attempt = 0; attempt < 4; attempt++){
    const reply = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: '', messages: [{role: 'user', content: 'Answer with the word OK.'}], max_tokens: 8, stream: false}),
      signal: AbortSignal.timeout(180000)
    }).catch(() => null);

    if(reply?.ok){
      return true;
    }

    await new Promise(done => setTimeout(done, 3000));
    await waitFor(async () => (await (await fetch(`${base}/api/pool`, {signal: AbortSignal.timeout(3000)})).json()).generation === 'ready', 300000, 2000);
  }

  return false;
};

const secondRoute = async () => {
  const routes = await readRoutes();

  if(!routes.some(item => item.kind === 'local' && new RegExp(`:${secondPort}(/|$)`).test(item.baseUrl))){
    await saveRoutes([...routes, {...routePresets.find(item => item.baseUrl.includes(`:${secondPort}`)), id: 'local-second'}]);
  }
};

router.get('/second', async (req, res) => {
  const pool = await askSecond('GET', '/api/pool').catch(() => null);
  const saved = await readSecond();
  const self = pool?.nodes?.find(item => item.self)?.stats || {};

  return res.status(200).json({
    running: Boolean(pool),
    starting: second.starting,
    step: second.step,
    port: secondPort,
    generation: pool?.generation || (second.starting ? 'starting' : 'off'),
    model: pool?.model?.name || '',
    path: pool?.model?.path || saved.model || '',
    gpu: saved.gpu ?? 0,
    first: saved.first || '',
    plan: second.plan || saved.plan || null,
    context: saved.context ?? 32768,
    madeRoom: Boolean(saved.restore) || Array.isArray(saved.restoreSlots),
    vramGb: Number(self.heldGpu || 0) / gib,
    ramGb: Number(self.heldRam || 0) / gib,
    error: second.error,
    log: await logTail(secondLog())
  });
});

// windows deja fijar para las tarjetas como mucho la mitad de la ram y lo comparten los dos motores:
// medido, IQ3_XXS sola fija 30 GB y un segundo modelo que quiso fijar otros 21 GB no pudo cargar en ninguna tarjeta,
// asi que la ram fijada y la ram se reparten antes de cargar y lo que no entra se lee del disco
export const planPair = ({firstBytes, secondBytes, firstVram, secondVram, totalRam = os.totalmem()}) => {
  const totalGb = totalRam / gib;
  // windows cuenta unos GB mas de lo que cada motor fija para expertos (medido: IQ3_S fija 35 y windows cuenta 39.7)
  const pinTotal = Math.max(8, Math.floor(totalGb / 2) - 6);
  const ramTotal = Math.max(16, Math.floor(totalGb) - 10);
  const needFirst = Math.max(1, Math.ceil(firstBytes / gib - firstVram));
  const needSecond = Math.max(1, Math.ceil(secondBytes / gib - secondVram));
  // el primero es el modelo principal: se queda con lo que necesita y al segundo le quedan al menos 12 GB fijados
  const pinFirst = Math.max(4, Math.min(needFirst, pinTotal - Math.min(12, needSecond)));
  const pinSecond = Math.max(2, Math.min(needSecond, pinTotal - pinFirst));
  const ramSecondFloor = Math.min(needSecond, Math.max(pinSecond + 4, 12));
  const ramFirst = Math.max(pinFirst + 2, Math.min(needFirst + 2, ramTotal - ramSecondFloor));
  const ramSecond = Math.max(pinSecond + 2, Math.min(needSecond + 2, ramTotal - ramFirst));
  return {pinFirst, pinSecond, ramFirst, ramSecond, pinTotal, ramTotal};
};

const modelEntry = async (file) => {
  const listed = await askEngine('GET', '/api/models', null, 10000).catch(() => null);
  const found = (listed?.models || []).find(item => item.path === file);

  if(found){
    return found;
  }

  return /\.gguf$/i.test(file) && fs.existsSync(file) ? {path: file, size: fs.statSync(file).size} : null;
};

const mainSlots = (main, change) => main.slots.map(slot => {
  if(slot.node !== main.self || !change[slot.kind]){
    return {node: slot.node, kind: slot.kind, gb: slot.gb, percent: slot.percent, keep: slot.keep};
  }

  return {node: slot.node, kind: slot.kind, gb: change[slot.kind], percent: 0, keep: 0};
});

router.post('/second/start', async (req, res) => {
  if(second.starting){
    return res.status(409).json({message: 'The models are already loading.'});
  }

  if(await askSecond('GET', '/api/pool').then(() => true).catch(() => false)){
    return res.status(409).json({message: 'Two models are already loaded, unload first.'});
  }

  const main = await askEngine('GET', '/api/pool', null, 5000).catch(() => null);

  if(!main){
    return res.status(503).json({message: 'The engine is not running.'});
  }

  const firstPath = String(req.body?.first?.path || main.model?.path || '');
  const modelPath = String(req.body?.second?.path || req.body?.path || '');
  const gpu = Number.isInteger(Number(req.body?.second?.gpu ?? req.body?.gpu)) ? Number(req.body?.second?.gpu ?? req.body?.gpu) : 0;
  const context = Math.min(262144, Math.max(4096, parseInt(req.body?.second?.context ?? req.body?.context, 10) || 32768));
  const masterExe = path.join(rootDir, 'master.exe');
  const mainGpu = main.mainGpu >= 0 ? main.mainGpu : 0;
  const card = (main.gpus || []).find(item => item.ordinal === mainGpu);
  const cardGb = Math.floor(Number(card?.vram || 0) / gib);
  const shared = gpu >= 0 && gpu === mainGpu;
  const secondOnly = Boolean(req.body?.secondOnly) && !shared && main.generation === 'ready' && firstPath === (main.model?.path || '');
  // en la misma tarjeta cada modelo tiene al menos 8 GB y entre los dos dejan 1 GB para windows
  const askedFirst = Number(req.body?.first?.gb) || 0;
  const askedSecond = Number(req.body?.second?.gb ?? req.body?.roomGb) || 0;
  const secondVram = shared ? Math.max(8, Math.min(cardGb - 9, askedSecond || 8)) : 0;
  const firstVram = shared ? Math.max(8, Math.min(cardGb - 1 - secondVram, askedFirst || cardGb - 1 - secondVram)) : 0;
  const firstModel = await modelEntry(firstPath);
  const secondModel = await modelEntry(modelPath);

  if(!firstModel || !secondModel){
    return res.status(400).json({message: 'Pick two models from the list of this PC.'});
  }

  if(!fs.existsSync(masterExe)){
    return res.status(404).json({message: 'master.exe not found, build the engine first.'});
  }

  if(shared && cardGb < 17){
    return res.status(400).json({message: `That card has ${cardGb} GB, two models need at least 8 GB each on it. Pick another card or the processor for the second model.`});
  }

  if(main.secondGpu && gpu >= 0 && gpu !== mainGpu && (main.helperGpu < 0 || main.helperGpu === gpu)){
    return res.status(400).json({message: 'The first model uses that card as a helper. Turn the helper card off in the Pool view or pick another card.'});
  }

  const plan = planPair({
    firstBytes: Number(firstModel.size || 0),
    secondBytes: Number(secondModel.size || 0),
    firstVram: shared ? firstVram : cardGb - 1,
    secondVram: gpu < 0 ? 0 : shared ? secondVram : Math.floor(Number((main.gpus || []).find(item => item.ordinal === gpu)?.vram || 0) / gib) - 1
  });
  const restore = secondOnly ? null : {slots: plainSlots(main.slots), pinGb: main.pinGb, vision: main.vision, model: main.model?.path || ''};
  const otherCardVision = main.vision && gpu >= 0 && gpu !== mainGpu && (main.visionGpu < 0 || main.visionGpu === gpu);

  second.starting = true;
  second.step = '';
  second.error = '';
  second.plan = {...plan, firstVram: shared ? firstVram : null, secondVram: shared ? secondVram : null, shared};
  res.status(202).json({starting: true, plan: second.plan});

  // si algo no llega a contestar se apaga el segundo y el primero vuelve a quedar como estaba
  const giveBack = async () => {
    await askSecond('POST', '/api/shutdown', null, 5000).catch(() => {});
    await waitFor(async () => !await askSecond('GET', '/api/monitor', null, 800).then(() => true).catch(() => false), 30000, 500);
    if(restore) await askEngine('POST', '/api/pool', {slots: restore.slots, pinGb: restore.pinGb, vision: restore.vision}, 20000).catch(() => {});
    await writeSecond({model: modelPath, gpu, context, restore: null});
    setLocalTurns(true);
  };

  try {
    await writeSecond({model: modelPath, first: firstPath, gpu, context, plan: second.plan, restore});

    if(!secondOnly){
      // primero el modelo principal con su parte, despues el segundo con lo que queda
      second.step = 'Loading the first model with its share';
      const settings = {slots: mainSlots(main, {...(shared ? {gpu: firstVram} : {}), ram: plan.ramFirst}), pinGb: plan.pinFirst};

      if(otherCardVision){
        settings.vision = false;
      }

      // otro modelo primero: se descarga el de ahora antes de cambiar los limites, si no se cargan dos seguidos y chocan
      // (medido: la recarga del de ahora y la carga del nuevo a la vez fallaron al leer los pesos a la video memoria)
      const changing = firstPath !== (main.model?.path || '');

      if(changing && main.generation !== 'off'){
        await askEngine('POST', '/api/models/unload', null, 180000);
      }

      await askEngine('POST', '/api/pool', settings, 20000);

      if(changing || main.generation !== 'ready'){
        await askEngine('POST', '/api/models/load', {path: firstPath}, 30000);
      }

      await waitFor(async () => (await askEngine('GET', '/api/pool', null, 3000)).generation === 'loading', 15000, 300);
      await waitFor(async () => ['ready', 'failed'].includes((await askEngine('GET', '/api/pool', null, 3000)).generation), 900000, 2000);

      if((await askEngine('GET', '/api/pool', null, 3000)).generation !== 'ready'){
        throw new Error('The first model could not load with its share, look at master.log');
      }

      // medido con IQ3_S y IQ1_M: si el segundo carga antes de la primera respuesta del primero, el primero ya no tiene donde
      // reservar lo suyo y todas sus respuestas fallan sin memoria, asi que primero contesta y despues se mide lo libre
      second.step = 'The first model gets ready';
      await warmUp(await engineBase());
    }

    // al segundo le toca lo que el primero de verdad dejo bajo la mitad de la ram: medido, windows le cuenta al primero
    // unos 4.7 GB mas que lo que fija para expertos (35 → 39.7, 29 → 33.7) y al segundo menos de 1 GB mas;
    // el contador de windows suma tarjetas repetidas (una pantalla virtual copia la 3090) y no sirve para esto
    const loaded = await askEngine('GET', '/api/pool', null, 5000).catch(() => null);
    const pinnedFirst = Number(loaded?.pinned || 0) / gib;
    const needSecond = Math.max(1, Math.ceil(Number(secondModel.size || 0) / gib - (shared ? secondVram : 0)));

    if(pinnedFirst > 0){
      plan.pinSecond = Math.max(2, Math.min(needSecond, Math.floor(os.totalmem() / 2 / gib - pinnedFirst - 7)));
    }

    plan.ramSecond = Math.max(plan.pinSecond + 2, Math.min(plan.ramSecond + 8, Math.floor(os.freemem() / gib - 8)));
    second.plan = {...second.plan, pinSecond: plan.pinSecond, ramSecond: plan.ramSecond, pinnedFirst: Math.round(pinnedFirst * 10) / 10};
    await writeSecond({model: modelPath, first: firstPath, gpu, context, plan: second.plan, restore});

    const pool = {
      acceptRequests: false,
      doorPort: secondDoor,
      apiListen: 'local',
      backend: gpu < 0 ? 'cpu' : 'auto',
      mainGpu: gpu < 0 ? -1 : gpu,
      secondGpu: false,
      helperGpu: -1,
      vision: false,
      context,
      pinGb: plan.pinSecond,
      model: modelPath,
      folder: path.dirname(modelPath),
      slots: [
        {node: 'this-pc', kind: 'gpu', gb: shared ? secondVram : 0, percent: 0, keep: 0},
        {node: 'this-pc', kind: 'ram', gb: plan.ramSecond, percent: 0, keep: 0},
        {node: 'this-pc', kind: 'ssd', gb: 64, percent: 0, keep: 0}
      ]
    };

    await fs.promises.writeFile(path.join(rootDir, secondPool), JSON.stringify(pool, null, 2), 'utf8');
    second.step = 'Starting the second engine';
    const log = fs.openSync(secondLog(), 'w');
    const child = spawn(masterExe, [String(secondPort), '-', '-', path.dirname(modelPath), modelPath, '--pool', secondPool], {cwd: rootDir, detached: true, stdio: ['ignore', log, log], windowsHide: true});
    child.unref();
    fs.closeSync(log);
    second.pid = child.pid;

    if(!await waitFor(async () => Boolean(await askSecond('GET', '/api/pool')), 60000)){
      throw new Error('The second engine did not start, look at master.second.log');
    }

    await secondRoute();
    second.step = 'Loading the second model';
    await waitFor(async () => ['ready', 'failed'].includes((await askSecond('GET', '/api/pool')).generation), 900000, 2000);
    const state = await askSecond('GET', '/api/pool').catch(() => null);

    if(state?.generation !== 'ready'){
      throw new Error('The second model could not load with what is left, look at master.second.log. Give it another card, a smaller model or less context.');
    }

    // los dos contestan una vez antes de decir que esta listo, asi una falta de memoria se ve aca y no en el primer chat
    second.step = 'Checking that both models answer';
    const secondAnswers = await warmUp(`http://127.0.0.1:${secondPort}`);
    const firstAnswers = await warmUp(await engineBase());

    if(!secondAnswers || !firstAnswers){
      throw new Error('The two models do not fit together on this PC. Give the second model another card, a smaller model or less context.');
    }

    // en la misma tarjeta o en el procesador se turnan, en tarjetas distintas escriben a la vez
    setLocalTurns(shared || gpu < 0);
  } catch(e) {
    second.error = String(e.message || e);
    await giveBack();
  } finally {
    second.starting = false;
    second.step = '';
  }
});

// si la app se cerro con dos modelos cargados, el primero quedo con su parte guardada en pool.json;
// al volver y no estar el segundo, el primero recupera su configuracion sola (devuelve false si el motor no esta)
export const healPair = async () => {
  const saved = await readSecond();

  if(!saved.restore || second.starting){
    return true;
  }

  if(await askSecond('GET', '/api/pool').then(() => true).catch(() => false)){
    return true;
  }

  const main = await askEngine('GET', '/api/pool', null, 3000).catch(() => null);

  if(!main){
    return false;
  }

  await askEngine('POST', '/api/pool', {slots: saved.restore.slots, pinGb: saved.restore.pinGb, vision: saved.restore.vision}, 20000).catch(() => {});
  await writeSecond({...saved, restore: null});
  setLocalTurns(true);
  console.log('the first model got its own settings back, the second model was not running anymore');
  return true;
};

// apaga el segundo y el primero vuelve a quedar como estaba, con toda la tarjeta y su ram
router.post('/second/stop', async (req, res) => {
  const saved = await readSecond();
  await askSecond('POST', '/api/shutdown', null, 5000).catch(() => {});
  const gone = await waitFor(async () => !await askSecond('GET', '/api/monitor', null, 800).then(() => true).catch(() => false), 30000, 500);

  if(!gone && second.pid){
    try {
      process.kill(second.pid);
    } catch(e) {
    }
  }

  second.pid = 0;
  second.error = '';
  const restore = saved.restore || (Array.isArray(saved.restoreSlots) ? {slots: saved.restoreSlots} : null);

  if(restore){
    const back = {slots: restore.slots};

    if(restore.pinGb !== undefined) back.pinGb = restore.pinGb;
    if(restore.vision !== undefined) back.vision = restore.vision;

    await askEngine('POST', '/api/pool', back, 20000).catch(e => {
      second.error = `The first model could not get its settings back: ${e.message}`;
    });
  }

  await writeSecond({...saved, restore: null, restoreSlots: null});
  setLocalTurns(true);
  return res.status(200).json({stopped: true, restored: Boolean(restore)});
});

export default router;
