import express from 'express';
import http from 'http';
import dgram from 'dgram';
import { execFile } from 'child_process';
import os from 'os';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import QRCode from 'qrcode';
import { paths } from './config.js';
import { cleanText, pathExists } from './storage.js';

// el celular entra por su propio puerto y solo desde la red de la casa o una vpn, cada celular con su llave
const phoneFile = () => path.join(paths.dataDir, 'phone.json');
const defaultPort = 3010;
const cookieName = 'sneo_phone';
const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const codeLife = 10 * 60 * 1000;

let current = null;
let pairing = null;
const attempts = new Map();

const cleanPort = value => {
  const port = parseInt(value, 10);
  return Number.isInteger(port) && port >= 1024 && port <= 65535 ? port : defaultPort;
};

export const readPhone = async () => {
  try {
    const data = JSON.parse(await fs.readFile(phoneFile(), 'utf8'));
    return {
      enabled: data.enabled === true,
      port: cleanPort(data.port),
      devices: Array.isArray(data.devices) ? data.devices.filter(item => item && item.id && item.tokenHash) : []
    };
  } catch(e) {
    return {enabled: false, port: defaultPort, devices: []};
  }
};

const savePhone = async (data) => {
  await fs.mkdir(paths.dataDir, {recursive: true});
  await fs.writeFile(phoneFile(), JSON.stringify(data, null, 2), 'utf8');
};

const hashToken = token => crypto.createHash('sha256').update(String(token)).digest('hex');

// solo direcciones de la red de la casa, de esta pc o de una vpn como tailscale
export const privateAddress = (address) => {
  const ip = String(address || '').replace(/^::ffff:/i, '');

  if(ip === '127.0.0.1' || ip === '::1'){
    return true;
  }

  const parts = ip.split('.');

  if(parts.length === 4 && parts.every(part => /^\d{1,3}$/.test(part) && Number(part) <= 255)){
    const [a, b] = parts.map(Number);
    return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
  }

  return /^f[cd][0-9a-f]{2}:/i.test(ip) || /^fe80:/i.test(ip);
};

// virtualbox vmware y hyper-v crean redes que el celular no ve, van al final
const virtualCards = /^(0a:00:27|08:00:27|00:50:56|00:0c:29|00:05:69|00:1c:14|00:15:5d)/i;
let routeAddress = '';

// la direccion que windows usa para salir a internet es la del wifi o el cable de la casa, se pregunta sin mandar nada
const findRoute = () => new Promise(done => {
  const probe = dgram.createSocket('udp4');
  const finish = () => {
    try {
      probe.close();
    } catch(e) {
    }

    done(routeAddress);
  };

  probe.on('error', finish);
  probe.connect(53, '1.1.1.1', () => {
    try {
      routeAddress = probe.address().address;
    } catch(e) {
    }

    finish();
  });
  setTimeout(finish, 500).unref?.();
});

export const phoneAddresses = () => {
  const cards = Object.values(os.networkInterfaces())
    .flat()
    .filter(item => item && item.family === 'IPv4' && !item.internal && privateAddress(item.address));
  // 169.254 es un cable directo entre dos pcs (el del pool) y el celular no lo ve, va al final como las redes virtuales
  const rank = item => item.address === routeAddress ? 0 : /^100\./.test(item.address) ? 2 : virtualCards.test(item.mac || '') || /^169\.254\./.test(item.address) ? 3 : 1;
  return cards.sort((a, b) => rank(a) - rank(b)).map(item => item.address);
};

const newCode = () => Array.from({length: 8}, () => letters[crypto.randomInt(letters.length)]).join('');

const sameText = (a, b) => {
  const left = Buffer.from(String(a));
  const right = Buffer.from(String(b));
  return left.length === right.length && crypto.timingSafeEqual(left, right);
};

const tooMany = (ip) => {
  const now = Date.now();
  const recent = (attempts.get(ip) || []).filter(at => now - at < codeLife);
  recent.push(now);
  attempts.set(ip, recent);
  return recent.length > 10;
};

const cookieOf = (req) => {
  const pair = String(req.headers.cookie || '').split(/;\s*/).find(item => item.startsWith(`${cookieName}=`));
  return pair ? pair.slice(cookieName.length + 1) : '';
};

export const pairedDevice = async (req) => await findDevice(req);

const findDevice = async (req) => {
  const token = cookieOf(req);

  if(!/^[a-f0-9]{64}$/.test(token)){
    return null;
  }

  const hash = hashToken(token);
  const state = await readPhone();
  return state.devices.find(item => sameText(item.tokenHash, hash)) || null;
};

// lo unico que el celular puede usar: chatear, aprobar, los agentes y ver o cargar modelos
const phoneRoutes = [
  ['GET', /^\/api\/sneo$/],
  ['GET', /^\/api\/settings$/],
  ['GET', /^\/api\/chats$/],
  ['POST', /^\/api\/chats$/],
  ['PUT', /^\/api\/chats\/[\w-]+$/],
  ['DELETE', /^\/api\/chats\/[\w-]+$/],
  ['GET', /^\/api\/projects$/],
  ['POST', /^\/api\/chat\/stream$/],
  ['POST', /^\/api\/chat\/(stop|steer)$/],
  ['GET', /^\/api\/pending$/],
  ['POST', /^\/api\/pending\/[\w-]+\/(approve|deny)$/],
  ['GET', /^\/api\/team\/(routes|agents)$/],
  ['PUT', /^\/api\/team\/agents$/],
  ['POST', /^\/api\/team\/(chat|run)$/],
  ['GET', /^\/api\/engine\/(monitor|files|pool|second)$/],
  ['POST', /^\/api\/engine\/files\/(load|stop)$/],
  ['POST', /^\/api\/engine\/second\/(start|stop)$/]
];

export const phoneAllows = (method, url) => {
  const route = String(url || '').split('?')[0];
  return phoneRoutes.some(([verb, pattern]) => verb === method && pattern.test(route));
};

const manifest = {
  name: 'Sneo',
  short_name: 'Sneo',
  start_url: '/',
  scope: '/',
  display: 'standalone',
  background_color: '#131316',
  theme_color: '#131316',
  icons: [
    {src: '/phone-icon-256.png', sizes: '256x256', type: 'image/png'},
    {src: '/phone-icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any'}
  ]
};

const iconFiles = {
  '/phone-icon-256.png': path.join(paths.appRoot, 'src-tauri', 'icons', '128x128@2x.png'),
  '/phone-icon-512.png': path.join(paths.appRoot, 'src-tauri', 'icons', 'icon.png')
};

const buildPhoneApp = ({apiRoutes, engineRoutes, port}) => {
  const app = express();
  app.disable('x-powered-by');

  app.use((req, res, next) => {
    if(!privateAddress(req.socket.remoteAddress)){
      return res.status(403).send('Forbidden');
    }

    const host = String(req.headers.host || '').toLowerCase();
    const allowed = new Set([`127.0.0.1:${port}`, `localhost:${port}`, ...phoneAddresses().map(address => `${address}:${port}`)]);

    if(!allowed.has(host)){
      return res.status(403).send('Forbidden');
    }

    if(req.headers.origin && req.headers.origin !== `http://${host}`){
      return res.status(403).json({message: 'Forbidden'});
    }

    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    return next();
  });

  app.use(express.json({limit: '25mb'}));

  app.get('/phone/me', async (req, res) => {
    const device = await findDevice(req);
    return res.status(200).json({paired: Boolean(device), name: device?.name || ''});
  });

  app.post('/phone/pair', async (req, res) => {
    const ip = String(req.socket.remoteAddress || '');

    if(tooMany(ip)){
      return res.status(429).json({message: 'Too many tries, wait a few minutes.'});
    }

    const code = cleanText(req.body?.code).toUpperCase().replace(/[^A-Z0-9]/g, '');

    if(!pairing || Date.now() > pairing.expiresAt || !sameText(code, pairing.code)){
      return res.status(403).json({message: 'That code is wrong or expired. Make a new one on the PC.'});
    }

    pairing = null;
    const token = crypto.randomBytes(32).toString('hex');
    const state = await readPhone();
    const device = {
      id: crypto.randomUUID(),
      name: cleanText(req.body?.name).slice(0, 40) || 'Phone',
      tokenHash: hashToken(token),
      createdAt: new Date().toISOString(),
      lastSeen: new Date().toISOString()
    };
    state.devices = [...state.devices, device].slice(-12);
    await savePhone(state);
    res.setHeader('Set-Cookie', `${cookieName}=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000`);
    return res.status(200).json({paired: true, name: device.name});
  });

  app.post('/phone/forget', async (req, res) => {
    const device = await findDevice(req);

    if(device){
      const state = await readPhone();
      state.devices = state.devices.filter(item => item.id !== device.id);
      await savePhone(state);
    }

    res.setHeader('Set-Cookie', `${cookieName}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0`);
    return res.status(200).json({paired: false});
  });

  app.get('/manifest.webmanifest', (req, res) => {
    res.setHeader('Content-Type', 'application/manifest+json');
    return res.status(200).send(JSON.stringify(manifest));
  });

  app.get(Object.keys(iconFiles), (req, res) => res.sendFile(iconFiles[req.path]));

  const sendPage = async (req, res) => {
    const page = path.join(paths.distDir, 'm.html');

    if(!(await pathExists(page))){
      return res.status(404).send('The phone page is not built yet. Run npm.cmd run build on the PC.');
    }

    res.setHeader('Cache-Control', 'no-cache');
    return res.sendFile(page);
  };

  app.get('/', sendPage);
  app.use('/assets', express.static(path.join(paths.distDir, 'assets'), {index: false}));

  let seenAt = 0;

  app.use('/api', async (req, res, next) => {
    const device = await findDevice(req);

    if(!device){
      return res.status(401).json({message: 'This phone is not paired, scan the code on the PC again.'});
    }

    if(!phoneAllows(req.method, req.originalUrl)){
      return res.status(403).json({message: 'This only works on the PC.'});
    }

    if(Date.now() - seenAt > 60000){
      seenAt = Date.now();
      const state = await readPhone();
      const found = state.devices.find(item => item.id === device.id);

      if(found){
        found.lastSeen = new Date().toISOString();
        await savePhone(state);
      }
    }

    return next();
  });

  app.use('/api/engine', engineRoutes);
  app.use('/api', apiRoutes);
  app.use((req, res) => res.status(404).send('Not found'));
  return app;
};

const stopListening = async () => {
  if(!current){
    return;
  }

  const server = current.server;
  current = null;
  server.closeAllConnections?.();
  await new Promise(done => server.close(() => done()));
};

// arranca o apaga el puerto del celular segun lo guardado
export const applyPhone = async (routers) => {
  const state = await readPhone();

  if(current && (!state.enabled || current.port !== state.port)){
    await stopListening();
  }

  if(!state.enabled || current){
    return {listening: Boolean(current), problem: ''};
  }

  const app = buildPhoneApp({...routers, port: state.port});
  const server = http.createServer(app);
  findRoute();

  return await new Promise(done => {
    server.once('error', error => {
      done({listening: false, problem: error.code === 'EADDRINUSE' ? `Port ${state.port} is taken by another program.` : error.message});
    });
    server.listen(state.port, process.env.SNEO_PHONE_HOST || '0.0.0.0', () => {
      current = {server, port: state.port};
      console.log(`Phone page on port ${state.port} for this network: ${phoneAddresses().map(address => `http://${address}:${state.port}`).join(' ') || 'no network found'}`);
      done({listening: true, problem: ''});
    });
  });
};

const phoneLinks = (port) => phoneAddresses().map(address => `http://${address}:${port}/`);

// windows tira lo que llega a un puerto sin una regla que lo deje pasar: el celular abre el link y la pagina nunca carga
const firewallRule = 'Sneo phone';

const runQuiet = (file, args) => new Promise(done => {
  execFile(file, args, {windowsHide: true, timeout: 120000}, (error, stdout) => done({ok: !error, out: String(stdout || '')}));
});

export const firewallOpen = async (port) => {
  if(process.platform !== 'win32'){
    return true;
  }

  const found = await runQuiet('netsh', ['advfirewall', 'firewall', 'show', 'rule', `name=${firewallRule}`]);
  return found.ok && found.out.includes(String(port));
};

// la regla solo deja entrar a este puerto en redes privadas como el wifi de la casa y windows pide permiso con su ventana
const openFirewall = async (port) => {
  await runQuiet('netsh', ['advfirewall', 'firewall', 'delete', 'rule', `name=${firewallRule}`]);
  const command = `Start-Process netsh -Verb RunAs -WindowStyle Hidden -Wait -ArgumentList 'advfirewall firewall add rule name="${firewallRule}" dir=in action=allow protocol=TCP localport=${port} profile=private'`;
  await runQuiet('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command]);
  return await firewallOpen(port);
};

const describe = async (problem = '') => {
  const state = await readPhone();

  if(!routeAddress){
    await findRoute();
  }

  const links = phoneLinks(state.port);
  const code = pairing && Date.now() < pairing.expiresAt ? pairing : null;
  const target = code && links[0] ? `${links[0]}#code=${code.code}` : links[0] || '';

  return {
    enabled: state.enabled,
    listening: Boolean(current),
    port: state.port,
    links,
    problem,
    code: code ? {code: code.code, expiresAt: code.expiresAt} : null,
    qr: target ? `data:image/svg+xml;base64,${Buffer.from(await QRCode.toString(target, {type: 'svg', margin: 1, color: {dark: '#000000', light: '#ffffff'}})).toString('base64')}` : '',
    target,
    firewall: state.enabled ? await firewallOpen(state.port) : true,
    devices: state.devices.map(item => ({id: item.id, name: item.name, createdAt: item.createdAt, lastSeen: item.lastSeen}))
  };
};

// lo que la ventana de la pc usa para prender el acceso, sacar un codigo y quitar celulares
export const phoneAdmin = (routers) => {
  const router = express.Router();

  router.get('/', async (req, res) => res.status(200).json(await describe()));

  router.post('/', async (req, res) => {
    const state = await readPhone();

    if(req.body?.enabled !== undefined){
      state.enabled = req.body.enabled === true;
    }

    if(req.body?.port !== undefined){
      state.port = cleanPort(req.body.port);
    }

    await savePhone(state);
    const result = await applyPhone(routers);

    if(state.enabled && result.listening && !pairing){
      pairing = {code: newCode(), expiresAt: Date.now() + codeLife};
    }

    return res.status(200).json(await describe(result.problem));
  });

  router.post('/code', async (req, res) => {
    pairing = {code: newCode(), expiresAt: Date.now() + codeLife};
    return res.status(200).json(await describe());
  });

  router.post('/firewall', async (req, res) => {
    const state = await readPhone();
    const open = await openFirewall(state.port);
    return res.status(200).json(await describe(open ? '' : 'Windows did not add the rule: accept its permission window to let the phone in.'));
  });

  router.delete('/devices/:id', async (req, res) => {
    const state = await readPhone();
    state.devices = state.devices.filter(item => item.id !== req.params.id);
    await savePhone(state);
    return res.status(200).json(await describe());
  });

  return router;
};
