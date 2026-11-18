import express from 'express';
import http from 'http';
import os from 'os';
import crypto from 'crypto';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { clientPort, defaultModel, paths, port } from './config.js';
import { pathExists, startData } from './storage.js';
import apiRoutes from './routes.js';
import engineRoutes, { healPair, readAllowNetwork } from './engine.js';
import { applyPhone, pairedDevice, phoneAdmin } from './phone.js';
import trainRoutes from './train.js';

dotenv.config();

process.on('uncaughtException', (err) => {
  console.error('uncaughtException (server keeps running):', err?.message || err);
});

process.on('unhandledRejection', (err) => {
  console.error('unhandledRejection (server keeps running):', err?.message || err);
});

const app = express();
const server = http.createServer(app);
const allowNetwork = await readAllowNetwork();
const host = allowNetwork ? '0.0.0.0' : '127.0.0.1';
const ownAddresses = Object.values(os.networkInterfaces()).flat().filter(item => item.family === 'IPv4').map(item => item.address);

app.set('sneoHost', host);

// una pagina de internet no puede usar el servidor: el host tiene que ser esta pc, el origen la ventana de sneo y la cookie la de esta sesion
const session = crypto.randomBytes(32).toString('hex');
const hosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`]);
const origins = new Set([`http://127.0.0.1:${port}`, `http://localhost:${port}`]);
const devOrigins = new Set([`http://127.0.0.1:${clientPort}`, `http://localhost:${clientPort}`]);

if(allowNetwork){
  for(const address of ownAddresses){
    hosts.add(`${address}:${port}`);
    origins.add(`http://${address}:${port}`);
    devOrigins.add(`http://${address}:${clientPort}`);
  }
}

const sessionOf = (req) => {
  const pair = String(req.headers.cookie || '').split(/;\s*/).find(item => item.startsWith('sneo_session='));
  return pair ? pair.slice('sneo_session='.length) : '';
};

const sameSession = (value) => value.length === session.length && crypto.timingSafeEqual(Buffer.from(value), Buffer.from(session));

const loopback = (address) => ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(String(address || ''));

app.use(async (req, res, next) => {
  if(!hosts.has(String(req.headers.host || '').toLowerCase())){
    return res.status(403).send('Forbidden');
  }

  const origin = req.headers.origin;

  if(origin && !origins.has(origin) && !devOrigins.has(origin)){
    return res.status(403).json({message: 'Forbidden'});
  }

  // otro equipo de la red solo entra si antes se emparejo con el codigo de la pagina del celular
  if(!loopback(req.socket.remoteAddress)){
    if(await pairedDevice(req)){
      return next();
    }

    return req.path.startsWith('/api/')
      ? res.status(403).json({message: 'Pair this device first: open Phone in Sneo on the PC and scan the code.'})
      : res.status(403).send('Pair this device first: open Phone in Sneo on the PC and scan the code.');
  }

  // la ventana manda su cookie y los programas de esta pc la cabecera x-sneo, que una pagina de otro origen no puede poner sin permiso
  if(!req.path.startsWith('/api/') || req.path === '/api/sneo' || (origin && devOrigins.has(origin)) || sameSession(sessionOf(req)) || (!origin && req.headers['x-sneo'])){
    return next();
  }

  return res.status(403).json({message: 'Open SNEO from its own window to use it.'});
});

app.use(express.json({limit: '25mb'}));

app.use(cors({
  origin: (origin, cb) => cb(null, !origin || origins.has(origin) || devOrigins.has(origin)),
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  credentials: true
}));

const sendPage = async (req, res) => {
  const indexPath = path.join(paths.distDir, 'index.html');

  if(!(await pathExists(indexPath))){
    return res.status(404).send('Frontend not built. Run npm.cmd run build first.');
  }

  if(loopback(req.socket.remoteAddress)){
    res.setHeader('Set-Cookie', `sneo_session=${session}; HttpOnly; SameSite=Strict; Path=/`);
  }

  return res.sendFile(indexPath);
};

app.get('/', sendPage);
app.use(express.static(paths.distDir, {index: false}));
app.get('/api/sneo', (req, res) => res.status(200).json({sneo: true}));
app.use('/api/phone', phoneAdmin({apiRoutes, engineRoutes}));
app.use('/api/engine', engineRoutes);
app.use('/api/train', trainRoutes);
app.use('/api', apiRoutes);
app.get(/.*/, sendPage);

(async () => {
  try {
    await startData();
    console.log(`Data: ${paths.dataDir}`);
    console.log(`Projects: ${paths.projectsRoot}`);
  } catch(err) {
    console.error('Error preparando data:', err.message);
  }
})();

server.on('error', (error) => {
  if(error.code === 'EADDRINUSE'){
    console.error(`FATAL: port ${port} already in use. Kill the other server first. Exiting.`);
    process.exit(1);
  }

  console.error('Server error:', error.message);
  process.exit(1);
});

server.listen(port, host, () => {
  console.log(allowNetwork ? `Server running on http://0.0.0.0:${port}, other devices on the network can open SNEO` : `Server running on http://localhost:${port}, only this PC can open SNEO`);
  console.log(`Default model: ${defaultModel}`);
  applyPhone({apiRoutes, engineRoutes}).then(result => {
    if(result.problem) console.log(`The phone page could not start: ${result.problem}`);
  });

  // espera hasta dos minutos a que el motor arranque para devolverle su configuracion si quedo a medias
  let tries = 0;
  const heal = async () => {
    const done = await healPair().catch(() => false);
    if(!done && tries++ < 24) setTimeout(heal, 5000);
  };
  setTimeout(heal, 3000);
});
