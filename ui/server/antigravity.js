import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { flattenChat, toolsText } from './claudecode.js';
import { sessionPart, keepSession, dropSession } from './textsession.js';

// antigravity instalado en esta pc: usa la cuenta de google del usuario sin api key, igual que claude code
let found = null;

const findAgy = () => {
  if(found && fs.existsSync(found)){
    return found;
  }

  const home = os.homedir();
  const places = [
    path.join(home, '.gemini', 'bin', 'agy.exe'),
    path.join(home, '.gemini', 'bin', 'agy')
  ];

  for(const dir of String(process.env.PATH || '').split(path.delimiter)){
    if(dir) places.push(path.join(dir, process.platform === 'win32' ? 'agy.exe' : 'agy'));
  }

  found = places.find(file => file && fs.existsSync(file)) || null;
  return found;
};

// carpeta vacia propia: sus herramientas no llegan a los archivos del usuario y las de sneo si
const emptyHome = () => {
  const home = path.join(os.tmpdir(), 'sneo-antigravity');
  fs.mkdirSync(home, {recursive: true});
  return home;
};

const plainModel = 'Another program uses you as a plain text model. Your own tools (files, terminal, browser) do not reach the user files, never use them. When the instructions below describe tools, call them only the way they describe, as text in your answer.';
const noOwnTools = 'Your last try used your own tools and gave no answer. Do not use them, answer with text only.';

const agyArgs = (route, conversation = '') => [
  '--input-format', 'stream-json', '--output-format', 'stream-json', '--disable-slash-commands',
  ...(route.model ? ['--model', route.model] : []),
  ...(conversation ? ['--conversation', conversation] : [])
];

const resultOf = (line) => {
  try {
    const event = JSON.parse(line);
    return event?.event === 'result' ? event.result || {} : null;
  } catch(e) {
    return null;
  }
};

// el texto va por la entrada como un mensaje stream-json: la linea de comandos de windows corta en 32k
const runAgy = (route, text, run) => new Promise((resolve, reject) => {
  const exe = findAgy();

  if(!exe){
    return reject(new Error('Antigravity is not installed on this PC. Install it and sign in once'));
  }

  const child = spawn(exe, agyArgs(route), {cwd: emptyHome(), windowsHide: true});
  let out = '';
  let err = '';
  const stop = () => child.kill();
  run?.signal?.addEventListener?.('abort', stop, {once: true});
  const timer = setTimeout(stop, 10 * 60 * 1000);

  child.stdout.on('data', chunk => out += chunk);
  child.stderr.on('data', chunk => err += chunk);
  child.on('error', reject);
  child.on('close', code => {
    clearTimeout(timer);
    run?.signal?.removeEventListener?.('abort', stop);
    if(run?.signal?.aborted) return reject(new Error('Stopped'));
    const result = out.split(/\r?\n/).map(resultOf).filter(Boolean).pop();
    if(!result) return reject(new Error((err || out).trim().slice(-400) || `agy exited with ${code}`));
    resolve(result);
  });
  child.stdin.end(JSON.stringify({event: 'user', message: {content: text}}) + '\n');
});

// un antigravity abierto por panel: cada mensaje nuevo entra por su entrada y la conversacion sigue en el mismo proceso
// medido: un turno desde cero 6.4 s y el siguiente en el mismo proceso 3.9 s con 9.5k tokens de cache
const live = new Map();
const idleClose = 10 * 60 * 1000;

const closeLive = (key) => {
  const item = live.get(key);

  if(!item){
    return;
  }

  live.delete(key);
  clearTimeout(item.timer);
  item.waiting?.reject(new Error('Stopped'));
  item.waiting = null;

  try {
    item.child.stdin.end();
  } catch(e) {}

  setTimeout(() => {
    if(item.child.exitCode === null) item.child.kill();
  }, 3000);
};

const openLive = (key, route, conversation) => {
  const exe = findAgy();

  if(!exe){
    throw new Error('Antigravity is not installed on this PC. Install it and sign in once');
  }

  const child = spawn(exe, agyArgs(route, conversation), {cwd: emptyHome(), windowsHide: true});
  const item = {child, route: route.id, model: route.model || '', buf: '', err: '', waiting: null, timer: null, total: null};
  const fail = (error) => {
    if(live.get(key) === item) live.delete(key);
    clearTimeout(item.timer);
    item.waiting?.reject(error);
    item.waiting = null;
  };

  child.stdout.on('data', chunk => {
    item.buf += chunk;
    let at;

    while((at = item.buf.indexOf('\n')) >= 0){
      const result = resultOf(item.buf.slice(0, at));
      item.buf = item.buf.slice(at + 1);
      if(!result || !item.waiting) continue;
      const waiting = item.waiting;
      item.waiting = null;
      waiting.resolve(result);
    }
  });
  child.stderr.on('data', chunk => item.err = (item.err + chunk).slice(-2000));
  child.on('error', fail);
  child.on('close', code => fail(new Error(item.err.trim().slice(-400) || `agy exited with ${code}`)));
  live.set(key, item);
  return item;
};

const askLive = (key, item, text, run) => new Promise((resolve, reject) => {
  clearTimeout(item.timer);
  const started = Date.now();
  const stop = () => {
    reject(new Error('Stopped'));
    closeLive(key);
  };
  const timer = setTimeout(stop, 10 * 60 * 1000);
  const done = () => {
    clearTimeout(timer);
    run?.signal?.removeEventListener?.('abort', stop);
  };

  if(run?.signal?.aborted){
    return stop();
  }

  run?.signal?.addEventListener?.('abort', stop, {once: true});
  item.waiting = {
    resolve: result => {
      done();
      item.timer = setTimeout(() => closeLive(key), idleClose);
      // en un proceso abierto los tokens vienen sumados de todos los turnos: se resta lo que ya se habia contado
      const total = result.usage || {};
      const before = item.total || {};
      item.total = total;
      const turn = {
        input_tokens: Math.max(0, (total.input_tokens || 0) - (before.input_tokens || 0)),
        output_tokens: Math.max(0, (total.output_tokens || 0) - (before.output_tokens || 0)),
        cache_read_tokens: Math.max(0, (total.cache_read_tokens || 0) - (before.cache_read_tokens || 0))
      };
      resolve({...result, usage: turn, spentMs: Date.now() - started});
    },
    reject: error => {
      done();
      reject(error);
    }
  };
  item.child.stdin.write(JSON.stringify({event: 'user', message: {content: text}}) + '\n');
});

const replyOf = (result, run) => {
  if(result.status !== 'SUCCESS'){
    throw new Error(String(result.error || 'Antigravity failed').slice(0, 400));
  }

  const usage = result.usage || {};
  if(run?.usage){
    // en un proceso abierto duration_seconds suma todos los turnos: se usa lo medido aca
    const spent = Number(result.spentMs || 0) || Number(result.duration_seconds || 0) * 1000;
    run.usage.rounds++;
    run.usage.inputTokens += (usage.input_tokens || 0) + (usage.cache_read_tokens || 0);
    run.usage.cachedTokens += usage.cache_read_tokens || 0;
    run.usage.outputTokens += usage.output_tokens || 0;
    run.usage.outputSpeed = spent > 0 && usage.output_tokens ? Math.round(usage.output_tokens / (spent / 1000) * 10) / 10 : run.usage.outputSpeed;
  }
  return {
    content: String(result.response || '').trim(),
    reasoning: '',
    tool_calls: [],
    finishReason: 'stop',
    usage: {
      prompt_tokens: (usage.input_tokens || 0) + (usage.cache_read_tokens || 0),
      completion_tokens: usage.output_tokens || 0
    }
  };
};

const askInPanel = async (route, messages, tools, run) => {
  const key = `agy|${run.sessionKey}`;
  const part = sessionPart(key, route, messages);
  let item = live.get(key);

  if(part.record && item && (item.route !== route.id || item.model !== (route.model || ''))){
    closeLive(key);
    item = null;
  }

  if(!part.record){
    closeLive(key);
    dropSession(key);
    item = null;
  }

  const {system, prompt} = flattenChat(messages);
  const text = part.record ? part.prompt : [plainModel, system, toolsText(tools), prompt].filter(Boolean).join('\n\n---\n\n');

  try {
    item = item || openLive(key, route, part.record?.conversation || '');
    let result = await askLive(key, item, text, run);

    // a veces igual prueba sus herramientas y termina sin texto: se le recuerda en la misma conversacion
    if(result.status === 'SUCCESS' && !String(result.response || '').trim()){
      result = await askLive(key, item, noOwnTools, run);
    }

    const reply = replyOf(result, run);
    keepSession(key, route, part.system, {conversation: result.conversation_id || part.record?.conversation || '', reply: reply.content});
    return reply;
  } catch(e) {
    closeLive(key);
    dropSession(key);
    throw e;
  }
};

export const askAntigravity = async (route, messages, tools, run) => {
  if(run?.sessionKey){
    return await askInPanel(route, messages, tools, run);
  }

  const {system, prompt} = flattenChat(messages);
  const text = [plainModel, system, toolsText(tools), prompt].filter(Boolean).join('\n\n---\n\n');
  let result = await runAgy(route, text, run);

  // a veces igual prueba sus herramientas y termina sin texto: una vuelta mas recordandoselo
  if(result.status === 'SUCCESS' && !String(result.response || '').trim()){
    result = await runAgy(route, `${text}\n\n---\n\n${noOwnTools}`, run);
  }

  return replyOf(result, run);
};

// los procesos abiertos se cierran con el servidor
process.on('exit', () => {
  for(const item of live.values()){
    try {
      item.child.kill();
    } catch(e) {}
  }
});

// los modelos que da la cuenta: agy models los lista uno por linea con el id y el nombre separados por tab
export const antigravityModels = () => new Promise(resolve => {
  const exe = findAgy();

  if(!exe){
    return resolve([]);
  }

  const child = spawn(exe, ['models'], {cwd: emptyHome(), windowsHide: true});
  let out = '';
  const timer = setTimeout(() => child.kill(), 60000);
  child.stdout.on('data', chunk => out += chunk);
  child.on('error', () => resolve([]));
  child.on('close', () => {
    clearTimeout(timer);
    resolve(out.split(/\r?\n/)
      .map(line => line.split('\t'))
      .filter(parts => parts.length >= 2 && parts[0].trim())
      .map(parts => ({id: parts[0].trim(), name: parts[1].trim()})));
  });
});
