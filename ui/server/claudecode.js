import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { paths } from './config.js';

// claude code instalado en esta pc: usa la cuenta del usuario sin api key
export const claudeCodeModels = [
  {id: '', name: 'Default'},
  {id: 'opus', name: 'Opus'},
  {id: 'sonnet', name: 'Sonnet'},
  {id: 'haiku', name: 'Haiku'}
];

let found = null;

const findClaude = () => {
  if(found && fs.existsSync(found)){
    return found;
  }

  const home = os.homedir();
  const places = [
    path.join(home, '.local', 'bin', 'claude.exe'),
    path.join(home, '.local', 'bin', 'claude'),
    path.join(process.env.APPDATA || '', 'npm', 'claude.cmd')
  ];

  for(const dir of String(process.env.PATH || '').split(path.delimiter)){
    if(dir) places.push(path.join(dir, 'claude.exe'));
  }

  const extensions = path.join(home, '.vscode', 'extensions');
  if(fs.existsSync(extensions)){
    const list = fs.readdirSync(extensions).filter(name => name.startsWith('anthropic.claude-code')).sort().reverse();
    for(const name of list){
      places.push(path.join(extensions, name, 'resources', 'native-binary', 'claude.exe'));
    }
  }

  found = places.find(file => file && fs.existsSync(file)) || null;
  return found;
};

const flatten = (messages) => {
  const system = [];
  const turns = [];

  for(const item of messages || []){
    const text = typeof item.content === 'string' ? item.content : Array.isArray(item.content) ? item.content.map(part => part.text || '').join('\n') : '';
    if(!text.trim()) continue;

    if(item.role === 'system'){
      system.push(text);
      continue;
    }

    turns.push(item.role === 'assistant' ? `Assistant: ${text}` : item.role === 'tool' ? `Tool result: ${text}` : `User: ${text}`);
  }

  // una sola pregunta va tal cual y una conversacion se pasa como transcripcion
  const prompt = turns.length === 1 ? turns[0].replace(/^User: /, '') : `${turns.join('\n\n')}\n\nAnswer the last user message.`;
  return {system: system.join('\n\n'), prompt};
};

const efforts = ['low', 'medium', 'high', 'xhigh', 'max'];

// carpeta vacia propia para que no cargue la memoria ni el CLAUDE.md de ningun proyecto
const plannerHome = () => {
  const home = path.join(os.tmpdir(), 'sneo-planner');
  fs.mkdirSync(home, {recursive: true});
  return home;
};

const runClaude = async (route, system, input, run, session = []) => {
  const exe = findClaude();

  if(!exe){
    throw new Error('Claude Code is not installed on this PC. Install it and sign in with claude once');
  }

  // la linea de comandos de windows corta en 32k, el sistema largo va dentro del texto
  // claude code le cuenta la carpeta y el sistema donde corre, que no son los del usuario
  system = `${system}\n\nIgnore the working directory, platform, git and environment details of this session: they belong to a helper process, not to the user or the project.`.trim();
  const shortSystem = system.length <= 8000;
  const args = ['-p', '--output-format', 'json', '--strict-mcp-config', '--tools', '',
    '--system-prompt', shortSystem ? system || 'You are a helpful assistant.' : 'Follow the instructions at the start of the message.'];
  args.push(...(session.length ? session : ['--no-session-persistence']));

  if(route.model){
    args.push('--model', route.model);
  }

  if(efforts.includes(route.effort)){
    args.push('--effort', route.effort);
  }

  const env = {...process.env};
  delete env.CLAUDECODE;
  delete env.ANTHROPIC_API_KEY;

  const text = await new Promise((resolve, reject) => {
    const child = spawn(exe, args, {env, cwd: plannerHome(), windowsHide: true});
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
      if(!out.trim()) return reject(new Error(err.trim() || `claude exited with ${code}`));
      resolve(out);
    });
    child.stdin.end(shortSystem || !system ? input : `${system}\n\n---\n\n${input}`);
  });

  let reply;
  try {
    reply = JSON.parse(text);
  } catch(e) {
    throw new Error(text.trim().slice(0, 400));
  }

  if(reply.is_error){
    throw new Error(String(reply.result || 'Claude Code failed').slice(0, 400));
  }

  const usage = reply.usage || {};
  if(run?.usage){
    const spent = Number(reply.duration_api_ms || reply.duration_ms || 0);
    run.usage.rounds++;
    run.usage.inputTokens += (usage.input_tokens || 0) + (usage.cache_read_input_tokens || 0) + (usage.cache_creation_input_tokens || 0);
    run.usage.cachedTokens += usage.cache_read_input_tokens || 0;
    run.usage.outputTokens += usage.output_tokens || 0;
    run.usage.outputSpeed = spent > 0 && usage.output_tokens ? Math.round(usage.output_tokens / (spent / 1000) * 10) / 10 : run.usage.outputSpeed;
  }
  return {
    content: String(reply.result || ''),
    reasoning: '',
    tool_calls: [],
    finishReason: 'stop',
    usage: {
      prompt_tokens: (usage.input_tokens || 0) + (usage.cache_read_input_tokens || 0) + (usage.cache_creation_input_tokens || 0),
      completion_tokens: usage.output_tokens || 0
    }
  };
};

export const askClaudeCode = async (route, messages, tools, run) => {
  const {system, prompt} = flatten(messages);
  return await runClaude(route, system, prompt, run);
};

// una sesion de claude por chat: la primera vez lleva el contexto entero y despues solo lo nuevo
const sessionsFile = () => path.join(paths.dataDir, 'claude-sessions.json');

const readSessions = () => {
  try {
    return JSON.parse(fs.readFileSync(sessionsFile(), 'utf8'));
  } catch(e) {
    return {};
  }
};

const writeSessions = (all) => {
  fs.mkdirSync(paths.dataDir, {recursive: true});
  fs.writeFileSync(sessionsFile(), JSON.stringify(all, null, 2), 'utf8');
};

export const chatSession = (chatId) => chatId ? readSessions()[chatId] || null : null;

export const askInSession = async ({route, chatId, system, first, next, run, note = {}}) => {
  const all = readSessions();
  const kept = all[chatId];
  const same = kept && kept.route === route.id && kept.model === (route.model || '');

  if(same){
    try {
      const reply = await runClaude(route, system, next, run, ['--resume', kept.id]);
      all[chatId] = {...kept, ...note, turns: (kept.turns || 0) + 1, at: Date.now()};
      writeSessions(all);
      return {...reply, resumed: true};
    } catch(e) {
      if(run?.signal?.aborted) throw e;
    }
  }

  if(kept) dropSessionFile(kept.id);
  const id = crypto.randomUUID();
  const reply = await runClaude(route, system, first, run, ['--session-id', id]);
  all[chatId] = {id, route: route.id, model: route.model || '', ...note, turns: 1, at: Date.now()};
  writeSessions(all);
  return {...reply, resumed: false};
};

const dropSessionFile = (id) => {
  const projects = path.join(os.homedir(), '.claude', 'projects');
  try {
    for(const dir of fs.readdirSync(projects)){
      if(!/sneo-planner$/i.test(dir)) continue;
      fs.rmSync(path.join(projects, dir, `${id}.jsonl`), {force: true});
      fs.rmSync(path.join(projects, dir, id), {recursive: true, force: true});
    }
  } catch(e) {}
};

export const dropChatSession = (chatId) => {
  const all = readSessions();
  if(!all[chatId]) return;
  dropSessionFile(all[chatId].id);
  delete all[chatId];
  writeSessions(all);
};
