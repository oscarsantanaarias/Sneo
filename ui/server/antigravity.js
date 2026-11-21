import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { flattenChat, toolsText } from './claudecode.js';

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
    if(dir) places.push(path.join(dir, 'agy.exe'));
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

// el texto va por la entrada como un mensaje stream-json: la linea de comandos de windows corta en 32k
const runAgy = (route, text, run) => new Promise((resolve, reject) => {
  const exe = findAgy();

  if(!exe){
    return reject(new Error('Antigravity is not installed on this PC. Install it and sign in once'));
  }

  const args = ['--input-format', 'stream-json', '--output-format', 'stream-json', '--disable-slash-commands'];

  if(route.model){
    args.push('--model', route.model);
  }

  const child = spawn(exe, args, {cwd: emptyHome(), windowsHide: true});
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
    const result = out.split(/\r?\n/)
      .map(line => {
        try {
          return JSON.parse(line);
        } catch(e) {
          return null;
        }
      })
      .filter(item => item?.event === 'result')
      .pop()?.result;
    if(!result) return reject(new Error((err || out).trim().slice(-400) || `agy exited with ${code}`));
    resolve(result);
  });
  child.stdin.end(JSON.stringify({event: 'user', message: {content: text}}) + '\n');
});

export const askAntigravity = async (route, messages, tools, run) => {
  const {system, prompt} = flattenChat(messages);
  const text = [plainModel, system, toolsText(tools), prompt].filter(Boolean).join('\n\n---\n\n');
  let result = await runAgy(route, text, run);

  // a veces igual prueba sus herramientas y termina sin texto: una vuelta mas recordandoselo
  if(result.status === 'SUCCESS' && !String(result.response || '').trim()){
    result = await runAgy(route, `${text}\n\n---\n\nYour last try used your own tools and gave no answer. Do not use them, answer with text only.`, run);
  }

  if(result.status !== 'SUCCESS'){
    throw new Error(String(result.error || 'Antigravity failed').slice(0, 400));
  }

  const usage = result.usage || {};
  if(run?.usage){
    const spent = Number(result.duration_seconds || 0) * 1000;
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
