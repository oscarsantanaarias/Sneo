import express from 'express';
import crypto from 'crypto';
import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { spawn, execFile } from 'child_process';
import { fileURLToPath } from 'url';
import { promisify } from 'util';
import { gitProgram, paths } from './config.js';
import { readSettings } from './storage.js';
import { askRoute, readRoutes, routeLabel } from './teams.js';
import { pickFolder } from './picker.js';

const router = express.Router();
const run = promisify(execFile);
const rootDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const trainDir = path.join(rootDir, 'train');
const dataDir = path.join(trainDir, 'data');
const sourcesDir = path.join(trainDir, 'sources');
const python = path.join(trainDir, 'venv', 'Scripts', 'python.exe');
const settingsFile = () => path.join(paths.dataDir, 'train.json');

const languages = {'.cpp': 'cpp', '.cc': 'cpp', '.h': 'cpp', '.hpp': 'cpp', '.c': 'c', '.cs': 'csharp', '.js': 'javascript', '.jsx': 'jsx', '.ts': 'typescript', '.tsx': 'tsx', '.py': 'python', '.lua': 'lua', '.java': 'java', '.go': 'go', '.rs': 'rust', '.ps1': 'powershell', '.bat': 'bat', '.kt': 'kotlin', '.swift': 'swift', '.php': 'php', '.rb': 'ruby'};

export const baseModels = [
  {id: 'Qwen/Qwen3.5-4B', label: 'Qwen3.5 4B', size: 4.7, speed: 1000},
  {id: 'Qwen/Qwen3.5-9B', label: 'Qwen3.5 9B', size: 9.7, speed: 450},
  {id: 'Qwen/Qwen3.8-27B', label: 'Qwen3.8 27B (4 bit)', size: 27.8, speed: 130},
  {id: 'Qwen/Qwen3.6-27B', label: 'Qwen3.6 27B (4 bit)', size: 27.8, speed: 130}
];

const defaults = {
  sources: [],
  extensions: Object.keys(languages),
  skipFolders: ['node_modules', '.git', 'build', 'dist', 'bin', 'obj', 'target', 'venv', '.venv', '__pycache__', 'runtime', 'third_party', 'deps', 'vendor', 'packages', 'out'],
  maxFileKb: 400,
  chunkLines: 80,
  baseModel: baseModels[0].id,
  teacher: {routeId: '', lessonsPerPiece: 1, maxPieces: 1500, language: 'Spanish', stylePairs: true},
  train: {epochs: 2, rank: 32, alpha: 64, learningRate: 0.0002, maxLength: 3072, accumulate: 8},
  export: {name: 'MyStyle', type: 'q8_0', converter: ''},
  lessonsKey: '',
  measured: {}
};

const job = {kind: '', running: false, phase: '', done: 0, total: 0, seconds: 0, left: 0, loss: null, evalLoss: null, problem: '', result: '', started: 0, log: []};
let child = null;
let stopAsked = false;

const readTrain = async () => {
  let saved = {};
  try {
    saved = JSON.parse(await fs.readFile(settingsFile(), 'utf8'));
  } catch(e) {
    saved = {};
  }
  const merged = {...defaults, ...saved, teacher: {...defaults.teacher, ...saved.teacher}, train: {...defaults.train, ...saved.train}, export: {...defaults.export, ...saved.export}, measured: {...saved.measured}};
  if(!merged.lessonsKey){
    merged.lessonsKey = crypto.randomBytes(18).toString('hex');
    await saveTrain(merged);
  }
  return merged;
};

const saveTrain = async (value) => {
  await fs.mkdir(paths.dataDir, {recursive: true});
  await fs.writeFile(settingsFile(), JSON.stringify(value, null, 2), 'utf8');
};

const note = (line) => {
  const text = String(line).replace(/\s+$/, '');
  if(!text) return;
  job.log.push(text);
  if(job.log.length > 300) job.log.splice(0, job.log.length - 300);
};

const startJob = (kind, total = 0) => {
  Object.assign(job, {kind, running: true, phase: '', done: 0, total, seconds: 0, left: 0, loss: null, evalLoss: null, problem: '', result: '', started: Date.now(), log: []});
  stopAsked = false;
};

const endJob = (problem = '', result = '') => {
  job.running = false;
  job.problem = stopAsked && !problem ? 'Stopped' : problem;
  job.result = result;
  job.seconds = (Date.now() - job.started) / 1000;
  child = null;
};

const readLines = async (file) => {
  if(!existsSync(file)) return [];
  const rows = [];
  // solo el salto de linea separa: el lector de lineas de node tambien corta en u2028 y u2029 que vienen dentro del codigo
  for(const line of (await fs.readFile(file, 'utf8')).split('\n')){
    if(!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch(e) {
      continue;
    }
  }
  return rows;
};

// cuanto hay para aprender y cuanto tardaria: los tokens salen de los caracteres (unos 3.3 por token en codigo)
const dataStats = async (settings) => {
  const pieces = await readLines(path.join(dataDir, 'pieces.jsonl'));
  const taught = await readLines(path.join(dataDir, 'teacher.jsonl'));
  const sent = await readLines(path.join(dataDir, 'external.jsonl'));
  const byId = new Map(pieces.map(item => [item.id, item]));
  let chars = 0;
  for(const p of pieces) chars += p.before.length + p.code.length + 120;
  let lessons = 0;
  for(const t of taught){
    const p = byId.get(t.id);
    if(!p) continue;
    if(settings.teacher.stylePairs && t.instruction) chars += t.instruction.length + p.code.length + 80;
    for(const l of t.lessons || []){
      if(!l?.question || !l?.answer) continue;
      chars += l.question.length + p.code.length + l.answer.length + 80;
      lessons++;
    }
  }
  for(const e of sent) chars += String(e.instruction || e.question || '').length + String(e.code || '').length + String(e.answer || '').length + 80;
  const files = new Set(pieces.map(item => item.source + '/' + item.file)).size;
  const examples = pieces.length + taught.filter(t => settings.teacher.stylePairs && t.instruction && byId.has(t.id)).length + lessons + sent.length;
  return {files, pieces: pieces.length, taught: taught.length, lessons, sent: sent.length, examples, tokens: Math.round(chars / 3.3)};
};

const estimate = (settings, stats, routes) => {
  const base = baseModels.find(item => item.id === settings.baseModel) || {speed: 1000};
  const measured = settings.measured[settings.baseModel];
  const speed = measured?.tokensPerSecond || base.speed;
  const trainSeconds = stats.tokens * settings.train.epochs / speed;
  const route = routes.find(item => item.id === settings.teacher.routeId);
  const wanted = Math.min(stats.pieces, settings.teacher.maxPieces);
  const each = settings.measured.teachSeconds?.[settings.teacher.routeId] || (route?.kind === 'local' ? 25 : 8);
  const teachSeconds = route ? Math.max(0, wanted - stats.taught) * each / Math.max(1, route.parallel || 1) : 0;
  return {trainSeconds, teachSeconds, exportSeconds: 240, speed, measured: !!measured};
};

router.get('/state', async (req, res) => {
  try {
    const settings = await readTrain();
    const routes = await readRoutes();
    const stats = await dataStats(settings);
    return res.status(200).json({
      settings,
      routes: routes.map(item => ({id: item.id, label: routeLabel(item), kind: item.kind})),
      baseModels,
      ready: {python: existsSync(python), converter: !!settings.export.converter && existsSync(path.join(settings.export.converter, 'convert_hf_to_gguf.py'))},
      stats,
      estimate: estimate(settings, stats, routes),
      job: {...job, log: job.log.slice(-80)}
    });
  } catch(e) {
    return res.status(500).json({message: e.message});
  }
});

router.post('/settings', async (req, res) => {
  const current = await readTrain();
  const body = req.body || {};
  const next = {
    ...current,
    baseModel: typeof body.baseModel === 'string' && body.baseModel.trim() ? body.baseModel.trim() : current.baseModel,
    teacher: {...current.teacher, ...(body.teacher || {})},
    train: {...current.train, ...(body.train || {})},
    export: {...current.export, ...(body.export || {})},
    extensions: Array.isArray(body.extensions) ? body.extensions : current.extensions
  };
  await saveTrain(next);
  return res.status(200).json({settings: next});
});

router.post('/sources/folder', async (req, res) => {
  try {
    const dir = await pickFolder({title: 'Pick a folder with code to learn from', button: 'Learn from this folder'});
    if(!dir) return res.status(200).json({cancelled: true});
    const settings = await readTrain();
    if(!settings.sources.some(item => item.path === dir)) settings.sources.push({id: crypto.randomBytes(6).toString('hex'), kind: 'folder', path: dir});
    await saveTrain(settings);
    return res.status(200).json({settings});
  } catch(e) {
    return res.status(500).json({message: e.message});
  }
});

router.post('/sources/git', async (req, res) => {
  const url = String(req.body?.url || '').trim();
  if(!/^(https:\/\/|git@)[\w.@:/~-]+$/.test(url)) return res.status(400).json({message: 'That does not look like a git address.'});
  const settings = await readTrain();
  const name = url.replace(/\.git$/, '').split(/[/:]/).filter(Boolean).slice(-2).join('-');
  if(!settings.sources.some(item => item.url === url)) settings.sources.push({id: crypto.randomBytes(6).toString('hex'), kind: 'git', url, path: path.join(sourcesDir, name)});
  await saveTrain(settings);
  return res.status(200).json({settings});
});

router.delete('/sources/:id', async (req, res) => {
  const settings = await readTrain();
  settings.sources = settings.sources.filter(item => item.id !== req.params.id);
  await saveTrain(settings);
  return res.status(200).json({settings});
});

// corta en pedazos que terminan en una linea vacia o en una llave que cierra para no partir funciones por la mitad
const pieces = (lines, size) => {
  const out = [];
  let start = 0;
  while(start < lines.length){
    let end = Math.min(lines.length, start + size);
    if(end < lines.length){
      for(let i = end - 1; i > start + size / 2; i--){
        const text = lines[i].trim();
        if(text === '' || text === '}' || text === '};' || text === 'end'){
          end = i + 1;
          break;
        }
      }
    }
    out.push([start, end]);
    start = end;
  }
  return out;
};

const walk = async (folder, settings, found) => {
  let entries = [];
  try {
    entries = await fs.readdir(folder, {withFileTypes: true});
  } catch(e) {
    return;
  }
  for(const entry of entries){
    const full = path.join(folder, entry.name);
    if(entry.isDirectory()){
      if(!entry.name.startsWith('.') && !settings.skipFolders.includes(entry.name.toLowerCase())) await walk(full, settings, found);
      continue;
    }
    const ext = path.extname(entry.name).toLowerCase();
    if(settings.extensions.includes(ext)) found.push(full);
  }
};

const collect = async (settings) => {
  const git = gitProgram() || 'git';
  for(const source of settings.sources.filter(item => item.kind === 'git')){
    if(stopAsked) return;
    job.phase = `Updating ${source.url}`;
    note(job.phase);
    try {
      if(existsSync(path.join(source.path, '.git'))) await run(git, ['-C', source.path, 'pull', '--ff-only'], {windowsHide: true});
      else {
        await fs.mkdir(sourcesDir, {recursive: true});
        await run(git, ['clone', '--depth', '1', source.url, source.path], {windowsHide: true});
      }
    } catch(e) {
      note(`Could not update ${source.url}: ${e.message.split('\n')[0]}`);
    }
  }
  job.phase = 'Reading the code';
  const seen = new Set();
  const rows = [];
  let fileCount = 0;
  for(const source of settings.sources){
    const files = [];
    await walk(source.path, settings, files);
    job.total += files.length;
    for(const file of files){
      if(stopAsked) return;
      job.done++;
      const stat = await fs.stat(file).catch(() => null);
      if(!stat || stat.size > settings.maxFileKb * 1024) continue;
      const raw = await fs.readFile(file);
      if(raw.subarray(0, 4096).includes(0)) continue;
      const lines = raw.toString('utf8').replace(/\r\n/g, '\n').split('\n');
      if(lines.reduce((sum, l) => sum + l.length, 0) / Math.max(1, lines.length) > 200) continue;
      fileCount++;
      const relative = path.relative(source.path, file).replace(/\\/g, '/');
      const sourceName = source.kind === 'git' ? source.url.split('/').pop().replace(/\.git$/, '') : path.basename(source.path);
      for(const [start, end] of pieces(lines, settings.chunkLines)){
        const code = lines.slice(start, end).join('\n').replace(/^\n+|\n+$/g, '');
        if(code.split('\n').filter(l => l.trim()).length < 8) continue;
        const id = crypto.createHash('sha1').update(code).digest('hex').slice(0, 16);
        if(seen.has(id)) continue;
        seen.add(id);
        const before = lines.slice(Math.max(0, start - 30), start).join('\n').replace(/^\n+|\n+$/g, '');
        rows.push({id, source: sourceName, file: relative, lang: languages[path.extname(file).toLowerCase()] || '', line: start + 1, before, code});
      }
    }
  }
  await fs.mkdir(dataDir, {recursive: true});
  // se escribe aparte y se cambia de nombre al final asi el estado nunca lee el archivo a medias
  const temporary = path.join(dataDir, 'pieces.jsonl.tmp');
  await fs.writeFile(temporary, rows.map(item => JSON.stringify(item)).join('\n') + (rows.length ? '\n' : ''), 'utf8');
  await fs.rename(temporary, path.join(dataDir, 'pieces.jsonl'));
  note(`${fileCount} files, ${rows.length} pieces of code`);
  return rows.length;
};

router.post('/collect', async (req, res) => {
  if(job.running) return res.status(409).json({message: 'Something else is running.'});
  const settings = await readTrain();
  if(!settings.sources.length) return res.status(400).json({message: 'Add a folder or a git repository first.'});
  startJob('collect');
  res.status(200).json({started: true});
  collect(settings).then(count => endJob(count === 0 ? 'No code found in those sources.' : '')).catch(e => endJob(e.message));
});

const lessonPrompt = (p, teacher) => `You are preparing lessons to fine-tune a small code model so it writes code exactly like one programmer.
Here is a piece of that programmer's code from ${p.file} (${p.lang}):

\`\`\`${p.lang}
${p.code}
\`\`\`

Return only JSON with this shape:
{"instruction": "...", "lessons": [{"question": "...", "answer": "..."}]}

- "instruction": the request, written in ${teacher.language}, that someone would type to get exactly this code. Name the functions, variables and behavior it must have, but do not paste the code.
- "lessons": ${teacher.lessonsPerPiece} item(s). Each is a question in ${teacher.language} a student would ask about this code (why it is done this way, a risk or bug, how to extend it) and a teaching answer in ${teacher.language}. Any code in an answer must follow this programmer's style exactly: same naming, braces, spacing, early returns and comment style.`;

// cada pedazo va al maestro elegido: tus rutas de agentes (sneo, claude, gemini, openai...) con sus claves
const teach = async (settings, route) => {
  const all = (await readLines(path.join(dataDir, 'pieces.jsonl'))).slice(0, settings.teacher.maxPieces);
  const target = path.join(dataDir, 'teacher.jsonl');
  const done = new Set((await readLines(target)).map(item => item.id));
  const left = all.filter(item => !done.has(item.id));
  job.total = all.length;
  job.done = done.size;
  job.phase = `Lessons from ${routeLabel(route)}`;
  const lanes = Math.max(1, Math.min(8, route.kind === 'local' ? 1 : route.parallel || 2));
  let next = 0;
  let timed = 0;
  let spent = 0;
  const worker = async () => {
    while(!stopAsked && next < left.length){
      const p = left[next++];
      const started = Date.now();
      try {
        const reply = await askRoute(route, [{role: 'user', content: lessonPrompt(p, settings.teacher)}], [], {});
        const found = String(reply.content || '').match(/\{[\s\S]*\}/);
        const answer = found ? JSON.parse(found[0]) : null;
        if(answer && typeof answer.instruction === 'string'){
          await fs.appendFile(target, JSON.stringify({id: p.id, instruction: answer.instruction, lessons: Array.isArray(answer.lessons) ? answer.lessons : []}) + '\n', 'utf8');
        } else note(`${p.file}:${p.line} the teacher did not answer in JSON, skipped`);
      } catch(e) {
        note(`${p.file}:${p.line} ${e.message.split('\n')[0]}`);
        if(/401|403|invalid.*key|unauthorized/i.test(e.message)) stopAsked = true;
      }
      job.done++;
      timed++;
      spent += (Date.now() - started) / 1000;
      job.left = spent / timed * (left.length - next) / lanes;
    }
  };
  await Promise.all(Array.from({length: lanes}, worker));
  const settingsNow = await readTrain();
  settingsNow.measured.teachSeconds = {...settingsNow.measured.teachSeconds, [route.id]: timed ? spent / timed : 0};
  await saveTrain(settingsNow);
};

router.post('/teach', async (req, res) => {
  if(job.running) return res.status(409).json({message: 'Something else is running.'});
  const settings = await readTrain();
  const route = (await readRoutes()).find(item => item.id === settings.teacher.routeId);
  if(!route) return res.status(400).json({message: 'Pick a teacher first.'});
  startJob('teach');
  res.status(200).json({started: true});
  teach(settings, route).then(() => endJob()).catch(e => endJob(e.message));
});

// lanza un guion de python y lee sus lineas @@ con el avance; el resto va al registro
const runPython = (kind, script, config, finish) => {
  startJob(kind);
  const configFile = path.join(dataDir, `${kind}-config.json`);
  fs.mkdir(dataDir, {recursive: true})
    .then(() => fs.writeFile(configFile, JSON.stringify(config, null, 2), 'utf8'))
    .then(() => {
      child = spawn(python, ['-u', path.join(trainDir, 'scripts', script), '--config', configFile], {
        cwd: trainDir,
        windowsHide: true,
        env: {...process.env, PYTHONWARNINGS: 'ignore', CUDA_DEVICE_ORDER: 'PCI_BUS_ID', HF_HUB_DISABLE_TELEMETRY: '1', PYTHONIOENCODING: 'utf-8'}
      });
      let last = null;
      let tail = '';
      const read = (chunk) => {
        tail += chunk.toString('utf8');
        const lines = tail.split(/\r?\n/);
        tail = lines.pop();
        for(const line of lines){
          if(!line.startsWith('@@')){
            if(!/Redirects are currently not supported|ResourceTracker|resource_tracker|_recursion_count|^\s*$/.test(line)) note(line);
            continue;
          }
          let step = null;
          try {
            step = JSON.parse(line.slice(2));
          } catch(e) {
            continue;
          }
          if(step.kind === 'phase') job.phase = step.text;
          if(step.kind === 'step'){
            Object.assign(job, {done: step.done, total: step.total, left: step.left, loss: step.loss});
            last = step;
          }
          if(step.kind === 'eval') job.evalLoss = step.loss;
          if(step.kind === 'done') job.result = step.adapter || step.target || '';
        }
      };
      child.stdout.on('data', read);
      child.stderr.on('data', read);
      child.on('close', async (code) => {
        const problem = code === 0 ? '' : job.log.slice(-1)[0] || `stopped with code ${code}`;
        const result = job.result;
        await finish(code === 0, last).catch(() => {});
        endJob(problem, result);
      });
    })
    .catch(e => endJob(e.message));
};

router.post('/train', async (req, res) => {
  if(job.running) return res.status(409).json({message: 'Something else is running.'});
  if(!existsSync(python)) return res.status(400).json({message: 'Install the training tools first.'});
  const settings = await readTrain();
  const quick = req.body?.quick === true;
  const runDir = path.join(trainDir, 'runs', new Date().toISOString().slice(0, 16).replace(/[-:T]/g, ''));
  runPython(quick ? 'test' : 'train', 'train.py', {baseModel: settings.baseModel, dataDir, runDir, stylePairs: settings.teacher.stylePairs, quick, train: settings.train}, async (ok, last) => {
    if(!ok || !last?.tokensPerSecond) return;
    const now = await readTrain();
    now.measured[now.baseModel] = {tokensPerSecond: last.tokensPerSecond, at: Date.now()};
    if(!quick) now.lastAdapter = path.join(runDir, 'adapter');
    await saveTrain(now);
  });
  return res.status(200).json({started: true});
});

router.post('/export', async (req, res) => {
  if(job.running) return res.status(409).json({message: 'Something else is running.'});
  const settings = await readTrain();
  if(!settings.lastAdapter || !existsSync(settings.lastAdapter)) return res.status(400).json({message: 'Train first: there is nothing to export yet.'});
  const size = settings.baseModel.split('/').pop();
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const name = String(settings.export.name || 'MyStyle').replace(/[^\w.-]+/g, '-');
  const target = path.join(rootDir, 'modelos', `${name}-${size}-${stamp}-${settings.export.type.toUpperCase()}.gguf`);
  runPython('export', 'export.py', {baseModel: settings.baseModel, adapter: settings.lastAdapter, converter: settings.export.converter, type: settings.export.type, target}, async () => {});
  return res.status(200).json({started: true});
});

router.post('/export/converter', async (req, res) => {
  try {
    const dir = await pickFolder({title: 'Pick the folder with convert_hf_to_gguf.py', button: 'Use this folder'});
    if(!dir) return res.status(200).json({cancelled: true});
    const settings = await readTrain();
    settings.export.converter = dir;
    await saveTrain(settings);
    return res.status(200).json({settings});
  } catch(e) {
    return res.status(500).json({message: e.message});
  }
});

router.post('/setup', (req, res) => {
  if(job.running) return res.status(409).json({message: 'Something else is running.'});
  startJob('setup');
  job.phase = 'Installing Python, PyTorch and the training libraries (about 4 GB)';
  child = spawn('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(trainDir, 'setup.ps1')], {cwd: trainDir, windowsHide: true});
  const read = (chunk) => chunk.toString('utf8').split(/\r?\n/).forEach(line => {
    if(!/Redirects are currently|NativeCommandError|CategoryInfo|^\+|At .*setup\.ps1/.test(line)) note(line);
  });
  child.stdout.on('data', read);
  child.stderr.on('data', read);
  child.on('close', (code) => endJob(existsSync(python) ? '' : `setup stopped with code ${code}`));
  return res.status(200).json({started: true});
});

router.post('/stop', (req, res) => {
  stopAsked = true;
  if(child){
    spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], {windowsHide: true});
  }
  return res.status(200).json({stopping: job.running});
});

router.post('/free-gpu', async (req, res) => {
  try {
    const settings = await readSettings().catch(() => ({}));
    const base = (settings.providerBaseUrl || 'http://localhost:8000').trim().replace(/\/+$/, '').replace(/\/v1$/, '');
    await fetch(`${base}/api/models/unload`, {method: 'POST', signal: AbortSignal.timeout(60000)});
    return res.status(200).json({freed: true});
  } catch(e) {
    return res.status(500).json({message: `The engine did not answer: ${e.message}`});
  }
});

router.post('/lessons/key', async (req, res) => {
  const settings = await readTrain();
  settings.lessonsKey = crypto.randomBytes(18).toString('hex');
  await saveTrain(settings);
  return res.status(200).json({settings});
});

router.delete('/lessons', async (req, res) => {
  await fs.rm(path.join(dataDir, 'external.jsonl'), {force: true});
  return res.status(200).json({cleared: true});
});

router.delete('/teacher', async (req, res) => {
  await fs.rm(path.join(dataDir, 'teacher.jsonl'), {force: true});
  return res.status(200).json({cleared: true});
});

// otros modelos o programas de esta pc mandan clases aca con la clave de la vista: cabecera x-sneo y Authorization: Bearer <clave>
router.post('/lessons', async (req, res) => {
  const settings = await readTrain();
  const given = String(req.headers.authorization || '').replace(/^Bearer\s+/i, '');
  if(!given || given.length !== settings.lessonsKey.length || !crypto.timingSafeEqual(Buffer.from(given), Buffer.from(settings.lessonsKey))){
    return res.status(401).json({message: 'Wrong or missing lessons key.'});
  }
  const list = Array.isArray(req.body?.lessons) ? req.body.lessons : [req.body];
  const clean = list.slice(0, 500).filter(item => item && typeof item.answer === 'string' && (typeof item.instruction === 'string' || typeof item.question === 'string')).map(item => ({
    instruction: String(item.instruction || '').slice(0, 20000),
    question: String(item.question || '').slice(0, 20000),
    code: String(item.code || '').slice(0, 60000),
    lang: String(item.lang || '').slice(0, 30),
    answer: String(item.answer).slice(0, 60000),
    from: String(item.from || '').slice(0, 80),
    at: Date.now()
  }));
  if(!clean.length) return res.status(400).json({message: 'Send {"lessons": [{"question" or "instruction": "...", "answer": "...", "code": "optional", "lang": "optional"}]}'});
  await fs.mkdir(dataDir, {recursive: true});
  await fs.appendFile(path.join(dataDir, 'external.jsonl'), clean.map(item => JSON.stringify(item)).join('\n') + '\n', 'utf8');
  const total = (await readLines(path.join(dataDir, 'external.jsonl'))).length;
  return res.status(200).json({saved: clean.length, total});
});

export default router;
