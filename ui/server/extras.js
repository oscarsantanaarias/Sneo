import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { paths } from './config.js';
import { cleanText, makeId, readSettings } from './storage.js';
import { askModel } from './model.js';
import { toolWebSearch, toolWebFetch } from './websearch.js';
import { onWindows } from './system.js';

const jsonPath = (file) => path.join(paths.dataDir, file);

const readJson = async (file, fallback) => {
  const full = jsonPath(file);

  if(!existsSync(full)){
    return fallback;
  }

  try {
    return JSON.parse(await readFile(full, 'utf8'));
  } catch(e) {
    return fallback;
  }
};

const writeJson = async (file, data) => {
  await writeFile(jsonPath(file), JSON.stringify(data, null, 2));
};

export const toolDeepResearch = async ({query}) => {
  const clean = cleanText(query);

  if(!clean){
    throw new Error('query requerido');
  }

  const planText = await askModel([
    {role: 'system', content: 'You break a research question into 3 focused web search queries. Return ONLY a JSON array of 3 short query strings, nothing else.'},
    {role: 'user', content: clean}
  ]);

  let queries = [clean];
  const match = planText.match(/\[[\s\S]*\]/);

  if(match){
    try {
      const arr = JSON.parse(match[0]);
      if(Array.isArray(arr) && arr.length){
        queries = arr.map(String).slice(0, 3);
      }
    } catch(e) {}
  }

  const findings = [];

  for(const q of queries){
    try {
      const search = await toolWebSearch({query: q, count: 3});
      const results = Array.isArray(search.results) ? search.results : [];
      findings.push(`## ${q}\n${results.slice(0, 3).map(r => `- ${r.title || ''}: ${r.description || r.url || ''}`).join('\n')}`);

      const top = results[0];

      if(top?.url){
        const page = await toolWebFetch({url: top.url});
        findings.push(`### Contenido de ${top.url}\n${String(page.content || page.text || '').slice(0, 2500)}`);
      }
    } catch(e) {
      findings.push(`## ${q}\n(busqueda fallo: ${e.message})`);
    }
  }

  const answer = await askModel([
    {role: 'system', content: 'You are a research analyst. From the gathered findings, write a clear, sourced synthesis answering the question. Note contradictions and what is uncertain. Reply in the user language.'},
    {role: 'user', content: `Question: ${clean}\n\nFindings:\n${findings.join('\n\n')}`}
  ]);

  return {query: clean, queries, answer};
};

export const toolReadDocument = async ({file}) => {
  const clean = cleanText(file);

  if(!clean || !path.isAbsolute(clean)){
    throw new Error('Pasa la ruta ABSOLUTA del documento (ej C:/carpeta/archivo.pdf)');
  }

  if(!existsSync(clean)){
    throw new Error(`No existe: ${clean}`);
  }

  const ext = path.extname(clean).toLowerCase();

  if(ext === '.pdf'){
    const { default: pdf } = await import('pdf-parse/lib/pdf-parse.js');
    const buf = await readFile(clean);
    const data = await pdf(buf);
    return {file: clean, pages: data.numpages, text: String(data.text || '').slice(0, 20000)};
  }

  if(ext === '.docx'){
    const { default: mammoth } = await import('mammoth');
    const result = await mammoth.extractRawText({path: clean});
    return {file: clean, text: String(result.value || '').slice(0, 20000)};
  }

  const text = await readFile(clean, 'utf8');
  return {file: clean, text: text.slice(0, 20000)};
};

export const toolSpeak = async ({text}) => {
  const clean = cleanText(text);

  if(!clean){
    throw new Error('text requerido');
  }

  if(!onWindows){
    return await new Promise((resolve) => {
      const child = spawn('spd-say', ['-w', clean.slice(0, 2000)]);
      child.on('close', (code) => resolve({spoken: true, engine: 'speech-dispatcher', ok: code === 0}));
      child.on('error', (e) => resolve({error: `TTS fallo: ${e.message}`}));
    });
  }

  return await new Promise((resolve) => {
    const child = spawn('powershell', [
      '-NoProfile', '-Command',
      'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak($env:OESA_TTS)'
    ], {env: {...process.env, OESA_TTS: clean.slice(0, 2000)}, windowsHide: true});

    child.stdin?.end();
    child.on('close', (code) => resolve({spoken: true, engine: 'windows-sapi', ok: code === 0}));
    child.on('error', (e) => resolve({error: `TTS fallo: ${e.message}`}));
  });
};

export const toolGenerateImage = async ({prompt, size}) => {
  const clean = cleanText(prompt);

  if(!clean){
    throw new Error('prompt requerido');
  }

  const settings = await readSettings();
  const baseUrl = cleanText(settings.imageBaseUrl) || cleanText(settings.providerBaseUrl);
  const apiKey = cleanText(settings.imageApiKey) || cleanText(settings.providerApiKey);

  if(!baseUrl){
    return {note: 'Falta backend de imagenes. Configura imageBaseUrl (endpoint OpenAI-compatible de imagenes, ej un server SD/ComfyUI o la API de OpenAI) y imageApiKey en settings.', prompt: clean};
  }

  const req = await fetch(`${baseUrl.replace(/\/+$/, '')}/images/generations`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', ...(apiKey ? {Authorization: `Bearer ${apiKey}`} : {})},
    body: JSON.stringify({model: cleanText(settings.imageModel) || 'dall-e-3', prompt: clean, n: 1, size: cleanText(size) || '1024x1024'})
  });

  const res = await req.json();

  if(!req.ok){
    return {error: res.error?.message || `Backend de imagenes respondio ${req.status}`};
  }

  const item = res.data?.[0] || {};
  const galleryDir = path.join(paths.dataDir, 'gallery');
  await mkdir(galleryDir, {recursive: true});

  if(item.b64_json){
    const outPath = path.join(galleryDir, `${makeId()}.png`);
    await writeFile(outPath, Buffer.from(item.b64_json, 'base64'));
    return {saved: outPath, prompt: clean};
  }

  return {url: item.url, prompt: clean};
};

export const toolNoteAdd = async ({text, title}) => {
  const clean = cleanText(text);

  if(!clean){
    throw new Error('text requerido');
  }

  const data = await readJson('notes.json', {notes: []});
  const note = {id: makeId(), title: cleanText(title) || clean.slice(0, 40), text: clean};
  data.notes.push(note);
  await writeJson('notes.json', data);

  return {saved: true, id: note.id, total: data.notes.length};
};

export const toolNoteList = async () => {
  const data = await readJson('notes.json', {notes: []});
  return {notes: data.notes.map(n => ({id: n.id, title: n.title, text: n.text.slice(0, 200)}))};
};

export const toolTaskAdd = async ({text}) => {
  const clean = cleanText(text);

  if(!clean){
    throw new Error('text requerido');
  }

  const data = await readJson('tasks.json', {tasks: []});
  const task = {id: makeId(), text: clean, done: false};
  data.tasks.push(task);
  await writeJson('tasks.json', data);

  return {saved: true, id: task.id, total: data.tasks.length};
};

export const toolTaskList = async () => {
  const data = await readJson('tasks.json', {tasks: []});
  return {tasks: data.tasks};
};

export const toolTaskDone = async ({id}) => {
  const data = await readJson('tasks.json', {tasks: []});
  const task = data.tasks.find(t => t.id === cleanText(id));

  if(!task){
    return {error: 'Tarea no encontrada'};
  }

  task.done = true;
  await writeJson('tasks.json', data);

  return {done: true, id: task.id};
};

export const toolCompareModels = async ({prompt, models}) => {
  const clean = cleanText(prompt);

  if(!clean){
    throw new Error('prompt requerido');
  }

  const list = Array.isArray(models) && models.length ? models.slice(0, 3).map(String) : [];

  if(list.length < 2){
    throw new Error('Pasa al menos 2 modelos en "models" (ej ["qwen3-coder:30b","qwen2.5-coder:14b"])');
  }

  const messages = [{role: 'user', content: clean}];
  const answers = await Promise.all(list.map(async (model) => {
    try {
      return {model, answer: await askModel(messages, null, model)};
    } catch(e) {
      return {model, answer: `error: ${e.message}`};
    }
  }));

  return {prompt: clean, answers};
};
