import express from 'express';
import fs from 'fs/promises';
import path from 'path';
import { exec, execFile, spawn } from 'child_process';
import { promisify } from 'util';
import { runnerUrl, paths } from './config.js';
import { getAllModels, getCurrentModel, readEngine } from './model.js';
import { contextGetProject, contextIndexProject, contextPickFiles } from './context.js';
import { getProjectPath, safeJoin } from './projects.js';
import { memoryAdd, memoryList, memorySearch } from './memory.js';
import { planCreate, planList, planUpdate } from './planner.js';
import { sandboxRead, sandboxStart, sandboxStop } from './sandbox.js';
import { cleanText, makeId, pathExists, pushUndo, readChats, readPendingData, readProjectsData, readSettings, readUndoData, saveProjectsData, saveSettings, saveUndoData, updateChats, updatePending } from './storage.js';
import { terminalList, terminalRead, terminalStart, terminalStop } from './terminal.js';
import { getAgentAnswer, runSubAgent, toolParallelAgents } from './agents.js';
import { askRoute, chatAgent, maskRoutes, readAgents, readRoutes, routeLabel, routePresets, runTeam, saveAgents, saveRoutes } from './teams.js';
import { claudeModels } from './claude.js';
import { answerApproval, approveRun, queueFor, runStore, stopChildren } from './runs.js';
import { pickFolder } from './picker.js';
import { accessModes, normalizeMode } from './gate.js';
import { toolCreateFolder, toolDeletePath, toolGrepFiles, toolListFiles, toolListProjects, toolProposePatch, toolReadFile, toolRenamePath, toolSandboxRun, toolWriteFile } from './tools.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);
const router = express.Router();

router.get('/health', async (req, res) => {
  try {
    const engine = await readEngine();
    const currentModel = engine.model || await getCurrentModel();
    const modelos = engine.model ? [] : await getAllModels();

    return res.status(200).json({
      message: 'Server OK',
      model: currentModel,
      runner: true,
      ready: !!engine.model || modelos.some(item => item.name === currentModel),
      projectsRoot: paths.projectsRoot
    });
  } catch(err) {
    return res.status(200).json({
      message: 'Server OK, but the model runner is not ready',
      model: await getCurrentModel(),
      runner: false,
      ready: false,
      projectsRoot: paths.projectsRoot
    });
  }
});

router.get('/models', async (req, res) => {
  try {
    const models = await getAllModels();
    return res.status(200).json({
      currentModel: await getCurrentModel(),
      models: models.map(item => ({
        name: item.name,
        size: item.size,
        modified_at: item.modified_at
      }))
    });
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/models/load', async (req, res) => {
  try {
    const { path: modelPath, name } = req.body || {};
    const settings = await readSettings().catch(() => ({}));
    const baseUrl = (settings.providerBaseUrl || 'http://localhost:8000/v1').replace(/\/v1\/?$/, '');

    const engineReq = await fetch(`${baseUrl}/api/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: modelPath, name })
    });

    const data = await engineReq.json();
    if(data.cargado && data.ruta){
      const current = await readSettings().catch(() => ({}));
      await saveSettings({ ...current, model: name || data.ruta });
    }
    return res.status(200).json(data);
  } catch(err) {
    return res.status(500).json({ message: err.message });
  }
});

router.post('/models/unload', async (req, res) => {
  try {
    const settings = await readSettings().catch(() => ({}));
    const baseUrl = (settings.providerBaseUrl || 'http://localhost:8000/v1').replace(/\/v1\/?$/, '');
    const reply = await fetch(`${baseUrl}/api/models/unload`, {method: 'POST'});
    return res.status(reply.status).json(await reply.json());
  } catch(err) {
    return res.status(500).json({ message: err.message });
  }
});

router.post('/models/scan', async (req, res) => {
  try {
    const { dir } = req.body || {};
    const settings = await readSettings().catch(() => ({}));
    const baseUrl = (settings.providerBaseUrl || 'http://localhost:8000/v1').replace(/\/v1\/?$/, '');

    const engineReq = await fetch(`${baseUrl}/api/models/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dir })
    });

    const data = await engineReq.json();
    return res.status(200).json(data);
  } catch(err) {
    return res.status(500).json({ message: err.message });
  }
});

router.get('/engine/status', async (req, res) => {
  try {
    const settings = await readSettings().catch(() => ({}));
    const baseUrl = (settings.providerBaseUrl || 'http://localhost:8000/v1').replace(/\/v1\/?$/, '');

    const engineReq = await fetch(`${baseUrl}/api/status`);
    const data = await engineReq.json();
    return res.status(200).json(data);
  } catch(err) {
    return res.status(500).json({ message: err.message });
  }
});

router.post('/engine/limits', async (req, res) => {
  try {
    const { vramReserveGb, ramReserveGb, ssdMaxGb } = req.body || {};
    const settings = await readSettings().catch(() => ({}));
    const baseUrl = (settings.providerBaseUrl || 'http://localhost:8000/v1').replace(/\/v1\/?$/, '');

    const reqLimits = await fetch(`${baseUrl}/api/limits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vramReserveGb, ramReserveGb, ssdMaxGb })
    });

    const data = await reqLimits.json();
    await saveSettings({ ...settings, vramReserveGb, ramReserveGb, ssdMaxGb });
    return res.status(200).json(data);
  } catch(err) {
    return res.status(500).json({ message: err.message });
  }
});

router.post('/engine/restart', async (req, res) => {
  try {
    const { vramReserveGb = 2, ramReserveGb = 8, ssdMaxGb = 64 } = req.body || {};
    const settings = await readSettings().catch(() => ({}));
    await saveSettings({ ...settings, vramReserveGb, ramReserveGb, ssdMaxGb });

    const motorRoot = path.resolve(paths.appRoot, '..');
    const masterExe = path.join(motorRoot, 'master.exe');

    if(!await pathExists(masterExe)){
      return res.status(404).json({ message: 'master.exe not found, build the engine first.' });
    }

    await new Promise(r => {
      execFile('powershell', ['-NoProfile', '-Command', `Get-Process master -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq '${masterExe}' } | Stop-Process -Force`], () => setTimeout(r, 1000));
    });

    const child = spawn(masterExe, ['8000', '127.0.0.1', '9090', 'modelos'], {
      cwd: motorRoot,
      detached: true,
      stdio: 'ignore'
    });
    child.unref();

    return res.status(200).json({ success: true, message: `Engine restarted. RAM kept free ${ramReserveGb} GB, VRAM kept free ${vramReserveGb} GB.` });
  } catch(e) {
    return res.status(500).json({ message: e.message });
  }
});

router.get('/settings', async (req, res) => {
  try {
    return res.status(200).json({settings: await readSettings()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

const updateEnvVar = async (key, value) => {
  const envPath = path.join(paths.appRoot, '.env');

  try {
    let text = '';

    try {
      text = await fs.readFile(envPath, 'utf8');
    } catch(e) {
      text = '';
    }

    const line = `${key}=${value}`;

    if(new RegExp(`^${key}=.*$`, 'm').test(text)){
      text = text.replace(new RegExp(`^${key}=.*$`, 'm'), line);
    } else {
      text = text.trimEnd() + `\n${line}\n`;
    }

    await fs.writeFile(envPath, text);
  } catch(e) {
  }
};

router.put('/settings', async (req, res) => {
  const {model, accessMode, allowNetwork, language, contextMessages, maxToolLoops, reverseTool, ghidraPort, idaPort, providerBaseUrl, providerApiKey, numCtx, temperature, thinkDepth, thinkCap, imageBaseUrl, imageModel, imageApiKey} = req.body ?? {};

  if(accessMode !== undefined && ![...accessModes, 'ask', 'full'].includes(accessMode)){
    return res.status(400).json({message: 'accessMode must be manual, plan or auto.'});
  }

  if(allowNetwork !== undefined && typeof allowNetwork !== 'boolean'){
    return res.status(400).json({message: 'allowNetwork must be true or false.'});
  }

  try {
    const current = await readSettings();
    const nextGhidraPort = Math.min(65535, Math.max(1, Number(ghidraPort || current.ghidraPort)));
    const nextIdaPort = Math.min(65535, Math.max(1, Number(idaPort || current.idaPort)));
    const nextNumCtx = numCtx === undefined
      ? current.numCtx
      : Number(numCtx) > 0
        ? Math.min(1048576, Math.max(2048, Math.round(Number(numCtx))))
        : null;
    const nextTemperature = temperature === undefined
      ? current.temperature
      : temperature !== null && temperature !== '' && Number.isFinite(Number(temperature))
        ? Math.min(2, Math.max(0, Number(temperature)))
        : null;
    const next = {
      ...current,
      model: cleanText(model) || current.model,
      accessMode: normalizeMode(accessMode, current.accessMode),
      allowNetwork: typeof allowNetwork === 'boolean' ? allowNetwork : current.allowNetwork === true,
      language: ['es', 'en'].includes(language) ? language : current.language,
      contextMessages: Math.min(60, Math.max(4, Number(contextMessages || current.contextMessages))),
      maxToolLoops: Math.min(50, Math.max(1, Number(maxToolLoops || current.maxToolLoops))),
      reverseTool: ['auto', 'ghidra', 'ida'].includes(reverseTool) ? reverseTool : current.reverseTool,
      ghidraPort: nextGhidraPort,
      idaPort: nextIdaPort,
      providerBaseUrl: cleanText(providerBaseUrl === undefined ? current.providerBaseUrl : providerBaseUrl) || 'http://localhost:8000/v1',
      providerApiKey: providerApiKey === undefined || providerApiKey === '***' ? (current.providerApiKey || '') : cleanText(providerApiKey),
      numCtx: nextNumCtx,
      temperature: nextTemperature,
      thinkDepth: ['off', 'light', 'normal', 'deep', 'full'].includes(thinkDepth) ? thinkDepth : current.thinkDepth,
      thinkCap: [0, 2048, 8192, 32768].includes(Number(thinkCap)) && thinkCap !== undefined ? Number(thinkCap) : current.thinkCap,
      imageBaseUrl: imageBaseUrl === undefined ? (current.imageBaseUrl || '') : cleanText(imageBaseUrl),
      imageModel: imageModel === undefined ? (current.imageModel || '') : cleanText(imageModel),
      imageApiKey: imageApiKey === undefined || imageApiKey === '***' ? (current.imageApiKey || '') : cleanText(imageApiKey),
      updatedAt: new Date().toISOString()
    };

    await saveSettings(next);
    await updateEnvVar('GHIDRA_PORT', nextGhidraPort);
    await updateEnvVar('IDA_PORT', nextIdaPort);
    const safeSettings = {...next, providerApiKey: next.providerApiKey ? '***' : '', imageApiKey: next.imageApiKey ? '***' : ''};
    return res.status(200).json({message: 'Settings guardados', settings: safeSettings});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.put('/settings/model', async (req, res) => {
  const {model} = req.body ?? {};
  const cleanModel = cleanText(model);

  if(!cleanModel){
    return res.status(400).json({message: 'Model required'});
  }

  try {
    const settings = await readSettings();
    settings.model = cleanModel;
    settings.updatedAt = new Date().toISOString();
    await saveSettings(settings);

    return res.status(200).json({message: 'Model updated', model: cleanModel});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/chats', async (req, res) => {
  try {
    return res.status(200).json(await readChats());
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/chats', async (req, res) => {
  const {title} = req.body ?? {};

  try {
    // el saludo queda guardado en el chat: se escribe en el idioma de la app
    const settings = await readSettings();
    const hello = settings.language === 'es' ? 'Chat nuevo. Elige un proyecto o dime qué revisar.' : 'New chat. Choose a project or tell me what to review.';
    const chat = {
      id: makeId(),
      title: cleanText(title) || 'New chat',
      project: '',
      messages: [{role: 'assistant', content: hello}],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    await updateChats(data => {
      data.chats.unshift(chat);
    });

    return res.status(200).json({chat});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.put('/chats/:id', async (req, res) => {
  const {title, project, messages, thinkDepth, thinkCap, accessMode} = req.body ?? {};

  try {
    const chat = await updateChats(data => {
      const found = data.chats.find(item => item.id === req.params.id);

      if(!found){
        return null;
      }

      if(title !== undefined) found.title = cleanText(title) || found.title;
      if(project !== undefined) found.project = cleanText(project);
      if(Array.isArray(messages)) found.messages = messages;
      if(['off', 'light', 'normal', 'deep', 'full'].includes(thinkDepth)) found.thinkDepth = thinkDepth;
      if(thinkCap !== undefined && [0, 2048, 8192, 32768].includes(Number(thinkCap))) found.thinkCap = Number(thinkCap);
      if([...accessModes, 'ask', 'full'].includes(accessMode)) found.accessMode = normalizeMode(accessMode);

      found.updatedAt = new Date().toISOString();
      return found;
    });

    if(!chat){
      return res.status(404).json({message: 'Chat no existe'});
    }

    return res.status(200).json({chat});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.delete('/chats/:id', async (req, res) => {
  try {
    await updateChats(data => {
      data.chats = data.chats.filter(item => item.id !== req.params.id);
    });

    return res.status(200).json({message: 'Chat eliminado'});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/projects', async (req, res) => {
  try {
    return res.status(200).json({projects: await toolListProjects(), root: paths.projectsRoot});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/projects', async (req, res) => {
  const {name, projectPath} = req.body ?? {};
  const cleanName = cleanText(name);
  const cleanPath = cleanText(projectPath);

  if(!cleanName && !cleanPath){
    return res.status(400).json({message: 'Nombre o ruta requerido'});
  }

  try {
    if(cleanPath){
      const finalPath = path.resolve(cleanPath);
      await fs.mkdir(finalPath, {recursive: true});

      const data = await readProjectsData();
      const finalName = cleanName || path.basename(finalPath);

      if(!data.projects.some(item => item.name === finalName)){
        data.projects.push({name: finalName, path: finalPath});
      }

      await saveProjectsData(data);
    } else {
      await fs.mkdir(safeJoin(paths.projectsRoot, cleanName), {recursive: true});
    }

    return res.status(200).json({projects: await toolListProjects()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/projects/pick', async (req, res) => {
  try {
    const selectedPath = await pickFolder({title: 'Open a project folder', start: cleanText(req.body?.start), button: 'Open project'});

    if(!selectedPath){
      return res.status(200).json({cancelled: true});
    }

    const data = await readProjectsData();
    const same = data.projects.find(item => path.resolve(item.path || '').toLowerCase() === path.resolve(selectedPath).toLowerCase());

    if(same){
      return res.status(200).json({project: {name: same.name, path: same.path}, projects: await toolListProjects()});
    }

    const base = path.basename(selectedPath) || selectedPath;
    let name = base;
    for(let n = 2; data.projects.some(item => item.name === name); n++){
      name = `${base} ${n}`;
    }
    data.projects.push({name, path: selectedPath});
    await saveProjectsData(data);

    return res.status(200).json({project: {name, path: selectedPath}, projects: await toolListProjects()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/projects/:project/files', async (req, res) => {
  try {
    return res.status(200).json({files: await toolListFiles({project: req.params.project})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/projects/:project/file', async (req, res) => {
  try {
    return res.status(200).json({content: await toolReadFile({project: req.params.project, file: req.query.path})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/projects/:project/search', async (req, res) => {
  try {
    return res.status(200).json({matches: await toolGrepFiles({project: req.params.project, query: req.query.q, pattern: req.query.pattern})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/projects/:project/changes', async (req, res) => {
  try {
    const plain = (value) => cleanText(value).replace(/\\/g, '/').replace(/^\.\//, '');
    const target = plain(req.query.path);
    const owner = cleanText(req.params.project);
    const data = await readUndoData();
    const mine = data.items.filter(item => item.project === owner && plain(item.target) === target && (item.action === 'restore' || item.action === 'delete-created'));

    if(!mine.length){
      return res.status(200).json({before: null});
    }

    let first = mine[0];
    let edits = 1;
    for(const item of mine.slice(1)){
      if(new Date(first.createdAt) - new Date(item.createdAt) > 15 * 60 * 1000) break;
      first = item;
      edits++;
    }

    const before = first.action === 'delete-created'
      ? ''
      : await fs.readFile(first.backupPath, 'utf8').catch(() => null);
    return res.status(200).json({before, since: first.createdAt, edits});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.put('/projects/:project/file', async (req, res) => {
  const {file, content} = req.body ?? {};

  try {
    return res.status(200).json({message: await toolWriteFile({project: req.params.project, file, content})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.delete('/projects/:project/path', async (req, res) => {
  try {
    return res.status(200).json({message: await toolDeletePath({project: req.params.project, target: req.query.path})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/projects/:project/folder', async (req, res) => {
  try {
    return res.status(200).json({message: await toolCreateFolder({project: req.params.project, folder: req.body?.folder})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.put('/projects/:project/rename', async (req, res) => {
  const {from, to} = req.body ?? {};

  try {
    return res.status(200).json({message: await toolRenamePath({project: req.params.project, from, to})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/undo', async (req, res) => {
  try {
    const data = await readUndoData();
    const item = data.items.shift();

    if(!item){
      return res.status(400).json({message: 'Nada para deshacer'});
    }

    const projectPath = await getProjectPath(item.project);

    if(item.action === 'restore'){
      const targetPath = safeJoin(projectPath, item.target);
      await fs.rm(targetPath, {recursive: true, force: true});
      await fs.mkdir(path.dirname(targetPath), {recursive: true});
      await fs.cp(item.backupPath, targetPath, {recursive: true});
    }

    if(item.action === 'delete-created'){
      await fs.rm(safeJoin(projectPath, item.target), {recursive: true, force: true});
    }

    if(item.action === 'rename'){
      const fromPath = safeJoin(projectPath, item.from);
      const toPath = safeJoin(projectPath, item.to);
      await fs.mkdir(path.dirname(toPath), {recursive: true});
      await fs.rename(fromPath, toPath);
    }

    await saveUndoData(data);
    return res.status(200).json({message: 'Deshecho', item});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/pending', async (req, res) => {
  try {
    const data = await readPendingData();
    return res.status(200).json({
      items: data.items.filter(item => item.status === 'pending').map(item => ({
        id: item.id,
        type: item.type,
        title: item.title,
        project: item.project,
        tool: item.tool || '',
        command: item.command || '',
        cwd: item.cwd || '',
        details: item.details || '',
        chatId: item.chatId || '',
        createdAt: item.createdAt,
        files: (item.files || []).map(file => ({
          file: file.file,
          conflict: file.conflict,
          reason: file.reason,
          diff: file.diff
        }))
      }))
    });
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/pending/:id/deny', async (req, res) => {
  try {
    const data = await readPendingData();
    const item = data.items.find(curr => curr.id === req.params.id);

    if(!item){
      return res.status(404).json({message: 'This request does not exist anymore.'});
    }

    await updatePending(item.id, {status: 'denied'});
    answerApproval(item.id, 'denied');

    return res.status(200).json({message: 'Denied'});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/pending/:id/approve', async (req, res) => {
  try {
    const data = await readPendingData();
    const item = data.items.find(curr => curr.id === req.params.id);

    if(!item){
      return res.status(404).json({message: 'This request does not exist anymore.'});
    }

    if(item.status !== 'pending'){
      return res.status(400).json({message: 'This request was already answered.'});
    }

    if(answerApproval(item.id, 'approved')){
      await updatePending(item.id, {status: 'approved'});
      return res.status(200).json({message: 'Approved'});
    }

    if(item.type !== 'patch'){
      await updatePending(item.id, {status: 'expired'});
      return res.status(410).json({message: 'This request belonged to a run that has already ended, so nothing was run. Ask the agent again.'});
    }

    if((item.files || []).some(file => file.conflict)){
      return res.status(409).json({message: 'This patch has a conflict and cannot be applied.'});
    }

    const projectPath = await getProjectPath(item.project);

    for(const file of item.files || []){
      const filePath = safeJoin(projectPath, file.file);
      const existed = await pathExists(filePath);
      const backupPath = path.join(paths.backupsDir, makeId());

      if(existed){
        await fs.cp(filePath, backupPath, {recursive: true});
      }

      await fs.mkdir(path.dirname(filePath), {recursive: true});
      await fs.writeFile(filePath, file.newContent);

      await pushUndo({
        action: existed ? 'restore' : 'delete-created',
        project: item.project,
        target: file.file,
        backupPath: existed ? backupPath : null
      });
    }

    await updatePending(item.id, {status: 'approved'});

    return res.status(200).json({message: 'Approved'});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/sandbox/run', async (req, res) => {
  const {project, command} = req.body ?? {};

  try {
    return res.status(200).json({message: 'Sandbox OK', result: await toolSandboxRun({project, command})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/sandbox/start', async (req, res) => {
  try {
    return res.status(200).json({session: await sandboxStart(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/sandbox/:id', async (req, res) => {
  try {
    return res.status(200).json({session: await sandboxRead({id: req.params.id})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/sandbox/:id/stop', async (req, res) => {
  try {
    return res.status(200).json({session: await sandboxStop({id: req.params.id})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/terminal/start', async (req, res) => {
  try {
    return res.status(200).json({session: await terminalStart(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/terminal', async (req, res) => {
  try {
    return res.status(200).json({sessions: await terminalList()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/terminal/:id', async (req, res) => {
  try {
    return res.status(200).json({session: await terminalRead({id: req.params.id})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/terminal/:id/stop', async (req, res) => {
  try {
    return res.status(200).json({session: await terminalStop({id: req.params.id})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/context/index', async (req, res) => {
  try {
    return res.status(200).json({context: await contextIndexProject(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/context/:project', async (req, res) => {
  try {
    return res.status(200).json({context: await contextGetProject({project: req.params.project})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/context/pick', async (req, res) => {
  try {
    return res.status(200).json({files: await contextPickFiles(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/memory', async (req, res) => {
  try {
    return res.status(200).json({item: await memoryAdd(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/memory', async (req, res) => {
  try {
    return res.status(200).json({items: await memorySearch({query: req.query.q})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/memory/all', async (req, res) => {
  try {
    return res.status(200).json({items: await memoryList()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/plans', async (req, res) => {
  try {
    return res.status(200).json({plan: await planCreate(req.body ?? {})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.get('/plans', async (req, res) => {
  try {
    return res.status(200).json({plans: await planList()});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.put('/plans/:planId/steps/:stepId', async (req, res) => {
  try {
    return res.status(200).json({plan: await planUpdate({
      planId: req.params.planId,
      stepId: req.params.stepId,
      ...req.body
    })});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/agents/parallel', async (req, res) => {
  const {project, task, agents} = req.body ?? {};

  try {
    return res.status(200).json({message: 'Subagents OK', results: await toolParallelAgents({project, task, agents})});
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

router.post('/patch/preview', async (req, res) => {
  const {project, changes, title} = req.body ?? {};

  try {
    return res.status(200).json({
      message: 'Patch pending',
      patch: await toolProposePatch({project, changes, title})
    });
  } catch(err) {
    return res.status(500).json({message: err.message});
  }
});

const liveRuns = new Map();
const queueLimit = Number(process.env.SNEO_QUEUE_LIMIT) || 8;

const openControl = () => {
  let finish = null;
  const control = {cancelled: false, steer: [], abort: new AbortController(), run: null};
  control.finished = new Promise(resolve => {
    finish = resolve;
  });
  control.finish = finish;
  return control;
};

const cancelControl = (control) => {
  if(!control || control.cancelled){
    return;
  }

  control.cancelled = true;
  control.abort.abort();
};

const runChat = async ({body, control, onEvent}) => {
  const {message, history, chatId, project, image, reverseTool, thinkDepth, thinkCap, accessMode} = body;
  const cleanMessage = cleanText(message);
  const cleanHistory = Array.isArray(history) ? history : [];
  const settings = await readSettings();
  const chats = chatId ? await readChats() : {chats: []};
  const chat = chats.chats.find(item => item.id === chatId) || null;
  const run = {
    id: makeId(),
    chatId: chatId || '',
    mode: normalizeMode(accessMode || chat?.accessMode, settings.accessMode),
    signal: control.abort.signal,
    onEvent,
    children: new Set()
  };

  control.run = run;
  control.abort.signal.addEventListener('abort', () => stopChildren(run), {once: true});

  const agent = await runStore.run(run, () => getAgentAnswer({
    message: cleanMessage,
    history: cleanHistory,
    project: cleanText(project),
    reverseTool: cleanText(reverseTool),
    onEvent,
    control,
    thinkDepth: ['off', 'light', 'normal', 'deep', 'full'].includes(thinkDepth) ? thinkDepth : null,
    thinkCap: [0, 2048, 8192, 32768].includes(Number(thinkCap)) ? Number(thinkCap) : 0,
    image: typeof image === 'string' && image ? image.replace(/^data:image\/[a-z]+;base64,/, '') : null,
    ratio: chat?.ratio,
    memo: chat?.memo
  }));

  const nextMessages = [
    ...cleanHistory,
    {role: 'user', content: cleanMessage},
    {role: 'assistant', content: agent.answer, tools: agent.tools, thoughts: agent.thoughts, steps: agent.steps, usage: agent.usage, mode: agent.mode}
  ];

  if(chatId){
    await updateChats(data => {
      const found = data.chats.find(item => item.id === chatId);

      if(!found){
        return null;
      }

      found.messages = nextMessages;
      found.project = cleanText(project);
      found.title = found.title === 'New chat' || found.title === 'Nuevo chat' ? cleanMessage.slice(0, 40) : found.title;
      found.ratio = agent.ratio;
      found.memo = agent.memo;
      found.updatedAt = new Date().toISOString();
      return found;
    });
  }

  return agent;
};

router.post('/chat/stop', async (req, res) => {
  const control = liveRuns.get(cleanText(req.body?.chatId));

  if(!control){
    return res.status(200).json({active: false});
  }

  cancelControl(control);
  await Promise.race([control.finished, new Promise(resolve => setTimeout(resolve, 8000))]);
  return res.status(200).json({active: true, stopped: true});
});

router.post('/chat/steer', async (req, res) => {
  const control = liveRuns.get(cleanText(req.body?.chatId));
  const text = cleanText(req.body?.message);
  const image = typeof req.body?.image === 'string' && req.body.image ? req.body.image.replace(/^data:image\/[a-z]+;base64,/, '') : '';

  if(!control){
    return res.status(200).json({active: false});
  }

  if(text || image){
    control.steer.push({text, image});
  }

  return res.status(200).json({active: true, queued: true});
});

router.post('/chat/mode', async (req, res) => {
  const control = liveRuns.get(cleanText(req.body?.chatId));

  if(!control?.run){
    return res.status(200).json({active: false});
  }

  const mode = normalizeMode(req.body?.mode, control.run.mode);
  control.run.mode = mode;
  const released = mode === 'auto' ? await approveRun(control.run.id) : 0;

  return res.status(200).json({active: true, mode, released});
});

router.get('/team/routes', async (req, res) => {
  return res.status(200).json({routes: maskRoutes(await readRoutes()), presets: routePresets, claudeModels});
});

router.put('/team/routes', async (req, res) => {
  try {
    return res.status(200).json({routes: maskRoutes(await saveRoutes(req.body?.routes))});
  } catch(e) {
    return res.status(400).json({message: String(e.message || e)});
  }
});

router.post('/team/routes/test', async (req, res) => {
  const route = (await readRoutes()).find(item => item.id === cleanText(req.body?.id));

  if(!route){
    return res.status(404).json({message: 'That route is not saved yet.'});
  }

  const startedAt = Date.now();

  try {
    const reply = await askRoute(route, [{role: 'user', content: 'Answer with the single word OK.'}], null, {signal: AbortSignal.timeout(90000)});
    return res.status(200).json({ok: true, answer: String(reply.content || '').trim().slice(0, 200), ms: Date.now() - startedAt});
  } catch(e) {
    return res.status(200).json({ok: false, message: String(e.message || e)});
  }
});

router.post('/team/models', async (req, res) => {
  const route = (await readRoutes()).find(item => item.id === cleanText(req.body?.id));

  if(!route){
    return res.status(404).json({message: 'That route is not saved yet.'});
  }

  if(route.kind === 'claude'){
    return res.status(200).json({models: claudeModels.map(item => item.id)});
  }

  try {
    const list = await fetch(`${route.baseUrl}/models`, {headers: route.apiKey ? {Authorization: `Bearer ${route.apiKey}`} : {}, signal: AbortSignal.timeout(20000)}).then(answer => answer.json());
    const models = (list.data || list.models || []).map(item => item.id || item.name).filter(Boolean).slice(0, 300);
    return res.status(200).json({models});
  } catch(e) {
    return res.status(200).json({models: [], message: String(e.message || e)});
  }
});

router.post('/team/run', async (req, res) => {
  const all = await readRoutes();
  const used = [req.body?.planner, req.body?.lead, ...(Array.isArray(req.body?.workers) ? req.body.workers : [])]
    .map(id => all.find(item => item.id === cleanText(id)))
    .filter(Boolean);

  for(const route of used){
    if(!await routeLoaded(route)){
      return res.status(409).json({message: `${route.name} has no model loaded yet. Load it in Agents, Models.`});
    }
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const stop = new AbortController();
  const send = (data) => {
    if(!res.writableEnded && !res.destroyed){
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  };
  const heartbeat = setInterval(() => send({type: 'ping'}), 15000);
  res.on('close', () => {
    if(!res.writableEnded){
      stop.abort();
    }
  });

  try {
    const body = req.body ?? {};
    const result = await runTeam({
      task: body.task,
      project: body.project,
      mode: body.mode,
      planner: body.planner,
      workers: body.workers,
      lead: body.lead,
      maxAgents: body.maxAgents,
      onEvent: send,
      signal: stop.signal,
      runWorker: runSubAgent
    });
    send({type: 'team-done', result});
  } catch(e) {
    send({type: 'team-error', message: stop.signal.aborted ? 'Stopped.' : String(e.message || e)});
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
});

// un motor de esta pc solo contesta a los agentes cuando su modelo esta cargado
const routeLoaded = async (route) => {
  if(route.kind !== 'local' || !/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/i.test(route.baseUrl || '')){
    return true;
  }

  const base = route.baseUrl.replace(/\/v1\/?$/, '');

  try {
    const pool = await fetch(`${base}/api/pool`, {signal: AbortSignal.timeout(3000)}).then(answer => answer.json());
    return pool.generation === 'ready';
  } catch(e) {
    return false;
  }
};

router.get('/team/agents', async (req, res) => {
  return res.status(200).json({agents: await readAgents()});
});

router.put('/team/agents', async (req, res) => {
  try {
    return res.status(200).json({agents: await saveAgents(req.body?.agents)});
  } catch(e) {
    return res.status(400).json({message: String(e.message || e)});
  }
});

// un mensaje a un agente de la vista de conversaciones: contesta por su ruta y avisa cada herramienta que usa
router.post('/team/chat', async (req, res) => {
  const body = req.body ?? {};
  const route = (await readRoutes()).find(item => item.id === cleanText(body.agent?.route));

  if(!route){
    return res.status(400).json({message: 'Pick the model this agent answers with first.'});
  }

  if(!await routeLoaded(route)){
    return res.status(409).json({message: 'The model of this agent is not loaded yet. Load it in Agents, Models.'});
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const stop = new AbortController();
  const send = (data) => {
    if(!res.writableEnded && !res.destroyed){
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  };
  const heartbeat = setInterval(() => send({type: 'ping'}), 15000);
  const startedAt = Date.now();
  res.on('close', () => {
    if(!res.writableEnded){
      stop.abort();
    }
  });

  try {
    const reply = await chatAgent({
      agent: body.agent,
      route,
      history: body.history,
      project: cleanText(body.project),
      peers: (Array.isArray(body.peers) ? body.peers : []).map(cleanText).filter(Boolean).slice(0, 8),
      run: {signal: stop.signal, usage: null},
      onStep: step => send({type: 'step', tool: step.tool}),
      runWorker: runSubAgent
    });
    send({type: 'reply', content: reply.content, tools: reply.tools, ms: Date.now() - startedAt, route: routeLabel(route)});
  } catch(e) {
    send({type: 'error', message: stop.signal.aborted ? 'Stopped.' : String(e.message || e)});
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
});

router.post('/chat/stream', async (req, res) => {
  const body = req.body ?? {};
  const chatId = cleanText(body.chatId);
  const key = chatId || `run-${makeId()}`;

  if(!cleanText(body.message) && !body.image){
    return res.status(400).json({message: 'Write a message first.'});
  }

  if(liveRuns.has(key)){
    return res.status(409).json({message: 'This chat is already working. Stop it or wait for it to finish.'});
  }

  const settings = await readSettings();
  const queue = queueFor(settings.providerBaseUrl || runnerUrl, queueLimit);

  if(queue.full){
    res.setHeader('Retry-After', '30');
    return res.status(429).json({message: `SNEO is busy: ${queue.limit} requests are already waiting for the model. Try again in a minute.`});
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const sendEvent = (data) => {
    if(!res.writableEnded && !res.destroyed){
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  };
  const heartbeat = setInterval(() => {
    if(!res.writableEnded && !res.destroyed){
      res.write(': ping\n\n');
    }
  }, 15000);
  const control = openControl();
  let leave = null;

  liveRuns.set(key, control);
  res.on('close', () => {
    if(!res.writableEnded){
      cancelControl(control);
    }
  });

  try {
    leave = await queue.enter(control.abort.signal, position => sendEvent({type: 'queued', position}));
    sendEvent({type: 'started'});

    const agent = await runChat({body: {...body, chatId}, control, onEvent: sendEvent});
    sendEvent({type: 'done', message: agent.answer, tools: agent.tools, thoughts: agent.thoughts, steps: agent.steps, usage: agent.usage, mode: agent.mode, stopped: !!agent.stopped});
  } catch(err) {
    if(control.cancelled){
      sendEvent({type: 'done', message: leave ? 'Stopped by you.' : 'Stopped before it started.', tools: [], steps: [], stopped: true});
    } else {
      console.log('chat stream failed', err.message);
      sendEvent({type: 'error', message: err.message || 'The message could not be completed. Try again.'});
    }
  } finally {
    leave?.();
    clearInterval(heartbeat);

    if(liveRuns.get(key) === control){
      liveRuns.delete(key);
    }

    control.finish();

    if(!res.writableEnded){
      res.end();
    }
  }
});

router.post('/chat', async (req, res) => {
  const body = req.body ?? {};
  const chatId = cleanText(body.chatId);
  const key = chatId || `run-${makeId()}`;

  if(!cleanText(body.message)){
    return res.status(400).json({message: 'Write a message first.'});
  }

  if(liveRuns.has(key)){
    return res.status(409).json({message: 'This chat is already working. Stop it or wait for it to finish.'});
  }

  const settings = await readSettings();
  const queue = queueFor(settings.providerBaseUrl || runnerUrl, queueLimit);

  if(queue.full){
    res.setHeader('Retry-After', '30');
    return res.status(429).json({message: `SNEO is busy: ${queue.limit} requests are already waiting for the model. Try again in a minute.`});
  }

  const control = openControl();
  let leave = null;

  liveRuns.set(key, control);
  res.on('close', () => {
    if(!res.writableEnded){
      cancelControl(control);
    }
  });

  try {
    leave = await queue.enter(control.abort.signal);
    const agent = await runChat({body: {...body, chatId, image: null}, control, onEvent: null});
    return res.status(200).json({message: agent.answer, tools: agent.tools, thoughts: agent.thoughts, steps: agent.steps, usage: agent.usage, mode: agent.mode});
  } catch(err) {
    console.log('chat failed', err.message);
    return res.status(control.cancelled ? 499 : 500).json({message: control.cancelled ? 'Cancelled partway through.' : err.message || 'The message could not be completed. Try again.'});
  } finally {
    leave?.();

    if(liveRuns.get(key) === control){
      liveRuns.delete(key);
    }

    control.finish();
  }
});

export default router;
