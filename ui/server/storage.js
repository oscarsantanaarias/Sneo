import fs from 'fs/promises';
import { defaultModel, paths } from './config.js';
import { normalizeMode } from './gate.js';

export const makeId = () => {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const cleanText = (value) => {
  return String(value ?? '').trim();
};

export const pathExists = async (targetPath) => {
  try {
    await fs.access(targetPath);
    return true;
  } catch(e) {
    return false;
  }
};

const readJson = async (file, fallback) => {
  try {
    const raw = await fs.readFile(file, 'utf8');
    return raw.trim() ? JSON.parse(raw) : fallback;
  } catch(e) {
    return fallback;
  }
};

const writeLocks = new Map();

const writeJson = (file, data) => {
  const run = async () => {
    const tmp = `${file}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2));

    for(let tries = 0; tries < 6; tries++){
      try {
        return await fs.rename(tmp, file);
      } catch(e) {
        if(!['EPERM', 'EBUSY', 'EACCES'].includes(e.code)){
          throw e;
        }

        await new Promise(resolve => setTimeout(resolve, 40 * (tries + 1)));
      }
    }

    await fs.copyFile(tmp, file);
    await fs.rm(tmp, {force: true});
  };

  const prev = writeLocks.get(file) || Promise.resolve();
  const next = prev.then(run, run);
  writeLocks.set(file, next.catch(() => {}));
  return next;
};

const ensureJson = async (file, data) => {
  try {
    await fs.access(file);
  } catch(e) {
    await writeJson(file, data);
  }
};

export const startData = async () => {
  await fs.mkdir(paths.dataDir, {recursive: true});
  await fs.mkdir(paths.backupsDir, {recursive: true});
  await fs.mkdir(paths.sandboxesDir, {recursive: true});
  await fs.mkdir(paths.projectsRoot, {recursive: true});

  await ensureJson(paths.chatsFile, {chats: []});
  await ensureJson(paths.projectsFile, {projects: []});
  await ensureJson(paths.undoFile, {items: []});
  await ensureJson(paths.agentLogsFile, {runs: []});
  await ensureJson(paths.pendingFile, {items: []});
  await ensureJson(paths.memoryFile, {items: []});
  await ensureJson(paths.plansFile, {plans: []});
  await ensureJson(paths.contextFile, {projects: {}});
  await ensureJson(paths.terminalFile, {sessions: []});
  await ensureJson(paths.sandboxSessionsFile, {sessions: []});

  await ensureJson(paths.settingsFile, {
    model: defaultModel,
    accessMode: 'manual',
    language: 'en',
    contextMessages: 16,
    maxToolLoops: 7
  });

  await ensureJson(paths.modelsFile, {
    models: [
      'qwen2.5-coder:14b',
      'qwen3.6:latest',
      'gemma4:latest'
    ]
  });
};

export const readChats = async () => {
  return await readJson(paths.chatsFile, {chats: []});
};

export const saveChats = async (data) => {
  await updateChats(current => {
    current.chats = data.chats;
  });
};

let chatsQueue = Promise.resolve();

export const updateChats = (change) => {
  const run = async () => {
    const data = await readJson(paths.chatsFile, {chats: []});

    if(!Array.isArray(data.chats)){
      data.chats = [];
    }

    const result = await change(data);

    if(result === null){
      return null;
    }

    await writeJson(paths.chatsFile, data);
    return result;
  };

  const next = chatsQueue.then(run, run);
  chatsQueue = next.catch(() => {});
  return next;
};

export const readSettings = async () => {
  const data = await readJson(paths.settingsFile, {});

  return {
    model: data.model || defaultModel,
    accessMode: normalizeMode(data.accessMode),
    allowNetwork: data.allowNetwork === true,
    language: ['es', 'en'].includes(data.language) ? data.language : 'en',
    contextMessages: Number(data.contextMessages || 16),
    maxToolLoops: Number(data.maxToolLoops || 7),
    numCtx: Number(data.numCtx) > 0 ? Number(data.numCtx) : null,
    temperature: typeof data.temperature === 'number' ? data.temperature : null,
    thinkDepth: ['off', 'light', 'normal', 'deep', 'full'].includes(data.thinkDepth) ? data.thinkDepth : 'normal',
    thinkCap: [0, 2048, 8192, 32768].includes(Number(data.thinkCap)) ? Number(data.thinkCap) : 0,
    reverseTool: ['auto', 'ghidra', 'ida'].includes(data.reverseTool) ? data.reverseTool : 'auto',
    agentEdits: data.agentEdits !== false,
    ghidraPort: Number(data.ghidraPort) || Number(process.env.GHIDRA_PORT) || 5093,
    idaPort: Number(data.idaPort) || Number(process.env.IDA_PORT) || 5073,
    providerBaseUrl: data.providerBaseUrl || 'http://localhost:8000/v1',
    providerApiKey: data.providerApiKey || '',
    imageBaseUrl: data.imageBaseUrl || '',
    imageApiKey: data.imageApiKey || '',
    imageModel: data.imageModel || '',
    plannerRoute: typeof data.plannerRoute === 'string' ? data.plannerRoute.slice(0, 80) : '',
    styleFiles: Array.isArray(data.styleFiles) ? data.styleFiles.filter(item => typeof item === 'string' && item.trim()).slice(0, 12) : [],
    updatedAt: data.updatedAt || null
  };
};

export const saveSettings = async (data) => {
  await writeJson(paths.settingsFile, data);
};

export const readProjectsData = async () => {
  return await readJson(paths.projectsFile, {projects: []});
};

export const saveProjectsData = async (data) => {
  await writeJson(paths.projectsFile, data);
};

export const readUndoData = async () => {
  return await readJson(paths.undoFile, {items: []});
};

export const saveUndoData = async (data) => {
  await writeJson(paths.undoFile, data);
};

export const readPendingData = async () => {
  return await readJson(paths.pendingFile, {items: []});
};

export const savePendingData = async (data) => {
  await writeJson(paths.pendingFile, data);
};

let pendingQueue = Promise.resolve();

export const updatePendingData = (change) => {
  const run = async () => {
    const data = await readPendingData();

    if(!Array.isArray(data.items)){
      data.items = [];
    }

    const result = await change(data);

    if(result === null){
      return null;
    }

    await savePendingData(data);
    return result;
  };

  const next = pendingQueue.then(run, run);
  pendingQueue = next.catch(() => {});
  return next;
};

export const pushPending = async (item) => {
  const pending = {
    id: makeId(),
    status: 'pending',
    createdAt: new Date().toISOString(),
    ...item
  };

  await updatePendingData(data => {
    data.items.unshift(pending);
    data.items = data.items.slice(0, 100);
  });

  return pending;
};

export const updatePending = async (id, patch) => {
  return await updatePendingData(data => {
    const found = data.items.find(item => item.id === id);

    if(!found){
      return null;
    }

    Object.assign(found, patch, {updatedAt: new Date().toISOString()});
    return found;
  });
};

export const pushUndo = async (item) => {
  const data = await readUndoData();
  data.items.unshift({
    id: makeId(),
    createdAt: new Date().toISOString(),
    ...item
  });
  data.items = data.items.slice(0, 50);
  await saveUndoData(data);
};

export const pushAgentLog = async (run) => {
  const data = await readJson(paths.agentLogsFile, {runs: []});
  data.runs.unshift({
    id: makeId(),
    createdAt: new Date().toISOString(),
    ...run
  });
  data.runs = data.runs.slice(0, 100);
  await writeJson(paths.agentLogsFile, data);
};

export const readManualModels = async () => {
  try {
    const data = await readJson(paths.modelsFile, {models: []});
    return data.models || [];
  } catch(e) {
    return [];
  }
};

export const readMemory = async () => {
  return await readJson(paths.memoryFile, {items: []});
};

export const saveMemory = async (data) => {
  await writeJson(paths.memoryFile, data);
};

export const readPlans = async () => {
  return await readJson(paths.plansFile, {plans: []});
};

export const savePlans = async (data) => {
  await writeJson(paths.plansFile, data);
};

export const readContextIndex = async () => {
  return await readJson(paths.contextFile, {projects: {}});
};

export const saveContextIndex = async (data) => {
  await writeJson(paths.contextFile, data);
};

export const readTerminalSessions = async () => {
  return await readJson(paths.terminalFile, {sessions: []});
};

export const saveTerminalSessions = async (data) => {
  await writeJson(paths.terminalFile, data);
};

export const readSandboxSessions = async () => {
  return await readJson(paths.sandboxSessionsFile, {sessions: []});
};

export const saveSandboxSessions = async (data) => {
  await writeJson(paths.sandboxSessionsFile, data);
};
