import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

const ownEnv = dotenv.config().parsed || {};

export const childEnv = () => Object.fromEntries(Object.entries(process.env).filter(([key]) => !(key in ownEnv)));

// para lo que corre sin preguntar cmd no busca programas en la carpeta actual, asi un proyecto no puede traer su propio git que se ejecute solo
export const safeEnv = () => ({...childEnv(), NoDefaultCurrentDirectoryInExePath: '1'});

let gitFound = null;

// git se busca solo en las carpetas del PATH, nunca en el proyecto
export const gitProgram = () => {
  if(gitFound !== null) return gitFound;
  const names = process.platform === 'win32' ? ['git.exe'] : ['git'];

  for(const dir of String(process.env.PATH || '').split(path.delimiter)){
    if(!dir || !path.isAbsolute(dir)) continue;

    for(const name of names){
      const full = path.join(dir, name);

      if(fs.existsSync(full)){
        gitFound = full;
        return gitFound;
      }
    }
  }

  gitFound = '';
  return gitFound;
};

const appRoot = process.cwd();
const dataDir = path.join(appRoot, 'data');

export const port = process.env.PORT || 3001;
export const clientPort = process.env.CLIENT_PORT || 5173;
export const runnerUrl = process.env.SNEO_RUNNER_URL || 'http://localhost:11434';
export const defaultModel = process.env.SNEO_DEFAULT_MODEL || 'gemma4:latest';

export const paths = {
  appRoot,
  dataDir,
  distDir: path.join(appRoot, 'dist'),
  chatsFile: path.join(dataDir, 'chats.json'),
  settingsFile: path.join(dataDir, 'settings.json'),
  modelsFile: path.join(dataDir, 'models.json'),
  projectsFile: path.join(dataDir, 'projects.json'),
  undoFile: path.join(dataDir, 'undo.json'),
  pendingFile: path.join(dataDir, 'pending-actions.json'),
  agentLogsFile: path.join(dataDir, 'agent-logs.json'),
  memoryFile: path.join(dataDir, 'memory.json'),
  plansFile: path.join(dataDir, 'plans.json'),
  contextFile: path.join(dataDir, 'context-index.json'),
  terminalFile: path.join(dataDir, 'terminal-sessions.json'),
  sandboxSessionsFile: path.join(dataDir, 'sandbox-sessions.json'),
  backupsDir: path.join(dataDir, 'backups'),
  sandboxesDir: path.join(dataDir, 'sandboxes'),
  projectsRoot: path.resolve(appRoot, process.env.PROJECTS_DIR || '../projectos')
};
