import { spawn } from 'child_process';
import { getProjectPath } from './projects.js';
import { childEnv } from './config.js';
import { endTree, trackChild } from './runs.js';
import { cleanText, makeId, readTerminalSessions, saveTerminalSessions } from './storage.js';

const liveSessions = new Map();

const pushOutput = (session, text) => {
  session.output += text;
  session.output = session.output.slice(-40000);
};

const followChild = async (session, child) => {
  session.pid = child.pid || null;
  child.stdout.on('data', data => pushOutput(session, data.toString()));
  child.stderr.on('data', data => pushOutput(session, data.toString()));
  child.on('error', err => {
    session.status = 'failed';
    pushOutput(session, `\n${err.message}\n`);
  });
  child.on('exit', code => {
    if(session.status === 'running'){
      session.status = 'exited';
    }

    session.exitCode = code;
    session.endedAt = new Date().toISOString();
  });

  liveSessions.set(session.id, {session, child});

  const data = await readTerminalSessions();
  data.sessions.unshift(session);
  data.sessions = data.sessions.slice(0, 100);
  await saveTerminalSessions(data);

  return session;
};

export const terminalAdopt = async ({project, command, cwd, child, output}) => {
  const session = {
    id: makeId(),
    project: cleanText(project),
    command,
    cwd,
    status: child.exitCode === null ? 'running' : 'exited',
    output: String(output || '').slice(-40000),
    createdAt: new Date().toISOString()
  };

  return await followChild(session, child);
};

export const terminalStart = async ({project, command}) => {
  const projectPath = await getProjectPath(project);
  const cleanCommand = cleanText(command) || 'cmd.exe';
  const session = {
    id: makeId(),
    project: cleanText(project),
    command: cleanCommand,
    cwd: projectPath,
    status: 'running',
    output: '',
    createdAt: new Date().toISOString()
  };

  const child = spawn(cleanCommand, {
    cwd: projectPath,
    shell: true,
    windowsHide: true,
    detached: process.platform !== 'win32',
    env: childEnv()
  });

  trackChild(child);
  return await followChild(session, child);
};

export const terminalRead = async ({id}) => {
  const live = liveSessions.get(id);

  if(live){
    return live.session;
  }

  const data = await readTerminalSessions();
  return data.sessions.find(item => item.id === id) || null;
};

export const terminalList = async () => {
  const data = await readTerminalSessions();
  const live = [...liveSessions.values()].map(item => item.session);
  const ids = new Set(live.map(item => item.id));

  return [...live, ...data.sessions.filter(item => !ids.has(item.id))].slice(0, 100);
};

export const terminalStop = async ({id}) => {
  const live = liveSessions.get(id);

  if(!live){
    return {id, status: 'not_running'};
  }

  live.session.status = 'stopped';
  live.session.endedAt = new Date().toISOString();
  liveSessions.delete(id);
  await endTree(live.child.pid);

  return live.session;
};
