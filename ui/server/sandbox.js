import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { childEnv, paths } from './config.js';
import { getProjectPath } from './projects.js';
import { endTree, trackChild } from './runs.js';
import { cleanText, makeId, readSandboxSessions, saveSandboxSessions } from './storage.js';

const liveSandboxes = new Map();

const pushOutput = (session, text) => {
  session.output += text;
  session.output = session.output.slice(-50000);
};

export const sandboxStart = async ({project, command, timeoutMs}) => {
  const projectPath = await getProjectPath(project);
  const sandboxName = `${cleanText(project)}-${makeId()}`.replace(/[^a-zA-Z0-9_.-]/g, '-');
  const sandboxPath = path.join(paths.sandboxesDir, sandboxName);
  const cleanCommand = cleanText(command) || 'dir';

  await fs.mkdir(sandboxPath, {recursive: true});
  await fs.cp(projectPath, sandboxPath, {
    recursive: true,
    filter: source => {
      const lower = source.toLowerCase();
      return !lower.includes(`${path.sep}node_modules${path.sep}`) &&
        !lower.endsWith(`${path.sep}node_modules`) &&
        !lower.includes(`${path.sep}.git${path.sep}`) &&
        !lower.endsWith(`${path.sep}.git`) &&
        !lower.includes(`${path.sep}dist${path.sep}`);
    }
  });

  const session = {
    id: makeId(),
    project: cleanText(project),
    command: cleanCommand,
    sandbox: sandboxPath,
    status: 'running',
    output: '',
    createdAt: new Date().toISOString()
  };

  const child = spawn(cleanCommand, {
    cwd: sandboxPath,
    shell: true,
    windowsHide: true,
    detached: process.platform !== 'win32',
    env: childEnv()
  });

  trackChild(child);
  session.pid = child.pid || null;

  const timer = setTimeout(() => {
    if(session.status === 'running'){
      session.status = 'timeout';
      session.endedAt = new Date().toISOString();
      endTree(child.pid);
    }
  }, Number(timeoutMs || 60000));

  child.stdout.on('data', data => pushOutput(session, data.toString()));
  child.stderr.on('data', data => pushOutput(session, data.toString()));
  child.on('error', err => {
    session.status = 'failed';
    pushOutput(session, `\n${err.message}\n`);
  });
  child.on('exit', code => {
    clearTimeout(timer);

    if(session.status === 'running'){
      session.status = 'exited';
    }

    session.exitCode = code;
    session.endedAt = new Date().toISOString();
  });

  liveSandboxes.set(session.id, {session, child, timer});

  const data = await readSandboxSessions();
  data.sessions.unshift(session);
  data.sessions = data.sessions.slice(0, 100);
  await saveSandboxSessions(data);

  return session;
};

export const sandboxRead = async ({id}) => {
  const live = liveSandboxes.get(id);

  if(live){
    return live.session;
  }

  const data = await readSandboxSessions();
  return data.sessions.find(item => item.id === id) || null;
};

export const sandboxStop = async ({id}) => {
  const live = liveSandboxes.get(id);

  if(!live){
    return {id, status: 'not_running'};
  }

  clearTimeout(live.timer);
  live.session.status = 'stopped';
  live.session.endedAt = new Date().toISOString();
  liveSandboxes.delete(id);
  await endTree(live.child.pid);

  return live.session;
};
