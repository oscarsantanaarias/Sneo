import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sneo-test-'));

fs.mkdirSync(path.join(root, 'data'), {recursive: true});
fs.mkdirSync(path.join(root, 'projects'), {recursive: true});
process.env.PROJECTS_DIR = path.join(root, 'projects');
process.env.SNEO_QUEUE_LIMIT = process.env.SNEO_QUEUE_LIMIT || '1';
process.chdir(root);

process.on('exit', () => {
  try {
    process.chdir(os.tmpdir());
    fs.rmSync(root, {recursive: true, force: true});
  } catch(e) {
    console.log('could not remove', root, e.message);
  }
});

export const testRoot = root;
export const projectsDir = path.join(root, 'projects');
export const fixturesDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

export const writeSettings = (extra = {}) => {
  fs.writeFileSync(path.join(root, 'data', 'settings.json'), JSON.stringify({
    model: 'mock-model',
    language: 'en',
    accessMode: 'manual',
    maxToolLoops: 8,
    reverseTool: 'auto',
    ghidraPort: 1,
    idaPort: 1,
    ...extra
  }, null, 2));
};

export const makeProject = (name, files = {}) => {
  const folder = path.join(root, 'projects', name);
  fs.mkdirSync(folder, {recursive: true});

  for(const [file, content] of Object.entries(files)){
    fs.mkdirSync(path.dirname(path.join(folder, file)), {recursive: true});
    fs.writeFileSync(path.join(folder, file), content);
  }

  return folder;
};

export const waitFor = async (check, timeoutMs = 10000) => {
  const started = Date.now();

  while(Date.now() - started < timeoutMs){
    if(await check()){
      return true;
    }

    await new Promise(resolve => setTimeout(resolve, 50));
  }

  return false;
};

export const isAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch(e) {
    return e.code === 'EPERM';
  }
};
