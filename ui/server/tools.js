import fs from 'fs/promises';
import path from 'path';
import net from 'net';
import { exec, execFile, spawn } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
import { childEnv, gitProgram, paths, safeEnv } from './config.js';
import { isReadOnlyCommand } from './gate.js';
import { contextGetProject, contextIndexProject, contextPickFiles, contextReadSelection } from './context.js';
import { codebaseIndex, codebaseSearch } from './embeddings.js';
import { toolVisualCheck } from './vision.js';
import { reAddDecomp, reIndex, reMap, reSearch } from './reindex.js';
import { toolWebFetch, toolWebSearch } from './websearch.js';
import { vectorRemember, vectorRecall, vectorForget } from './vectormem.js';
import { toolDeepResearch, toolReadDocument, toolSpeak, toolGenerateImage, toolNoteAdd, toolNoteList, toolTaskAdd, toolTaskList, toolTaskDone, toolCompareModels } from './extras.js';
import { memoryAdd, memoryList, memorySearch } from './memory.js';
import { planCreate, planList, planUpdate } from './planner.js';
import { cleanText, makeId, pathExists, pushPending, pushUndo, readProjectsData, readSettings, saveProjectsData } from './storage.js';
import { sandboxRead, sandboxStart, sandboxStop } from './sandbox.js';
import { terminalAdopt, terminalList, terminalRead, terminalStart, terminalStop } from './terminal.js';
import { getProjectPath, getRegisteredProjects, listDir, relativePath, safeJoin } from './projects.js';
import { askApproval, endTree, runStore, trackChild } from './runs.js';
import { decideTool, toolKind } from './gate.js';

const execAsync = promisify(exec);

const serverTalk = /\b(listening|running|started|ready|serving)\b[^\n]{0,80}(\bport\b|https?:\/\/|localhost|127\.0\.0\.1|:\d{2,5}\b)|https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?/i;

const execNoInput = (command, {cwd, timeout, maxBuffer, env}) => new Promise((resolve, reject) => {
  const child = spawn(command, {cwd, shell: true, windowsHide: true, detached: process.platform !== 'win32', env: env || childEnv()});
  const began = Date.now();
  let stdout = '';
  let stderr = '';
  let heardAt = began;
  let timedOut = false;
  let settled = false;

  const timer = setTimeout(() => {
    timedOut = true;
    endTree(child.pid);
  }, timeout);

  const watch = setInterval(() => {
    if(Date.now() - began < 8000 || Date.now() - heardAt < 5000 || !serverTalk.test(`${stdout}\n${stderr}`.slice(-6000))) return;
    settle(() => {
      child.stdout.removeAllListeners('data');
      child.stderr.removeAllListeners('data');
      resolve({stdout, stderr, child, served: true});
    });
  }, 1000);

  const settle = (act) => {
    if(settled) return;
    settled = true;
    clearTimeout(timer);
    clearInterval(watch);
    act();
  };

  trackChild(child);
  child.stdout.on('data', data => {
    stdout = (stdout + data).slice(-maxBuffer);
    heardAt = Date.now();
  });
  child.stderr.on('data', data => {
    stderr = (stderr + data).slice(-maxBuffer);
    heardAt = Date.now();
  });
  child.on('error', err => settle(() => reject(Object.assign(err, {stdout, stderr}))));
  child.on('close', code => settle(() => {
    if(code === 0){
      return resolve({stdout, stderr});
    }

    const why = timedOut ? `Command stopped after ${Math.round(timeout / 1000)} s without finishing` : 'Command failed';
    reject(Object.assign(new Error(`${why}: ${command}\n${(stderr.trim() ? stderr : stdout).slice(-8000)}`), {code, stdout, stderr}));
  }));
  child.stdin?.end();
});

const interactiveScaffolder = (cmd) => /\b(npm|yarn|pnpm|bun)(\.cmd|\.exe)?\s+create\b|\bcreate-(vite|react-app|next-app|svelte)\b|\bnpx(\.cmd|\.exe)?\b.*\b(create-|tailwindcss\s+init|shadcn)\b|\bnpm(\.cmd|\.exe)?\s+init\b(?!.*\s-y\b)/i.test(cmd);

const skipNames = ['node_modules', '.git', 'dist', 'data'];

const walkFiles = async (dirPath, base, found = []) => {
  const entries = await fs.readdir(dirPath, {withFileTypes: true});

  for(const item of entries){
    if(skipNames.includes(item.name)){
      continue;
    }

    const itemPath = path.join(dirPath, item.name);

    if(item.isDirectory()){
      await walkFiles(itemPath, base, found);
      continue;
    }

    found.push(relativePath(base, itemPath));
  }

  return found;
};

const patternToRegex = (pattern) => {
  const escaped = String(pattern || '*')
    .replace(/\*\*\//g, 'DBLSTARSLASH')
    .replace(/\*\*/g, 'DBLSTAR')
    .replace(/\*/g, 'SNGSTAR')
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/DBLSTARSLASH/g, '(?:.*/)?')
    .replace(/DBLSTAR/g, '.*')
    .replace(/SNGSTAR/g, '[^/]*');

  return new RegExp(`^${escaped}$`, 'i');
};

const getLineStats = (oldText, newText) => {
  const oldLines = String(oldText ?? '') ? String(oldText ?? '').split('\n') : [];
  const newLines = String(newText ?? '') ? String(newText ?? '').split('\n') : [];
  const max = Math.max(oldLines.length, newLines.length);
  let added = 0;
  let removed = 0;
  let changed = 0;

  for(let i = 0; i < max; i++){
    if(oldLines[i] === newLines[i]){
      continue;
    }

    if(oldLines[i] === undefined){
      added++;
      continue;
    }

    if(newLines[i] === undefined){
      removed++;
      continue;
    }

    changed++;
  }

  return {
    added,
    removed,
    changed,
    delta: newLines.length - oldLines.length
  };
};

const patchFile = (file, oldContent, newContent, exists) => ({
  file,
  exists,
  conflict: false,
  reason: '',
  oldContent,
  newContent,
  diff: makeDiff(oldContent, newContent),
  lines: getLineStats(oldContent, newContent)
});

const waitForWrite = async (project, files, title) => {
  if(runStore.getStore()?.mode !== 'manual'){
    return 'approved';
  }

  return await askApproval({type: 'patch', project: cleanText(project), title, files});
};

const deniedWrite = (file) => ({
  file,
  action: 'denied',
  error: 'The user denied this change, so the file was not touched. Do not send the same change again: ask the user what to change or continue another way.'
});

const actionTitle = (name, args) => {
  return name === 'delete_path'
    ? `Delete ${cleanText(args.target)}`
    : name === 'rename_path'
      ? `Rename ${cleanText(args.from)} to ${cleanText(args.to)}`
      : name === 'create_folder'
        ? `Create folder ${cleanText(args.folder)}`
        : name === 'git_commit'
          ? `Commit: ${cleanText(args.message).slice(0, 80)}`
          : name === 'ghidra' || name === 'ida'
            ? `${name === 'ida' ? 'IDA' : 'Ghidra'}: ${cleanText(args.command).slice(0, 120)}`
            : `Use ${name}`;
};

export const checkTool = async (name, args = {}) => {
  const run = runStore.getStore();

  if(!run){
    return;
  }

  let cwd = '';

  if(cleanText(args.project)){
    try {
      cwd = await getProjectPath(args.project);
    } catch(e) {
      cwd = '';
    }
  }

  if(toolKind(name) === 'command' && !cleanText(args.command)){
    args.command = name === 'terminal_start'
      ? 'cmd.exe'
      : name === 'sandbox_start'
        ? 'dir'
        : name === 'sandbox_run' && cwd
          ? await getDefaultSandboxCommand(cwd)
          : '';
  }

  if(toolKind(name) === 'command' && !args.command){
    return;
  }

  const decision = decideTool(run.mode, name, args, cwd);

  if(decision === 'run' || (decision === 'ask' && toolKind(name) === 'file')){
    return;
  }

  if(decision === 'refuse'){
    throw new Error(`Plan mode is read-only, so ${name} was not run and nothing changed. Keep this step in the plan; the user can run the plan afterwards.`);
  }

  const command = toolKind(name) === 'command' ? cleanText(args.command) : '';
  const answer = await askApproval({
    type: command ? 'command' : 'action',
    tool: name,
    project: cleanText(args.project),
    title: command ? `Run ${command}` : actionTitle(name, args),
    command,
    cwd,
    details: command ? '' : JSON.stringify(args, null, 2).slice(0, 2000)
  });

  if(answer !== 'approved'){
    throw new Error(`The user denied ${name}, so nothing was run. Do not send it again: continue another way or ask the user.`);
  }
};

export const toolQueueWrite = async ({project, file, content}) => {
  const projectPath = await getProjectPath(project);
  const filePath = safeJoin(projectPath, cleanText(file));
  const existed = await pathExists(filePath);
  const oldContent = existed ? await fs.readFile(filePath, 'utf8') : '';
  const newContent = String(content ?? '');
  const pending = await pushPending({
    type: 'patch',
    project: cleanText(project),
    title: `Write ${cleanText(file)}`,
    files: [patchFile(cleanText(file), oldContent, newContent, existed)]
  });

  return {pending: true, id: pending.id, file: cleanText(file), status: 'waiting for the user to approve', proposed: getLineStats(oldContent, newContent)};
};

export const toolListProjects = async () => {
  return await getRegisteredProjects();
};

export const toolAddProject = async ({projectPath, name}) => {
  const cleanPath = cleanText(projectPath);

  if(!cleanPath){
    throw new Error('projectPath is required (an absolute path, e.g. D:\\work\\my-app)');
  }

  const finalPath = path.resolve(cleanPath);

  if(!(await pathExists(finalPath))){
    throw new Error(`La ruta no existe: ${finalPath}`);
  }

  const data = await readProjectsData();
  const finalName = cleanText(name) || path.basename(finalPath);

  if(!data.projects.some(item => item.name === finalName)){
    data.projects.push({name: finalName, path: finalPath});
    await saveProjectsData(data);
  }

  return {registered: finalName, path: finalPath};
};

export const toolListFiles = async ({project}) => {
  const projectPath = await getProjectPath(project);
  await fs.mkdir(projectPath, {recursive: true});

  return await listDir(projectPath, 3, projectPath);
};

export const toolReadFile = async ({project, file}) => {
  const projectPath = await getProjectPath(project);
  const filePath = safeJoin(projectPath, cleanText(file));
  const stat = await fs.stat(filePath);

  if(!stat.isFile()){
    throw new Error('No es archivo');
  }

  if(stat.size > 500000){
    throw new Error('Archivo muy grande para leer aqui');
  }

  return await fs.readFile(filePath, 'utf8');
};

export const toolReadRange = async ({project, file, start, end}) => {
  const content = await toolReadFile({project, file});
  const lines = content.split('\n');
  const first = Math.max(1, Number(start || 1));
  const last = Math.min(lines.length, Number(end || first + 120));

  return {
    file: cleanText(file),
    start: first,
    end: last,
    totalLines: lines.length,
    content: lines.slice(first - 1, last).map((line, index) => `${first + index}: ${line}`).join('\n')
  };
};

export const toolWriteFile = async ({project, file, content}) => {
  const projectPath = await getProjectPath(project);
  const filePath = safeJoin(projectPath, cleanText(file));
  const existed = await pathExists(filePath);
  const oldContent = existed ? await fs.readFile(filePath, 'utf8') : '';
  const newContent = content !== null && typeof content === 'object'
    ? JSON.stringify(content, null, 2)
    : String(content ?? '');

  if(existed && newContent === oldContent){
    return {
      file: relativePath(projectPath, filePath),
      action: 'no-change',
      warning: 'NOTHING CHANGED: the content you wrote is identical to what is already in the file. This did nothing. Stop repeating it. Make a REAL change or tell the user you could not fix it.'
    };
  }

  if(await waitForWrite(project, [patchFile(cleanText(file), oldContent, newContent, existed)], `Write ${cleanText(file)}`) !== 'approved'){
    return deniedWrite(cleanText(file));
  }

  const backupPath = path.join(paths.backupsDir, makeId());

  if(existed){
    await fs.cp(filePath, backupPath, {recursive: true});
  }

  await fs.mkdir(path.dirname(filePath), {recursive: true});
  await fs.writeFile(filePath, newContent);

  await pushUndo({
    action: existed ? 'restore' : 'delete-created',
    project: cleanText(project),
    target: cleanText(file),
    backupPath: existed ? backupPath : null
  });

  const rows = diffRows(oldContent, newContent);
  const total = oldContent.split('\n').length;
  const touched = rows.filter(row => row.kind === 'add' || row.kind === 'del').length;
  // reescribir un archivo entero para cambiar pocas lineas son minutos escribiendo lo mismo asi que se le avisa al modelo para la proxima
  const wasteful = existed && total >= 60 && touched * 10 <= total;

  return {
    file: relativePath(projectPath, filePath),
    action: existed ? 'updated' : 'created',
    lines: getLineStats(oldContent, newContent),
    change: {file: relativePath(projectPath, filePath), created: !existed, rows},
    ...(wasteful ? {note: `You rewrote all ${total} lines of this file to change ${touched}. Next time send only the lines that change with replace_in_file {file, search, replace}: writing the whole file again takes minutes.`} : {})
  };
};

export const toolDeletePath = async ({project, target}) => {
  const projectPath = await getProjectPath(project);
  const targetPath = safeJoin(projectPath, cleanText(target));
  const backupPath = path.join(paths.backupsDir, makeId());

  if(targetPath === projectPath){
    throw new Error('Cannot delete project root');
  }

  if(await pathExists(targetPath)){
    await fs.cp(targetPath, backupPath, {recursive: true});
  }

  await fs.rm(targetPath, {recursive: true, force: true});

  await pushUndo({
    action: 'restore',
    project: cleanText(project),
    target: cleanText(target),
    backupPath
  });

  return `Eliminado: ${relativePath(projectPath, targetPath)}`;
};

export const toolCreateFolder = async ({project, folder}) => {
  const projectPath = await getProjectPath(project);
  const folderPath = safeJoin(projectPath, cleanText(folder));
  const existed = await pathExists(folderPath);

  await fs.mkdir(folderPath, {recursive: true});

  if(!existed){
    await pushUndo({
      action: 'delete-created',
      project: cleanText(project),
      target: cleanText(folder)
    });
  }

  return `Carpeta creada: ${relativePath(projectPath, folderPath)}`;
};

export const toolRenamePath = async ({project, from, to}) => {
  const projectPath = await getProjectPath(project);
  const fromPath = safeJoin(projectPath, cleanText(from));
  const toPath = safeJoin(projectPath, cleanText(to));

  await fs.mkdir(path.dirname(toPath), {recursive: true});
  await fs.rename(fromPath, toPath);

  await pushUndo({
    action: 'rename',
    project: cleanText(project),
    from: cleanText(to),
    to: cleanText(from)
  });

  return `Renombrado: ${cleanText(from)} -> ${cleanText(to)}`;
};

export const toolSearchFiles = async ({project, query}) => {
  const projectPath = await getProjectPath(project);
  const cleanQuery = cleanText(query).toLowerCase();
  const found = [];

  if(!cleanQuery){
    throw new Error('query requerido');
  }

  const walk = async (dirPath) => {
    const entries = await fs.readdir(dirPath, {withFileTypes: true});

    for(const item of entries){
      if(['node_modules', '.git', 'dist', 'data'].includes(item.name)){
        continue;
      }

      const itemPath = path.join(dirPath, item.name);

      if(item.isDirectory()){
        await walk(itemPath);
        continue;
      }

      const rel = relativePath(projectPath, itemPath);

      if(item.name.toLowerCase().includes(cleanQuery)){
        found.push({file: rel, match: 'filename'});
        continue;
      }

      const stat = await fs.stat(itemPath);

      if(stat.size > 300000){
        continue;
      }

      const content = await fs.readFile(itemPath, 'utf8');

      content.split('\n').forEach((line, index) => {
        if(line.toLowerCase().includes(cleanQuery) && found.length < 60){
          found.push({
            file: rel,
            line: index + 1,
            text: line.slice(0, 220)
          });
        }
      });
    }
  };

  await walk(projectPath);
  return found.slice(0, 60);
};

export const toolGlobFiles = async ({project, pattern}) => {
  const projectPath = await getProjectPath(project);
  const cleanPattern = cleanText(pattern) || '**/*';
  const regex = patternToRegex(cleanPattern);
  const rootRegex = cleanPattern.startsWith('**/') ? patternToRegex(cleanPattern.slice(3)) : null;
  const files = await walkFiles(projectPath, projectPath);

  return files.filter(file => regex.test(file) || rootRegex?.test(file)).slice(0, 300);
};

export const toolGrepFiles = async ({project, query, pattern}) => {
  const projectPath = await getProjectPath(project);
  const files = pattern ? await toolGlobFiles({project, pattern}) : await walkFiles(projectPath, projectPath);
  const cleanQuery = cleanText(query).toLowerCase();
  const found = [];

  if(!cleanQuery){
    throw new Error('query requerido');
  }

  for(const file of files){
    const filePath = safeJoin(projectPath, file);
    const stat = await fs.stat(filePath);

    if(stat.size > 300000){
      continue;
    }

    const content = await fs.readFile(filePath, 'utf8');
    const lines = content.split('\n');

    lines.forEach((line, index) => {
      if(line.toLowerCase().includes(cleanQuery) && found.length < 120){
        found.push({
          file,
          line: index + 1,
          text: line.slice(0, 240)
        });
      }
    });
  }

  return found;
};

const blankEdges = (lines) => {
  let first = 0;
  let last = lines.length;
  while(first < last && !lines[first].trim()) first++;
  while(last > first && !lines[last - 1].trim()) last--;
  return [first, lines.length - last];
};

export const swapText = (content, search, replace, all = false) => {
  const from = String(search ?? '');
  const to = String(replace ?? '');

  if(!from){
    return null;
  }

  const tries = [[from, to]];

  if(content.includes('\r\n') && from.includes('\n') && !from.includes('\r\n')){
    tries.push([from.replace(/\n/g, '\r\n'), to.replace(/\r?\n/g, '\r\n')]);
  }

  for(const [a, b] of tries){
    if(!content.includes(a)) continue;
    const many = content.split(a).length - 1;
    return {
      content: all ? content.split(a).join(b) : content.replace(a, () => b),
      count: all ? many : 1,
      many
    };
  }

  // mismas lineas aunque cambien los espacios al final o las lineas vacias de las puntas
  const eol = content.includes('\r\n') ? '\r\n' : '\n';
  const lines = content.split(eol);
  let wanted = from.replace(/\r\n/g, '\n').split('\n');
  let added = to.replace(/\r\n/g, '\n').split('\n');
  const [headWanted, tailWanted] = blankEdges(wanted);
  const [headAdded, tailAdded] = blankEdges(added);
  wanted = wanted.slice(headWanted, wanted.length - tailWanted);
  added = added.slice(Math.min(headWanted, headAdded), added.length - Math.min(tailWanted, tailAdded));

  if(!wanted.length){
    return null;
  }

  const hits = [];

  for(let i = 0; i + wanted.length <= lines.length && hits.length < 2; i++){
    let same = true;
    for(let j = 0; j < wanted.length && same; j++) same = lines[i + j].trimEnd() === wanted[j].trimEnd();
    if(same) hits.push(i);
  }

  if(hits.length !== 1){
    return null;
  }

  return {
    content: [...lines.slice(0, hits[0]), ...added, ...lines.slice(hits[0] + wanted.length)].join(eol),
    count: 1,
    many: 1,
    loose: true
  };
};

// filas del cambio para que la app lo muestre lado a lado
export const diffRows = (before, after, context = 3, most = 300) => {
  const a = String(before ?? '').replace(/\r\n/g, '\n').split('\n');
  const b = String(after ?? '').replace(/\r\n/g, '\n').split('\n');
  if(before === '' || before === undefined || before === null) a.length = 0;
  let head = 0;
  while(head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while(tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const midA = a.slice(head, a.length - tail);
  const midB = b.slice(head, b.length - tail);
  const rows = a.slice(0, head).map(text => ({kind: 'same', text}));

  if(midA.length * midB.length > 2000000){
    for(const text of midA) rows.push({kind: 'del', text});
    for(const text of midB) rows.push({kind: 'add', text});
  } else {
    const width = midB.length + 1;
    const table = new Uint32Array((midA.length + 1) * width);
    for(let i = midA.length - 1; i >= 0; i--){
      for(let j = midB.length - 1; j >= 0; j--){
        table[i * width + j] = midA[i] === midB[j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
      }
    }
    let i = 0;
    let j = 0;
    while(i < midA.length || j < midB.length){
      if(i < midA.length && j < midB.length && midA[i] === midB[j]){
        rows.push({kind: 'same', text: midA[i]});
        i++;
        j++;
      } else if(i < midA.length && (j >= midB.length || table[(i + 1) * width + j] >= table[i * width + j + 1])){
        rows.push({kind: 'del', text: midA[i]});
        i++;
      } else {
        rows.push({kind: 'add', text: midB[j]});
        j++;
      }
    }
  }

  for(const text of a.slice(a.length - tail)) rows.push({kind: 'same', text});

  const near = rows.map(() => false);
  rows.forEach((row, index) => {
    if(row.kind === 'same') return;
    for(let k = Math.max(0, index - context); k <= Math.min(rows.length - 1, index + context); k++) near[k] = true;
  });

  const shown = [];
  let skipped = 0;
  let cut = 0;
  rows.forEach((row, index) => {
    if(!near[index]){
      skipped++;
      return;
    }
    if(skipped){
      shown.push({kind: 'gap', count: skipped});
      skipped = 0;
    }
    if(shown.length >= most){
      cut++;
      return;
    }
    shown.push({kind: row.kind, text: row.text.length > 400 ? `${row.text.slice(0, 400)}…` : row.text});
  });
  if(skipped) shown.push({kind: 'gap', count: skipped});
  if(cut) shown.push({kind: 'more', count: cut});
  return shown;
};

const missingText = (file) => `The search text was not found in ${file}. Read the file again (read_file or read_range) and copy only the few exact lines that change, indentation included, then retry: a short search matches, a big block copied from memory does not. To rewrite the whole file use write_file.`;

export const toolReplaceInFile = async ({project, file, search, replace, all}) => {
  const projectPath = await getProjectPath(project);
  const filePath = safeJoin(projectPath, cleanText(file));
  const backupPath = path.join(paths.backupsDir, makeId());
  const cleanSearch = String(search ?? '');

  if(!cleanSearch){
    throw new Error('search is required: the exact text to replace.');
  }

  const content = await fs.readFile(filePath, 'utf8');
  const swapped = swapText(content, cleanSearch, replace, all === true);

  if(!swapped){
    throw new Error(missingText(cleanText(file)));
  }

  const nextContent = swapped.content;

  if(nextContent === content){
    return {
      file: cleanText(file),
      action: 'no-change',
      warning: 'NOTHING CHANGED: your search and replace produce identical content. This edit did nothing. Stop repeating it. Read the file, find the EXACT line to change, and use a replace that is actually different - or tell the user you could not locate the cause.'
    };
  }

  if(await waitForWrite(project, [patchFile(cleanText(file), content, nextContent, true)], `Edit ${cleanText(file)}`) !== 'approved'){
    return deniedWrite(cleanText(file));
  }

  await fs.cp(filePath, backupPath, {recursive: true});
  await fs.writeFile(filePath, nextContent);

  await pushUndo({
    action: 'restore',
    project: cleanText(project),
    target: cleanText(file),
    backupPath
  });

  return {
    file: cleanText(file),
    action: 'patched',
    lines: getLineStats(content, nextContent),
    change: {file: cleanText(file), rows: diffRows(content, nextContent)}
  };
};

const makeDiff = (oldText, newText) => {
  const oldLines = String(oldText ?? '').split('\n');
  const newLines = String(newText ?? '').split('\n');
  const max = Math.max(oldLines.length, newLines.length);
  const lines = [];

  for(let i = 0; i < max; i++){
    if(oldLines[i] === newLines[i]){
      continue;
    }

    if(oldLines[i] !== undefined){
      lines.push(`- ${oldLines[i]}`);
    }

    if(newLines[i] !== undefined){
      lines.push(`+ ${newLines[i]}`);
    }

    if(lines.length > 120){
      lines.push('...diff truncated...');
      break;
    }
  }

  return lines.join('\n');
};

export const toolProposePatch = async ({project, changes, title}) => {
  const cleanProject = cleanText(project);
  const projectPath = await getProjectPath(cleanProject);
  const cleanChanges = Array.isArray(changes) ? changes : [];
  const byFile = new Map();

  if(cleanChanges.length === 0){
    throw new Error('changes is required: a list of {file, search, replace}.');
  }

  // varios cambios al mismo archivo se aplican uno sobre otro
  for(const change of cleanChanges){
    const file = cleanText(change.file);

    if(!byFile.has(file)){
      const filePath = safeJoin(projectPath, file);
      const exists = await pathExists(filePath);
      const oldContent = exists ? await fs.readFile(filePath, 'utf8') : '';
      byFile.set(file, {file, exists, conflict: false, reason: '', oldContent, newContent: oldContent});
    }

    const item = byFile.get(file);

    if(item.conflict) continue;

    if(change.content !== undefined){
      item.newContent = String(change.content ?? '');
      continue;
    }

    const swapped = swapText(item.newContent, change.search, change.replace, change.all === true);

    if(!swapped){
      item.conflict = true;
      item.reason = `search text not found: ${String(change.search ?? '').slice(0, 120)}`;
      continue;
    }

    item.newContent = swapped.content;
  }

  const files = [...byFile.values()].map(item => ({
    ...item,
    diff: makeDiff(item.oldContent, item.newContent),
    lines: getLineStats(item.oldContent, item.newContent)
  }));
  const conflicts = files.filter(item => item.conflict);

  if(conflicts.length){
    return {
      applied: false,
      error: `Nothing was applied: ${conflicts.length} change(s) did not match the file. Read the file again, copy the exact lines into search and retry.`,
      files: files.map(item => ({file: item.file, conflict: item.conflict, reason: item.reason}))
    };
  }

  if(await waitForWrite(cleanProject, files, cleanText(title) || 'Proposed patch') !== 'approved'){
    return {applied: false, denied: true, error: 'The user denied this patch, so no file was touched. Do not send it again: ask the user what to change.', files: files.map(item => ({file: item.file}))};
  }

  for(const item of files){
    const filePath = safeJoin(projectPath, item.file);
    const existed = await pathExists(filePath);
    const backupPath = path.join(paths.backupsDir, makeId());

    if(existed){
      await fs.cp(filePath, backupPath, {recursive: true});
    }

    await fs.mkdir(path.dirname(filePath), {recursive: true});
    await fs.writeFile(filePath, item.newContent);

    await pushUndo({
      action: existed ? 'restore' : 'delete-created',
      project: cleanProject,
      target: item.file,
      backupPath: existed ? backupPath : null
    });
  }

  return {
    applied: true,
    title: cleanText(title) || 'Patch',
    files: files.map(item => ({file: item.file, action: item.exists ? 'updated' : 'created', lines: item.lines})),
    changes: files.map(item => ({file: item.file, created: !item.exists, rows: diffRows(item.oldContent, item.newContent)}))
  };
};

const pcSkip = new Set(['node_modules', '.git', 'windows', '$recycle.bin', 'appdata', 'programdata', 'system volume information', 'winsxs', '.cache', 'temp']);

const pcTextExt = new Set(['.txt', '.md', '.log', '.json', '.xml', '.csv', '.ini', '.cfg', '.yml', '.yaml', '.html', '.htm', '.js', '.jsx', '.ts', '.tsx', '.css', '.py', '.c', '.cpp', '.h', '.cs', '.lua', '.java', '.go', '.rs', '.php', '.sql', '.sh', '.bat', '.ps1', '']);

export const toolPcSearch = async ({query, root, content}) => {
  const cleanQuery = cleanText(query).toLowerCase();
  const wantContent = cleanText(content).toLowerCase();

  if(!cleanQuery && !wantContent){
    throw new Error('give query (filename) and/or content (text inside the file) to search for');
  }

  const roots = root ? [cleanText(root)] : [
    path.join(process.env.USERPROFILE || 'C:\\Users', 'Desktop'),
    path.join(process.env.USERPROFILE || 'C:\\Users', 'Downloads'),
    path.join(process.env.USERPROFILE || 'C:\\Users', 'Documents')
  ];

  const results = [];
  const deadline = Date.now() + 25000;

  const walk = async (dir, depth) => {
    if(results.length >= 100 || Date.now() > deadline || depth > 8){
      return;
    }

    let entries;

    try {
      entries = await fs.readdir(dir, {withFileTypes: true});
    } catch(e) {
      return;
    }

    for(const item of entries){
      if(results.length >= 100 || Date.now() > deadline){
        return;
      }

      const lower = item.name.toLowerCase();

      if(item.isDirectory()){
        if(pcSkip.has(lower) || lower.startsWith('$')){
          continue;
        }

        await walk(path.join(dir, item.name), depth + 1);
        continue;
      }

      const genericQuery = /^\.?[a-z0-9]{1,4}$/.test(cleanQuery);
      const nameMatch = cleanQuery && lower.includes(cleanQuery);
      const full = path.join(dir, item.name);

      if(wantContent){
        const ext = path.extname(lower);

        if(!pcTextExt.has(ext)){
          continue;
        }

        if(cleanQuery && !genericQuery && !nameMatch){
          continue;
        }

        try {
          const stat = await fs.stat(full);

          if(stat.size < 3000000 && (await fs.readFile(full, 'utf8')).toLowerCase().includes(wantContent)){
            results.push(full);
          }
        } catch(e) {
        }
      } else if(nameMatch){
        results.push(full);
      }
    }
  };

  for(const base of roots){
    if(Date.now() > deadline){
      break;
    }

    await walk(base, 0);
  }

  return {query: cleanQuery || null, content: wantContent || null, roots, found: results.length, timedOut: Date.now() > deadline, files: results.slice(0, 100)};
};

const runGit = async (projectPath, args, maxBuffer) => {
  const git = gitProgram();

  if(!git){
    throw new Error('git was not found in the PATH');
  }

  return await execFileAsync(git, args, {cwd: projectPath, timeout: 15000, maxBuffer, windowsHide: true, env: safeEnv()});
};

export const toolGitStatus = async ({project}) => {
  const projectPath = await getProjectPath(project);
  const result = await runGit(projectPath, ['status', '--short', '--branch'], 1024 * 400);

  return result.stdout || result.stderr;
};

export const toolGitDiff = async ({project, staged}) => {
  const projectPath = await getProjectPath(project);
  const result = await runGit(projectPath, staged ? ['diff', '--staged'] : ['diff'], 1024 * 900);

  return (result.stdout || result.stderr).slice(-20000);
};

export const toolGitCommit = async ({project, message}) => {
  const projectPath = await getProjectPath(project);
  const cleanMessage = cleanText(message);

  if(!cleanMessage){
    throw new Error('a commit message is required');
  }

  await runGit(projectPath, ['add', '-A'], 1024 * 900);
  const result = await runGit(projectPath, ['commit', '-m', cleanMessage], 1024 * 900);

  return result.stdout || result.stderr;
};

export const toolRunCommand = async ({project, command}) => {
  const projectPath = await getProjectPath(project);
  const cleanCommand = cleanText(command);

  if(!cleanCommand){
    throw new Error('command required');
  }

  if(interactiveScaffolder(cleanCommand)){
    return {
      note: 'Interactive scaffolder: it prompts (directory not empty, framework, "Ok to proceed?") and HANGS when run without a keyboard. Do NOT use it. Instead create the project files directly with write_file: package.json (with scripts and deps), vite.config.js, tailwind.config.js, postcss.config.js, index.html and src/*. Then run "npm.cmd install" and "npm.cmd run build". That is fully non-interactive.',
      command: cleanCommand,
      blocked: true
    };
  }

  const result = await execNoInput(cleanCommand, {
    cwd: projectPath,
    timeout: 300000,
    maxBuffer: 1024 * 1024 * 10,
    windowsHide: true,
    env: isReadOnlyCommand(cleanCommand, projectPath) ? safeEnv() : childEnv()
  });

  if(result.served){
    const heard = `${result.stdout}\n${result.stderr}`;
    const session = await terminalAdopt({project, command: cleanCommand, cwd: projectPath, child: result.child, output: heard});
    const port = heard.match(/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\bport)\D{0,3}(\d{2,5})/i);

    return {
      background: true,
      terminalId: session.id,
      ...(port ? {port: Number(port[1])} : {}),
      note: 'This command started a server that keeps running, so it was moved to the background as a terminal session. It is running now and it works: do not start it again. Read its output with terminal_read and stop it with terminal_stop, both with this terminalId.',
      stdout: result.stdout.slice(-6000),
      stderr: result.stderr.slice(-6000)
    };
  }

  return {
    stdout: result.stdout.slice(-12000),
    stderr: result.stderr.slice(-12000)
  };
};

const queryTcpBridge = async ({command, port, label}) => {
  const cleanCommand = cleanText(command);

  if(!cleanCommand){
    throw new Error(`command requerido para ${label}`);
  }

  const targetPort = Number(port);

  return await new Promise((resolve, reject) => {
    const socket = net.createConnection({host: '127.0.0.1', port: targetPort});
    let data = '';
    let settled = false;
    const fail = (error) => {
      if(settled){
        return;
      }

      settled = true;
      reject(error);
    };

    socket.setTimeout(15000);
    socket.on('connect', () => socket.write(`${cleanCommand}\n`));
    socket.on('data', chunk => {
      data += chunk.toString('utf8');
    });
    socket.on('end', () => {
      if(settled){
        return;
      }

      settled = true;
      resolve({port: targetPort, command: cleanCommand, response: data.trim().slice(0, 30000)});
    });
    socket.on('timeout', () => {
      socket.destroy();
      fail(`${label} no respondio en el puerto ${targetPort} (timeout)`);
    });
    socket.on('error', err => fail(`No pude conectar a ${label} en 127.0.0.1:${targetPort} (${err.code || err.message})`));
  });
};

export const toolGhidra = async ({command, port}) => {
  const settings = await readSettings();
  const result = await queryTcpBridge({command, port: Number(port) || settings.ghidraPort, label: 'Ghidra'});
  const decompMatch = /^DECOMP\s+([0-9a-fx]+)/i.exec(result.command);

  if(decompMatch && result.response.includes('"ok": true')){
    reAddDecomp(decompMatch[1], result.response);
  }

  return result;
};

export const toolIda = async ({command, port}) => {
  const settings = await readSettings();
  return await queryTcpBridge({command, port: Number(port) || settings.idaPort, label: 'IDA'});
};

const getDefaultSandboxCommand = async (sandboxPath) => {
  const packagePath = path.join(sandboxPath, 'package.json');

  if(await pathExists(packagePath)){
    const raw = await fs.readFile(packagePath, 'utf8');
    const data = JSON.parse(raw);

    if(data.scripts?.build) return 'npm.cmd run build';
    if(data.scripts?.test) return 'npm.cmd test';
    if(data.scripts?.dev) return 'npm.cmd run dev -- --help';
  }

  return 'dir';
};

export const toolSandboxRun = async ({project, command}) => {
  const projectPath = await getProjectPath(project);
  const sandboxName = `${cleanText(project)}-${makeId()}`.replace(/[^a-zA-Z0-9_.-]/g, '-');
  const sandboxPath = path.join(paths.sandboxesDir, sandboxName);

  await fs.mkdir(sandboxPath, {recursive: true});
  await fs.cp(projectPath, sandboxPath, {
    recursive: true,
    filter: (source) => {
      const lower = source.toLowerCase();
      return !lower.includes(`${path.sep}node_modules${path.sep}`) &&
        !lower.endsWith(`${path.sep}node_modules`) &&
        !lower.includes(`${path.sep}.git${path.sep}`) &&
        !lower.endsWith(`${path.sep}.git`) &&
        !lower.includes(`${path.sep}dist${path.sep}`);
    }
  });

  const cleanCommand = cleanText(command) || await getDefaultSandboxCommand(sandboxPath);

  if(/\b(run dev|npm start|yarn dev|yarn start|vite|next dev|serve|nodemon|http-server)\b/i.test(cleanCommand) && !/build|--help|-v|version/i.test(cleanCommand)){
    return {
      note: 'Ese comando arranca un servidor que no termina, no uso sandbox_run (se colgaria). Para verificar una web, usa "npm.cmd run build" para confirmar que compila, o sandbox_start para levantar un server en background y sandbox_read para leer su salida.',
      command: cleanCommand,
      blocked: true
    };
  }

  if(interactiveScaffolder(cleanCommand)){
    return {
      note: 'Interactive scaffolder: it prompts and HANGS without a keyboard. Do NOT use it. Create the project files directly with write_file (package.json, vite.config.js, tailwind.config.js, index.html, src/*), then "npm.cmd install" and "npm.cmd run build".',
      command: cleanCommand,
      blocked: true
    };
  }

  const result = await execNoInput(cleanCommand, {
    cwd: sandboxPath,
    timeout: 300000,
    maxBuffer: 1024 * 1024 * 10,
    windowsHide: true
  });

  if(result.served){
    await endTree(result.child.pid);

    return {
      sandbox: sandboxPath,
      command: cleanCommand,
      note: 'This command started a server that keeps running. It started fine and was stopped so the sandbox run could end. Use sandbox_start to keep a server running in the background.',
      stdout: result.stdout.slice(-6000),
      stderr: result.stderr.slice(-6000)
    };
  }

  return {
    sandbox: sandboxPath,
    command: cleanCommand,
    stdout: result.stdout.slice(-16000),
    stderr: result.stderr.slice(-16000)
  };
};

const subTools = ['list_files', 'read_file', 'read_range', 'search_files', 'glob', 'grep', 'ghidra', 'ida', 're_search', 're_map', 'pc_search', 'web_search', 'web_fetch', 'sandbox_run', 'git_status', 'git_diff', 'context_index', 'context_get', 'context_pick', 'context_read', 'codebase_search', 'memory_search', 'memory_list'];

const shellTools = ['run_command', 'sandbox_run', 'sandbox_start', 'terminal_start'];
const guarded = '(node|npm|master|worker|sneo|cmd|powershell|pwsh|conhost|explorer)';
const killsByName = new RegExp([
  `\\b(taskkill|tskill)\\b[^|&;]*\\/im\\s+"?${guarded}(\\.exe)?\\b`,
  `\\b(stop-process|spps)\\b[^|&;]*-(name|processname)\\s+"?${guarded}\\b`,
  `\\bget-process\\s+(-name\\s+)?"?${guarded}\\b[^|]*\\|\\s*(stop-process|spps|kill)\\b`,
  `\\b(killall|pkill)\\b[^|&;]*\\b${guarded}\\b`,
  `\\bwmic\\b[^|&;]*\\b${guarded}\\.exe\\b[^|&;]*\\b(delete|terminate)\\b`
].join('|'), 'i');

const killsSneo = (command) => {
  const text = String(command || '');
  const own = [process.pid, process.ppid].filter(Boolean).join('|');
  return killsByName.test(text) || new RegExp(`\\b(taskkill\\b[^|&;]*\\/pid\\s+|stop-process\\b[^|&;]*-id\\s+|\\bkill\\s+(-\\w+\\s+)*)(${own})\\b`, 'i').test(text);
};

export const runTool = async (name, args = {}) => {
  await checkTool(name, args);

  if(shellTools.includes(name) && killsSneo(args?.command)){
    return {
      blocked: true,
      command: args.command,
      note: 'Blocked: this kills programs by name, which would close Sneo itself (it runs on Node) and every other Node or shell program on this PC. To stop a server you started use terminal_list and terminal_stop with its id. To free a port run "netstat -ano | findstr :PORT" to get the one PID that listens there and then "taskkill /PID that_pid /F".'
    };
  }

  if(name === 'list_projects') return await toolListProjects();
  if(name === 'add_project') return await toolAddProject(args);
  if(name === 'list_files') return await toolListFiles(args);
  if(name === 'read_file') return await toolReadFile(args);
  if(name === 'read_range') return await toolReadRange(args);
  if(name === 'write_file') return await toolWriteFile(args);
  if(name === 'delete_path') return await toolDeletePath(args);
  if(name === 'create_folder') return await toolCreateFolder(args);
  if(name === 'rename_path') return await toolRenamePath(args);
  if(name === 'search_files') return await toolSearchFiles(args);
  if(name === 'glob') return await toolGlobFiles(args);
  if(name === 'grep') return await toolGrepFiles(args);
  if(name === 'replace_in_file') return await toolReplaceInFile(args);
  if(name === 'propose_patch') return await toolProposePatch(args);
  if(name === 'run_command') return await toolRunCommand(args);
  if(name === 'ghidra') return await toolGhidra(args);
  if(name === 'ida') return await toolIda(args);
  if(name === 're_index') return await reIndex(args);
  if(name === 're_search') return await reSearch(args);
  if(name === 're_map') return await reMap(args);
  if(name === 'pc_search') return await toolPcSearch(args);
  if(name === 'web_search') return await toolWebSearch(args);
  if(name === 'web_fetch') return await toolWebFetch(args);
  if(name === 'visual_check') return await toolVisualCheck(args);
  if(name === 'sandbox_run') return await toolSandboxRun(args);
  if(name === 'git_status') return await toolGitStatus(args);
  if(name === 'git_diff') return await toolGitDiff(args);
  if(name === 'git_commit') return await toolGitCommit(args);
  if(name === 'context_index') return await contextIndexProject(args);
  if(name === 'context_get') return await contextGetProject(args);
  if(name === 'context_pick') return await contextPickFiles(args);
  if(name === 'context_read') return await contextReadSelection(args);
  if(name === 'codebase_index') return await codebaseIndex(args);
  if(name === 'codebase_search') return await codebaseSearch(args);
  if(name === 'plan_create') return await planCreate(args);
  if(name === 'plan_update') return await planUpdate(args);
  if(name === 'plan_list') return await planList(args);
  if(name === 'memory_add') return await memoryAdd(args);
  if(name === 'memory_search') return await memorySearch(args);
  if(name === 'memory_list') return await memoryList(args);
  if(name === 'terminal_start') return await terminalStart(args);
  if(name === 'terminal_read') return await terminalRead(args);
  if(name === 'terminal_stop') return await terminalStop(args);
  if(name === 'terminal_list') return await terminalList(args);
  if(name === 'sandbox_start') return await sandboxStart(args);
  if(name === 'sandbox_read') return await sandboxRead(args);
  if(name === 'sandbox_stop') return await sandboxStop(args);
  if(name === 'vector_remember') return await vectorRemember(args);
  if(name === 'vector_recall') return await vectorRecall(args);
  if(name === 'vector_forget') return await vectorForget(args);
  if(name === 'read_document') return await toolReadDocument(args);
  if(name === 'deep_research') return await toolDeepResearch(args);
  if(name === 'speak') return await toolSpeak(args);
  if(name === 'generate_image') return await toolGenerateImage(args);
  if(name === 'note_add') return await toolNoteAdd(args);
  if(name === 'note_list') return await toolNoteList(args);
  if(name === 'task_add') return await toolTaskAdd(args);
  if(name === 'task_list') return await toolTaskList(args);
  if(name === 'task_done') return await toolTaskDone(args);
  if(name === 'compare_models') return await toolCompareModels(args);

  throw new Error(`There is no tool named "${name}" here. To change a file use replace_in_file {file, search, replace} or write_file {file, content}; to run a command use run_command {command}.`);
};

export const runSubTool = async (name, args = {}) => {
  if(!subTools.includes(name)){
    throw new Error(`Subagent tool not allowed: ${name}`);
  }

  await checkTool(name, args);

  if(name === 'list_files') return await toolListFiles(args);
  if(name === 'read_file') return await toolReadFile(args);
  if(name === 'read_range') return await toolReadRange(args);
  if(name === 'search_files') return await toolSearchFiles(args);
  if(name === 'glob') return await toolGlobFiles(args);
  if(name === 'grep') return await toolGrepFiles(args);
  if(name === 'ghidra') return await toolGhidra(args);
  if(name === 'ida') return await toolIda(args);
  if(name === 're_search') return await reSearch(args);
  if(name === 're_map') return await reMap(args);
  if(name === 'pc_search') return await toolPcSearch(args);
  if(name === 'web_search') return await toolWebSearch(args);
  if(name === 'web_fetch') return await toolWebFetch(args);
  if(name === 'sandbox_run') return await toolSandboxRun(args);
  if(name === 'git_status') return await toolGitStatus(args);
  if(name === 'git_diff') return await toolGitDiff(args);
  if(name === 'context_index') return await contextIndexProject(args);
  if(name === 'context_get') return await contextGetProject(args);
  if(name === 'context_pick') return await contextPickFiles(args);
  if(name === 'context_read') return await contextReadSelection(args);
  if(name === 'codebase_search') return await codebaseSearch(args);
  if(name === 'memory_search') return await memorySearch(args);
  if(name === 'memory_list') return await memoryList(args);

  throw new Error(`Subagent tool not allowed: ${name}`);
};
