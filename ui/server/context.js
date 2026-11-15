import fs from 'fs/promises';
import path from 'path';
import { getProjectPath, relativePath, safeJoin } from './projects.js';
import { cleanText, readContextIndex, saveContextIndex } from './storage.js';

const skipNames = ['node_modules', '.git', 'dist', 'data'];

const walk = async (dirPath, base, files = []) => {
  const entries = await fs.readdir(dirPath, {withFileTypes: true});

  for(const item of entries){
    if(skipNames.includes(item.name)){
      continue;
    }

    const itemPath = path.join(dirPath, item.name);

    if(item.isDirectory()){
      await walk(itemPath, base, files);
      continue;
    }

    files.push(itemPath);
  }

  return files;
};

const summarizeFile = async (projectPath, filePath) => {
  const stat = await fs.stat(filePath);
  const rel = relativePath(projectPath, filePath);

  if(stat.size > 500000){
    return {
      file: rel,
      size: stat.size,
      lines: null,
      summary: 'large file skipped'
    };
  }

  const content = await fs.readFile(filePath, 'utf8');
  const lines = content.split('\n');
  const imports = lines.filter(line => /^\s*import\s|^\s*const\s|^\s*function\s|^\s*class\s|^\s*export\s/.test(line)).slice(0, 30);

  return {
    file: rel,
    size: stat.size,
    lines: lines.length,
    summary: imports.join('\n').slice(0, 2000)
  };
};

export const contextIndexProject = async ({project}) => {
  const cleanProject = cleanText(project);
  const projectPath = await getProjectPath(cleanProject);
  const files = await walk(projectPath, projectPath);
  const summaries = [];

  for(const filePath of files.slice(0, 400)){
    summaries.push(await summarizeFile(projectPath, filePath));
  }

  const data = await readContextIndex();
  data.projects[cleanProject] = {
    project: cleanProject,
    updatedAt: new Date().toISOString(),
    files: summaries
  };

  await saveContextIndex(data);

  return data.projects[cleanProject];
};

export const contextGetProject = async ({project}) => {
  const data = await readContextIndex();
  return data.projects[cleanText(project)] || null;
};

export const contextPickFiles = async ({project, query}) => {
  const index = await contextGetProject({project}) || await contextIndexProject({project});
  const cleanQuery = cleanText(query).toLowerCase();

  if(!cleanQuery){
    return index.files.slice(0, 40);
  }

  return index.files.filter(item => {
    return `${item.file} ${item.summary}`.toLowerCase().includes(cleanQuery);
  }).slice(0, 40);
};

export const contextReadSelection = async ({project, files}) => {
  const projectPath = await getProjectPath(project);
  const list = Array.isArray(files) ? files : [];
  const result = [];

  for(const file of list.slice(0, 12)){
    const filePath = safeJoin(projectPath, file);
    const stat = await fs.stat(filePath);

    if(stat.size > 250000){
      result.push({file, content: 'large file skipped'});
      continue;
    }

    result.push({
      file,
      content: await fs.readFile(filePath, 'utf8')
    });
  }

  return result;
};
