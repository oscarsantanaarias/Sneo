import fs from 'fs/promises';
import path from 'path';
import { paths } from './config.js';
import { cleanText, readProjectsData } from './storage.js';

export const safeJoin = (base, target = '') => {
  const cleanTarget = String(target || '').replace(/^[/\\]+/, '');
  const finalPath = path.resolve(base, cleanTarget);

  if(finalPath !== base && !finalPath.startsWith(`${base}${path.sep}`)){
    throw new Error('Path outside project blocked');
  }

  return finalPath;
};

export const relativePath = (base, filePath) => {
  return path.relative(base, filePath).replace(/\\/g, '/');
};

export const getRegisteredProjects = async () => {
  const data = await readProjectsData();
  const entries = await fs.readdir(paths.projectsRoot, {withFileTypes: true});
  const localProjects = entries
    .filter(item => item.isDirectory())
    .map(item => ({
      name: item.name,
      path: path.join(paths.projectsRoot, item.name)
    }));

  const all = [...data.projects, ...localProjects];
  const seen = new Set();

  return all.filter(item => {
    const key = item.name.toLowerCase();

    if(seen.has(key)){
      return false;
    }

    seen.add(key);
    return true;
  }).sort((a, b) => a.name.localeCompare(b.name));
};

export const isAbsolutePath = (value) => /^[a-zA-Z]:[\\/]/.test(value) || value.startsWith('\\\\') || value.startsWith('/');

// el nombre como se compara: sin diferencias de espacios ni de mayusculas
const plainName = (value) => cleanText(value).replace(/\s+/g, ' ').toLowerCase();

export const getProjectPath = async (project) => {
  const cleanProject = cleanText(project);

  if(!cleanProject){
    throw new Error('project requerido');
  }

  if(isAbsolutePath(cleanProject)){
    return path.resolve(cleanProject);
  }

  const projects = await getRegisteredProjects();
  const found = projects.find(item => item.name === cleanProject)
    || projects.find(item => plainName(item.name) === plainName(cleanProject));

  if(found){
    return path.resolve(found.path);
  }

  return safeJoin(paths.projectsRoot, cleanProject);
};

// un chat con proyecto trabaja en ese proyecto: el modelo a veces reescribe el nombre con otros espacios o inventa uno
// y eso abria otra carpeta en projectos en vez de la del chat. solo sale de ahi con otro proyecto que existe o una ruta completa
export const chatProject = async (asked, active) => {
  const wanted = cleanText(asked);
  const own = cleanText(active);

  if(!wanted || !own){
    return wanted || own;
  }

  if(plainName(wanted) === plainName(own)){
    return own;
  }

  if(isAbsolutePath(wanted)){
    const home = await getProjectPath(own).catch(() => '');
    return home && plainName(path.resolve(wanted)) === plainName(home) ? own : wanted;
  }

  const projects = await getRegisteredProjects();
  return projects.some(item => item.name === wanted) ? wanted : own;
};

const readCleanDir = async (dirPath) => {
  const entries = await fs.readdir(dirPath, {withFileTypes: true});
  return entries
    .filter(item => !['node_modules', '.git', 'dist', 'data'].includes(item.name))
    .sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name));
};

export const listDir = async (dirPath, deep = 0, base = dirPath, budget = null) => {
  if(!budget){
    const cleanEntries = await readCleanDir(dirPath);
    const items = [];
    for(const item of cleanEntries){
      const itemPath = path.join(dirPath, item.name);
      const itemInfo = {
        name: item.name,
        path: relativePath(base, itemPath),
        type: item.isDirectory() ? 'folder' : 'file'
      };
      if(item.isDirectory() && deep > 0){
        itemInfo.children = await listDir(itemPath, deep - 1, base, null);
      }
      items.push(itemInfo);
    }
    return items;
  }

  const rootItems = [];
  const queue = [{dir: dirPath, depth: deep, target: rootItems}];
  while(queue.length > 0 && budget.left > 0){
    const {dir, depth, target} = queue.shift();
    const cleanEntries = await readCleanDir(dir);
    const share = queue.length > 0 ? Math.max(25, Math.ceil(budget.left / (queue.length + 1))) : budget.left;
    const cap = Math.min(budget.left, share);
    let added = 0;
    for(const item of cleanEntries){
      if(added >= cap){
        budget.omitted += cleanEntries.length - added;
        break;
      }
      budget.left--;
      added++;
      const itemPath = path.join(dir, item.name);
      const itemInfo = {
        name: item.name,
        path: relativePath(base, itemPath),
        type: item.isDirectory() ? 'folder' : 'file'
      };
      if(item.isDirectory() && depth > 0){
        itemInfo.children = [];
        queue.push({dir: itemPath, depth: depth - 1, target: itemInfo.children});
      }
      target.push(itemInfo);
    }
  }
  return rootItems;
};
