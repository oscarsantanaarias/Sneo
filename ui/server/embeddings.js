import fs from 'fs/promises';
import path from 'path';
import { paths } from './config.js';
import { getProjectPath, relativePath } from './projects.js';
import { getEmbedding } from './model.js';
import { cleanText } from './storage.js';

const indexFile = path.join(paths.dataDir, 'embeddings.json');
const skipNames = ['node_modules', '.git', 'dist', 'data', 'build', '.next'];
const textExt = ['.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json', '.md', '.txt', '.py', '.c', '.cpp', '.h', '.cs', '.lua', '.xml', '.vue', '.svelte', '.go', '.rs', '.java', '.php'];
const CHUNK_LINES = 40;
const MAX_CHUNKS = 600;

const readIndex = async () => {
  try {
    return JSON.parse(await fs.readFile(indexFile, 'utf8'));
  } catch(e) {
    return {projects: {}};
  }
};

const writeIndex = async (data) => {
  await fs.writeFile(indexFile, JSON.stringify(data));
};

const walk = async (dirPath, files = []) => {
  const entries = await fs.readdir(dirPath, {withFileTypes: true});

  for(const item of entries){
    if(skipNames.includes(item.name)){
      continue;
    }

    const itemPath = path.join(dirPath, item.name);

    if(item.isDirectory()){
      await walk(itemPath, files);
      continue;
    }

    if(textExt.includes(path.extname(item.name).toLowerCase())){
      files.push(itemPath);
    }
  }

  return files;
};

const chunkFile = (content, rel) => {
  const lines = content.split('\n');
  const chunks = [];

  for(let i = 0; i < lines.length; i += CHUNK_LINES){
    const slice = lines.slice(i, i + CHUNK_LINES);
    const text = slice.join('\n').trim();

    if(text){
      chunks.push({file: rel, start: i + 1, end: Math.min(i + CHUNK_LINES, lines.length), text: text.slice(0, 4000)});
    }
  }

  return chunks;
};

const cosine = (a, b) => {
  let dot = 0;
  let na = 0;
  let nb = 0;

  for(let i = 0; i < a.length; i++){
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }

  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
};

export const codebaseIndex = async ({project}) => {
  const cleanProject = cleanText(project);
  const projectPath = await getProjectPath(cleanProject);
  const files = await walk(projectPath);
  const chunks = [];

  for(const filePath of files){
    const stat = await fs.stat(filePath);

    if(stat.size > 400000){
      continue;
    }

    const content = await fs.readFile(filePath, 'utf8');
    chunks.push(...chunkFile(content, relativePath(projectPath, filePath)));

    if(chunks.length >= MAX_CHUNKS){
      break;
    }
  }

  const limited = chunks.slice(0, MAX_CHUNKS);

  for(const chunk of limited){
    chunk.vec = await getEmbedding(`search_document: ${chunk.text}`);
  }

  const data = await readIndex();
  data.projects[cleanProject] = {
    project: cleanProject,
    updatedAt: new Date().toISOString(),
    chunks: limited
  };

  await writeIndex(data);

  return {project: cleanProject, filesIndexed: files.length, chunksIndexed: limited.length, truncated: chunks.length >= MAX_CHUNKS};
};

export const codebaseSearch = async ({project, query, k}) => {
  const cleanProject = cleanText(project);
  const cleanQuery = cleanText(query);

  if(!cleanQuery){
    throw new Error('query requerido');
  }

  const data = await readIndex();
  let entry = data.projects[cleanProject];

  if(!entry || !entry.chunks?.length){
    await codebaseIndex({project: cleanProject});
    entry = (await readIndex()).projects[cleanProject];
  }

  if(!entry || !entry.chunks?.length){
    return {results: [], hint: 'No hay indice. El proyecto tiene archivos de texto?'};
  }

  const queryVec = await getEmbedding(`search_query: ${cleanQuery}`);
  const top = Number(k) || 6;

  const scored = entry.chunks.map(chunk => ({
    file: chunk.file,
    start: chunk.start,
    end: chunk.end,
    score: Number(cosine(queryVec, chunk.vec).toFixed(3)),
    snippet: chunk.text.slice(0, 400)
  }));

  scored.sort((a, b) => b.score - a.score);

  return {query: cleanQuery, results: scored.slice(0, top)};
};
