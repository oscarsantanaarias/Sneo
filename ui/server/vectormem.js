import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { paths } from './config.js';
import { getEmbedding } from './model.js';
import { cleanText, makeId } from './storage.js';

const storePath = path.join(paths.dataDir, 'vectormem.json');

const load = async () => {
  if(!existsSync(storePath)){
    return {items: []};
  }

  try {
    const raw = await readFile(storePath, 'utf8');
    const data = JSON.parse(raw);
    return {items: Array.isArray(data.items) ? data.items : []};
  } catch(e) {
    return {items: []};
  }
};

const save = async (data) => {
  await writeFile(storePath, JSON.stringify(data));
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

export const vectorRemember = async ({text, tags}) => {
  const clean = cleanText(text);

  if(!clean){
    throw new Error('text requerido');
  }

  const vec = await getEmbedding(`search_document: ${clean}`);
  const data = await load();
  const item = {
    id: makeId(),
    text: clean.slice(0, 4000),
    tags: Array.isArray(tags) ? tags.map(String) : [],
    vec
  };

  data.items.push(item);
  await save(data);

  return {saved: true, id: item.id, total: data.items.length};
};

export const vectorRecall = async ({query, k}) => {
  const clean = cleanText(query);

  if(!clean){
    throw new Error('query requerido');
  }

  const data = await load();

  if(!data.items.length){
    return {results: [], note: 'La memoria vectorial esta vacia. Usa vector_remember para guardar cosas.'};
  }

  const queryVec = await getEmbedding(`search_query: ${clean}`);
  const limit = Math.min(10, Math.max(1, Number(k) || 5));

  const ranked = data.items
    .map(item => ({
      id: item.id,
      text: item.text,
      tags: item.tags,
      score: Number(cosine(queryVec, item.vec).toFixed(3))
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  return {results: ranked};
};

export const vectorForget = async ({id}) => {
  const data = await load();
  const before = data.items.length;
  data.items = data.items.filter(item => item.id !== cleanText(id));
  await save(data);

  return {removed: before - data.items.length, total: data.items.length};
};
