import fs from 'fs/promises';
import path from 'path';
import { paths } from './config.js';
import { getEmbedding } from './model.js';
import { toolGhidra } from './tools.js';
import { cleanText } from './storage.js';

const indexFile = path.join(paths.dataDir, 're_index.json');

const readIndex = async () => {
  try {
    return JSON.parse(await fs.readFile(indexFile, 'utf8'));
  } catch(e) {
    return {functions: []};
  }
};

const writeIndex = async (data) => {
  await fs.writeFile(indexFile, JSON.stringify(data));
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

export const reIndex = async ({addresses, port}) => {
  const list = Array.isArray(addresses) ? addresses.map(cleanText).filter(Boolean) : [];

  if(!list.length){
    throw new Error('addresses requerido: lista de direcciones hex, ej ["01b25f2a","01c33a10"]');
  }

  const data = await readIndex();
  const known = new Set(data.functions.map(item => item.addr));
  let added = 0;
  const failed = [];

  for(const addr of list.slice(0, 300)){
    try {
      const res = await toolGhidra({command: `DECOMP ${addr}`, port});
      const text = String(res.response || '').slice(0, 6000);

      if(!text){
        failed.push(addr);
        continue;
      }

      const vec = await getEmbedding(`search_document: ${text}`);
      const existing = data.functions.find(item => item.addr === addr);

      if(existing){
        existing.text = text;
        existing.vec = vec;
      } else {
        data.functions.push({addr, text, vec});
        added++;
      }

      known.add(addr);
    } catch(e) {
      failed.push(addr);
    }
  }

  await writeIndex(data);

  return {added, total: data.functions.length, failed};
};

export const reAddDecomp = async (addr, text) => {
  try {
    const cleanAddr = cleanText(addr);
    const cleanTextData = String(text || '').slice(0, 6000);

    if(!cleanAddr || !cleanTextData){
      return;
    }

    const data = await readIndex();
    const vec = await getEmbedding(`search_document: ${cleanTextData}`);
    const existing = data.functions.find(item => item.addr === cleanAddr);

    if(existing){
      existing.text = cleanTextData;
      existing.vec = vec;
    } else {
      data.functions.push({addr: cleanAddr, text: cleanTextData, vec});
    }

    await writeIndex(data);
  } catch(e) {
    return;
  }
};

export const reMap = async ({address, depth, port}) => {
  const start = cleanText(address).replace(/^0x/i, '');

  if(!start){
    throw new Error('address requerido: la funcion raiz, ej 0178d190');
  }

  const maxDepth = Math.min(Number(depth) || 2, 3);
  const maxFuncs = 40;
  const seen = new Set();
  const map = [];
  const queue = [{addr: start, level: 0}];

  while(queue.length && map.length < maxFuncs){
    const {addr, level} = queue.shift();

    if(seen.has(addr) || level > maxDepth){
      continue;
    }

    seen.add(addr);

    let res;

    try {
      res = await toolGhidra({command: `DECOMP ${addr}`, port});
    } catch(e) {
      map.push({addr, level, error: String(e.message || e)});
      continue;
    }

    let code = '';

    try {
      code = JSON.parse(res.response).code || '';
    } catch(e) {
      code = res.response || '';
    }

    const callees = [...new Set((code.match(/FUN_([0-9a-f]{6,})/gi) || []).map(m => m.replace(/^FUN_/i, '')))].filter(a => a !== addr);

    map.push({addr, level, callees: callees.slice(0, 20), snippet: code.replace(/\s+/g, ' ').slice(0, 200)});

    if(level < maxDepth){
      for(const callee of callees.slice(0, 12)){
        if(!seen.has(callee)){
          queue.push({addr: callee, level: level + 1});
        }
      }
    }
  }

  return {root: start, depth: maxDepth, functionsMapped: map.length, truncated: map.length >= maxFuncs, map};
};

export const reSearch = async ({query, k}) => {
  const cleanQuery = cleanText(query);

  if(!cleanQuery){
    throw new Error('query requerido');
  }

  const data = await readIndex();

  if(!data.functions.length){
    return {results: [], hint: 'No hay funciones indexadas. Usa re_index con una lista de direcciones primero.'};
  }

  const queryVec = await getEmbedding(`search_query: ${cleanQuery}`);
  const top = Number(k) || 6;

  const scored = data.functions.map(item => ({
    addr: item.addr,
    score: Number(cosine(queryVec, item.vec).toFixed(3)),
    snippet: item.text.slice(0, 500)
  }));

  scored.sort((a, b) => b.score - a.score);

  return {query: cleanQuery, results: scored.slice(0, top)};
};
