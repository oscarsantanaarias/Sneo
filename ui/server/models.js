import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { once } from 'events';

const repos = [
  {repo: 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF', family: 'Qwen3.8 Flash Next', kind: 'Coder', arch: 'qwen4exp', tuned: true},
  {repo: 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', family: 'Qwen3.8 Flash Next', kind: 'General', arch: 'qwen4exp', tuned: true},
  {repo: 'unsloth/Qwen3.8-Flash-Next-GGUF', family: 'Qwen3.8 Flash Next', kind: 'General', arch: 'qwen4exp', tuned: false, base: 'Qwen3.8-Flash-Next-'},
  {repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', family: 'Qwen3.6 35B-A3B', kind: 'General', arch: 'qwen35moe', tuned: false, base: 'Qwen3.6-35B-A3B-'}
];
const knownTypes = new Set([0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 42]);
const valueSizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8};
const gb = 1024 ** 3;

let catalog = null;
let stopper = null;
let job = {state: 'idle', id: '', label: '', file: '', done: 0, total: 0, speed: 0, problem: '', note: ''};

const hubJson = async (url) => {
  const res = await fetch(url, {signal: AbortSignal.timeout(30000)});

  if(!res.ok){
    throw new Error(`Hugging Face answered ${res.status} for ${url}`);
  }

  return await res.json();
};

const readCatalog = async () => {
  if(catalog && Date.now() - catalog.when < 600000){
    return catalog.options;
  }

  const options = [];

  for(const {repo, family, kind, arch, tuned, base} of repos){
    const info = await hubJson(`https://huggingface.co/api/models/${repo}`);
    const tree = await hubJson(`https://huggingface.co/api/models/${repo}/tree/${info.sha}?recursive=1`);
    const groups = new Map();

    for(const entry of tree){
      const name = path.posix.basename(entry.path || '');

      if(entry.type !== 'file' || !name.endsWith('.gguf') || /^(mmproj|mtp)/i.test(name)){
        continue;
      }

      const quant = entry.path.includes('/')
        ? entry.path.split('/')[0]
        : name.replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '').slice(base && name.startsWith(base) ? base.length : 0);

      if(/^(BF16|F16|F32|MTP)$/i.test(quant)){
        continue;
      }

      const id = `${repo}:${quant}`;
      const group = groups.get(id) || {id, repo, revision: info.sha, family, kind, arch, tuned, quant, label: `${kind} ${quant}`, files: [], total: 0};
      group.files.push({path: entry.path, name, size: entry.size || entry.lfs?.size || 0, oid: entry.lfs?.oid || entry.oid || ''});
      group.total += entry.size || entry.lfs?.size || 0;
      groups.set(id, group);
    }

    const readers = tree.filter(entry => entry.type === 'file' && /^mmproj.*\.gguf$/i.test(path.posix.basename(entry.path || '')));
    const reader = [/bf16/i, /(^|[^b])f16/i, /f32/i].map(kind => readers.find(entry => kind.test(path.posix.basename(entry.path)))).find(Boolean);

    for(const group of groups.values()){
      group.files.sort((a, b) => a.name.localeCompare(b.name));
      if(reader && arch === 'qwen4exp'){
        const name = path.posix.basename(reader.path);
        group.reader = {path: path.posix.join(path.posix.dirname(group.files[0].path), name), from: reader.path, name, size: reader.size || reader.lfs?.size || 0, oid: reader.lfs?.oid || reader.oid || ''};
      }
      options.push(group);
    }
  }

  catalog = {when: Date.now(), options};
  return options;
};

const localFiles = async (folder) => {
  const found = new Map();
  const walk = async (dir, depth) => {
    const entries = await fsp.readdir(dir, {withFileTypes: true}).catch(() => []);

    for(const entry of entries){
      const full = path.join(dir, entry.name);

      if(entry.isDirectory() && depth < 3){
        await walk(full, depth + 1);
        continue;
      }

      if(!entry.isFile() || !entry.name.endsWith('.gguf')){
        continue;
      }

      const stat = await fsp.stat(full).catch(() => null);

      if(stat){
        found.set(entry.name.toLowerCase(), {full, size: stat.size});
      }
    }
  };

  await walk(folder, 0);
  return found;
};

const engineInfo = async (baseUrl) => {
  const pool = await fetch(`${baseUrl}/api/pool`, {signal: AbortSignal.timeout(10000)}).then(res => res.json()).catch(() => ({}));
  const models = await fetch(`${baseUrl}/api/models`, {signal: AbortSignal.timeout(10000)}).then(res => res.json()).catch(() => ({}));
  const stats = pool.nodes?.[0]?.stats || {};
  return {
    folder: models.folder || '',
    architectures: Array.isArray(models.architectures) ? models.architectures : ['qwen4exp'],
    ramTotal: Number(stats.ramTotal) || 0,
    vramTotal: Number(stats.gpu?.vramTotal) || 0
  };
};

const sameFile = (file, options, local) => {
  for(const option of options){
    for(const other of option.files){
      if(!file.oid || other.oid !== file.oid){
        continue;
      }

      const have = local.get(other.name.toLowerCase());

      if(have && have.size === other.size){
        return have;
      }
    }
  }

  return null;
};

export const modelCatalog = async (baseUrl) => {
  const options = await readCatalog();
  const facts = await engineInfo(baseUrl);
  const local = facts.folder ? await localFiles(facts.folder) : new Map();
  const disk = facts.folder ? await fsp.statfs(facts.folder).catch(() => null) : null;
  const diskFree = disk ? disk.bavail * disk.bsize : 0;
  const memory = Math.max(0, facts.ramTotal - 6 * gb) + Math.max(0, facts.vramTotal - gb);

  const list = options.filter(option => facts.architectures.includes(option.arch)).map(option => {
    let present = 0;
    let shared = 0;
    let named = 0;

    for(const file of option.files){
      const have = local.get(file.name.toLowerCase());
      named += have ? 1 : 0;

      if(have && have.size === file.size){
        present += file.size;
        continue;
      }

      if(sameFile(file, options, local)){
        shared += file.size;
      }
    }

    const ratio = memory ? option.total / memory : 99;
    const needed = option.total - present - shared;
    const readerThere = !option.reader || !facts.folder || [path.join(facts.folder, ...option.reader.path.split('/')), path.join(facts.folder, option.reader.name)].some(place => fs.existsSync(place) && fs.statSync(place).size === option.reader.size);
    const readerNeeded = readerThere ? 0 : option.reader.size;
    return {
      id: option.id,
      label: option.label,
      family: option.family,
      repo: option.repo,
      quant: option.quant,
      kind: option.kind,
      tuned: option.tuned,
      size: option.total,
      parts: option.files.length,
      installed: present === option.total,
      older: present !== option.total && named === option.files.length,
      shared,
      needed,
      readerNeeded,
      diskOk: needed + readerNeeded <= Math.max(0, diskFree - 2 * gb),
      ratio,
      runs:
        ratio <= 1 ? 'Runs from memory' :
        ratio <= 2.3 ? 'Hot experts stay in memory, the rest come from the SSD' :
        ratio <= 4 ? 'Reads many experts from the SSD each token (slower)' :
        'Mostly read from the SSD each token (slow)'
    };
  }).sort((a, b) => repos.findIndex(item => item.family === a.family) - repos.findIndex(item => item.family === b.family) || a.size - b.size);

  for(const family of new Set(list.map(item => item.family))){
    const own = list.filter(item => item.family === family);
    const best = (limit) => {
      const tuned = own.filter(item => item.tuned && item.ratio <= limit);
      const any = own.filter(item => item.ratio <= limit);
      return tuned.length ? tuned[tuned.length - 1] : any[any.length - 1];
    };
    const pick = best(1) || best(2.3) || own[0];

    if(pick){
      pick.recommended = true;
    }
  }

  return {folder: facts.folder, ramTotal: facts.ramTotal, vramTotal: facts.vramTotal, memory, diskFree, options: list, job};
};

const parseHeader = (data) => {
  let at = 0;
  const need = (count) => {
    if(at + count > data.length){
      throw new RangeError('short');
    }
  };
  const u32 = () => {
    need(4);
    at += 4;
    return data.readUInt32LE(at - 4);
  };
  const u64 = () => {
    need(8);
    at += 8;
    return Number(data.readBigUInt64LE(at - 8));
  };
  const text = () => {
    const length = u64();
    need(length);
    at += length;
    return data.toString('utf8', at - length, at);
  };
  const skipValue = (type) => {
    if(type === 8){
      text();
      return;
    }

    if(type !== 9){
      need(valueSizes[type] ?? 0);
      at += valueSizes[type] ?? 0;
      return;
    }

    const inner = u32();
    const count = u64();

    if(valueSizes[inner]){
      need(count * valueSizes[inner]);
      at += count * valueSizes[inner];
      return;
    }

    for(let i = 0; i < count; i++){
      skipValue(inner);
    }
  };

  need(24);

  if(data.toString('latin1', 0, 4) !== 'GGUF'){
    throw new Error('this file is not a GGUF model');
  }

  at = 4;
  u32();
  const tensorCount = u64();
  const valueCount = u64();
  let architecture = '';

  try {
    for(let i = 0; i < valueCount; i++){
      const key = text();
      const type = u32();

      if(key === 'general.architecture' && type === 8){
        architecture = text();
        continue;
      }

      skipValue(type);
    }

    const types = new Set();

    for(let i = 0; i < tensorCount; i++){
      text();
      const dims = u32();
      need(dims * 8 + 12);
      at += dims * 8;
      types.add(u32());
      at += 8;
    }

    return {architecture, types: [...types], tensors: tensorCount};
  } catch(e) {
    if(e instanceof RangeError){
      return null;
    }

    throw e;
  }
};

const checkHeader = async (url, arch, signal) => {
  for(const megabytes of [4, 16, 64]){
    const res = await fetch(url, {headers: {Range: `bytes=0-${megabytes * 1048576 - 1}`}, signal});

    if(!res.ok){
      throw new Error(`the download link answered ${res.status}`);
    }

    const header = parseHeader(Buffer.from(await res.arrayBuffer()));

    if(!header){
      continue;
    }

    if(header.architecture && header.architecture !== arch){
      throw new Error(`this file is ${header.architecture}, expected ${arch}`);
    }

    const unknown = header.types.filter(type => !knownTypes.has(type));

    if(unknown.length){
      throw new Error(`this model uses tensor types the engine does not read (${unknown.join(', ')})`);
    }

    return header;
  }

  throw new Error('the model header is bigger than 64 MB, it does not look like a GGUF model');
};

const fetchFile = async (url, file, target, signal) => {
  const part = `${target}.part`;
  const have = await fsp.stat(part).then(stat => stat.size).catch(() => 0);
  let start = have > file.size ? 0 : have;
  const res = await fetch(url, {headers: start ? {Range: `bytes=${start}-`} : {}, signal});

  if(!res.ok){
    throw new Error(`${file.name}: the download answered ${res.status}`);
  }

  if(start && res.status !== 206){
    start = 0;
  }

  job.done += start;
  const out = fs.createWriteStream(part, {flags: start ? 'a' : 'w'});
  let windowStart = Date.now();
  let windowBytes = 0;

  try {
    for await (const chunk of res.body){
      if(!out.write(chunk)){
        await once(out, 'drain');
      }

      job.done += chunk.length;
      windowBytes += chunk.length;

      if(Date.now() - windowStart >= 1000){
        job.speed = Math.round(windowBytes * 1000 / (Date.now() - windowStart));
        windowStart = Date.now();
        windowBytes = 0;
      }
    }
  } finally {
    out.end();
    await once(out, 'close').catch(() => {});
  }

  const size = (await fsp.stat(part)).size;

  if(size !== file.size){
    throw new Error(`${file.name}: got ${size} bytes of ${file.size}, press Download again to resume`);
  }

  await fsp.rename(part, target);
};

const runDownload = async (option, folder, baseUrl) => {
  const options = await readCatalog();
  const signal = stopper.signal;
  const link = (file) => `https://huggingface.co/${option.repo}/resolve/${option.revision}/${(file.from || file.path).split('/').map(encodeURIComponent).join('/')}`;
  const files = option.reader ? [...option.files, option.reader] : option.files;
  job = {state: 'checking', id: option.id, label: option.label, file: option.files[0].name, done: 0, total: files.reduce((sum, file) => sum + file.size, 0), speed: 0, problem: '', note: 'checking the model header'};

  await checkHeader(link(option.files[0]), option.arch, signal);
  job.state = 'downloading';
  job.note = '';

  for(const file of files){
    const target = path.join(folder, ...file.path.split('/'));
    await fsp.mkdir(path.dirname(target), {recursive: true});
    job.file = file.name;

    const there = await fsp.stat(target).catch(() => null);
    const above = file === option.reader ? await fsp.stat(path.join(folder, file.name)).catch(() => null) : null;

    if((there && there.size === file.size) || (above && above.size === file.size)){
      job.done += file.size;
      continue;
    }

    const local = await localFiles(folder);
    const twin = sameFile(file, options, local);

    if(twin){
      job.note = `${file.name} is the same file as ${path.basename(twin.full)}, linking it instead of downloading`;
      await fsp.link(twin.full, target).catch(async () => await fsp.copyFile(twin.full, target));
      job.done += file.size;
      job.note = '';
      continue;
    }

    await fetchFile(link(file), file, target, signal);
  }

  job.state = 'done';
  job.speed = 0;
  job.note = `${option.label} is ready in ${folder}`;
  catalog = null;
  await fetch(`${baseUrl}/api/models/scan`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({dir: folder}), signal: AbortSignal.timeout(30000)}).catch(() => {});
};

export const startDownload = async (id, baseUrl) => {
  if(['checking', 'downloading'].includes(job.state)){
    throw new Error(`${job.label} is already downloading`);
  }

  const options = await readCatalog();
  const option = options.find(item => item.id === id);

  if(!option){
    throw new Error('that model is not in the list');
  }

  const facts = await engineInfo(baseUrl);

  if(!facts.folder){
    throw new Error('pick a models folder first in Settings, Engine');
  }

  stopper = new AbortController();
  runDownload(option, facts.folder, baseUrl).catch(e => {
    job.speed = 0;
    job.state = stopper?.signal.aborted ? 'paused' : 'error';
    job.problem = stopper?.signal.aborted ? '' : e.message;
    job.note = stopper?.signal.aborted ? 'paused, press Download to resume' : '';
  });
  return job;
};

export const stopDownload = () => {
  stopper?.abort();
  return job;
};

export const downloadState = () => job;
