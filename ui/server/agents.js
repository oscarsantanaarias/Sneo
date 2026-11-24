import path from 'path';
import { readdir, readFile } from 'fs/promises';
import { existsSync, readFileSync } from 'fs';
import { paths } from './config.js';
import { askModel, askModelChat, blindReason, contextWindow, readEngine } from './model.js';
import { askRoute, readRoutes, routeLabel } from './teams.js';
import { askInSession, chatSession } from './claudecode.js';
import { cleanText, makeId, pushAgentLog, readSettings } from './storage.js';
import { checkTool, runSubTool, runTool, toolQueueWrite } from './tools.js';
import { runStore, stopChildren } from './runs.js';
import { normalizeMode, toolKind } from './gate.js';
import { LoopWatch, silentNote } from './watch.js';
import { calibrateRatio, clipText, countChars, defaultRatio, estimateTokens, fitMessages, outputLimit, placeSummary, withoutThinking } from './fit.js';
import { subAgentToolSchemas, toolSchemas } from './toolSchemas.js';
import { memoryList } from './memory.js';
import { ensurePluginsLoaded, getPluginTools, isPluginTool, runPluginTool, reloadPlugins, listPlugins } from './plugins.js';
import { chatProject, getProjectPath } from './projects.js';
import { resolveCall, unknownTool, writesFiles } from './toolAliases.js';

const dispatchTool = async (name, callArgs, run) => {
  if(isPluginTool(name) || name === 'plugin_reload'){
    await checkTool(name, callArgs);
  }

  if(isPluginTool(name)) return await runPluginTool(name, callArgs);
  if(name === 'plugin_list') return await listPlugins();
  if(name === 'plugin_reload') return await reloadPlugins();
  if(name === 'parallel_agents') return await toolParallelAgents(callArgs, run);
  if(name === 'orchestrate') return await orchestrate({...callArgs, onEvent: run?.onEvent, run});
  return await runTool(name, callArgs);
};

const toon = (value) => {
  if(value === null || value === undefined || typeof value === 'string'){
    return String(value ?? '');
  }
  return JSON.stringify(value);
};

const buildMemoryBlock = async () => {
  try {
    const items = await memoryList();

    if(!items.length){
      return '';
    }

    const lines = items.slice(0, 40).map(item => `- ${item.text}`).join('\n');
    return `\n\nMEMORY (what you learned before, respect it):\n${lines}`;
  } catch(e) {
    return '';
  }
};

const parseNativeArgs = (rawArgs) => {
  if(typeof rawArgs !== 'string'){
    return rawArgs && typeof rawArgs === 'object' ? rawArgs : {};
  }

  if(!rawArgs.trim()){
    return {};
  }

  try {
    const parsed = JSON.parse(rawArgs);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch(e) {
    return null;
  }
};

// la carpeta de una llamada es la del chat, salvo que el modelo pida a proposito otro proyecto que existe
const placeCall = async (name, callArgs, project) => {
  const home = isPluginTool(name)
    ? callArgs.project || project
    : await chatProject(callArgs.project, project);

  if(home){
    callArgs.project = home;
  }
};

const normalizeToolArgs = (name, rawArgs, projectRoot) => {
  const args = {...(rawArgs || {})};
  const aliases = {
    write_file: 'file',
    read_file: 'file',
    read_range: 'file',
    replace_in_file: 'file',
    visual_check: 'file',
    create_folder: 'folder',
    delete_path: 'target',
    add_project: 'projectPath'
  };
  const key = aliases[name];

  if(key && !args[key] && args.path){
    args[key] = args.path;
  }

  if(key && typeof args[key] === 'string' && path.isAbsolute(args[key]) && projectRoot){
    const root = path.resolve(projectRoot);
    const candidate = path.resolve(args[key]);

    if(candidate.toLowerCase().startsWith(`${root.toLowerCase()}${path.sep}`)){
      args[key] = path.relative(root, candidate);
    } else if(['file', 'folder', 'target'].includes(key)){
      args[key] = path.basename(candidate);
    }
  }

  delete args.path;
  return args;
};

const loadSkills = async () => {
  try {
    const raw = await readFile(path.join(paths.dataDir, 'skills.json'), 'utf8');
    const data = JSON.parse(raw || '{}');
    return Array.isArray(data.skills) ? data.skills : [];
  } catch(e) {
    return [];
  }
};

const buildSkillsBlock = (skills) => {
  if(!skills.length){
    return '';
  }

  const blocks = skills.map(skill => `### ${skill.name}: ${skill.description}\n${skill.steps}`).join('\n\n');

  return `\n\nBUILT-IN CAPABILITIES (apply them automatically when the task calls for it, without waiting for user commands. Pick the one that fits the request):\n\n${blocks}`;
};

const parseXmlToolCall = (text) => {
  const source = String(text || '');
  const fnMatch = /<function\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>/.exec(source);

  if(!fnMatch){
    return null;
  }

  const tool = fnMatch[1];

  if(!isKnownTool(tool)){
    return null;
  }

  const args = {};
  const paramRe = /<parameter\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>\s*([\s\S]*?)(?=\s*<parameter\s*=|\s*<\/?function|$)/g;
  let m;

  while((m = paramRe.exec(source))){
    let value = m[2].replace(/<\/parameter>\s*$/i, '').trim();

    if(/^-?\d+$/.test(value)){
      value = Number(value);
    } else if(value === 'true' || value === 'false'){
      value = value === 'true';
    }

    args[m[1]] = value;
  }

  return {tool, args};
};

const TOOL_NAMES = new Set(toolSchemas.map(item => item.function?.name).filter(Boolean));

const isKnownTool = (name) => TOOL_NAMES.has(name) || isPluginTool(name);

const parsePyStyleCall = (text) => {
  const source = String(text || '').trim().replace(/^CALLED:\s*/i, '').replace(/^`+|`+$/g, '');
  const m = /^\s*\[?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([\s\S]*)\)\s*\]?\s*$/.exec(source);

  if(!m || !isKnownTool(m[1])){
    return null;
  }

  const tool = m[1];
  const inner = m[2].trim();

  if(inner.startsWith('{') && inner.endsWith('}')){
    try {
      const parsed = JSON.parse(inner);

      if(!parsed || typeof parsed !== 'object' || Array.isArray(parsed)){
        return null;
      }

      return {tool, args: parsed};
    } catch(e) {
      return null;
    }
  }

  const args = {};
  const argRe = /([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|-?\d+(?:\.\d+)?|true|false)/g;
  let a;

  while((a = argRe.exec(m[2]))){
    let value = a[2];

    if(value === 'true' || value === 'false'){
      value = value === 'true';
    } else if(/^-?\d+(\.\d+)?$/.test(value)){
      value = Number(value);
    } else {
      value = value.slice(1, -1).replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\\\/g, '\\');
    }

    args[a[1]] = value;
  }

  if(m[2].trim() && Object.keys(args).length === 0){
    return null;
  }

  return {tool, args};
};

const parsePythonIdaCall = (text) => {
  const source = String(text || '').trim().replace(/^`+|`+$/g, '');
  const m = /^\s*ida\s*\.\s*(name|decompile|disasm|xrefs|find|str)\s*\(\s*["']([^"',)]+)["']\s*(?:,\s*(\d+)\s*)?\)\s*;?\s*$/i.exec(source);

  if(!m){
    return null;
  }

  const commands = {name: 'NAME', decompile: 'DECOMPILE', disasm: 'DISASM', xrefs: 'XREFS', find: 'FIND', str: 'STR'};
  return {tool: 'ida', args: {command: `${commands[m[1].toLowerCase()]} ${m[2]}${m[3] ? ` ${m[3]}` : ''}`.trim()}};
};

// varias llamadas en una respuesta: un arreglo json o un objeto por linea, cada una leida como parseToolCall
// si alguna no se entiende se lee como antes una sola
export const parseToolCalls = (text) => {
  const bare = String(text || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();

  if(bare.startsWith('[') && bare.endsWith(']')){
    try {
      const list = JSON.parse(bare);
      const calls = (Array.isArray(list) ? list : []).slice(0, 8).map(item => parseToolCall(JSON.stringify(item)));
      if(calls.length && calls.every(Boolean)) return calls;
    } catch(e) {}
  }

  const lines = bare.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if(lines.length > 1 && lines.every(line => line.startsWith('{') && line.endsWith('}'))){
    const calls = lines.slice(0, 8).map(line => parseToolCall(line));
    if(calls.every(Boolean)) return calls;
  }

  const one = parseToolCall(text);
  return one ? [one] : [];
};

export const parseToolCall = (text) => {
  const bare = String(text || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
  const blocks = bare.startsWith('{') && bare.endsWith('}') ? [bare] : [];

  for(const raw of blocks){
    try {
      const parsed = JSON.parse(raw);

      if(typeof parsed?.tool === 'string' && isKnownTool(parsed.tool)){
        return {tool: parsed.tool, args: parsed.args || parsed.arguments || {}};
      }

      if(typeof parsed?.name === 'string' && 'arguments' in parsed && isKnownTool(parsed.name)){
        return {tool: parsed.name, args: parsed.arguments || {}};
      }
    } catch(e) {
      continue;
    }
  }

  const tagged = [...String(text || '').matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi)];

  for(const match of tagged){
    try {
      const parsed = JSON.parse(match[1]);

      if(typeof parsed?.name === 'string' && isKnownTool(parsed.name)){
        return {tool: parsed.name, args: parsed.arguments || parsed.args || {}};
      }
    } catch(e) {
      continue;
    }
  }

  return parseXmlToolCall(text) || parsePyStyleCall(text) || parsePythonIdaCall(text) || parseWriteTag(text) || parseReverseCommand(text);
};

const reverseCommands = 'DECOMP_XREF|DECOMPILE|DECOMP|DISASM|XREFS|XREF|XFROM|FUNC|CURRENT|HELP|PING|NAME|TRACE|MAKEFUNC|RENAME_FUNC|RENAME_SYMBOL|RENAME_LABEL|RENAME_LOCAL|RENAME|NEARSTR|READPTR|WRITERS|READERS|CALLERS|FINDPTR|STRUCT|VTABLE|VTDUMP|PCODE|RETYPE|DEFSTRUCT|SETFIELD|MKPTRS|RTTI|SETCONV';
const reverseCommandRe = new RegExp(`^\\s*\`*\\s*(ida|IDA|ghidra|Ghidra|GHIDRA)\\s*:?\\s+["'\`]?((?:STR|WSTR|FIND|EXPORT|IMPORTXREF)(?:[ \\t]+[^\\n"'\`]+)?|(?:BYTES|PATCHBYTES|PATCH)(?:[ \\t]+[0-9A-Fa-f?]{1,16})+|(?:${reverseCommands})(?:[ \\t]+(?=[^\\s"'\`]*[0-9A-Z_:])[^\\s"'\`]+(?:[ \\t]+[^\\s"'\`]+){0,2})?)["'\`]?\\s*\`*\\s*$`);

const parseReverseCommand = (text) => {
  const m = reverseCommandRe.exec(String(text || ''));

  if(!m){
    return null;
  }

  return {tool: m[1].toLowerCase(), args: {command: m[2].trim()}};
};

const addressRe = /(?:\b0x|\b(?:FUN|DAT|LAB|sub|loc|off|unk)_)([0-9a-f]{4,16})\b|(?<![#\w])((?=[0-9]*[a-f])(?=[a-f]*[0-9])[0-9a-f]{6,16}|[0-9]{8,16})(?!\w)/i;
const reverseAskRe = /^\s*(?:(?:please|pls|por favor)\s+)?(?:(?:in|en|with|con|using|usa|usando)\s+)?(?:(?:ghidra|ida)\s*[:,]?\s+)?(?:decompil\w*|decomp|disassembl\w*|desensambl\w*|xrefs?|analy[sz]e|analiza\w*|read|lee\w*|show|muestra\w*|mostra\w*|open|abr\w*|look at|mira|check|revisa\w*|explain|explica\w*|what does|what is|qu[eé] hace|qu[eé] es)\b/i;

export const parseDirectGhidraRequest = (text) => {
  const source = String(text || '');
  const found = addressRe.exec(source);
  const port = source.match(/\b(?:port|puerto)\s*[:=]?\s*(\d{2,5})\b/i)?.[1];

  if(!found || !reverseAskRe.test(source)){
    return null;
  }

  return {
    tool: 'ghidra',
    args: {command: `DECOMP ${found[1] || found[2]}`, ...(port ? {port: Number(port)} : {})}
  };
};

export const parseDirectIdaRequest = (text) => {
  const source = String(text || '');
  const found = addressRe.exec(source);
  const port = source.match(/\b(?:port|puerto)\s*[:=]?\s*(\d{2,5})\b/i)?.[1];
  const stringTarget = source.match(/[`"']([^`"']+)[`"']/)?.[1]
    || source.match(/\b([\w.-]+\.(?:s4hd|dll|exe|pak|bin))\b/i)?.[1];

  if(!/\bida\b|ida_bridge/i.test(source)){
    return null;
  }

  if(!found){
    if(stringTarget && /\b(find|search|where|open|opened|used|reference|references|buscar|d[oó]nde|abierto|usado)\b/i.test(source)){
      return {tool: 'ida', args: {command: `STR ${stringTarget}`, ...(port ? {port: Number(port)} : {})}};
    }

    if(/\b(access|available|connect|ping|help|acceso|conectar)\b/i.test(source)){
      return {tool: 'ida', args: {command: 'PING', ...(port ? {port: Number(port)} : {})}};
    }

    return null;
  }

  if(!reverseAskRe.test(source)){
    return null;
  }

  return {
    tool: 'ida',
    args: {command: `DECOMPILE ${found[1] || found[2]}`, ...(port ? {port: Number(port)} : {})}
  };
};

export const parseDirectIntent = (text, project) => {
  const source = String(text || '').trim();
  const cleanProject = cleanText(project);

  if(!cleanProject){
    return null;
  }

  if(/\b(how many|what files|which files|list files|cu[aá]ntos archivos|qu[eé] archivos|lista los archivos|what('?s| is) in (the|this) project|i don'?t see it|no (lo )?veo|no est[aá]|check files|check what|veri(fy|fica))\b/i.test(source)){
    return {tool: 'list_files', args: {project: cleanProject}};
  }

  const folderMatch = source.match(/\b(?:create|make|crea|haz|arm[aá])\s+(?:a\s+|una\s+)?(?:folder|directory|carpeta|directorio)\s+(?:called|named|name|llamada|de nombre)?\s*[`"']?([^\s`"']+)[`"']?/i)
    || source.match(/\b(?:make|crea)\s+(?:a\s+|una\s+)?(?:folder|carpeta)\s+(?:name|called|llamada)\s+[`"']?([^\s`"']+)[`"']?/i)
    || source.match(/\b(?:there\s+)?(?:make|create|crea)\s+(?:a\s+|una\s+)?(?:folder|carpeta)\s+(?:name|called)?\s*[`"']?([^\s`"']+)[`"']?/i);

  if(folderMatch){
    return {tool: 'create_folder', args: {project: cleanProject, folder: folderMatch[1]}};
  }

  const readMatch = source.match(/\b(?:read|show|open|lee|muestra|abre)\s+(?:the\s+|el\s+)?(?:file\s+|archivo\s+)?[`"']?([^\s`"']+\.[a-z0-9]+)[`"']?/i);

  if(readMatch){
    return {tool: 'read_file', args: {project: cleanProject, file: readMatch[1]}};
  }

  const writeMatch = source.match(/\b(?:create|make|write|crea|haz|escribe)\b[\s\S]*?\b(?:file|archivo)\s+(?:called|named|llamado|de nombre)?\s*[`"']?([^\s`"']+)[`"']?[\s\S]*?\b(?:inside|with|containing|dentro|contenido|con)\b\s*[:=-]?\s*[`"']?([\s\S]+?)[`"']?$/i);

  if(writeMatch){
    return {
      tool: 'write_file',
      args: {project: cleanProject, file: writeMatch[1], content: writeMatch[2].trim()}
    };
  }

  return null;
};

const parseWriteTag = (text) => {
  const s = String(text || '');

  if(!/<\s*(?:file_created|create_file|createfile|new_file|write_file|save_file)\b/i.test(s)){
    return null;
  }

  const pathM = /\b(?:path|file|filename|filepath)\s*=\s*"([^"]*)"/i.exec(s);
  const contentM = /\bcontent\s*=\s*"([\s\S]*)"\s*\/?>?\s*$/i.exec(s) || /\bcontent\s*=\s*"([\s\S]*)"/i.exec(s);

  if(!pathM){
    return null;
  }

  const unesc = (v) => String(v).replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\r/g, '\r').replace(/\\\\/g, '\\');

  return {tool: 'write_file', args: {file: unesc(pathM[1]), content: contentM ? unesc(contentM[1]) : ''}};
};

const brokenToolRe = new RegExp('^\\s*(?:CALLED:\\s*)?(?:<\\/?\\s*(?:' + [...TOOL_NAMES].join('|') + ')\\s*[/>(\\s]|(?:' + [...TOOL_NAMES].join('|') + ')\\s*\\()', 'im');

const hasBrokenToolCall = (text) => {
  const source = String(text || '');
  return /<\/?tool_call>|<function\s*=|<parameter\s*=|^\s*\{\s*"tool"\s*:|"arguments"\s*:\s*[{"]|<\s*(?:file_created|create_file|new_file|save_file|write_file)\s+[a-z]+\s*=/im.test(source) || brokenToolRe.test(source);
};

const toolJsonFixMessage = `
Your last response tried to call a tool but the format was invalid.
Call the tool using native function calling. If native calling is unavailable, reply with ONLY one valid JSON object,
no markdown and no explanation, in this format:
{"tool":"ghidra","args":{"command":"DECOMP 0179aca0"}}
Use the correct tool and arguments for the user's request. Do not announce what you will do.
`.trim();

const getToolSummary = (tools) => {
  const lines = [];

  for(const item of tools || []){
    const result = item.result || {};
    const changedFiles = result.files || (result.file ? [result] : []);

    for(const file of changedFiles){
      if(!file?.lines){
        continue;
      }

      lines.push(`- ${file.file}: +${file.lines.added || 0} -${file.lines.removed || 0} changed ${file.lines.changed || 0}`);
    }
  }

  if(!lines.length){
    return '';
  }

  return `\n\nChanges:\n${lines.join('\n')}`;
};

const getSearchSummary = (tools) => {
  const searchTools = ['pc_search', 'search_files', 'glob', 're_search', 'codebase_search'];

  for(let i = (tools || []).length - 1; i >= 0; i--){
    const item = tools[i];

    if(!searchTools.includes(item.tool)){
      continue;
    }

    const result = item.result || {};
    const files = Array.isArray(result.files) ? result.files : Array.isArray(result.results) ? result.results.map(r => r.file || r.addr || r) : (Array.isArray(result) ? result : null);

    if(!files){
      continue;
    }

    if(!files.length){
      return `\n\n_(real ${item.tool} result: 0 matches found${result.timedOut ? ', search timed out' : ''})_`;
    }

    const shown = files.slice(0, 20).map(f => `- ${typeof f === 'string' ? f : JSON.stringify(f)}`).join('\n');
    return `\n\nReal ${item.tool} results (${files.length}):\n${shown}`;
  }

  return '';
};

const finishAnswer = (answer, tools) => {
  const changeSummary = String(answer || '').includes('Changes:') ? '' : getToolSummary(tools);
  const searchSummary = getSearchSummary(tools);

  return `${answer}${changeSummary}${searchSummary}`;
};

const isRefusal = (text) => {
  const s = String(text || '');
  return /no tengo (la )?capacidad|como asistente (de ia|virtual)|no puedo (instalar|acceder|interactuar|leer)/i.test(s)
    || /as an ai (assistant|language model)|i (can'?t|cannot|do not have|don'?t have|am unable|'?m unable).{0,45}(access|browse|retrieve|read|open|filesystem|file system|your (pc|computer|desktop|files))|without direct (filesystem|file system|access)|(search|tool).{0,20}(is limited|doesn'?t|does not) (provide|allow|support|have)|manual search|windows search|file explorer|(system|project|environment|access) (access )?(restrictions|limitations)|due to .{0,25}(restrictions|limitations)|copy and paste the (text|content|file)/i.test(s);
};

const keepInHistory = (item) => {
  if(!['user', 'assistant'].includes(item?.role)){
    return false;
  }

  if(item.role !== 'assistant'){
    return true;
  }

  if(isRefusal(item.content)){
    return false;
  }

  if(/^No recibi respuesta del modelo/.test(String(item.content || ''))){
    return false;
  }

  return !(hasBrokenToolCall(item.content) && !parseToolCall(item.content));
};

export const toolParallelAgents = async ({project, task, agents}, run) => {
  const cleanProject = cleanText(project);
  const cleanTask = cleanText(task) || 'Review the project and give recommendations.';
  const cleanAgents = Array.isArray(agents) && agents.length > 0 ? agents : [
    {name: 'Explorer', role: 'Map structure, key files and context.'},
    {name: 'Searcher', role: 'Find symbols, patterns and related files.'},
    {name: 'Builder', role: 'Assess implementation and possible changes.'},
    {name: 'Reviewer', role: 'Look for risks, bugs and missing tests.'},
    {name: 'Tester', role: 'Test in sandbox and summarize results.'}
  ];

  const results = await Promise.all(cleanAgents.slice(0, 5).map(agent => runSubAgent({
    name: cleanText(agent.name) || 'Subagent',
    role: cleanText(agent.role) || 'Review',
    task: cleanTask,
    project: cleanProject,
    run
  })));

  await pushAgentLog({
    project: cleanProject,
    task: cleanTask,
    results
  });

  return results;
};

// archivos de codigo del usuario para que el agente escriba igual que el y no a su manera
const styleBlock = async (files) => {
  if(!files?.length) return '';
  const each = Math.floor(12000 / files.length);
  const parts = [];
  for(const file of files){
    const text = await readFile(file, 'utf8').catch(() => '');
    if(!text.trim()) continue;
    parts.push(`--- ${path.basename(file)} ---\n${text.slice(0, each)}`);
  }
  if(!parts.length) return '';
  return `

THE USER'S CODE STYLE (always follow it):
These are files the user wrote. Whenever you write or change code, write it exactly the way they write it: the same naming, indentation, brace placement, spacing, early returns, line length, comment language, comment style and how much they comment, and the same habits. Copy the style, not the content.
${parts.join('\n\n')}`;
};

// las herramientas que suma un agente que escribe codigo, con los mismos permisos que el workspace
const editTools = ['replace_in_file', 'write_file', 'create_folder', 'rename_path', 'run_command'];

// lo que se muestra al lado de cada herramienta en el panel del agente
const stepDetail = (args) => {
  if(!args || typeof args !== 'object') return '';
  const value = args.path || args.file || args.query || args.pattern || args.command || args.url || args.address || args.text || '';
  return String(value).replace(/\s+/g, ' ').slice(0, 90);
};

// los resultados viejos se recortan para que un agente con muchas vueltas no se pase del contexto
const toolResultBudget = 150000;
const isToolResult = (item) => item.role === 'tool' || (item.role === 'user' && /^Tool \S+ result:\n/.test(String(item.content || '')));
const shrinkResults = (messages) => {
  let total = messages.reduce((sum, item) => sum + String(item.content || '').length, 0);

  for(let i = 0; i < messages.length - 6 && total > toolResultBudget; i++){
    const item = messages[i];
    const text = String(item.content || '');
    if(!isToolResult(item) || text.length <= 600) continue;
    const kept = `${text.slice(0, 500)}\n(older result shortened, run the tool again if you need it whole)`;
    total -= text.length - kept.length;
    messages[i] = {...item, content: kept};
  }
};

export const runSubAgent = async ({name, role, task, project, model, run, route = null, onStep = null, history = [], persona = '', rounds = 6, canEdit = false}) => {
  const system = `
${persona || `You are subagent ${name}. Role: ${role}.\nYou work in parallel with other subagents.`}
Active project: ${project || 'none'}.

Allowed tools (call them with native function calling):
list_files, read_file, read_range, glob, grep, search_files, ghidra, ida, re_search, sandbox_run, git_status, git_diff, codebase_search${canEdit ? ', replace_in_file, write_file, create_folder, rename_path, run_command' : ''}.

${canEdit ? 'Change files with replace_in_file (exact text to replace) and write_file only for new files. Read a file before you edit it. After editing, run it to check it works.' : 'Do not write or delete files.'}
Deliver clear findings, risks and the next step.
`.trim();
  const toolList = canEdit ? [...subAgentToolSchemas, ...toolSchemas.filter(item => editTools.includes(item.function?.name))] : subAgentToolSchemas;

  const messages = [
    {role: 'system', content: system},
    ...(Array.isArray(history) ? history : []),
    {role: 'user', content: task}
  ];
  const tools = [];
  const subRun = run ? {...run, onEvent: null} : null;

  const repeats = new Map();
  const repeated = (tool, args) => {
    const key = `${tool} ${JSON.stringify(args || {})}`;
    const count = (repeats.get(key) || 0) + 1;
    repeats.set(key, count);
    return count > 2;
  };

  for(let i = 0; i < rounds; i++){
    shrinkResults(messages);
    const reply = route
      ? await askRoute(route, messages, toolList, subRun)
      : await askModelChat(messages, toolList, subRun, model);
    const rawAnswer = reply.content || '';
    const answer = rawAnswer
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .replace(/<think>[\s\S]*$/gi, '')
      .trim();
    const nativeCalls = Array.isArray(reply.tool_calls) ? reply.tool_calls : [];
    const thinking = String(reply.reasoning || (rawAnswer.match(/<think>([\s\S]*?)(<\/think>|$)/i) || [])[1] || '').trim();

    if(thinking){
      onStep?.({thought: thinking.slice(-2000)});
    }

    if(nativeCalls.length){
      const calls = nativeCalls.map(item => ({item, args: parseNativeArgs(item.function?.arguments)}));
      messages.push({role: 'assistant', content: answer, tool_calls: calls.map(call => call.args ? call.item : {...call.item, function: {...call.item.function, arguments: '{}'}}), ...(reply.raw ? {_claude: reply.raw} : {})});

      for(const call of calls){
        const asked = call.item.function?.name || '';
        onStep?.({tool: asked, detail: stepDetail(call.args)});
        const resolved = resolveCall(asked, call.args || {}, '');
        const toolName = resolved.name || asked;
        const callArgs = normalizeToolArgs(toolName, resolved.args || {}, null);

        await placeCall(toolName, callArgs, project);

        let result;
        const toolStart = Date.now();

        if(!call.args){
          result = {error: 'Arguments were not valid JSON, nothing was run. Send the call again with valid JSON arguments.'};
        } else if(resolved.error){
          result = {error: resolved.error, hint: resolved.hint};
        } else if(resolved.calls){
          result = {error: 'Subagents can only read. Report the change you would make instead.'};
        } else {
          try {
            result = repeated(toolName, callArgs)
              ? {error: 'You already ran this exact call twice. Use what it returned or try something different.'}
              : canEdit && editTools.includes(toolName)
                ? await runTool(toolName, callArgs)
                : await runSubTool(toolName, callArgs);
          } catch(e) {
            result = {error: String(e.message || e)};
          }
        }

        if(run?.usage){
          run.usage.toolMs += Date.now() - toolStart;
          run.usage.toolCalls++;
        }

        tools.push({tool: toolName, args: callArgs, result});
        messages.push({role: 'tool', tool_call_id: call.item.id, content: toon(result).slice(0, 6000)});
      }

      continue;
    }

    const calls = parseToolCalls(answer);

    if(!calls.length && hasBrokenToolCall(answer)){
      messages.push({role: 'assistant', content: answer});
      messages.push({role: 'user', content: toolJsonFixMessage});
      continue;
    }

    if(!calls.length){
      return {name, role, answer, tools};
    }

    // los modelos de solo texto pueden pedir varias herramientas juntas: cada vuelta les cuesta una llamada entera
    const results = [];

    for(const call of calls){
      const callArgs = {...(call.args || {})};

      await placeCall(call.tool, callArgs, project);

      let result;
      const toolStart = Date.now();

      onStep?.({tool: call.tool, detail: stepDetail(callArgs)});

      try {
        result = repeated(call.tool, callArgs)
          ? {error: 'You already ran this exact call twice. Use what it returned or try something different.'}
          : canEdit && editTools.includes(call.tool)
            ? await runTool(call.tool, callArgs)
            : await runSubTool(call.tool, callArgs);
      } catch(e) {
        result = {error: String(e.message || e)};
      }

      if(run?.usage){
        run.usage.toolMs += Date.now() - toolStart;
        run.usage.toolCalls++;
      }

      tools.push({tool: call.tool, args: callArgs, result});
      results.push(`Tool ${call.tool} result:\n${toon(result).slice(0, 6000)}`);
    }

    i += calls.length - 1;
    messages.push({role: 'assistant', content: answer});
    messages.push({role: 'user', content: `${results.join('\n\n')}\n\nFinish your answer or request another tool.`});
  }

  // sin vueltas no se tira lo leido: se le pide la conclusion con lo que ya tiene
  try {
    shrinkResults(messages);
    messages.push({role: 'user', content: 'You used all your tool calls. Do not call any tool. Write your final answer now from what you already read: findings with file, function and line, and what is still unverified.'});
    const reply = route
      ? await askRoute(route, messages, null, subRun)
      : await askModelChat(messages, null, subRun, model);
    const answer = String(reply.content || '').replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*$/gi, '').trim();
    if(answer) return {name, role, answer, tools};
  } catch(e) {
    if(run?.signal?.aborted) throw e;
  }

  return {
    name,
    role,
    answer: 'The agent used all its tool calls without a final answer.',
    tools
  };
};

const parsePlan = (text, maxN) => {
  const match = String(text || '').match(/\[[\s\S]*\]/);

  if(match){
    try {
      const arr = JSON.parse(match[0]);

      if(Array.isArray(arr)){
        return arr
          .filter(item => item && (item.goal || item.role))
          .map((item, i) => ({role: cleanText(item.role) || `Agent ${i + 1}`, goal: cleanText(item.goal) || cleanText(item.role)}))
          .slice(0, maxN);
      }
    } catch(e) {}
  }

  return [];
};

const MODE_PLAN = {
  audit: 'Split the review of this project into independent inspection subtasks (bugs, security, performance, tests, dead code, structure). Each subtask = one focused area.',
  build: 'Split building this feature into independent pieces (each piece = one file or one clear component). Each subtask investigates the current code and DESIGNS its piece with concrete code, but does NOT write files.',
  reverse: 'Split this reverse-engineering task into independent targets (each = one function/packet/address to investigate with ghidra and follow to its root).',
  research: 'Split this research into independent angles (each subtask covers one angle: code, decompilation, web, data) so together they answer the question.',
  auto: 'Split this task into independent subtasks that different agents can do in parallel without depending on each other.'
};

export const orchestrate = async ({project, task, mode, maxAgents, onEvent, run}) => {
  const settings = await readSettings();
  const subRun = run ? {...run, onEvent: null} : null;
  const bigModel = cleanText(settings.model) || 'qwen3-coder:30b';
  const workerModel = 'qwen2.5-coder:14b';
  const wave = 3;
  const maxN = Math.min(Math.max(2, parseInt(maxAgents, 10) || 15), 15);
  const cleanProject = cleanText(project);
  const cleanTask = cleanText(task) || 'Review the project.';
  const kind = ['audit', 'build', 'reverse', 'research'].includes(mode) ? mode : 'auto';

  onEvent?.({type: 'tool', tool: `orchestrator: planning with ${bigModel}`});

  const planText = await askModel([
    {role: 'system', content: `You are the planner of a multi-agent team. ${MODE_PLAN[kind]}\nProject: ${cleanProject || 'none'}.\nReturn ONLY a JSON array (max ${maxN} items) of {"role": short name, "goal": one concrete instruction for that agent}. No prose, just the JSON array.`},
    {role: 'user', content: cleanTask}
  ], subRun, bigModel);

  let plan = parsePlan(planText, maxN);

  if(!plan.length){
    plan = [
      {role: 'Explorer', goal: `Map structure and key files relevant to: ${cleanTask}`},
      {role: 'Searcher', goal: `Find symbols, patterns and related files for: ${cleanTask}`},
      {role: 'Analyzer', goal: `Assess implementation, risks and what is missing for: ${cleanTask}`}
    ];
  }

  onEvent?.({type: 'tool', tool: `orchestrator: ${plan.length} workers (${workerModel}), waves of ${wave}`});

  const results = [];

  for(let i = 0; i < plan.length; i += wave){
    const batch = plan.slice(i, i + wave);
    onEvent?.({type: 'tool', tool: `orchestrator: wave ${Math.floor(i / wave) + 1}/${Math.ceil(plan.length / wave)} (${batch.map(item => item.role).join(', ')})`});

    const batchResults = await Promise.all(batch.map(item => runSubAgent({
      name: item.role,
      role: item.role,
      task: `${item.goal}\n\n(Overall goal: ${cleanTask}). You are READ-ONLY: investigate and report concrete findings; if this is a build task, include the exact code you would write. Do NOT write files.`,
      project: cleanProject,
      model: workerModel,
      run
    }).catch(e => ({name: item.role, role: item.role, answer: `error: ${e.message || e}`, tools: []}))));

    results.push(...batchResults);
  }

  let judge = null;

  if(kind === 'build' && cleanProject){
    onEvent?.({type: 'tool', tool: 'orchestrator: judge (build)'});

    try {
      judge = await runTool('sandbox_run', {project: cleanProject, command: 'npm.cmd run build'});
    } catch(e) {
      judge = {error: String(e.message || e)};
    }
  }

  onEvent?.({type: 'tool', tool: `orchestrator: synthesizing with ${bigModel}`});

  const findings = results.map((item, i) => `### Agent ${i + 1} - ${item.role}\n${String(item.answer || '').slice(0, 3000)}`).join('\n\n');
  const judgeBlock = judge ? `\n\nBUILD/JUDGE RESULT:\n${toon(judge).slice(0, 2000)}` : '';

  const answer = await askModel([
    {role: 'system', content: 'You are the lead of a multi-agent team. Synthesize the agents findings into ONE clear, concrete answer for the user (in the user language). Merge, remove duplicates, resolve contradictions, give the conclusion and next steps. If it was a build task, present the assembled solution/code so it can be applied.'},
    {role: 'user', content: `TASK: ${cleanTask}\n\nAGENT FINDINGS:\n${findings}${judgeBlock}`}
  ], subRun, bigModel);

  await pushAgentLog({project: cleanProject, task: cleanTask, results});

  return {mode: kind, plan, agents: results.map(item => ({role: item.role, answer: item.answer})), judge, answer};
};

const reverseTask = /\b(ghidra|ida|decompil\w*|disassembl\w*|desensambl\w*|xrefs?|FUN_[0-9a-f]{4,}|sub_[0-9a-f]{4,}|0x[0-9a-f]{4,})\b/i;
const changeVerbs = /\b(create|make|build|add|update|fix|change|modify|implement|write|generate|set ?up|scaffold|refactor|redesign|improve|put|insert|crea|cre[aá]|creame|haz|hac[eé]|hazme|agrega|agreg[aá]|a[nñ]ade|arregla|arregl[aá]|pon|pone|pon[eé]|ponle|cambia|cambi[aá]|modifica|actualiza|implementa|escribe|genera|arma|arm[aá]|construye|mejora)\b/i;
const questionStart = /^\s*(what|why|how|where|when|which|who|whose|is|are|should|explain|(?:do|does|did)\s+(?:you|i|we|it|this|that|they)|qu[eé]|por qu[eé]|c[oó]mo|d[oó]nde|cu[aá]ndo|cu[aá]l|qui[eé]n|explica\w*)\b/i;
const decompiledCode = /\b(undefined[1248]?|FUN_[0-9a-f]{4,}|DAT_[0-9a-f]{4,}|param_\d+|local_[0-9a-f]+|__thiscall|__fastcall|in_stack_\w+|unaff_\w+|extraout_\w+)\b/;
const languageFiles = {html: 'index.html', htm: 'index.html', css: 'style.css', scss: 'style.scss', less: 'style.less', js: 'script.js', javascript: 'script.js', mjs: 'script.js', jsx: 'App.jsx', ts: 'script.ts', typescript: 'script.ts', tsx: 'App.tsx', json: 'data.json', py: 'main.py', python: 'main.py', sql: 'query.sql', md: 'README.md', markdown: 'README.md', vue: 'App.vue', svelte: 'App.svelte', php: 'index.php', cs: 'Program.cs', csharp: 'Program.cs', java: 'Main.java', go: 'main.go', rs: 'main.rs', rust: 'main.rs', c: 'main.c', cpp: 'main.cpp', lua: 'main.lua', yaml: 'config.yml', yml: 'config.yml', xml: 'data.xml', svg: 'image.svg'};

export const detectBuildIntent = (message, project = '') => {
  const source = String(message || '').trim();
  const folder = source.match(/\b(?:in|inside|into|en|dentro de)\s+(?:the\s+|a\s+|la\s+|el\s+|una\s+)?(?:folder|directory|carpeta|directorio)\s+(?:called\s+|named\s+|llamada\s+)?[`"']?([\w.\-\\/:]*[\w-])[`"']?/i)?.[1] || '';
  const absolute = source.match(/\b([a-zA-Z]:[\\/][^\s`"'<>|]+)/)?.[1] || '';

  if(!source || reverseTask.test(source) || !changeVerbs.test(source)){
    return null;
  }

  if(questionStart.test(source) && !/^\s*(can|could|would|will)\s+you\b/i.test(source)){
    return null;
  }

  const target = cleanText(project) || absolute || (/^[a-zA-Z]:[\\/]/.test(folder) ? folder : '') || folder;

  if(!target){
    return null;
  }

  return {
    project: target,
    folder: cleanText(project) && folder && !/^[a-zA-Z]:[\\/]/.test(folder) ? folder : ''
  };
};

export const codeBlocksOf = (text) => {
  return [...String(text || '').matchAll(/```([^\n`]*)\n([\s\S]*?)```/g)]
    .map(match => ({raw: match[0], info: match[1].trim(), code: match[2].replace(/\s+$/, ''), at: match.index}))
    .filter(block => block.code.trim().length > 40)
    .filter(block => !decompiledCode.test(block.code))
    .filter(block => !/^(bash|sh|shell|zsh|cmd|bat|powershell|ps1|pwsh|console|terminal|text|txt|plaintext|output|log|diff)$/i.test(block.info.split(/[\s:{]/)[0]));
};

const blockName = (block, blocks, answer, message, used) => {
  const language = block.info.split(/[\s:{]/)[0].toLowerCase();
  const fallback = languageFiles[language] || `file${blocks.indexOf(block) + 1}.${/^[a-z0-9]{1,6}$/.test(language) ? language : 'txt'}`;
  const extension = fallback.split('.').pop().toLowerCase();
  const fits = (name) => !!name && (!languageFiles[language] || name.split('.').pop().toLowerCase() === extension);
  const before = answer.slice(Math.max(0, block.at - 240), block.at).split('\n').filter(line => line.trim()).pop() || '';
  const html = blocks.find(item => /<html|<!doctype|<head|<body/i.test(item.code))?.code || '';
  const linked = extension === 'css'
    ? html.match(/<link[^>]+href=["']([^"':]+\.css)["']/i)?.[1]
    : extension === 'js'
      ? html.match(/<script[^>]+src=["']([^"':]+\.js)["']/i)?.[1]
      : '';
  const asked = [...String(message || '').matchAll(/([\w.\/\\-]+\.([a-z0-9]{1,6}))\b/gi)].find(item => item[2].toLowerCase() === extension)?.[1];
  const named = block.info.match(/([\w.\-\/\\]*[\w-]\.[a-z0-9]{1,6})/i)?.[1]
    || block.code.split('\n')[0].match(/^\s*(?:<!--|\/\/|\/\*|#|--)\s*([\w.\-\/\\]+\.[a-z0-9]{1,6})\s*(?:-->|\*\/)?\s*$/i)?.[1]
    || [...before.matchAll(/([\w.\-\/\\]*[\w-]\.([a-z0-9]{1,6}))/gi)].map(item => item[1]).find(fits)
    || (fits(linked) ? linked : '')
    || asked
    || fallback;
  const clean = named.replace(/\\/g, '/').replace(/^(\.\/)+/, '').replace(/^\/+/, '');

  if(!used.has(clean.toLowerCase())){
    return clean;
  }

  const dot = clean.lastIndexOf('.');

  for(let n = 2; n < 100; n++){
    const next = `${clean.slice(0, dot)}${n}${clean.slice(dot)}`;

    if(!used.has(next.toLowerCase())){
      return next;
    }
  }

  return `${makeId()}-${clean}`;
};

const projectFiles = async (root, most = 300) => {
  const found = [];
  const skip = new Set(['node_modules', '.git', 'dist', 'build', 'target', '.next', 'venv', '__pycache__']);
  const walk = async (dir, rel, depth) => {
    if(found.length >= most || depth > 5) return;
    let list = [];
    try {
      list = await readdir(dir, {withFileTypes: true});
    } catch(e) {
      return;
    }
    for(const item of list){
      if(found.length >= most) return;
      if(item.name.startsWith('.')) continue;
      if(item.isDirectory()){
        if(!skip.has(item.name)) await walk(path.join(dir, item.name), rel + item.name + '/', depth + 1);
      } else {
        found.push(rel + item.name);
      }
    }
  };
  await walk(root, '', 0);
  return found;
};

// otro modelo arma el plan y el de esta pc lo ejecuta con sus herramientas
const planFirst = async ({settings, message, history, projectRoot, chatId, run, onEvent}) => {
  if(!settings.plannerRoute || String(message || '').trim().length < 12){
    return '';
  }

  const route = (await readRoutes()).find(item => item.id === settings.plannerRoute);
  if(!route){
    return '';
  }

  const files = projectRoot ? await projectFiles(projectRoot) : [];
  const past = history.slice(-4)
    .map(item => `${item.role === 'assistant' ? 'Assistant' : 'User'}: ${String(typeof item.content === 'string' ? item.content : '').slice(0, 1500)}`)
    .filter(line => line.length > 12)
    .join('\n\n');
  const ask = [
    past ? `Recent conversation:\n${past}` : '',
    files.length ? `Project files:\n${files.join('\n')}` : 'No project is open.',
    `New request from the user:\n${message}`
  ].filter(Boolean).join('\n\n');

  const system = 'You plan coding work for another model that runs on the user PC with tools to read, write and run files. You cannot see file contents, only names, and you see what that model reported after each task. Write a short numbered plan (at most 8 steps): which files to read first, what to change in each, how to verify it runs, and traps to avoid. No code longer than a few lines. Write it in the language of the user. If the request is just chat or a question that needs no work, answer only NO_PLAN.';
  onEvent?.({type: 'tool', tool: `planning with ${routeLabel(route)}`});
  try {
    let reply;

    // claude code guarda una sesion por chat y recuerda los planes y lo que hizo el modelo local
    if(route.kind === 'claudecode' && chatId){
      const filesKey = `${projectRoot}|${files.length}|${files.join('|').length}`;
      const kept = chatSession(chatId);
      const lastDone = [...history].reverse().find(item => item?.role === 'assistant' && typeof item.content === 'string' && item.content.trim());
      const next = [
        lastDone ? `What the local model did and answered after the last request:\n${lastDone.content.slice(0, 4000)}` : '',
        kept?.filesKey !== filesKey ? (files.length ? `Project files now:\n${files.join('\n')}` : 'No project is open now.') : '',
        `New request from the user:\n${message}`
      ].filter(Boolean).join('\n\n');
      reply = await askInSession({route, chatId, system, first: ask, next, run: {signal: run.signal}, note: {filesKey}});
      onEvent?.({type: 'tool', tool: reply.resumed ? 'planner: same Claude session as this chat' : 'planner: new Claude session for this chat'});
    } else {
      reply = await askRoute(route, [{role: 'system', content: system}, {role: 'user', content: ask}], null, {signal: run.signal});
    }

    const plan = String(reply.content || '').trim();
    if(!plan || /^NO_PLAN\b/i.test(plan)){
      onEvent?.({type: 'tool', tool: 'planner: no plan needed'});
      return '';
    }
    onEvent?.({type: 'thought', thought: `Plan from ${routeLabel(route)}:\n${plan}`});
    return `\n\nPLAN FROM THE PLANNER (${routeLabel(route)}): follow it, but check each step against the real code and fix the plan if the code says otherwise.\n${plan.slice(0, 6000)}`;
  } catch(e) {
    if(run.signal?.aborted) throw e;
    onEvent?.({type: 'tool', tool: `planner failed: ${String(e.message || e).slice(0, 120)}`});
    return '';
  }
};

export const getAgentAnswer = async (options = {}) => {
  const live = runStore.getStore();

  if(!live){
    const defaults = await readSettings();
    const fresh = {
      id: makeId(),
      chatId: '',
      mode: normalizeMode(options.mode, defaults.accessMode),
      signal: options.control?.abort?.signal,
      onEvent: options.onEvent,
      children: new Set()
    };

    fresh.signal?.addEventListener('abort', () => stopChildren(fresh), {once: true});
    return await runStore.run(fresh, () => getAgentAnswer(options));
  }

  const {message, history = [], project, reverseTool = 'auto', onEvent, control, image, thinkDepth = null, thinkCap = 0, ratio: savedRatio = 0, memo = null} = options;
  let mode = live.mode;
  const settings = await readSettings();
  await ensurePluginsLoaded();

  const run = {
    signal: control?.abort?.signal,
    onEvent,
    thinkDepth: ['off', 'light', 'normal', 'deep', 'full'].includes(thinkDepth) ? thinkDepth : null,
    thinkCap: Number(thinkCap) > 0 ? Number(thinkCap) : 0,
    usage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedTokens: 0,
      lastPromptTokens: 0,
      contextLimit: await contextWindow(settings),
      thinkMs: 0,
      writeMs: 0,
      toolMs: 0,
      firstTokenMs: 0,
      outputSpeed: 0,
      promptSpeed: 0,
      rounds: 0,
      toolCalls: 0
    }
  };

  let projectRoot = '';

  if(project){
    try {
      projectRoot = await getProjectPath(project);
    } catch(e) {
      projectRoot = '';
    }
  }

  let effectiveMessage = message;
  const pictures = [];

  if(image){
    const blind = blindReason(await readEngine());
    if(!blind){
      pictures.push(`data:image/png;base64,${image}`);
      effectiveMessage = message || '(the user sent a picture with no text)';
    } else {
      effectiveMessage = `${message || ''}\n\n[the user attached a picture but it could not be read: ${blind}]`;
    }
  }

  const skills = await loadSkills();
  const skillsBlock = buildSkillsBlock(skills);
  const memoryBlock = await buildMemoryBlock();
  const reverseAsked = [message, ...history.filter(item => item?.role === 'user').slice(-3).map(item => typeof item.content === 'string' ? item.content : '')].join('\n');
  const reverseTalk = /\b(ghidra|ida|ida_bridge|decompil\w*|disassembl\w*|desensambl\w*|xrefs?|reversing|reverse engineering|re_search|re_map|re_index)\b/i.test(reverseAsked)
    || (addressRe.test(String(message || '')) && reverseAskRe.test(String(message || '')));
  const reverseBlock = !reverseTalk
    ? ''
    : `\n\nReverse-engineering bridge preference: ${['auto', 'ghidra', 'ida'].includes(reverseTool) ? reverseTool : 'auto'}\n\n`
      + `REVERSE ENGINEERING: if the user talks about Ghidra, functions, addresses (hex like 01b25f2a), decompiling, disassembling, xrefs or reversing a binary, use the configured reverse-engineering tool. If the user explicitly names a tool or gives a port, obey that exact tool and port; never substitute the other bridge or its configured default. When the configured tool is IDA, do not call Ghidra; when it is Ghidra, do not call IDA. Start with "HELP" or "CURRENT" for Ghidra if unsure. IDA commands include DECOMPILE, DISASM, NAME, XREFS, FIND and STR. To map many functions at once, use parallel_agents (subagents have ghidra and ida).
If the user asks WHERE something is in the binary ("where the license is validated", "which function decrypts X") and you don't know the address: first re_search (maybe you already investigated it), and if not, collect candidate addresses with XREF/TRACE/DECOMP_XREF and follow the chain. Every DECOMP you do is auto-saved for future searches.

`
      + (reverseTool === 'ida' ? '' : `GHIDRA ARSENAL (when to use each command; ONLY when Ghidra is selected — ignore this section for IDA):
- CURRENT: decompiled C++ and disassembly of the current active cursor/view in Ghidra.
- DECOMP <addr>: function pseudo-C. Your primary decompilation tool.
- FUNC <addr>: disassembly listing. Use it to verify assembly instructions, opcode writes (mov [reg+off],..) or reads (mov ..,[reg+off]).
- XREF <addr>: who REFERENCES that address (callers of a func, or accesses to a data/global).
- XFROM <addr>: what that function calls (callees). To map outgoing calls.
- CALLERS <addr>: filtered references finding functions that CALL this address (direct & resolved indirect calls).
- WRITERS <addr>: filtered references finding who WRITES to a memory address/global variable.
- READERS <addr>: filtered references finding who READS from a memory address/global variable.
- FINDPTR <addr>: scans entire binary memory for 32-bit little-endian pointers pointing to <addr> (finds hidden vtables, callbacks, dispatch tables).
- STR <ascii text>: find raw ASCII text in memory and list all referencing functions.
- WSTR <utf16 text>: find raw UTF-16LE text in memory and list all referencing functions.
- BYTES <hex bytes>: raw memory pattern scan (e.g. BYTES 55 8B EC or crypto constants).
- DECOMP_XREF <addr>: decompiles directly all functions that reference this address. Shortcut.
- TRACE <addr>: follow the flow. Combines XREF, DECOMP_XREF, and NEARSTR to map the entire context of an address.
- NEARSTR <addr>: strings near an address (within ±128 bytes) to identify unnamed functions.
- READPTR <addr> [count]: follow a pointer chain and dereference memory slots. For nested structs.
- STRUCT <addr> [n]: dumps fields in memory resolving int, ptr, func, and string types automatically.
- VTABLE <funcaddr>: finds vtables containing this function + neighboring virtual method slots with names.
- VTDUMP <addr> [count]: dumps entire vtable slot-by-slot, mapping addresses to method names.
- MKPTRS <addr> [count]: marks sequence in memory as pointers in Ghidra (fixes raw byte views).
- RTTI <addr|Clase|Ns::Clase>: walks MSVC 32-bit RTTI (TypeDescriptor -> CompleteObjectLocator -> vftable) to discover class names, constructors, destructors!
- DEFSTRUCT <name> [size]: dynamically defines a new structure in Ghidra's DataTypeManager.
- SETFIELD <struct> <off> <type> [field]: defines field offset, type, and name in a structure.
- RETYPE <func> <var> <type>: retypes local variable or parameter in the decompiler AST (HighFunctionDBUtil).
- SETCONV <func> <convention>: sets calling convention (__thiscall, __stdcall, __fastcall, __cdecl).
- PCODE <addr>: high-level P-code operations from decompiler (STOREs, CALLIND, BRANCHIND).
- PATCH <addr> <hex bytes>: in-memory live byte patching (clears code units, writes bytes, flushes cache).
- MAKEFUNC <addr>: forces creation/disassembly of a function at address.
- RENAME_FUNC <addr> <name>: renames function symbol (SourceType.USER_DEFINED).
- RENAME_SYMBOL <addr> <name>: renames data or global label.
- RENAME_LABEL <addr> <name>: renames code label.
- RENAME_LOCAL <func_addr> <old_name> <new_name>: renames local variable in decompiler.
- IMPORTXREF <name>: who uses a DLL import (e.g. send, recv, memcpy, VirtualProtect).
- QUIT / SHUTDOWN / EXIT: safely closes socket and frees server port.

`)
      + (reverseTool === 'ghidra' ? '' : `IDA COMMANDS (when IDA is selected; never send Ghidra commands to IDA):
- PING: health check (returns "pong").
- DECOMPILE <addr>: Hex-Rays pseudo-C for a function.
- DISASM <addr> [count]: disassembly listing.
- NAME <addr>: symbol/function name at address.
- RENAME <addr> <name>: rename function or label.
- XREFS <addr>: callers and data cross-references.
- FIND <text>: search function and symbol names.
- STR <text>: search strings recognized in the IDA database.
- EXPORT [path]: export disassembly or database dump.
- QUIT: close socket connection.

`)
      + `RE RULES (mandatory):
- Questions about a binary (functions, addresses, structures, strings in an exe or dll) go to the selected reverse-engineering tool, not to grep over source code.
- Never state what a function does, who fills a field or what a value means without having seen it in the tool output in THIS response.
- Each function or address is investigated from scratch; do not carry conclusions over from earlier questions.

EXAMPLE of correct investigation (imitate this methodology, not the result):
Question: "what does FUN_xxxx do and who calls it?"
1. ghidra "DECOMP xxxx"  -> read the real pseudo-C. See it reads a global DAT_yyyy and calls vtable+0x70.
2. ghidra "XREF xxxx"  -> see it is called by FUN_zzzz from such site.
3. ghidra "DECOMP zzzz"  -> understand the caller's context.
4. To know who FILLS the data: ghidra "XREF DAT_yyyy" -> find the WRITEs, decompile that writer.
5. Only after seeing all that, conclude. If something is unverified, say it: "I did not confirm X".

`;
  const seniorBlock = `

SENIOR WORKFLOW (always follow it; you are not a junior who dumps code and leaves):
1. PLAN: if the task has more than one step, first look at the relevant code (read_file/grep) and form a short mental plan. For big tasks use plan_create and execute step by step.
2. READ BEFORE WRITING: never edit an EXISTING file without reading it first. BUT if you are CREATING something new from scratch (e.g. "make a store", empty/new project), do NOT waste calls on list_files/read_file - just write_file directly. Only explore when you need to understand existing code.
2b. WEB PAGES: for "make a store/landing/site", write ONE self-contained index.html (HTML+CSS+JS inline, or Tailwind via CDN <script src="https://cdn.tailwindcss.com">). Do it in a single write_file. Do not over-plan or list files first - just build it, then visual_check.
3. VERIFY FOR REAL: after writing code, RUN it with sandbox_run (or run_command) to confirm it does not break. If it errors, read the error, fix it, run again. Do not say "done" until it runs.
3b. IF IT IS A WEB PAGE (html/css/front): after editing use visual_check to SEE how it looks (screenshot + console errors + design critique). If something overflows, overlaps, is unreadable or looks bad, fix it and look again. Never deliver a web page without having looked at it.
4. SELF-CRITIQUE: before delivering, re-read your own change (read_range of the edited file) looking for bugs, typos, edge cases or half-done work. If you find something, fix it before answering.
5. REPORT SHORT: at the end, summarize what you changed, in which file, and what you verified. Do not paste the whole code.
If you learn a durable fact about the project or a user preference, save it with memory_add.

LONG TASKS (autonomous): if the request is big (several files, "build the whole app", "implement X fully"), do NOT stop after one step. Create a plan_create with all the steps and execute them ONE BY ONE in the same response: do the step, plan_update to done, continue with the next, until the list is complete. Do not hand control back to the user mid-task unless you need a real decision from them. Keep working until the plan is complete.

EXAMPLE of solving a vague request well (imitate this flow):
User: "make me a store website that works"
Bad (what you do NOT do): write 15 files, say "done, all working" without running anything.
Good (what you DO):
1. Pick a simple stack that runs on its own: a single index.html with HTML+CSS+JS (no build, no deps that can fail). Assume that and say it in one line.
2. write_file the complete, working index.html.
3. visual_check the index.html -> see how it looks, whether there are console errors or broken layout.
4. If something is wrong, fix it and visual_check again.
5. Report: "I created the store in index.html (project X). I opened and verified it: looks good, no errors. It has catalog, cart and checkout."
If the user asks for React/Node with deps: do NOT use interactive scaffolders (npm create vite, create-react-app, npx tailwindcss init) - they PROMPT and HANG when run without a keyboard. Instead write the project files directly with write_file: package.json (with deps + scripts), vite.config.js, tailwind.config.js, postcss.config.js, index.html, and src/ files. Then run "npm.cmd install" and verify with "npm.cmd run build" (NEVER "npm run dev"/"npm start", they never exit). Read the REAL output; only say it works once you saw it compile without error. To SEE the web page use visual_check.`;
  const planBlock = await planFirst({settings, message: effectiveMessage, history, projectRoot, chatId: live.chatId, run, onEvent});
  const replyLang = settings.language === 'en' ? 'English' : 'Spanish';
  const system = `
You are an autonomous senior engineer working on the user's machine. You act like a professional with judgment, not an autocomplete. Always reply to the user in ${replyLang}, clear and direct.

HOW YOU WORK (your constitution, non-negotiable):
1. UNDERSTAND first. If the request is a vague TASK ("improve this", "make it good"), do NOT ask up front: look at the code/context with tools, infer the best interpretation and START. BUT if the input is a single ambiguous word, gibberish, a greeting, or names a project/file that does not exist, do NOT invent work or create anything: use list_projects and ask the user what they mean. NEVER create a new project/files from a lone unclear word (e.g. "hgola") - creating a project needs a clear request like "create a Go project called X". A typo is not an instruction.
2. BE DECISIVE. Pick the best path and execute it. No "you could do X or Y, tell me which"; you pick the best and do it, noting in one line what you assumed.
3. PLAN if it is big. Break it into steps and execute them straight through, without handing control back midway.
3b. ORCHESTRATE the giant ones. If the task is LARGE and splits into independent parts (audit the WHOLE project, build a full multi-file app/feature, reverse-engineer SEVERAL functions/packets, or wide research across many sources), call the orchestrate tool (mode: audit|build|reverse|research, up to 15 agents) instead of doing it all yourself. For normal single-file or single-step tasks, do NOT use orchestrate - it is slow and wasteful for small work; just do it directly.
4. ALWAYS VERIFY. After writing code, RUN it (sandbox_run/run_command). If it fails, read the error, fix it, run again. Never say "done/works" without having executed it and seen it pass. Lying about this is the worst possible mistake.
5. DO NOT HALLUCINATE. Only state what you saw in a tool's real output. If you did not verify it, say so.
6. DELIVER RESULTS, not promises. The work is done when it actually works, not when the code "should" work.
7. REPORT SHORT. At the end: what you did, where, and what you verified. Do not repeat the whole code, no filler.
8. USE THE WEB if you lack info. If you do not know an API, a format, an error message, a library version: web_search + web_fetch instead of guessing/hallucinating. Better to search than to invent.

GENERAL PROGRAMMING (rigor): before editing, read the file and the ones that depend on it (grep the symbol). When you finish non-trivial code, run whatever exists: tests (npm.cmd test / node --test / pytest), typecheck (npx.cmd tsc --noEmit), or the build. Read the REAL error and fix it; do not assume. Keep the existing code style.

Use tools when you need to see, create, edit, test or coordinate.
Allowed root folder: ${paths.projectsRoot}
Active project: ${project || 'none'}

Tools:
list_projects, list_files, read_file, read_range, glob, grep, write_file, delete_path,
create_folder, rename_path, search_files, replace_in_file, propose_patch, run_command,
sandbox_run, sandbox_start, sandbox_read, sandbox_stop, terminal_start, terminal_read,
terminal_stop, terminal_list, git_status, git_diff, git_commit, context_index,
context_get, context_pick, context_read, plan_create, plan_update, plan_list,
memory_add, memory_search, memory_list, parallel_agents, orchestrate,
vector_remember, vector_recall, read_document, deep_research, speak,
generate_image, note_add, note_list, task_add, task_list, task_done,
compare_models, plugin_list, plugin_reload${reverseTalk ? ', ghidra, ida, re_index, re_search, re_map' : ''}.

EXTRA CAPABILITIES: you have long-term semantic memory (vector_remember to save, vector_recall to search by meaning); you can read PDFs/DOCX (read_document with an absolute path); do multi-step web research (deep_research); talk out loud (speak); take notes and tasks; compare models; and use external plugins/MCP servers (plugin_list). Reach for these when useful.

TOOL PROTOCOL:
Call tools with native function calling. NEVER describe the tool in text or say "I'm going to use X": CALL it directly.
FILES GO TO DISK, NOT TO THE CHAT: when the user asks you to create or change something in a project, every file you produce must be written inside that project with write_file (new or whole files) or replace_in_file (edits). Never answer with the code pasted in the chat instead of writing it. When you are done, reply with a short summary: which files you wrote or changed and what you checked.
Never ANNOUNCE without delivering: no answers that are just "Building..."/"Writing..."/"I'll create...". For a NEW file call write_file with the entire content in one call; for a very long new file write the first part and extend it with replace_in_file.
To CHANGE an existing file use replace_in_file with only the lines that change (search = the current lines, replace = the new ones). Never rewrite a whole file with write_file to change a few lines: writing hundreds of unchanged lines again takes minutes.
If a tool returns an error, read the message, fix the arguments and retry. Do not pass the raw error to the user.
Chain all the tools you need before answering. Reply to the user only when the task is done.

PROJECT RULES:
The active project is "${project || 'none'}". When the user mentions a file WITHOUT naming a project, ALWAYS use the active project. Do NOT guess the project from the file name.
If the user gives an ABSOLUTE folder path (e.g. D:\\work\\my-app) that is not a project: register it ONCE with add_project, then pass THAT path (or the name) as "project" in grep/search_files/codebase_search/read_file. Tools accept absolute paths. Do NOT use run_command to search text: use grep or codebase_search. If a tool fails, read the error and fix it; do not repeat the same call 10 times.
If read_file returns ENOENT, do NOT give up: use list_files of the active project, find the similarly named file and retry.
GROUND TRUTH:
- For current project, path, file list, file count, or whether something exists, call list_projects/list_files in THIS response first.
- Memory, old chats, summaries, and context indexes are hints only; never present them as current filesystem facts.
- If the tool says a project is empty or a file is missing, report that result plainly. Do not merge it with older project contents.
- For a greeting or casual message with no task, greet briefly. Do not volunteer a project name, old files, or a project summary unless the user asks.

IMPORTANT - YOUR REAL CAPABILITIES:
YOU CAN install software, dependencies and run commands: use run_command (e.g. npm.cmd install express, pip install flask, npm.cmd uninstall package).
YOU CAN access, create, edit and delete project files: read_file, write_file, delete_path.
NEVER say "I don't have the ability to install" or "I can't access files" or "security restrictions". It is FALSE: you have tools for all of that.
NEVER give manual step-by-step guides for the user to install something: do it YOURSELF with run_command.
NEVER say "I created/saved the file" without having called write_file in THIS response. Saying you did something you did not do is a serious failure.
NEVER paste the full code in the chat: write it to the file with write_file and in the chat only summarize what you changed and where.
AFTER writing a file, VERIFY: use list_files or read_range to confirm it was written, or sandbox_run to test that it works. Only then report to the user.
If the request is vague (e.g. "you do it"), look at the project with list_files and read_file, decide and act.
On Windows use npm.cmd and npx.cmd instead of npm/npx.

${reverseBlock}If the user asks for subagents, parallel, split work or multiple roles, use parallel_agents.
If the user asks to validate or test without touching the project, use sandbox_run.

GOTCHAS CHEATSHEET (avoid them):
- Vite needs index.html at the project ROOT (not in public/) with <script type="module" src="/src/main.jsx">. CRA puts it in public/. Do not mix them.
- Every package you import must be in package.json AND installed with npm install. Declaring it is not enough.
- Vite entry = src/main.jsx; CRA entry = src/index.js. A project is one or the other, not both.

DESIGN (do NOT leave it generic or bland): a web page that "works" but looks ugly is half-done work. Always apply:
- Cohesive palette: one strong accent color (not the default bootstrap blue) + neutrals. Layered subtle backgrounds, not flat white/gray.
- Breathing room: generous padding and margins, plenty of whitespace. Nothing cramped.
- Hierarchy: large, heavy titles, smaller and dimmer secondary text. Clear type scale.
- Modern details: rounded corners (rounded-xl/2xl), soft shadows, transitions and hover states on buttons/cards, subtle gradients where they help.
- Product cards: image with fixed aspect-ratio, hover that lifts (scale/shadow), highlighted price.
- Truly responsive: grid that collapses, mobile-first.
- If you use Tailwind: do NOT leave minimal classes; use spacing, shadow, ring, hover:, transition, gradients. Default Tailwind without work looks generic.
- Before saying "done", use visual_check and see if it looks pro or bland; if bland, improve it.

If the task is big, create a plan with plan_create.
If you need context, use context_index/context_pick/context_read.
Save preferences or durable facts with memory_add.
Use terminal_start for long sessions and for servers that keep running. If run_command answers with background: true, the server is already running in a terminal session: do not start it again, go on with the task.
Ports 3001, 8000, 9090, 9096 and 9097 belong to Sneo on this PC: a project server must never listen on them. Use the port the user asks for, or a free one such as 5050-5059. Never stop programs by name (taskkill /IM node.exe, Stop-Process -Name node, killall node): Sneo itself runs on Node and would die with them. Stop your own servers with terminal_stop, or kill the single PID that holds a port.
To edit code with preview/approve, use propose_patch.
After editing, report file, lines added/removed/changed and location.
Do not delete files unless the user clearly asks.
`.trim() + seniorBlock + skillsBlock + await styleBlock(settings.styleFiles) + planBlock;

  const window = await contextWindow(settings);
  const guard = new LoopWatch({mode, window});
  const toolResults = [];
  const thoughts = [];
  const steps = [];
  const memoUsable = !!memo?.text && Number(memo.count) > 0 && Number(memo.count) <= history.length;
  const firstKept = memoUsable ? Number(memo.count) : 0;
  const origin = new Map();
  const carriers = new Set();
  const past = [];
  let ratio = Number(savedRatio) > 0 ? Number(savedRatio) : defaultRatio;
  let covered = memoUsable ? Number(memo.count) : 0;
  let lastSummary = memoUsable ? memo.text : '';
  let stopReason = null;
  let warnedRoom = false;

  run.usage.contextLimit = window;

  history.slice(firstKept).forEach((item, offset) => {
    if(!keepInHistory(item)){
      return;
    }

    // un chat viejo puede traer razonamiento guardado como respuesta: al modelo vuelve solo lo que contesto
    const entry = {role: item.role, content: item.role === 'assistant' ? withoutThinking(item.content) || item.content : item.content};
    origin.set(entry, firstKept + offset);
    past.push(entry);
  });

  const modeRule = mode === 'plan'
    ? 'PERMISSION MODE: Plan. You can read, search and run read-only commands, but you cannot change anything: no file writes, no deletes, no installs, no commands that change state. Investigate what you need, then answer with a numbered plan: which files to create or change, what goes in each, which commands to run and how to verify it. Do not paste whole files. Do not claim you changed anything. The user can run the plan afterwards.'
    : mode === 'manual'
      ? 'PERMISSION MODE: Manual. Reading and read-only commands run directly. Every change (writing, renaming or deleting files, commands that change something, installs, terminals, sandboxes) waits for the user to approve it. If the user denies a change, do not send it again: continue another way or explain what you need.'
      : 'PERMISSION MODE: Auto. You can change files and run commands without asking. Prefer delete_path over shell deletes, it keeps a backup.';
  const buildIntent = detectBuildIntent(message, project);
  const buildRule = buildIntent
    ? `THIS REQUEST CHANGES PROJECT FILES: write every file you produce with write_file or replace_in_file inside the project "${buildIntent.project}"${buildIntent.folder ? `, in the folder "${buildIntent.folder}"` : ''}. Folders are created automatically. Do not paste the files in the chat; when you finish, answer only with a short summary of the files you wrote and what you checked.`
    : '';
  const lateContext = [memoryBlock.trim(), modeRule, mode === 'plan' ? '' : buildRule].filter(Boolean).join('\n\n');
  let currentUser = {role: 'user', content: `${lateContext}\n\nCURRENT MESSAGE FROM THE USER:\n${effectiveMessage}`, ...(pictures.length ? {images: pictures} : {})};
  let messages = [{role: 'system', content: system}, ...past, currentUser];

  if(memoUsable){
    const placed = placeSummary(messages, memo.text);

    if(placed.mergedWith && origin.has(placed.mergedWith)){
      origin.set(placed.carrier, origin.get(placed.mergedWith));
    }

    if(placed.mergedWith === currentUser){
      currentUser = placed.carrier;
    }

    carriers.add(placed.carrier);
    messages = placed.messages;
  }

  const fitNow = async (tools) => {
    const fitted = await fitMessages({
      messages,
      tools,
      window,
      ratio,
      pinned: currentUser,
      summarize: text => askModel([
        {role: 'system', content: 'You condense the older part of a conversation between a user and a coding agent so the agent can keep working without it. Write a compact summary with: what the user asked and decided, the files and paths involved, the commands run and their results, errors still open, and what was about to happen next. Keep paths, names, numbers and error text exactly as written. Say a tool failed only when its TOOL RESULT shows an error, never guess one. Tell what each tool did in words, never as a call like name(args): the agent reads this summary and keeps calling tools the normal way. No reasoning of your own, no preamble.'},
        {role: 'user', content: text}
      ], {signal: run.signal, thinkDepth: 'off'})
    });

    if(!fitted.compacted){
      return fitted;
    }

    for(const item of fitted.region){
      covered = origin.has(item)
        ? Math.max(covered, origin.get(item) + 1)
        : carriers.has(item)
          ? covered
          : history.length;
    }

    if(fitted.mergedWith && origin.has(fitted.mergedWith)){
      origin.set(fitted.carrier, origin.get(fitted.mergedWith));
    }

    if(fitted.mergedWith === currentUser){
      currentUser = fitted.carrier;
    }

    carriers.add(fitted.carrier);
    lastSummary = fitted.summary;
    messages = fitted.messages;
    onEvent?.({type: 'tool', tool: `older messages summarized to fit the context window (${Math.round(fitted.before / window * 100)}% to ${Math.round(fitted.after / window * 100)}%)`});
    return fitted;
  };

  const formatStep = (tool, args, status = 'done', diffStats = null) => {
    let action = 'Analyzed';
    let label = tool;
    let fileType = '';
    let icon = 'wrench';
    let file = args?.file || args?.path || '';
    let lines = (args?.startLine || args?.endLine) ? `#L${args?.startLine || 1}-${args?.endLine || ''}` : '';

    if(file){
      const lower = file.toLowerCase();
      if(/\.(jsx|tsx)$/i.test(lower)) fileType = 'react';
      else if(/\.(js|mjs|cjs)$/i.test(lower)) fileType = 'js';
      else if(/\.(css|scss|less)$/i.test(lower)) fileType = 'css';
      else if(/\.(json|html|xml|md)$/i.test(lower)) fileType = 'code';
      else fileType = 'doc';
    }

    const plain = text => String(text || '').replace(/\s+/g, ' ').trim();
    const name = typeof tool === 'string' ? tool : JSON.stringify(tool);

    if(['read_file', 'read_range', 'read_document', 'context_read'].includes(name)){
      action = 'Analyzed';
      icon = 'book-open';
      label = file ? path.basename(file) : 'file';
    } else if(['write_file', 'replace_in_file', 'patch_file', 'propose_patch'].includes(name)){
      action = 'Edited';
      icon = 'pen-to-square';
      label = file ? path.basename(file) : 'file';
    } else if(name === 'list_files' || name === 'glob'){
      action = 'Explored';
      icon = 'folder-open';
      label = name === 'glob' && args?.pattern ? plain(args.pattern).slice(0, 60) : `files in ${args?.project || 'workspace'}`;
    } else if(['search_files', 'grep_code', 'grep', 'codebase_search', 're_search', 'pc_search', 'memory_search', 'vector_recall'].includes(name)){
      action = 'Searched';
      icon = 'magnifying-glass';
      label = `"${plain(args?.query || args?.pattern).slice(0, 60)}"`;
    } else if(['web_search', 'web_fetch', 'deep_research'].includes(name)){
      action = 'Browsed';
      icon = 'globe';
      label = plain(args?.query || args?.url).slice(0, 80);
    } else if(['sandbox_run', 'run_command', 'sandbox_start', 'terminal_start'].includes(name)){
      action = 'Ran';
      icon = 'terminal';
      label = plain(args?.command).slice(0, 90);
    } else if(name.startsWith('git_')){
      action = 'Ran';
      icon = 'terminal';
      label = `git ${name.slice(4)}`;
    } else if(name.startsWith('plan_')){
      action = 'Planned';
      icon = 'list-check';
      label = name === 'plan_create'
        ? plain(args?.title).slice(0, 70) || `${Array.isArray(args?.steps) ? args.steps.length : 0} steps`
        : name === 'plan_update'
          ? `step ${plain(args?.status) || 'updated'}`
          : 'the plan';
    } else if(name === 'delete_path'){
      action = 'Deleted';
      icon = 'trash';
      label = path.basename(String(args?.target || '')) || 'a path';
    } else if(name === 'create_folder'){
      action = 'Created';
      icon = 'folder-plus';
      label = plain(args?.folder) || 'a folder';
    } else if(name === 'rename_path'){
      action = 'Renamed';
      icon = 'pen';
      label = `${path.basename(String(args?.from || ''))} to ${path.basename(String(args?.to || ''))}`;
    } else if(tool === 'ghidra'){
      action = 'Ghidra';
      icon = 'microchip';
      label = `${args?.command || ''} ${(args?.target || args?.name || '').slice(0, 25)}`;
    } else if(tool === 'ida'){
      action = 'IDA';
      icon = 'microchip';
      label = `${args?.command || ''} ${(args?.target || args?.name || '').slice(0, 25)}`;
    } else if(tool === 'list_projects'){
      action = 'Checked';
      icon = 'list-check';
      label = 'active projects';
    } else {
      action = 'Used';
      label = name.replace(/_/g, ' ');
    }

    const added = diffStats?.added || 0;
    const removed = diffStats?.removed || 0;

    return {
      action,
      label,
      file: file ? path.basename(file) : '',
      fileType,
      lines,
      added,
      removed,
      tool,
      args,
      status,
      timestamp: Date.now()
    };
  };

  const addStep = (tool, args, status = 'done', diffStats = null, changes = null) => {
    const item = formatStep(tool, args, status, diffStats);
    if(changes?.length) item.changes = changes;
    const existingIdx = typeof steps.findLastIndex === 'function'
      ? steps.findLastIndex(s => s.tool === tool && s.file === item.file && s.status === 'running')
      : -1;
    if(existingIdx >= 0){
      steps[existingIdx] = item;
    } else {
      steps.push(item);
    }
    onEvent?.({type: 'step', step: item});
    return item;
  };

  const useTool = async (name, callArgs, failHint = '', preset = null) => {
    addStep(name, callArgs, 'running');
    onEvent?.({type: 'tool', tool: name, args: callArgs});

    let result = preset;
    const toolStart = Date.now();

    if(!preset){
      try {
        result = await dispatchTool(name, callArgs, run);
      } catch(e) {
        result = {error: String(e.message || e), hint: failHint || 'Fix the arguments or use another tool and retry.'};
      }
    }

    run.usage.toolMs += Date.now() - toolStart;
    run.usage.toolCalls++;
    // las filas del cambio son para la app, el modelo no las necesita
    const changes = result?.change ? [result.change] : Array.isArray(result?.changes) ? result.changes : null;
    const visible = changes ? (({change, changes: rows, ...rest}) => rest)(result) : result;
    const counted = changes ? changes.reduce((sum, item) => {
      for(const row of item.rows || []){
        if(row.kind === 'add') sum.added++;
        if(row.kind === 'del') sum.removed++;
      }
      return sum;
    }, {added: 0, removed: 0}) : null;
    toolResults.push({tool: name, args: callArgs, result: visible});
    addStep(name, callArgs, result?.error ? 'error' : 'done', counted || (result?.lines ? {added: result.lines.added || 0, removed: result.lines.removed || 0} : null), changes);

    if(toolKind(name) === 'file' && result && !result.error && !result.pending){
      onEvent?.({type: 'files', project});
    }

    const verdict = guard.tool(name, callArgs, result);
    const shown = clipText(toon(visible), outputLimit(window, ratio));

    if(verdict?.stop){
      stopReason = verdict.stop;
    }

    return {
      result,
      shown: verdict?.redirect ? `${shown}\n\n[SNEO] ${verdict.redirect}` : shown
    };
  };

  // los nombres y argumentos de otros agentes se traducen antes de correr
  const runCall = async (asked, rawArgs, failHint = '', preset = null) => {
    const lastFile = [...toolResults].reverse().find(item => typeof item.args?.file === 'string' && item.args.file)?.args.file || '';
    const resolved = preset || isPluginTool(asked)
      ? {name: asked, args: rawArgs || {}}
      : resolveCall(asked, rawArgs || {}, lastFile);
    const parts = resolved.calls || [resolved];
    const shown = [];

    for(const part of parts){
      const name = part.name || asked;
      const callArgs = normalizeToolArgs(name, part.args || {}, projectRoot);

      await placeCall(name, callArgs, project);

      const own = preset
        || (part.error ? {error: part.error, hint: part.hint} : null)
        || (part.unknown ? unknownTool(asked) : null);
      const used = await useTool(name, callArgs, failHint, own);
      shown.push(parts.length > 1 ? `${name}: ${used.shown}` : used.shown);

      if(stopReason){
        break;
      }
    }

    return {shown: shown.join('\n\n')};
  };

  const endRun = (text, extra = {}) => ({
    answer: finishAnswer(String(text || '').trim() || silentNote(toolResults.length > 0), toolResults),
    tools: toolResults,
    thoughts: thoughts.join('\n\n'),
    steps,
    usage: run.usage,
    mode,
    ratio,
    memo: lastSummary && covered > 0 ? {count: covered, text: lastSummary} : null,
    ...extra
  });

  const priorAssistant = history.filter(item => item.role === 'assistant').map(item => String(item.content || '').replace(/\s+/g, ' ').trim().toLowerCase());
  const isRepeat = (text) => {
    const norm = String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();

    if(norm.length < 120){
      return false;
    }

    const probe = norm.slice(40, 220);
    return priorAssistant.some(prev => prev.length > 120 && prev.includes(probe));
  };
  let repeatRetries = 0;

  const writeTools = ['write_file', 'replace_in_file', 'propose_patch', 'delete_path', 'create_folder', 'rename_path', 'run_command', 'git_commit'];
  const claimsWriteWork = (text) => {
    const s = String(text || '');
    return /he (a[nñ]adido|creado|guardado|modificado|actualizado|escrito)|ha sido (cread|guardad|a[nñ]adid|actualizad|modificad)|(archivo|c[oó]digo|carpeta|directorio|folder).{0,40}(est[aá]|queda|qued[oó]) guardado|guardado en\s*[`"']?[A-Za-z]:\\/i.test(s)
      || /\b(?:done\s*[-—:]?\s*)?(?:created|added|wrote|written|saved|made|built|set up|implemented|generated)\b.{0,60}\b(?:folder|directory|file|project|app)\b/i.test(s)
      || /\b(?:the|el)\s+(?:`?[a-zA-Z0-9_-]+`?\s+)?(?:folder|directory|carpeta|directorio)\s+is\s+(?:now\s+)?created\b/i.test(s)
      || /i(?:'ve| have) (created|added|wrote|written|saved|made|built|set up|implemented|generated)|i (created|added|wrote|saved|built|implemented|generated)\b|(files?|folders?|directory|directories|store|site|page|project|app|component|code).{0,40}(have been|has been|were|was|is) (created|added|written|saved|built|generated)|created (a|an|the|your|all) .{0,40}(store|site|page|app|project|file|folder|directory|section)/i.test(s);
  };
  let fakeWorkRetries = 0;
  let emptyRetries = 0;
  let verifyRetries = 0;
  let reGroundRetries = 0;
  let buildFailRetries = 0;
  let webVerifyRetries = 0;
  let refusalRetries = 0;
  let announceRetries = 0;
  let codeRetries = 0;
  let toolIntentRetries = 0;
  let brokenRetries = 0;
  let lengthRetries = 0;
  let loopRetries = 0;
  const loopNudge = 'Your last reply got stuck writing the same text again and again, so it was cut. Do not write it again. Decide the next concrete step and do it now with a tool call; if nothing is left to do, answer in a few lines.';
  const lengthNudge = 'Your last reply was cut at the output limit while writing a file, so that write did not complete. Check what is on disk now (read_range or read_file of that file), then continue in smaller pieces: at most 150 lines per call. Write the first part with write_file and extend it with replace_in_file (search the last lines already written, replace them with those lines plus the next part). Do not resend the whole file at once.';
  const selectedReverseTool = ['ghidra', 'ida'].includes(reverseTool) ? reverseTool : 'auto';
  const directGhidraCall = selectedReverseTool === 'ida' ? null : parseDirectGhidraRequest(message);
  const directIdaCall = selectedReverseTool === 'ghidra' ? null : parseDirectIdaRequest(message);
  const directPick = directGhidraCall || directIdaCall || parseDirectIntent(message, project);
  const directIntent = directPick && mode === 'plan' && ['file', 'change'].includes(toolKind(directPick.tool)) ? null : directPick;
  const blockedReverseTools = selectedReverseTool === 'ghidra'
    ? new Set(['ida'])
    : selectedReverseTool === 'ida'
      ? new Set(['ghidra', 're_index', 're_search', 're_map'])
      : new Set();
  const reverseCommandDescriptions = {
    ghidra: 'Query Ghidra. Commands ONLY: HELP, CURRENT, FUNC <addr>, DECOMP <addr>, XREF <addr>, XFROM <addr>, STR <text>, WSTR <text>, BYTES <hex>, DECOMP_XREF <addr>, TRACE <addr>, NEARSTR <addr>, IMPORTXREF <name>, MAKEFUNC <addr>, RENAME_FUNC/RENAME_SYMBOL/RENAME_LABEL <addr> <name>.',
    ida: 'Query IDA Pro through ida_bridge.py. Commands ONLY: PING, NAME <addr>, RENAME <addr> <name>, DECOMPILE <addr>, DISASM <addr> [count], FIND <text>, XREFS <addr>, STR <text>, EXPORT [path], QUIT. Do not use Ghidra commands such as FUNC or DECOMP.'
  };
  const configuredToolSchemas = toolSchemas.map(item => {
    const name = item.function?.name;
    if(!['ghidra', 'ida'].includes(name) || !item.function){
      return item;
    }

    return {
      ...item,
      function: {...item.function, description: reverseCommandDescriptions[name]}
    };
  });
  const reverseNames = ['ghidra', 'ida', 're_index', 're_search', 're_map'];
  const offeredTools = [...configuredToolSchemas, ...getPluginTools()].filter(item => !blockedReverseTools.has(item.function?.name) && (reverseTalk || !reverseNames.includes(item.function?.name)));
  const availableTools = mode === 'plan'
    ? offeredTools.filter(item => ['free', 'reverse'].includes(toolKind(item.function?.name)) || item.function?.name === 'run_command')
    : offeredTools;
  const asksForTool = /\b(ghidra|ida|decompil|disassembl|xref|read_file|list_files|search_files|grep|sandbox|terminal|run_command|web_search|web_fetch|plugin|memory|re_search|re_map)\b/i.test(String(message || ''));
  const maxToolLoops = Math.max(1, Number(settings.maxToolLoops) || 50);
  const verifyTools = ['sandbox_run', 'run_command'];
  const announcesWork = (text) => /\b(i['’ ]?will|i'?ll|let me|i'?m going to|i am going to|i need to|i have to|tengo que|debo|let me actually|i apologize.*let me|you['’]re right.*let me|first,?\s+i|next,?\s+i|voy a|vamos a|empiezo|primero (voy|creo|armo|hago)|let'?s (start|build|create|set ?up|do)|building|writing|creating|generating|implementing|coding|scaffolding|setting up|construyendo|creando|escribiendo|armando|generando|implementando)\b/i.test(String(text || ''));

  const saveCodeBlocks = async (answer, blocks) => {
    const target = buildIntent.project;
    const used = new Set();
    const saved = [];
    const failed = [];
    let text = answer;

    for(const block of blocks){
      const name = blockName(block, blocks, answer, message, used);
      const file = buildIntent.folder ? `${buildIntent.folder.replace(/[\\/]+$/, '')}/${name}` : name;
      used.add(name.toLowerCase());

      if(mode === 'plan'){
        saved.push(file);
        continue;
      }

      addStep('write_file', {project: target, file}, 'running');
      onEvent?.({type: 'tool', tool: mode === 'manual' ? `queueing ${file} for your approval` : `saving ${file} from the reply`});

      try {
        const result = mode === 'manual'
          ? await toolQueueWrite({project: target, file, content: block.code})
          : await runTool('write_file', {project: target, file, content: block.code});
        toolResults.push({tool: 'write_file', args: {project: target, file}, result});
        addStep('write_file', {project: target, file}, result?.error ? 'error' : 'done', result?.lines ? {added: result.lines.added || 0, removed: result.lines.removed || 0} : null);

        if(result?.error){
          failed.push(`${file} (${result.error})`);
          continue;
        }

        saved.push(file);
        text = text.replace(block.raw, mode === 'manual' ? `_(waiting for your approval: \`${file}\`)_` : `_(saved to \`${file}\`)_`);
      } catch(e) {
        toolResults.push({tool: 'write_file', args: {project: target, file}, result: {error: String(e.message || e)}});
        addStep('write_file', {project: target, file}, 'error');
        failed.push(`${file} (${e.message || e})`);
      }
    }

    if(mode === 'plan'){
      return `${answer}\n\nPlan mode is read-only, so nothing was written. Running this plan would create: ${saved.map(item => `\`${item}\``).join(', ')}. Press Run this plan to apply it.`;
    }

    if(saved.length && mode === 'auto'){
      onEvent?.({type: 'files', project: target});
    }

    const list = saved.map(item => `- \`${item}\``).join('\n');
    const report = !saved.length
      ? ''
      : mode === 'manual'
        ? `\n\nThese files are waiting for your approval in "${target}" (nothing is written until you approve them):\n${list}`
        : `\n\nFiles written to "${target}":\n${list}`;

    return `${text}${report}${failed.length ? `\n\nCould not write: ${failed.join('; ')}` : ''}`;
  };

  for(let i = 0; i < maxToolLoops; i++){
    if(control?.cancelled){
      return endRun('Stopped by you. ⏹', {stopped: true});
    }

    mode = live.mode;

    if(control?.steer?.length){
      const steers = control.steer.splice(0).map(item => typeof item === 'string' ? {text: item, image: ''} : item);
      // una foto mandada mientras trabaja entra como la del primer mensaje: si el motor ve imagenes va adjunta y si no se avisa
      const blind = steers.some(item => item.image) ? blindReason(await readEngine()) : '';
      const seeing = !blind;

      for(const item of steers){
        const unread = item.image && !seeing ? `\n[the user attached a picture but it could not be read: ${blind}]` : '';
        messages.push({
          role: 'user',
          content: `[NEW MESSAGE FROM THE USER, while you work - take it into account now]: ${item.text || '(a picture with no text)'}${unread}`,
          ...(item.image && seeing ? {images: [`data:image/png;base64,${item.image}`]} : {})
        });
      }

      onEvent?.({type: 'tool', tool: `steer: ${(steers[0].text || 'picture').slice(0, 50)}`});
    }

    if(directIntent && toolResults.length === 0){
      const callArgs = normalizeToolArgs(directIntent.tool, directIntent.args, projectRoot);
      const used = await useTool(directIntent.tool, callArgs, `Check that ${directIntent.tool} is listening on the configured port, or try another command.`);
      messages.push({role: 'assistant', content: ''});
      messages.push({role: 'user', content: `Tool ${directIntent.tool} result:\n${used.shown}\n\n${used.result?.error ? 'The tool failed. Fix the arguments and retry, use another tool, or tell the user plainly what failed.' : 'Now answer only from this result, or use another tool if verification is needed.'}`});
      continue;
    }

    if(toolResults.slice(-2).filter(item => item.result?.action === 'no-change').length >= 2){
      onEvent?.({type: 'tool', tool: 'edits are not changing anything, stopping'});
      return endRun('I could not fix it: my edits do not change the file (what I try to replace stays the same). Point me to the exact line, or describe the bug in more detail, so I do not repeat changes that do nothing.');
    }

    let fitted = null;

    try {
      fitted = await fitNow(availableTools);
    } catch(e) {
      if(e.name === 'AbortError' || control?.cancelled){
        return endRun('Stopped by you. ⏹', {stopped: true});
      }

      throw e;
    }

    if(fitted.hopeless && !warnedRoom){
      warnedRoom = true;
      onEvent?.({type: 'tool', tool: 'the system prompt and tools alone do not fit the context window, raise Context window in Settings'});
    }

    const roomVerdict = fitted.hopeless ? null : guard.context(estimateTokens(messages, availableTools, ratio));

    if(roomVerdict?.stop){
      stopReason = roomVerdict.stop;
      break;
    }

    let reply;
    const sentChars = countChars(messages, availableTools);
    const roundsBefore = run.usage.rounds;

    onEvent?.({type: 'tool', tool: toolResults.length ? 'thinking (processing results)...' : 'generating response...'});

    try {
      reply = await askModelChat(messages, availableTools, run);
    } catch(e) {
      if(e.name === 'AbortError' || control?.cancelled){
        return endRun('Stopped by you. ⏹', {stopped: true});
      }

      throw e;
    }

    if(run.usage.rounds > roundsBefore && run.usage.lastPromptExact && run.usage.lastPromptTokens > 0){
      ratio = calibrateRatio(ratio, sentChars, run.usage.lastPromptTokens);
    }

    if(reply.reasoning?.trim() && !thoughts.includes(reply.reasoning.trim())){
      thoughts.push(reply.reasoning.trim());
    }

    // las marcas de pensar con etiqueta de un modelo se leen como las comunes
    const replyText = String(reply.content || '').replace(/<(\/?)think:[\w-]+>/gi, '<$1think>');
    const closeAt = replyText.indexOf('</think>');
    const rawAnswer = closeAt >= 0 && !/<think>/i.test(replyText.slice(0, closeAt)) ? `<think>${replyText}` : replyText;
    const thinkMatches = [...rawAnswer.matchAll(/<think>([\s\S]*?)(?:<\/think>|$)/gi)];
    for(const m of thinkMatches){
      const t = m[1].trim();
      if(t && !thoughts.includes(t)){
        thoughts.push(t);
        onEvent?.({type: 'thought', thought: t});
      }
    }
    const answer = rawAnswer
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .replace(/<think>[\s\S]*$/gi, '')
      .trim();
    const nativeCalls = Array.isArray(reply.tool_calls) ? reply.tool_calls : [];
    const wasCut = reply.finishReason === 'length';
    const planVerdict = guard.text(answer);

    if(planVerdict?.stop){
      return endRun(`${answer}\n\n${planVerdict.note}`, {guard: planVerdict.stop});
    }

    if(reply.cut === 'repeating' && !nativeCalls.length){
      if(loopRetries < 1){
        loopRetries++;
        onEvent?.({type: 'tool', tool: 'the reply got stuck repeating itself, asking again'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: loopNudge});
        continue;
      }

      return endRun(`${answer}\n\n_(cut here: the reply kept repeating the same text)_`);
    }

    if(nativeCalls.length){
      const calls = nativeCalls.map(item => ({item, args: parseNativeArgs(item.function?.arguments)}));
      messages.push({role: 'assistant', content: answer, tool_calls: calls.map(call => call.args ? call.item : {...call.item, function: {...call.item.function, arguments: '{}'}})});

      for(const call of calls){
        const name = call.item.function?.name || '';
        const preset = control?.cancelled
          ? {error: 'Stopped by the user before this call ran.'}
          : !call.args
            ? wasCut
              ? {error: 'This call was cut at the output limit, its arguments are incomplete JSON. Nothing was run.'}
              : {error: `The arguments are not valid JSON, nothing was run: ${String(call.item.function?.arguments || '').slice(0, 300)}`, hint: 'Send the call again with valid JSON arguments.'}
            : null;
        const used = await runCall(name, call.args || {}, 'Fix the arguments or use another tool and retry.', preset);

        messages.push({role: 'tool', tool_call_id: call.item.id, content: used.shown});
      }

      if(stopReason){
        break;
      }

      const cutWrite = calls.some(call => !call.args && writesFiles(call.item.function?.name));

      if(wasCut && cutWrite && lengthRetries < 2){
        lengthRetries++;
        onEvent?.({type: 'tool', tool: 'reply cut at the output limit, continuing in pieces'});
        messages.push({role: 'user', content: lengthNudge});
      }

      continue;
    }

    if(wasCut && lengthRetries < 2 && ((answer.match(/```/g) || []).length % 2 === 1 || hasBrokenToolCall(answer))){
      lengthRetries++;
      onEvent?.({type: 'tool', tool: 'reply cut at the output limit, continuing in pieces'});
      messages.push({role: 'assistant', content: answer});
      messages.push({role: 'user', content: lengthNudge});
      continue;
    }

    const call = parseToolCall(answer);

    if(!answer.trim() && !call){
      if(emptyRetries < 2){
        emptyRetries++;
        messages.push({role: 'user', content: 'Continue: reply to the user with what you did, or call the next tool. Do not return empty.'});
        continue;
      }

      return endRun(toolResults.some(item => writeTools.includes(item.tool) && !item.result?.error && !item.result?.blocked)
        ? 'Done. I applied the changes (see the tool details below).'
        : wasCut
          ? 'The model used its whole reply thinking and never wrote an answer. Try again with a lower Effort or a thinking limit, or ask for a smaller first step.'
          : 'I did not get a response from the model. Try again with a more direct instruction.');
    }

    if(!call && hasBrokenToolCall(answer) && brokenRetries < 2){
      brokenRetries++;
      messages.push({role: 'assistant', content: answer});
      messages.push({role: 'user', content: toolJsonFixMessage});
      continue;
    }

    if(!call){
      if(asksForTool && toolResults.length === 0 && toolIntentRetries < 2){
        toolIntentRetries++;
        onEvent?.({type: 'tool', tool: 'you requested a tool; retrying with a direct call'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'The user requested a tool-backed action. Your last reply called no tool. Call the best matching tool NOW using native function calling. Do not explain, announce, or output XML.'});
        continue;
      }

      if(isRepeat(answer) && repeatRetries < 2){
        repeatRetries++;
        onEvent?.({type: 'tool', tool: 'you are repeating your previous answer, retrying'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: `You are COPYING your previous answer (same text). Do NOT repeat old messages. The user wrote: "${message}". Reply ONLY to that, short and concrete. If "${message}" is not a clear instruction, ask them to clarify. If they sent an image, describe what you see and act on it.`});
        continue;
      }

      if(isRefusal(answer) && refusalRetries < 2){
        refusalRetries++;
        onEvent?.({type: 'tool', tool: 'you CAN do it, use the tools'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'FALSE: you have FULL access. There are NO "system/project/access restrictions" - you invented that. You run on the user PC with real filesystem tools. To search the PC: pc_search (query and/or content). To READ any file, including an absolute path like C:\\Users\\sneo\\Desktop\\aqui.txt: call read_file with file set to that FULL path (read_file accepts absolute paths). To search PC content use pc_search with content. Do it NOW. Never say you cannot access/read files or to copy-paste - just call read_file with the path.'});
        continue;
      }

      const usedReverse = toolResults.some(item => ['ghidra', 'ida', 're_search', 're_map', 're_index'].includes(item.tool));
      const wroteFiles = toolResults.some(item => toolKind(item.tool) === 'file' && item.result && !item.result.error);
      const deniedChange = toolResults.some(item => item.result?.action === 'denied' || item.result?.denied);
      const codeBlocks = buildIntent && !usedReverse && !wroteFiles && !deniedChange ? codeBlocksOf(answer) : [];

      if(codeBlocks.length && mode !== 'plan' && codeRetries < 2){
        codeRetries++;
        onEvent?.({type: 'tool', tool: 'code was pasted in the chat, asking to write it to the project'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: `You put code in the chat but did not write any file. Write every one of those files into the project "${buildIntent.project}"${buildIntent.folder ? ` inside the folder "${buildIntent.folder}"` : ''} now with write_file, one call per file with its full content (folders are created automatically). Then answer with a short summary of the files you wrote. Do not paste the code in the chat again.`});
        continue;
      }

      if(codeBlocks.length){
        return endRun(await saveCodeBlocks(answer, codeBlocks));
      }

      if(mode !== 'plan' && toolResults.length === 0 && announcesWork(answer) && announceRetries < 2){
        announceRetries++;
        onEvent?.({type: 'tool', tool: 'you announced work but did nothing, act now'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'You DESCRIBED what you will do but did NOT do anything - you called zero tools. Stop announcing intentions or apologizing. START NOW: call the tool (create_folder, write_file, run_command, list_files) NOW using native function calling to actually do it. Do not reply with text until the tool has executed.'});
        continue;
      }

      const didWrite = toolResults.some(item => writeTools.includes(item.tool) && !item.result?.error);

      if(mode !== 'plan' && claimsWriteWork(answer) && !didWrite && fakeWorkRetries < 2){
        fakeWorkRetries++;
        onEvent?.({type: 'tool', tool: 'verification: you wrote nothing, retrying'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'VERIFICATION FAILED: You say you created/saved something but you did NOT run any write/folder tool (create_folder, write_file, replace_in_file, etc). DO NOT INVENT. Do it NOW for real by calling the right tool with native function calling. Do not reply to the user until you have actually executed it.'});
        continue;
      }

      const claimsRE = reverseTalk && /FUN_[0-9a-f]{5,}|\bDECOMP\b|\bXREF\b|\bopcode\b|desensambl|decompil/i.test(answer);
      const reverseTools = selectedReverseTool === 'ida' ? ['ida'] : selectedReverseTool === 'ghidra' ? ['ghidra', 're_search', 're_index'] : ['ghidra', 'ida', 're_search', 're_index'];
      const usedReTool = toolResults.some(item => reverseTools.includes(item.tool) && !item.result?.error);

      if(claimsRE && !usedReTool && reGroundRetries < 2){
        reGroundRetries++;
        const reverseName = selectedReverseTool === 'ida' ? 'IDA' : selectedReverseTool === 'ghidra' ? 'Ghidra' : 'Ghidra or IDA';
        onEvent?.({type: 'tool', tool: `anti-hallucination: you did not investigate with ${reverseName}, retrying`});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: `STOP: you are stating things about functions, addresses or packets WITHOUT investigating with ${reverseName} in this response. DO NOT INVENT. Use the selected reverse-engineering tool now, then state only what its output confirms.`});
        continue;
      }

      const lastBuild = [...toolResults].reverse().find(item => ['sandbox_run', 'run_command'].includes(item.tool));
      const buildFailed = lastBuild && /\berror\b|failed|could not resolve|cannot find|no such file|is not recognized|missing script|not found|exit code [1-9]/i.test(JSON.stringify(lastBuild.result || ''));
      const claimsSuccess = /funciona|funcional|\blisto\b|compila( sin| correct)|complet(ado|amente)|exitos|correctamente|todo (bien|ok)/i.test(answer)
        || /\b(works|working|builds? (fine|successfully|without)|compiles?|compiled|is (ready|done|complete)|successfully|all set|fully functional)\b/i.test(answer);

      if(mode !== 'plan' && buildFailed && claimsSuccess && buildFailRetries < 3){
        buildFailRetries++;
        onEvent?.({type: 'tool', tool: 'build failed, not letting it claim success'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'STOP: the last build/command FAILED (check its output: there is an error). Do NOT say it works or is done. Fix it for real NOW: if deps are missing install them with "npm.cmd install <packages>", fix whatever breaks, and run "npm.cmd run build" again until it passes WITHOUT error. Only then reply, and only with what the clean build confirms.'});
        continue;
      }

      // solo cuenta como pagina lo que se dibuja: un .js de logica (cart.js) mandaba a mirar un diseno que no existe
      const webFileEdited = toolResults.some(item => writeTools.includes(item.tool) && !item.result?.error && /\.(html?|css|scss|jsx|tsx|vue|svelte)\b/i.test(JSON.stringify(item.args || '')));
      const sawDesign = toolResults.some(item => item.tool === 'visual_check' && !item.result?.error);
      // palabras enteras: UI sin limites encontraba "incluido" o "construido" en casi toda respuesta en espanol
      const claimsDesign = /\b(dise[nñ]o|se ve|visual|moderno|atractiv\w*|estilo|interfaz|look)\b/i.test(answer) || /\bUI\b/.test(answer);

      if(webFileEdited && (claimsSuccess || claimsDesign) && !sawDesign && webVerifyRetries < 1){
        webVerifyRetries++;
        onEvent?.({type: 'tool', tool: 'you did not look at the design, use visual_check'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'You edited a web page but did NOT look at it. You have the visual_check tool: use it NOW on the main file (e.g. index.html) to SEE how it really looks (screenshot + errors + critique). If it looks bland, empty, with unreadable text (low contrast) or broken, IMPROVE it and look again. Do not say the design is good without having seen it.'});
        continue;
      }

      const writeIndex = toolResults.findIndex(item => writeTools.includes(item.tool) && !item.result?.error);
      const verifiedAfterWrite = writeIndex >= 0 && toolResults.slice(writeIndex + 1).some(item => verifyTools.includes(item.tool));
      const pkgPath = projectRoot ? path.join(projectRoot, 'package.json') : '';
      const hasBuild = !!pkgPath && existsSync(pkgPath) && (() => { try { return !!JSON.parse(readFileSync(pkgPath, 'utf8')).scripts?.build; } catch { return false; } })();
      const wroteCode = toolResults.some(item => writeTools.includes(item.tool) && !item.result?.error && /\.(jsx?|tsx?|mjs|cjs|vue|svelte|py|cs|go|rs|java)\b/i.test(JSON.stringify(item.args || '')));

      if(mode === 'auto' && didWrite && wroteCode && hasBuild && !verifiedAfterWrite && verifyRetries < 2){
        verifyRetries++;
        onEvent?.({type: 'tool', tool: 'auto-verification: reviewing the change'});
        messages.push({role: 'assistant', content: answer});
        messages.push({role: 'user', content: 'You wrote code but did NOT run it. Reading the file is NOT verification. RUN it now: sandbox_run with "npm.cmd run build" (for a JS/web project) or the right test/run command, and READ the real output. If it errors, fix it and run again. Do not say it works until a build/run passes cleanly. Only then reply.'});
        continue;
      }

      return endRun(answer);
    }

    const used = await runCall(call.tool, call.args);
    messages.push({role: 'assistant', content: answer});
    messages.push({role: 'user', content: `Tool ${call.tool} result:\n${used.shown}\n\nNow reply to the user or use another tool if needed.`});

    if(stopReason){
      break;
    }
  }

  const stopNote = stopReason ? guard.stop(stopReason).note : '';

  if(control?.cancelled){
    return endRun('Stopped by you. ⏹', {stopped: true});
  }

  if(toolResults.length && stopReason !== 'context'){
    onEvent?.({type: 'tool', tool: stopReason ? `stopped (${stopReason}), writing what was found` : 'final synthesis: concluding with what was investigated'});
    messages.push({role: 'user', content: stopReason
      ? `STOP using tools now: ${stopNote} With what you already found, give the user your best concrete answer: what you found, in which files or functions, and what is still unconfirmed. Do not call tools.`
      : 'You can no longer use tools (step budget is used up). With EVERYTHING you investigated so far, give the user your BEST concrete answer/conclusion: what you found, in which files/functions, and your hypothesis. If something is unconfirmed, say it explicitly. Do NOT ask for a more concrete instruction, do NOT call tools: reply with what you have.'});

    let finalAnswer = '';

    try {
      await fitNow(availableTools);
      const reply = await askModelChat(messages, availableTools, run);
      finalAnswer = String(reply.content || '')
        .replace(/<think>[\s\S]*?<\/think>/gi, '')
        .replace(/^[\s\S]*?<\/think>/i, '')
        .trim();

      if(reply.reasoning?.trim()){
        thoughts.push(reply.reasoning.trim());
      }

      if(!finalAnswer){
        finalAnswer = (await askModel(messages, run)).trim();
      }
    } catch(e) {
      if(e.name === 'AbortError' || control?.cancelled){
        return endRun('Stopped by you. ⏹', {stopped: true});
      }

      throw e;
    }

    if(finalAnswer || stopNote){
      return endRun(`${finalAnswer}${stopNote ? `${finalAnswer ? '\n\n' : ''}${stopNote}` : ''}`, {guard: stopReason});
    }
  }

  if(stopNote){
    return endRun(stopNote, {guard: stopReason});
  }

  const errorTools = toolResults.filter(item => item.result?.error);

  return endRun(errorTools.length
    ? `I ran out of attempts and several tools failed. Last error: ${errorTools[errorTools.length - 1].result.error}. Rephrase or give me the correct path/project.`
    : 'I did many steps without closing the task. Give me a more concrete goal (which file, which project, what to search exactly).');
};
