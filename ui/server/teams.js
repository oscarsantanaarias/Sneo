import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { askClaude, claudeModels } from './claude.js';
import { askClaudeCode, claudeCodeModels } from './claudecode.js';
import { askAntigravity } from './antigravity.js';
import { paths } from './config.js';
import { askPlain } from './model.js';
import { cleanText } from './storage.js';
import { getProjectPath } from './projects.js';

const routesFile = () => path.join(paths.dataDir, 'agent-routes.json');
const agentsFile = () => path.join(paths.dataDir, 'agent-chat.json');

// una ruta es un proveedor con su modelo: el motor de esta pc, un segundo motor local, claude o cualquier api de chat estilo openai
export const routePresets = [
  {kind: 'local', name: 'Sneo (this PC)', baseUrl: 'http://127.0.0.1:8000/v1', model: '', parallel: 1},
  {kind: 'local', name: 'Sneo second engine', baseUrl: 'http://127.0.0.1:8001/v1', model: '', parallel: 1},
  {kind: 'claude', name: 'Claude', baseUrl: '', model: 'claude-opus-5-5', parallel: 4},
  {kind: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: '', parallel: 4},
  {kind: 'openai', name: 'DeepSeek', baseUrl: 'https://api.deepseek.com/v1', model: '', parallel: 4},
  {kind: 'openai', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', model: '', parallel: 4},
  {kind: 'claudecode', name: 'Claude Code (your subscription)', baseUrl: '', model: '', parallel: 1},
  {kind: 'openai', name: 'Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-flash-latest', parallel: 4},
  {kind: 'agy', name: 'Antigravity (your Google account)', baseUrl: '', model: '', parallel: 1}
];

const cleanRoute = (item, kept = {}) => {
  const kind = ['local', 'claude', 'claudecode', 'agy', 'openai'].includes(item?.kind) ? item.kind : 'openai';
  const key = item?.apiKey === '***' ? kept.apiKey || '' : cleanText(item?.apiKey);
  const account = kind === 'claudecode' || kind === 'agy';

  return {
    id: cleanText(item?.id) || crypto.randomUUID(),
    kind,
    name: cleanText(item?.name).slice(0, 60) || 'Route',
    baseUrl: kind === 'claude' || account ? '' : cleanText(item?.baseUrl).replace(/\/+$/, ''),
    apiKey: kind === 'local' || account ? '' : key,
    model: cleanText(item?.model).slice(0, 120),
    effort: ['low', 'medium', 'high', 'xhigh', 'max'].includes(item?.effort) ? item.effort : 'medium',
    parallel: Math.min(16, Math.max(1, parseInt(item?.parallel, 10) || (kind === 'local' ? 1 : 4)))
  };
};

export const readRoutes = async () => {
  let saved = [];

  try {
    const data = JSON.parse(await fs.readFile(routesFile(), 'utf8'));
    saved = Array.isArray(data.routes) ? data.routes : [];
  } catch(e) {
    saved = [];
  }

  if(saved.length){
    return saved.map(item => cleanRoute(item, item));
  }

  return [cleanRoute({...routePresets[0], id: 'local-main'})];
};

export const maskRoutes = (routes) => routes.map(item => ({...item, apiKey: item.apiKey ? '***' : ''}));

export const saveRoutes = async (list) => {
  const current = await readRoutes();
  const routes = (Array.isArray(list) ? list : []).slice(0, 24).map(item => cleanRoute(item, current.find(old => old.id === item?.id) || {}));
  await fs.mkdir(paths.dataDir, {recursive: true});
  await fs.writeFile(routesFile(), JSON.stringify({routes}, null, 2), 'utf8');
  return routes;
};

export const routeLabel = (route) => route.kind === 'claudecode'
  ? `${route.name} · ${claudeCodeModels.find(item => item.id === route.model)?.name || route.model || 'Default'}`
  : route.kind === 'claude'
  ? `${route.name} · ${claudeModels.find(item => item.id === route.model)?.name || route.model}`
  : route.model ? `${route.name} · ${route.model}` : route.name;

// en esta pc el modelo elegido en la ruta manda: si es el cargado contesta y si es otro se cambia antes
export const ensureLocalModel = async (route, signal) => {
  if(route.kind !== 'local' || !thisPc.test(route.baseUrl || '') || !route.model){
    return;
  }

  const base = route.baseUrl.replace(/\/v1$/, '');
  const state = await fetch(`${base}/api/models`, {signal: AbortSignal.timeout(5000)}).then(answer => answer.json());
  const wanted = (state.models || []).find(item => item.name === route.model || item.path === route.model);

  if(!wanted){
    throw new Error(`${route.model} is not in the models folder of this PC`);
  }

  if(state.currentModel === wanted.path && (state.ready || state.warming)){
    if(state.ready) return;
  } else {
    const answer = await fetch(`${base}/api/models/load`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path: wanted.path})}).then(item => item.json());
    if(answer.success === false) throw new Error(`${wanted.name} did not load`);
  }

  const until = Date.now() + 15 * 60 * 1000;
  while(Date.now() < until){
    if(signal?.aborted) throw new Error('Stopped');
    await new Promise(resolve => setTimeout(resolve, 1000));
    const now = await fetch(`${base}/api/models`, {signal: AbortSignal.timeout(5000)}).then(answer => answer.json()).catch(() => ({}));
    if(now.problem) throw new Error(`${wanted.name} did not load: ${now.problem}`);
    if(now.currentModel === wanted.path && now.ready) return;
    if(now.currentModel && now.currentModel !== wanted.path && !now.warming) throw new Error(`another model was loaded while waiting for ${wanted.name}`);
  }
  throw new Error(`${wanted.name} took too long to load`);
};

export const askRoute = async (route, messages, tools, run) => {
  await ensureLocalModel(route, run?.signal);
  if(route.kind === 'claude'){
    return await askClaude(route, messages, tools, run);
  }

  if(route.kind === 'claudecode'){
    return await askClaudeCode(route, messages, tools, run);
  }

  if(route.kind === 'agy'){
    return await askAntigravity(route, messages, tools, run);
  }

  return await askPlain(route, messages, tools, run);
};

// cuantos agentes corren a la vez en cada ruta: las apis aguantan varios y los motores de esta pc se turnan,
// medido con dos modelos cargados: a la vez escriben 25 y 25 tok/s, por turnos 74 y 67
// en tarjetas distintas es al reves: IQ3_XXS en la 3090 y IQ3_S en la 1070 a la vez escriben 61 y 16, mas que de a uno
const busy = new Map();
const thisPc = /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/i;
let localTurns = true;

export const setLocalTurns = (value) => {
  localTurns = value !== false;
};

export const slotOf = (route) => route.kind === 'local' && thisPc.test(route.baseUrl || '')
  ? localTurns ? {key: 'this-pc', most: 1} : {key: route.id, most: 1}
  : {key: route.id, most: route.parallel};

const withSlot = async (route, job) => {
  const {key, most} = slotOf(route);
  const slot = busy.get(key) || {running: 0, waiting: []};
  busy.set(key, slot);

  // el lugar pasa directo al que esperaba, asi nadie se cuela entre medio
  if(slot.running >= most){
    await new Promise(resolve => slot.waiting.push(resolve));
  } else {
    slot.running++;
  }

  try {
    return await job();
  } finally {
    const next = slot.waiting.shift();

    if(next){
      next();
    } else {
      slot.running--;
    }
  }
};

const MODE_PLAN = {
  audit: 'Split the review of this project into independent inspection subtasks (bugs, security, performance, tests, dead code, structure). Each subtask = one focused area.',
  build: 'Split building this feature into independent pieces (each piece = one file or one clear component). Each subtask investigates the current code and DESIGNS its piece with concrete code, but does NOT write files.',
  reverse: 'Split this reverse-engineering task into independent targets (each = one function, packet or address to investigate with ghidra or ida and follow to its root).',
  research: 'Split this research into independent angles (each subtask covers one angle: code, decompilation, web, data) so together they answer the question.',
  auto: 'Split this task into independent subtasks that different agents can do in parallel without depending on each other.'
};

const parsePlan = (text, most) => {
  const match = String(text || '').match(/\[[\s\S]*\]/);

  if(!match){
    return [];
  }

  try {
    const list = JSON.parse(match[0]);
    return Array.isArray(list)
      ? list.filter(item => item && (item.goal || item.role)).map((item, i) => ({role: cleanText(item.role) || `Agent ${i + 1}`, goal: cleanText(item.goal) || cleanText(item.role)})).slice(0, most)
      : [];
  } catch(e) {
    return [];
  }
};

// los agentes de la vista de conversaciones: cada uno con su papel y la ruta del modelo que le contesta,
// tantos como se quiera (el tope solo evita que un archivo roto llene la pantalla)
export const mostAgents = 100;

export const defaultAgents = [
  {id: 'orchestrator', name: 'Orchestrator', focus: 'Task manager · Main agent', route: '', tools: false},
  {id: 'coder', name: 'Coder', focus: 'Code generation · Developer', route: '', tools: true, edits: true},
  {id: 'researcher', name: 'Researcher', focus: 'Research · Information retrieval', route: '', tools: true},
  {id: 'reviewer', name: 'Reviewer', focus: 'Quality control · Testing', route: '', tools: true}
];

export const cleanAgent = (item, i = 0) => ({
  id: cleanText(item?.id).slice(0, 60) || crypto.randomUUID(),
  name: cleanText(item?.name).slice(0, 40) || `Agent ${i + 1}`,
  focus: cleanText(item?.focus).slice(0, 80),
  route: cleanText(item?.route).slice(0, 120),
  tools: item?.tools === true || item?.edits === true || (item?.edits === undefined && cleanText(item?.id) === 'coder'),
  // el coder escribe codigo de entrada y los demas solo si se les prende
  edits: item?.edits === true || (item?.edits === undefined && cleanText(item?.id) === 'coder')
});

export const readAgents = async () => {
  try {
    const data = JSON.parse(await fs.readFile(agentsFile(), 'utf8'));

    // una lista guardada vale aunque este vacia: los agentes por defecto solo aparecen la primera vez
    if(Array.isArray(data.agents)){
      return data.agents.slice(0, mostAgents).map(cleanAgent);
    }
  } catch(e) {
  }

  return defaultAgents.map(cleanAgent);
};

export const saveAgents = async (list) => {
  const agents = (Array.isArray(list) ? list : []).slice(0, mostAgents).map(cleanAgent);
  await fs.mkdir(paths.dataDir, {recursive: true});
  await fs.writeFile(agentsFile(), JSON.stringify({agents}, null, 2), 'utf8');
  return agents;
};

const dropThinking = (text) => String(text || '')
  .replace(/<think>[\s\S]*?<\/think>/gi, '')
  .replace(/<think>[\s\S]*$/gi, '')
  .trim();

// una vuelta de chat con un agente: sin proyecto contesta directo y con proyecto puede leer el codigo
export const chatAgent = async ({agent, route, history, project, peers = [], rounds = 100, run, onStep, runWorker}) => {
  const clean = cleanAgent(agent);
  const handoff = peers.length
    ? ` The other agents are ${peers.join(', ')}. To give one of them work, write a line that starts with @Name: and the message, for example "@${peers[0]}: check the login code and tell me what breaks". Only those lines reach them, so never say you sent, asked or dispatched something without writing that line. Their answer comes back to you as a message that starts with "Report from <name>:". Messages that start with "From <name>:" were sent to you by that agent.`
    : '';
  const where = project ? await getProjectPath(project).catch(() => '') : '';
  const place = project ? ` The team works on the project "${project}"${where ? ` in ${where}` : ''}.` : '';
  const persona = `You are ${clean.name}${clean.focus ? ` (${clean.focus})` : ''}, one agent of a team that works with the user.${place}${handoff} Answer as ${clean.name}: concrete, short when the question is short, in the language of the user.`;
  const past = (Array.isArray(history) ? history : [])
    .filter(item => item && String(item.content || '').trim())
    .slice(-30)
    .map(item => ({role: item.role === 'assistant' ? 'assistant' : 'user', content: String(item.content).slice(0, 16000)}));

  if(!past.length || past[past.length - 1].role !== 'user'){
    throw new Error('Write a message for the agent first');
  }

  // cada panel sigue su propia conversacion con claude code o antigravity y en cada turno solo manda lo nuevo
  run = run ? {...run, sessionKey: `${clean.id}|${project || ''}|${route.id}|${route.model || ''}`} : run;

  if(clean.tools && project && runWorker){
    const last = past.pop();
    const writer = clean.edits
      ? ' You are the one who changes the code: apply fixes yourself in the project files. Use replace_in_file with the exact text to change for edits, write_file only for new files, and run the code with sandbox_run or run_command to check it works. Never ask another agent to apply a change you can make, and say which files and lines you changed.'
      : ' You can only read the project. When something must change, describe the exact change and hand it to the agent that edits code. To review changes start with git_diff and read a whole file only when the diff does not show enough.';
    const result = await withSlot(route, () => runWorker({name: clean.name, role: clean.focus || clean.name, task: last.content, project, route, run, onStep, history: past, persona: persona + writer, rounds, canEdit: clean.edits}));
    return {content: dropThinking(result.answer), tools: (result.tools || []).map(item => item.tool), thinking: ''};
  }

  const reply = await withSlot(route, () => askRoute(route, [{role: 'system', content: persona}, ...past], null, run));
  const thinking = String(reply.reasoning || (String(reply.content || '').match(/<think>([\s\S]*?)(<\/think>|$)/i) || [])[1] || '').trim();
  return {content: dropThinking(reply.content), tools: [], thinking: thinking.slice(-4000)};
};

export const runTeam = async ({task, project, mode, planner, workers, lead, maxAgents, onEvent, signal, runWorker}) => {
  const routes = await readRoutes();
  const find = id => routes.find(item => item.id === id);
  const plannerRoute = find(planner) || routes[0];
  const leadRoute = find(lead) || plannerRoute;
  const workerRoutes = (Array.isArray(workers) ? workers : []).map(find).filter(Boolean);
  const kind = MODE_PLAN[mode] ? mode : 'auto';
  const most = Math.min(Math.max(2, parseInt(maxAgents, 10) || 4), 32);
  const cleanTask = cleanText(task);
  const run = {signal, usage: null};
  const say = event => onEvent?.(event);

  if(!cleanTask){
    throw new Error('Write the task for the team first');
  }

  if(!workerRoutes.length){
    workerRoutes.push(plannerRoute);
  }

  say({type: 'team-start', planner: {id: 'planner', route: routeLabel(plannerRoute), kind: plannerRoute.kind}, lead: {id: 'lead', route: routeLabel(leadRoute), kind: leadRoute.kind}});
  say({type: 'team-agent', id: 'planner', status: 'thinking'});

  const planReply = await withSlot(plannerRoute, () => askRoute(plannerRoute, [
    {role: 'system', content: `You are the planner of a multi-agent team. ${MODE_PLAN[kind]}\nProject: ${cleanText(project) || 'none'}.\nReturn ONLY a JSON array (at most ${most} items) of {"role": short name, "goal": one concrete instruction for that agent}. No prose, just the JSON array.`},
    {role: 'user', content: cleanTask}
  ], null, run));

  let plan = parsePlan(planReply.content, most);

  if(!plan.length){
    plan = [
      {role: 'Explorer', goal: `Map the structure and key files relevant to: ${cleanTask}`},
      {role: 'Analyzer', goal: `Assess the implementation, the risks and what is missing for: ${cleanTask}`}
    ];
  }

  const agents = plan.map((item, i) => ({...item, id: `w${i + 1}`, route: workerRoutes[i % workerRoutes.length]}));
  say({type: 'team-agent', id: 'planner', status: 'done'});
  say({type: 'team-plan', agents: agents.map(item => ({id: item.id, role: item.role, goal: item.goal, route: routeLabel(item.route), kind: item.route.kind}))});

  for(const item of agents){
    say({type: 'team-message', from: 'planner', to: item.id, text: item.goal});
  }

  const results = await Promise.all(agents.map(item => withSlot(item.route, async () => {
    say({type: 'team-agent', id: item.id, status: 'working'});
    const startedAt = Date.now();

    try {
      const result = await runWorker({
        name: item.role,
        role: item.role,
        task: `${item.goal}\n\n(Overall goal: ${cleanTask}). You are READ-ONLY: investigate and report concrete findings; if this is a build task, include the exact code you would write. Do NOT write files.`,
        project: cleanText(project),
        route: item.route,
        run,
        onStep: step => say({type: 'team-agent', id: item.id, status: 'tool', tool: step.tool})
      });
      say({type: 'team-agent', id: item.id, status: 'done', ms: Date.now() - startedAt});
      say({type: 'team-message', from: item.id, to: 'lead', text: String(result.answer || '').slice(0, 800)});
      return {...result, id: item.id, route: routeLabel(item.route)};
    } catch(e) {
      const message = e.name === 'AbortError' ? 'stopped' : String(e.message || e);
      say({type: 'team-agent', id: item.id, status: 'error', error: message});
      return {id: item.id, name: item.role, role: item.role, answer: `error: ${message}`, tools: [], route: routeLabel(item.route)};
    }
  })));

  say({type: 'team-agent', id: 'lead', status: 'thinking'});
  const findings = results.map((item, i) => `### Agent ${i + 1} - ${item.role} (${item.route})\n${String(item.answer || '').slice(0, 4000)}`).join('\n\n');
  const final = await withSlot(leadRoute, () => askRoute(leadRoute, [
    {role: 'system', content: 'You are the lead of a multi-agent team. Merge the findings of the agents into ONE clear, concrete answer for the user, in the language of the task. Remove duplicates, resolve contradictions, give the conclusion and the next steps. If it was a build task, present the assembled code so it can be applied.'},
    {role: 'user', content: `TASK: ${cleanTask}\n\nAGENT FINDINGS:\n${findings}`}
  ], null, run));
  say({type: 'team-agent', id: 'lead', status: 'done'});
  say({type: 'team-final', answer: final.content || ''});

  return {mode: kind, agents: results.map(item => ({id: item.id, role: item.role, route: item.route, answer: item.answer})), answer: final.content || ''};
};
