import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { tr } from './i18n.js';
import { colorOf, readEvents, short, useLoaded } from './agentsShared.js';
import AgentOffice from './AgentOffice.jsx';

const storeKey = 'sneo.agentChats';
const mostAgents = 100;

const loadChats = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(storeKey) || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch(e) {
    return {};
  }
};

const keepChats = (chats) => {
  try {
    localStorage.setItem(storeKey, JSON.stringify(chats));
  } catch(e) {
  }
};

const html = (text) => ({__html: DOMPurify.sanitize(marked.parse(String(text || '')))});

const number = (i) => String(i + 1).padStart(2, '0');

// los encargos que un agente le deja a otro con una linea @Nombre: mensaje
const mostHops = 8;
const parseHandoffs = (text, agents, selfId) => {
  const found = [];
  let open = null;

  for(const line of String(text || '').split('\n')){
    const match = line.match(/^\s*(?:[-*]\s*)?(?:\*\*)?@([^:\n*]{1,40}?)(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.*)$/);
    const target = match && agents.find(item => item.id !== selfId && item.name.toLowerCase() === match[1].trim().toLowerCase());

    if(target){
      open = {id: target.id, name: target.name, text: match[2]};
      found.push(open);
      continue;
    }

    if(match){
      open = null;
      continue;
    }

    if(open && line.trim()){
      open.text += '\n' + line;
    } else {
      open = null;
    }
  }

  return found.filter(item => item.text.trim());
};

const count = (n) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n || 0);

function Speed({usage}){
  if(!usage || (!usage.outputSpeed && !usage.outputTokens)){
    return null;
  }

  const parts = [
    usage.outputSpeed ? `${usage.outputSpeed} tok/s` : '',
    usage.promptSpeed ? `${tr('prompt')} ${usage.promptSpeed} tok/s` : '',
    usage.outputTokens ? `${count(usage.outputTokens)} ${tr('out')}` : ''
  ].filter(Boolean);

  return <span className="chat-speed" title={tr('{0} in, {1} out, {2} rounds', count(usage.inputTokens), count(usage.outputTokens), usage.rounds || 0)}><i className="fa-solid fa-gauge-high"></i> {parts.join(' · ')}</span>;
}

const stripFrom = (text) => String(text || '').replace(/^(From|Report from) [^:\n]+:\n/, '');

function Trace({items, thinking, live}){
  const list = items || [];

  if(!list.length && !thinking){
    return null;
  }

  const rows = (
    <>
      {list.map((row, i) => row.kind === 'thought'
        ? <div key={i} className="chat-trace-thought">{live && i === list.length - 1 ? row.text.slice(-600) : short(row.text, 400)}</div>
        : <div key={i} className={`chat-trace-row chat-trace-${row.kind}`}><i className={row.kind === 'handoff' ? 'fa-solid fa-share' : 'fa-solid fa-wrench'}></i> {row.text}</div>)}
      {thinking && <div className="chat-trace-thought">{thinking}</div>}
    </>
  );

  if(live){
    return <div className="chat-trace chat-trace-live">{rows}</div>;
  }

  return (
    <details className="chat-trace">
      <summary>{tr('What it did ({0})', list.length + (thinking ? 1 : 0))}</summary>
      {rows}
    </details>
  );
}

function AgentCard({agent, index, routes, ready, labelOf, messages, step, live, speed, asks = [], onAnswer, draft, others, editing, onDraft, onSend, onStop, onForward, onEdit, onChange, onRemove, onClear}){
  const listRef = useRef(null);
  const route = routes.find(item => item.id === agent.route);
  const color = colorOf(route || {});
  const last = messages[messages.length - 1];
  const second = route?.kind === 'local' && /:8001(\/|$)/.test(route.baseUrl || '');
  const state = !route ? 'off' : step ? 'busy' : !ready ? 'wait' : last?.role === 'error' ? 'bad' : 'ok';
  const stateText = state === 'off' ? tr('No model') : state === 'busy' ? step : state === 'wait' ? (second ? tr('Second model off') : tr('No model loaded')) : state === 'bad' ? tr('Failed') : tr('Ready');

  useEffect(() => {
    listRef.current?.scrollTo({top: listRef.current.scrollHeight, behavior: 'smooth'});
  }, [messages.length, step, live?.length]);

  return (
    <section className={step ? 'card chat-card chat-busy' : 'card chat-card'} style={{'--agent': color}}>
      <div className="chat-head">
        <div className="chat-title">
          <strong>{number(index)} · {agent.name}</strong>
          <span className={`chat-state chat-state-${state}`}>{state === 'busy' && speed?.outputSpeed ? <span className="chat-live-speed">{speed.outputSpeed} tok/s</span> : null}<span className="chat-dot"></span>{stateText}</span>
        </div>
        <div className="chat-sub">
          <span className="chat-focus">{agent.focus || tr('Agent')}</span>
          <select className="chat-route" value={agent.route} onChange={(e) => onChange({route: e.target.value})} aria-label={tr('Model this agent answers with')}>
            {!route && <option value="">{tr('Pick a model...')}</option>}
            {routes.map(item => <option key={item.id} value={item.id}>{labelOf(item)}</option>)}
          </select>
          <button type="button" className="icon-btn" title={tr('Clear this conversation: the agent forgets everything said here')} disabled={!messages.length && !step} onClick={onClear}><i className="fa-solid fa-broom"></i></button>
          <button type="button" className={editing ? 'icon-btn active' : 'icon-btn'} title={tr('Edit this agent')} onClick={onEdit}><i className="fa-solid fa-sliders"></i></button>
          <button type="button" className="icon-btn icon-danger" title={tr('Remove this agent')} onClick={onRemove}><i className="fa-solid fa-trash"></i></button>
        </div>
      </div>

      <div className="chat-body">
      {editing && (
        <div className="chat-edit">
          <div className="chat-edit-row">
            <input value={agent.name} onChange={(e) => onChange({name: e.target.value})} placeholder={tr('Name')} aria-label={tr('Name')} />
            <input value={agent.focus} onChange={(e) => onChange({focus: e.target.value})} placeholder={tr('What it does, for example Code generation · Developer')} aria-label={tr('What it does')} />
          </div>
          <div className="chat-edit-row">
            <label className="chat-check">
              <input type="checkbox" checked={agent.tools} onChange={(e) => onChange(e.target.checked ? {tools: true} : {tools: false, edits: false})} />
              {tr('Can read the project (files, search, ghidra and ida)')}
            </label>
            <label className="chat-check">
              <input type="checkbox" checked={agent.edits === true || (agent.edits === undefined && agent.id === 'coder')} onChange={(e) => onChange(e.target.checked ? {tools: true, edits: true} : {edits: false})} />
              {tr('Can change the code (edits files and runs commands)')}
            </label>
            <button type="button" className="agents-small agents-danger" onClick={onRemove}><i className="fa-solid fa-trash"></i> {tr('Remove agent')}</button>
            <button type="button" className="agents-small" onClick={onEdit}>{tr('Done')}</button>
          </div>
        </div>
      )}

      <div className="chat-list" ref={listRef}>
        {!messages.length && (
          <div className="chat-bubble chat-intro">
            {!route
              ? tr('Pick the model this agent answers with.')
              : ready
                ? tr('{0} answers with {1}. Write below or use the command center.', agent.name, labelOf(route))
                : second
                  ? tr('{0} answers with the second model on this PC, which is off. Start it in Agents, Models.', agent.name)
                  : tr('{0} answers with the model loaded in Sneo on this PC, and none is loaded. Load one in Engine.', agent.name)}
          </div>
        )}

        {messages.map((item, i) => {
          if(item.role === 'user'){
            const from = item.from ? others.find(other => other.id === item.from) : null;
            return (
              <div key={i} className={from ? 'chat-bubble chat-forward' : 'chat-bubble chat-user'} style={from ? {'--from': colorOf(routes.find(one => one.id === from.route) || {})} : undefined}>
                {from && <span className="chat-from">{item.kind === 'report' ? tr('Report from {0}', from.name) : tr('From {0}', from.name)}</span>}
                {from ? stripFrom(item.content) : item.content}
              </div>
            );
          }

          if(item.role === 'error'){
            return <div key={i} className="chat-bubble chat-error"><i className="fa-solid fa-circle-exclamation"></i> {item.content}</div>;
          }

          return (
            <div key={i} className="chat-bubble chat-agent">
              <Trace items={item.activity} thinking={item.thinking} />
              <div className="markdown-body" dangerouslySetInnerHTML={html(item.content)} />
              {item.handoffs?.length ? (
                <div className="chat-handoffs">
                  {item.handoffs.map((name, k) => <span key={k} className="chat-handoff"><i className="fa-solid fa-share"></i> {tr('Sent to {0}', name)}</span>)}
                </div>
              ) : null}
              <div className="chat-meta">
                {item.ms ? <span>{(item.ms / 1000).toFixed(1)} s</span> : null}
                <Speed usage={item.usage} />
                {item.tools?.length ? <span title={item.tools.join(', ')}><i className="fa-solid fa-wrench"></i> {item.tools.length}</span> : null}
                <span className="rail-spacer"></span>
                {others.length > 0 && (
                  <select className="chat-pass" value="" onChange={(e) => { if(e.target.value) onForward(e.target.value, item.content); }} aria-label={tr('Send this answer to another agent')}>
                    <option value="">{tr('Send to...')}</option>
                    {others.map(other => <option key={other.id} value={other.id}>{other.name}</option>)}
                  </select>
                )}
                <button type="button" className="icon-btn" title={tr('Copy')} onClick={() => navigator.clipboard?.writeText(item.content)}><i className="fa-regular fa-copy"></i></button>
              </div>
            </div>
          );
        })}

        {asks.map(ask => (
          <div key={ask.id} className="chat-bubble chat-ask">
            <span className="chat-ask-title"><i className="fa-solid fa-hand"></i> {tr('Waiting for your approval')}: {ask.title}</span>
            {ask.diff && <details><summary>{tr('See the change')}</summary><pre className="chat-ask-diff">{ask.diff}</pre></details>}
            <div className="chat-ask-row">
              <button type="button" className="agents-small" onClick={() => onAnswer(ask.id, 'approve')}><i className="fa-solid fa-check"></i> {tr('Approve')}</button>
              <button type="button" className="agents-small agents-danger" onClick={() => onAnswer(ask.id, 'deny')}><i className="fa-solid fa-xmark"></i> {tr('Deny')}</button>
            </div>
          </div>
        ))}
        {step && live?.length ? <Trace items={live} live /> : null}
        {step && <div className="chat-bubble chat-typing"><span></span><span></span><span></span></div>}
      </div>
      </div>

      <div className="chat-input">
        <input
          value={draft}
          placeholder={ready ? tr('Message {0}...', agent.name) : tr('Load its model first')}
          onChange={(e) => onDraft(e.target.value)}
          onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); onSend(); } }}
          disabled={!route || !ready}
        />
        {step
          ? <button type="button" className="chat-send" title={tr('Stop')} onClick={onStop}><i className="fa-solid fa-stop"></i></button>
          : <button type="button" className="chat-send" title={tr('Send')} disabled={!route || !ready || !draft.trim()} onClick={onSend}><i className="fa-solid fa-arrow-up"></i></button>}
      </div>
    </section>
  );
}

export default function AgentChats({apiUrl, routes, projects = [], project = '', saveFirst}){
  const [agents, setAgents] = useState([]);
  const [chats, setChats] = useState(loadChats);
  const [steps, setSteps] = useState({});
  const [live, setLive] = useState({});
  const [speeds, setSpeeds] = useState({});
  const [visits, setVisits] = useState([]);
  const [view, setView] = useState(() => {
    try {
      return localStorage.getItem('sneo.agentView') === 'office' ? 'office' : 'panels';
    } catch(e) {
      return 'panels';
    }
  });

  const pickView = (next) => {
    setView(next);
    try {
      localStorage.setItem('sneo.agentView', next);
    } catch(e) {}
  };
  const liveRef = useRef({});
  const waiting = useRef({});
  const [drafts, setDrafts] = useState({});
  const [editing, setEditing] = useState('');
  const [target, setTarget] = useState('all');
  const [command, setCommand] = useState('');
  const [chatProject, setChatProject] = useState(project);
  // cuantas vueltas de herramientas tiene cada agente antes de tener que contestar
  const [asks, setAsks] = useState({});
  const [agentMode, setAgentMode] = useState(() => {
    try {
      return ['auto', 'manual', 'plan'].includes(localStorage.getItem('sneo.agentMode')) ? localStorage.getItem('sneo.agentMode') : 'manual';
    } catch(e) {
      return 'manual';
    }
  });

  const pickMode = (next) => {
    setAgentMode(next);
    try {
      localStorage.setItem('sneo.agentMode', next);
    } catch(e) {}
  };

  const answerAsk = async (agentId, askId, answer) => {
    setAsks(prev => ({...prev, [agentId]: (prev[agentId] || []).filter(item => item.id !== askId)}));
    try {
      await fetch(`${apiUrl}/api/pending/${askId}/${answer}`, {method: 'POST'});
    } catch(e) {}
  };

  const [rounds, setRounds] = useState(() => {
    try {
      return Math.min(1000, Math.max(1, Number(localStorage.getItem('sneo.agentRounds')) || 100));
    } catch(e) {
      return 100;
    }
  });
  const projectPath = projects.find(item => item.name === chatProject)?.path || '';

  const pickRounds = (value) => {
    const next = Math.min(1000, Math.max(1, Math.round(Number(value)) || 1));
    setRounds(next);
    try {
      localStorage.setItem('sneo.agentRounds', String(next));
    } catch(e) {}
  };
  const [notice, setNotice] = useState('');
  const loaded = useRef(false);
  const stops = useRef({});
  const stepsRef = useRef({});
  const chatsRef = useRef(chats);
  const agentsRef = useRef([]);
  agentsRef.current = agents;
  const {readyOf, labelOf} = useLoaded(apiUrl);
  const routeOf = agent => routes.find(item => item.id === agent?.route);
  const readyRef = useRef(readyOf);
  readyRef.current = readyOf;
  // los encargos entre agentes salen despues: usan el modo, las vueltas y el proyecto de ahora y no los de cuando se escribio el primer mensaje
  const pickedRef = useRef({});
  pickedRef.current = {mode: agentMode, rounds, project: chatProject};

  useEffect(() => {
    fetch(`${apiUrl}/api/team/agents`)
      .then(answer => answer.json())
      .then(res => {
        setAgents(res.agents || []);
        loaded.current = true;
      })
      .catch(e => setNotice(tr('The app server did not answer: {0}', e.message)));
  }, []);

  useEffect(() => {
    if(project){
      setChatProject(project);
    }
  }, [project]);

  useEffect(() => {
    chatsRef.current = chats;
    keepChats(chats);
  }, [chats]);

  // cada agente sin ruta valida toma la primera que exista para poder contestar
  useEffect(() => {
    if(!routes.length || !agents.length){
      return;
    }

    const valid = id => routes.some(item => item.id === id);

    if(agents.some(item => !valid(item.route))){
      setAgents(prev => prev.map(item => valid(item.route) ? item : {...item, route: routes[0].id}));
    }
  }, [routes, agents]);

  useEffect(() => {
    if(!loaded.current){
      return;
    }

    const timer = setTimeout(() => {
      fetch(`${apiUrl}/api/team/agents`, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({agents})}).catch(() => {});
    }, 600);

    return () => clearTimeout(timer);
  }, [agents]);

  const push = (id, message) => {
    setChats(prev => ({...prev, [id]: [...(prev[id] || []), message].slice(-80)}));
  };

  const setStep = (id, text) => {
    stepsRef.current = {...stepsRef.current, [id]: text};
    setSteps(stepsRef.current);
  };

  const addLive = (id, row) => {
    const now = liveRef.current[id] || [];
    const last = now[now.length - 1];
    // el pensamiento que sigue creciendo reemplaza al anterior en vez de repetirse
    const next = row.kind === 'thought' && last?.kind === 'thought' ? [...now.slice(0, -1), row] : [...now, row];
    liveRef.current = {...liveRef.current, [id]: next.slice(-40)};
    setLive(liveRef.current);
  };

  const nameOf = id => agentsRef.current.find(item => item.id === id)?.name || id;

  const sendTo = async (id, text, from = '', hops = 0, kind = '') => {
    const agent = agentsRef.current.find(item => item.id === id);
    const clean = String(text || '').trim();

    if(!agent || !clean){
      return;
    }

    if(stepsRef.current[id]){
      // lo que manda otro agente espera su turno y lo que escribe el usuario avisa
      if(kind){
        waiting.current[id] = [...(waiting.current[id] || []), [id, text, from, hops, kind]];
        return;
      }
      push(id, {role: 'error', content: tr('{0} is still answering, wait or stop it first.', agent.name)});
      return;
    }

    if(!routes.some(item => item.id === agent.route)){
      push(id, {role: 'error', content: tr('Pick the model this agent answers with first.')});
      return;
    }

    if(!readyRef.current(routeOf(agent))){
      push(id, {role: 'error', content: tr('The model of {0} is not loaded yet. Load it in Agents, Models.', agent.name)});
      return;
    }

    if(saveFirst && !await saveFirst()){
      push(id, {role: 'error', content: tr('The models could not be saved, look at the Models tab.')});
      return;
    }

    const message = {role: 'user', content: from ? `${kind === 'report' ? 'Report from' : 'From'} ${nameOf(from)}:\n${clean}` : clean, ...(from ? {from, kind} : {})};
    const history = [...(chatsRef.current[id] || []), message]
      .filter(item => item.role === 'user' || item.role === 'assistant')
      .map(item => ({role: item.role, content: item.content}));
    chatsRef.current = {...chatsRef.current, [id]: [...(chatsRef.current[id] || []), message]};
    push(id, message);
    if(from){
      setVisits(prev => [...prev, {id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, from, to: id, text: clean, kind}].slice(-30));
    }

    const stop = new AbortController();
    stops.current[id] = stop;
    setStep(id, tr('Thinking...'));
    liveRef.current = {...liveRef.current, [id]: []};
    setLive(liveRef.current);

    try {
      const req = await fetch(`${apiUrl}/api/team/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent, history, project: pickedRef.current.project, rounds: pickedRef.current.rounds, mode: pickedRef.current.mode, peers: agentsRef.current.filter(item => item.id !== id).map(item => item.name)}),
        signal: stop.signal
      });
      let answered = false;

      await readEvents(req, event => {
        if(event.type === 'step' && event.usage){
          setSpeeds(prev => ({...prev, [id]: event.usage}));
        }

        if(event.type === 'step' && event.thought){
          setStep(id, tr('Thinking...'));
          addLive(id, {kind: 'thought', text: event.thought});
        }

        if(event.type === 'step' && event.tool){
          setStep(id, tr('Using {0}', event.tool));
          addLive(id, {kind: 'tool', text: event.detail ? `${event.tool} · ${event.detail}` : event.tool});
        }

        if(event.type === 'reply'){
          answered = true;
          const content = event.content || tr('(no answer)');
          const handoffs = hops < mostHops ? parseHandoffs(content, agentsRef.current, id) : [];
          const sentBack = handoffs.some(item => item.id === from);
          push(id, {role: 'assistant', content, ms: event.ms, tools: event.tools || [], usage: event.usage || null, thinking: event.thinking || '', activity: liveRef.current[id] || [], handoffs: handoffs.map(item => item.name)});

          // los encargos salen de verdad y la respuesta a un encargo vuelve sola al que lo pidio
          handoffs.forEach(item => setTimeout(() => sendTo(item.id, item.text.trim(), id, hops + 1, 'task'), 0));
          if(kind === 'task' && from && !sentBack && hops < mostHops){
            setTimeout(() => sendTo(from, content, id, hops + 1, 'report'), 0);
          }
          if(hops >= mostHops && parseHandoffs(content, agentsRef.current, id).length){
            push(id, {role: 'error', content: tr('The agents passed work {0} times in a row, stopped here so they do not loop.', mostHops)});
          }
        }

        if(event.type === 'approval' && event.item){
          const item = {id: event.item.id, title: event.item.title, diff: ''};
          setAsks(prev => ({...prev, [id]: [...(prev[id] || []), item]}));
          setStep(id, tr('Waiting for your approval'));
          fetch(`${apiUrl}/api/pending`).then(answer => answer.json()).then(res => {
            const found = (res.items || []).find(one => one.id === item.id);
            const diff = (found?.files || []).map(file => `${file.file}\n${file.diff || ''}`).join('\n\n') || found?.command || found?.details || '';
            if(diff) setAsks(prev => ({...prev, [id]: (prev[id] || []).map(one => one.id === item.id ? {...one, diff: diff.slice(0, 12000)} : one)}));
          }).catch(() => {});
        }

        if(event.type === 'approval-done'){
          setAsks(prev => ({...prev, [id]: (prev[id] || []).filter(one => one.id !== event.id)}));
          setStep(id, tr('Thinking...'));
        }

        if(event.type === 'error'){
          answered = true;
          push(id, {role: 'error', content: event.message});
        }
      });

      if(!answered){
        push(id, {role: 'error', content: tr('The answer was cut off.')});
      }
    } catch(e) {
      push(id, {role: 'error', content: e.name === 'AbortError' ? tr('Stopped.') : e.message});
    } finally {
      delete stops.current[id];
      setStep(id, '');
      liveRef.current = {...liveRef.current, [id]: []};
      setLive(liveRef.current);
      const next = (waiting.current[id] || []).shift();
      if(next) setTimeout(() => sendTo(...next), 0);
    }
  };

  const sendDraft = (id) => {
    const text = drafts[id] || '';

    if(!text.trim()){
      return;
    }

    setDrafts(prev => ({...prev, [id]: ''}));
    sendTo(id, text);
  };

  const sendCommand = () => {
    const text = command.trim();

    if(!text){
      return;
    }

    setCommand('');

    if(target === 'all'){
      agentsRef.current.forEach(item => sendTo(item.id, text));
      return;
    }

    sendTo(target, text);
  };

  const changeAgent = (id, change) => {
    setAgents(prev => prev.map(item => item.id === id ? {...item, ...change} : item));
  };

  const addAgent = () => {
    if(agents.length >= mostAgents){
      return;
    }

    const agent = {id: `agent-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, name: tr('Agent {0}', agents.length + 1), focus: '', route: routes[0]?.id || '', tools: false};
    setAgents(prev => [...prev, agent]);
    setEditing(agent.id);
  };

  // quitar un agente se guarda en el momento y no despues de la pausa de los cambios comunes
  const removeAgent = (id) => {
    stops.current[id]?.abort();
    const next = agentsRef.current.filter(item => item.id !== id);
    agentsRef.current = next;
    setAgents(next);
    fetch(`${apiUrl}/api/team/agents`, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({agents: next})}).catch(() => {});
    setChats(prev => {
      const next = {...prev};
      delete next[id];
      return next;
    });

    if(target === id){
      setTarget('all');
    }
  };

  // la memoria de cada agente es su conversacion, borrarla es empezar de cero
  const clearAgent = (id) => {
    stops.current[id]?.abort();
    delete waiting.current[id];
    Object.keys(waiting.current).forEach(key => {
      waiting.current[key] = waiting.current[key].filter(item => item[2] !== id);
    });
    chatsRef.current = {...chatsRef.current, [id]: []};
    setChats(prev => ({...prev, [id]: []}));
  };

  const clearLogs = () => {
    waiting.current = {};
    Object.values(stops.current).forEach(stop => stop.abort());
    chatsRef.current = {};
    setChats({});
  };

  const columns = agents.length <= 1 ? 1 : agents.length <= 4 ? 2 : agents.length <= 9 ? 3 : 4;
  // el centro de comandos solo escribe cuando estan cargados los modelos de todos los agentes que reciben la orden
  const commandReady = (target === 'all' ? agents : agents.filter(item => item.id === target)).every(item => readyOf(routeOf(item)));

  return (
    <div className="agents-chats">
      <div className="chats-head">
        <div>
          <h2>{tr('Agent conversations')}</h2>
          <span>{tr('{0} separate message streams, each one on its own model', agents.length)}</span>
        </div>
        <div className="chats-tools">
          <label className="chats-project" title={tr('Agents that can read the project look at this one')}>
            <i className="fa-solid fa-folder"></i>
            <select value={chatProject} onChange={(e) => setChatProject(e.target.value)} aria-label={tr('Project')}>
              <option value="">{tr('No project')}</option>
              {projects.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
            {projectPath && <span className="chats-path" title={projectPath}>{projectPath}</span>}
          </label>
          <label className="chats-rounds" title={tr('What agents that change code may do: Auto changes files and runs commands alone, Ask waits for your approval on each change, Plan only reads')}>
            <i className="fa-solid fa-shield-halved"></i>
            {tr('Permissions')}
            <select value={agentMode} onChange={(e) => pickMode(e.target.value)}>
              <option value="auto">{tr('Auto')}</option>
              <option value="manual">{tr('Ask')}</option>
              <option value="plan">{tr('Plan')}</option>
            </select>
          </label>
          <label className="chats-rounds" title={tr('How many tool calls each agent can make before it has to answer')}>
            <i className="fa-solid fa-wrench"></i>
            {tr('Tool calls')}
            <input type="number" min="1" max="1000" value={rounds} onChange={(e) => pickRounds(e.target.value)} />
          </label>
          <div className="segmented chats-view">
            <button type="button" className={view === 'panels' ? 'active' : ''} onClick={() => pickView('panels')}><i className="fa-solid fa-table-columns"></i> {tr('Panels')}</button>
            <button type="button" className={view === 'office' ? 'active' : ''} onClick={() => pickView('office')}><i className="fa-solid fa-building"></i> {tr('Office')}</button>
          </div>
          <button type="button" className="agents-small" disabled={agents.length >= mostAgents} onClick={addAgent}><i className="fa-solid fa-plus"></i> {tr('Add agent')}</button>
          <button type="button" className="agents-small" title={tr('Stops every agent and clears all conversations: they forget everything and start from zero')} onClick={clearLogs}>{tr('Clear logs')}</button>
        </div>
      </div>

      {notice && <div className="view-alert alert-bad">{notice}</div>}

      {view === 'office' && (
        <AgentOffice
          agents={agents}
          routes={routes}
          steps={steps}
          live={live}
          speeds={speeds}
          chats={chats}
          visits={visits}
          readyOf={route => readyOf(route)}
          labelOf={labelOf}
          onSend={(id, text) => sendTo(id, text)}
          onStop={id => stops.current[id]?.abort()}
        />
      )}

      {view === 'panels' && <div className="chats-grid" style={{'--chat-cols': columns, '--chat-rows': Math.max(1, Math.ceil(agents.length / columns))}}>
        {agents.map((agent, i) => (
          <AgentCard
            key={agent.id}
            agent={agent}
            index={i}
            routes={routes}
            ready={readyOf(routeOf(agent))}
            labelOf={labelOf}
            messages={chats[agent.id] || []}
            step={steps[agent.id] || ''}
            live={live[agent.id] || []}
            speed={speeds[agent.id] || null}
            asks={asks[agent.id] || []}
            onAnswer={(askId, answer) => answerAsk(agent.id, askId, answer)}
            draft={drafts[agent.id] || ''}
            others={agents.filter(item => item.id !== agent.id)}
            editing={editing === agent.id}
            onDraft={text => setDrafts(prev => ({...prev, [agent.id]: text}))}
            onSend={() => sendDraft(agent.id)}
            onStop={() => stops.current[agent.id]?.abort()}
            onForward={(to, text) => sendTo(to, text, agent.id)}
            onEdit={() => setEditing(prev => prev === agent.id ? '' : agent.id)}
            onChange={change => changeAgent(agent.id, change)}
            onRemove={() => removeAgent(agent.id)}
            onClear={() => clearAgent(agent.id)}
          />
        ))}
      </div>}

      <section className="card chats-command" title={tr('Send an instruction to the entire team or to one agent')}>
        <div className="agents-row">
          <span className="chats-command-title"><i className="fa-solid fa-tower-broadcast"></i> {tr('Command center')}</span>
          <select className="chats-target" value={target} onChange={(e) => setTarget(e.target.value)} aria-label={tr('Who gets the command')}>
            <option value="all">{tr('All agents · Broadcast')}</option>
            {agents.map((item, i) => <option key={item.id} value={item.id}>{`${number(i)} · ${short(item.name, 30)}`}</option>)}
          </select>
          <input
            className="chats-command-input"
            value={command}
            placeholder={commandReady ? tr('Instruction for the team...') : tr('Load the models of these agents first')}
            disabled={!commandReady}
            onChange={(e) => setCommand(e.target.value)}
            onKeyDown={(e) => { if(e.key === 'Enter'){ e.preventDefault(); sendCommand(); } }}
          />
          <button type="button" className="agents-run" disabled={!command.trim() || !agents.length || !commandReady} onClick={sendCommand}>
            {tr('Send command')} <i className="fa-solid fa-arrow-up-right-from-square"></i>
          </button>
        </div>
      </section>
    </div>
  );
}
