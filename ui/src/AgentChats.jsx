import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { tr } from './i18n.js';
import { colorOf, readEvents, short, useLoaded } from './agentsShared.js';

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

function AgentCard({agent, index, routes, ready, labelOf, messages, step, draft, others, editing, onDraft, onSend, onStop, onForward, onEdit, onChange, onRemove}){
  const listRef = useRef(null);
  const route = routes.find(item => item.id === agent.route);
  const color = colorOf(route || {});
  const last = messages[messages.length - 1];
  const second = route?.kind === 'local' && /:8001(\/|$)/.test(route.baseUrl || '');
  const state = !route ? 'off' : step ? 'busy' : !ready ? 'wait' : last?.role === 'error' ? 'bad' : 'ok';
  const stateText = state === 'off' ? tr('No model') : state === 'busy' ? step : state === 'wait' ? (second ? tr('Second model off') : tr('No model loaded')) : state === 'bad' ? tr('Failed') : tr('Ready');

  useEffect(() => {
    listRef.current?.scrollTo({top: listRef.current.scrollHeight, behavior: 'smooth'});
  }, [messages.length, step]);

  return (
    <section className={step ? 'card chat-card chat-busy' : 'card chat-card'} style={{'--agent': color}}>
      <div className="chat-head">
        <div className="chat-title">
          <strong>{number(index)} · {agent.name}</strong>
          <span className={`chat-state chat-state-${state}`}><span className="chat-dot"></span>{stateText}</span>
        </div>
        <div className="chat-sub">
          <span className="chat-focus">{agent.focus || tr('Agent')}</span>
          <select className="chat-route" value={agent.route} onChange={(e) => onChange({route: e.target.value})} aria-label={tr('Model this agent answers with')}>
            {!route && <option value="">{tr('Pick a model...')}</option>}
            {routes.map(item => <option key={item.id} value={item.id}>{labelOf(item)}</option>)}
          </select>
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
              <input type="checkbox" checked={agent.tools} onChange={(e) => onChange({tools: e.target.checked})} />
              {tr('Can read the project (files, search, ghidra and ida)')}
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
                {from && <span className="chat-from">{tr('From {0}', from.name)}</span>}
                {from ? String(item.content || '').replace(/^From [^:\n]+:\n/, '') : item.content}
              </div>
            );
          }

          if(item.role === 'error'){
            return <div key={i} className="chat-bubble chat-error"><i className="fa-solid fa-circle-exclamation"></i> {item.content}</div>;
          }

          return (
            <div key={i} className="chat-bubble chat-agent">
              <div className="markdown-body" dangerouslySetInnerHTML={html(item.content)} />
              <div className="chat-meta">
                {item.ms ? <span>{(item.ms / 1000).toFixed(1)} s</span> : null}
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
  const [drafts, setDrafts] = useState({});
  const [editing, setEditing] = useState('');
  const [target, setTarget] = useState('all');
  const [command, setCommand] = useState('');
  const [chatProject, setChatProject] = useState(project);
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

  const nameOf = id => agentsRef.current.find(item => item.id === id)?.name || id;

  const sendTo = async (id, text, from = '') => {
    const agent = agentsRef.current.find(item => item.id === id);
    const clean = String(text || '').trim();

    if(!agent || !clean){
      return;
    }

    if(stepsRef.current[id]){
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

    const message = {role: 'user', content: from ? `From ${nameOf(from)}:\n${clean}` : clean, ...(from ? {from} : {})};
    const history = [...(chatsRef.current[id] || []), message]
      .filter(item => item.role === 'user' || item.role === 'assistant')
      .map(item => ({role: item.role, content: item.content}));
    chatsRef.current = {...chatsRef.current, [id]: [...(chatsRef.current[id] || []), message]};
    push(id, message);

    const stop = new AbortController();
    stops.current[id] = stop;
    setStep(id, tr('Thinking...'));

    try {
      const req = await fetch(`${apiUrl}/api/team/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent, history, project: chatProject, peers: agentsRef.current.filter(item => item.id !== id).map(item => item.name)}),
        signal: stop.signal
      });
      let answered = false;

      await readEvents(req, event => {
        if(event.type === 'step'){
          setStep(id, tr('Using {0}', event.tool));
        }

        if(event.type === 'reply'){
          answered = true;
          push(id, {role: 'assistant', content: event.content || tr('(no answer)'), ms: event.ms, tools: event.tools || []});
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

  const clearLogs = () => {
    Object.values(stops.current).forEach(stop => stop.abort());
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
          </label>
          <button type="button" className="agents-small" disabled={agents.length >= mostAgents} onClick={addAgent}><i className="fa-solid fa-plus"></i> {tr('Add agent')}</button>
          <button type="button" className="agents-small" onClick={clearLogs}>{tr('Clear logs')}</button>
        </div>
      </div>

      {notice && <div className="view-alert alert-bad">{notice}</div>}

      <div className="chats-grid" style={{'--chat-cols': columns, '--chat-rows': Math.max(1, Math.ceil(agents.length / columns))}}>
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
          />
        ))}
      </div>

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
