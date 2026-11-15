import React from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { tr } from './i18n.js';
import { colorOf, readEvents, short, useLoaded } from './agentsShared.js';

const modes = [
  {value: 'auto', label: 'Any task'},
  {value: 'build', label: 'Build'},
  {value: 'audit', label: 'Review'},
  {value: 'research', label: 'Research'},
  {value: 'reverse', label: 'Reverse engineering'}
];

const statusText = {
  waiting: 'Waiting',
  thinking: 'Thinking',
  working: 'Working',
  tool: 'Using a tool',
  done: 'Done',
  error: 'Failed'
};

// un punto que viaja de un agente a otro cuando se mandan un mensaje
function Flight({from, to, color}){
  const ref = useRef(null);

  useEffect(() => {
    const motion = ref.current?.animate?.([
      {transform: `translate(${from.x}px, ${from.y}px)`, opacity: 0},
      {opacity: 1, offset: 0.12},
      {opacity: 1, offset: 0.88},
      {transform: `translate(${to.x}px, ${to.y}px)`, opacity: 0}
    ], {duration: 1200, easing: 'ease-in-out', fill: 'forwards'});

    return () => motion?.cancel();
  }, []);

  return <circle ref={ref} r="6" fill={color} className="agents-flight" style={{transform: `translate(${from.x}px, ${from.y}px)`, opacity: 0}} />;
}

export default function AgentTeam({apiUrl, routes, projects = [], project = '', saveFirst}){
  const [planner, setPlanner] = useState('');
  const [lead, setLead] = useState('');
  const [workers, setWorkers] = useState([]);
  const [mode, setMode] = useState('auto');
  const [maxAgents, setMaxAgents] = useState(4);
  const [teamProject, setTeamProject] = useState(project);
  const [task, setTask] = useState('');
  const [running, setRunning] = useState(false);
  const [agents, setAgents] = useState({});
  const [order, setOrder] = useState([]);
  const [log, setLog] = useState([]);
  const [flights, setFlights] = useState([]);
  const [answer, setAnswer] = useState('');
  const [notice, setNotice] = useState('');
  const stopRef = useRef(null);
  const {readyOf} = useLoaded(apiUrl);
  // el equipo solo arranca cuando estan cargados los modelos de esta pc que va a usar
  const teamReady = [planner, lead, ...workers].filter(Boolean).every(id => readyOf(routes.find(item => item.id === id)));
  const logRef = useRef(null);

  // si una ruta elegida desaparece se usa la primera
  useEffect(() => {
    const has = id => routes.some(item => item.id === id);
    const first = routes[0]?.id || '';
    setPlanner(prev => has(prev) ? prev : first);
    setLead(prev => has(prev) ? prev : first);
    setWorkers(prev => {
      const kept = prev.filter(has);
      return kept.length ? kept : first ? [first] : [];
    });
  }, [routes]);

  useEffect(() => {
    if(project){
      setTeamProject(project);
    }
  }, [project]);

  useEffect(() => {
    logRef.current?.scrollTo({top: logRef.current.scrollHeight, behavior: 'smooth'});
  }, [log]);

  const fly = (from, to) => {
    const key = `${from}-${to}-${Date.now()}-${Math.random()}`;
    setFlights(prev => [...prev, {key, from, to}]);
    setTimeout(() => setFlights(prev => prev.filter(item => item.key !== key)), 1300);
  };

  const handle = (event) => {
    if(event.type === 'team-start'){
      setAgents({
        planner: {id: 'planner', role: tr('Planner'), route: event.planner.route, kind: event.planner.kind, status: 'waiting', mark: 'P'},
        lead: {id: 'lead', role: tr('Lead'), route: event.lead.route, kind: event.lead.kind, status: 'waiting', mark: 'L'}
      });
      setOrder([]);
      return;
    }

    if(event.type === 'team-plan'){
      setAgents(prev => {
        const next = {...prev};
        event.agents.forEach((item, i) => {
          next[item.id] = {...item, status: 'waiting', mark: String(i + 1)};
        });
        return next;
      });
      setOrder(event.agents.map(item => item.id));
      return;
    }

    if(event.type === 'team-agent'){
      setAgents(prev => ({...prev, [event.id]: {...(prev[event.id] || {id: event.id, role: event.id}), status: event.status, tool: event.tool || '', error: event.error || ''}}));

      if(event.status === 'tool'){
        setLog(prev => [...prev, {kind: 'tool', who: event.id, text: event.tool}]);
      }

      if(event.status === 'error'){
        setLog(prev => [...prev, {kind: 'error', who: event.id, text: event.error}]);
      }

      return;
    }

    if(event.type === 'team-message'){
      fly(event.from, event.to);
      setLog(prev => [...prev, {kind: 'message', who: event.from, to: event.to, text: event.text}]);
      return;
    }

    if(event.type === 'team-final'){
      setAnswer(event.answer || '');
      return;
    }

    if(event.type === 'team-error'){
      setNotice(event.message);
    }
  };

  const runTeam = async () => {
    if(running || !task.trim()){
      return;
    }

    if(!await saveFirst()){
      return;
    }

    const stop = new AbortController();
    stopRef.current = stop;
    setRunning(true);
    setAnswer('');
    setLog([]);
    setAgents({});
    setOrder([]);
    setNotice('');

    try {
      const req = await fetch(`${apiUrl}/api/team/run`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task, project: teamProject, mode, planner, workers, lead, maxAgents}),
        signal: stop.signal
      });
      await readEvents(req, handle);
    } catch(e) {
      if(e.name !== 'AbortError'){
        setNotice(e.message);
      }
    } finally {
      setRunning(false);
      stopRef.current = null;
    }
  };

  // planificador arriba lider abajo y los agentes en filas de hasta 7 en el medio, tantas filas como hagan falta
  const rowsOf = count => Math.max(1, Math.ceil(count / 7));
  const layout = useMemo(() => {
    const count = order.length;
    const rows = rowsOf(count);
    const perRow = Math.ceil(count / rows) || 1;
    const top = rows === 1 ? 200 : 170;
    const spots = {planner: {x: 300, y: 52}, lead: {x: 300, y: rows === 1 ? 352 : top + rows * 96 + 30}};

    order.forEach((id, i) => {
      const row = Math.floor(i / perRow);
      const inRow = row === rows - 1 ? count - perRow * (rows - 1) : perRow;
      const step = 500 / inRow;
      spots[id] = {x: 50 + step * (i - row * perRow + 0.5), y: top + row * 96};
    });

    return spots;
  }, [order]);

  const height = rowsOf(order.length) === 1 ? 430 : 170 + rowsOf(order.length) * 96 + 110;
  const nameOf = (id) => agents[id]?.role || id;
  const nodes = Object.values(agents).filter(item => layout[item.id]);
  const busyCount = nodes.filter(item => ['thinking', 'working', 'tool'].includes(item.status)).length;

  return (
    <div className="agents-grid">
      <div className="agents-col">
        <section className="card">
          <div className="card-head">
            <h3>{tr('Team')}</h3>
            {running && <span className="state-pill pill-busy"><span className="state-dot dot-live"></span>{tr('{0} working', busyCount)}</span>}
          </div>

          <label className="agents-field">
            <span>{tr('Plans the work')}</span>
            <select value={planner} onChange={(e) => setPlanner(e.target.value)}>
              {routes.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </label>

          <div className="agents-field">
            <span>{tr('Agents work on')}</span>
            <div className="agents-chips">
              {routes.map(item => (
                <button
                  key={item.id}
                  type="button"
                  className={workers.includes(item.id) ? 'agents-chip on' : 'agents-chip'}
                  style={{'--route': colorOf(item)}}
                  aria-pressed={workers.includes(item.id)}
                  onClick={() => setWorkers(prev => prev.includes(item.id) ? prev.filter(id => id !== item.id) : [...prev, item.id])}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>

          <label className="agents-field">
            <span>{tr('Merges the answers')}</span>
            <select value={lead} onChange={(e) => setLead(e.target.value)}>
              {routes.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </label>

          <div className="agents-row">
            <label className="agents-field">
              <span>{tr('Kind of task')}</span>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                {modes.map(item => <option key={item.value} value={item.value}>{tr(item.label)}</option>)}
              </select>
            </label>
            <label className="agents-field agents-count">
              <span>{tr('Agents')}</span>
              <input type="number" min="2" max="32" value={maxAgents} onChange={(e) => setMaxAgents(Number(e.target.value) || 4)} />
            </label>
          </div>

          <label className="agents-field">
            <span>{tr('Project')}</span>
            <select value={teamProject} onChange={(e) => setTeamProject(e.target.value)}>
              <option value="">{tr('No project')}</option>
              {projects.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
          </label>

          <span className="agents-hint">{tr('A planner splits the task, each agent works on its own model and a lead merges the answers. Agents only read, they never write files.')}</span>
        </section>
      </div>

      <div className="agents-col agents-wide">
        <section className="card agents-stage">
          <svg className="agents-graph" viewBox={`0 0 600 ${height}`} role="img" aria-label={tr('Agents and their messages')}>
            {order.map(id => {
              const status = agents[id]?.status;
              return (
                <g key={`edges-${id}`}>
                  <line className={status && status !== 'waiting' ? 'agents-edge on' : 'agents-edge'} x1={layout.planner.x} y1={layout.planner.y} x2={layout[id].x} y2={layout[id].y} />
                  <line className={status === 'done' ? 'agents-edge on' : 'agents-edge'} x1={layout[id].x} y1={layout[id].y} x2={layout.lead.x} y2={layout.lead.y} />
                </g>
              );
            })}

            {nodes.map(item => (
              <g key={item.id} className={`agents-node agents-${item.status || 'waiting'}`} transform={`translate(${layout[item.id].x},${layout[item.id].y})`} style={{'--route': colorOf(item)}}>
                <title>{`${item.role}${item.route ? ` · ${item.route}` : ''}${item.goal ? `\n${item.goal}` : ''}`}</title>
                <circle r="31" className="agents-halo" />
                <circle r="27" className="agents-ring" />
                <circle r="21" className="agents-core" />
                <text y="5" className="agents-mark">{item.mark || '?'}</text>
                <text y="47" className="agents-label">{short(item.role, 16)}</text>
                <text y="61" className="agents-sub">{item.status === 'tool' && item.tool ? short(item.tool, 18) : tr(statusText[item.status] || 'Waiting')}</text>
              </g>
            ))}

            {flights.map(item => layout[item.from] && layout[item.to] && (
              <Flight key={item.key} from={layout[item.from]} to={layout[item.to]} color={colorOf(agents[item.from] || {})} />
            ))}

            {!nodes.length && (
              <g className="agents-preview">
                {[150, 300, 450].map(x => (
                  <g key={`line-${x}`}>
                    <line x1="300" y1="52" x2={x} y2="200" stroke="currentColor" strokeDasharray="4 5" />
                    <line x1={x} y1="200" x2="300" y2="352" stroke="currentColor" strokeDasharray="4 5" />
                  </g>
                ))}
                {[[300, 52, 'P'], [150, 200, '1'], [300, 200, '2'], [450, 200, '3'], [300, 352, 'L']].map(([x, y, mark]) => (
                  <g key={mark}>
                    <circle cx={x} cy={y} r="25" />
                    <text x={x} y={y + 5}>{mark}</text>
                  </g>
                ))}
              </g>
            )}

            {!nodes.length && (
              <text x="300" y={height - 14} className="agents-empty">{running ? tr('Starting the team...') : tr('Pick the team, write a task and run it')}</text>
            )}
          </svg>
        </section>

        <section className="card">
          <textarea className="agents-task" rows="3" value={task} placeholder={tr('What should the team do?')} onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if(e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runTeam(); }} />
          <div className="agents-row agents-actions">
            {notice && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {notice}</div>}
            {!notice && !teamReady && !running && <div className="agents-notice"><i className="fa-solid fa-hourglass-half"></i> {tr('Load the models of the team first, in Models')}</div>}
            <span className="rail-spacer"></span>
            {running && <button type="button" className="agents-small" onClick={() => stopRef.current?.abort()}>{tr('Stop')}</button>}
            <button type="button" className="agents-run" disabled={running || !task.trim() || !routes.length || !teamReady} onClick={runTeam}>
              <i className={running ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-play'}></i> {running ? tr('Working...') : tr('Run the team')}
            </button>
          </div>
        </section>

        {log.length > 0 && (
          <section className="card">
            <div className="card-head">
              <h3>{tr('Messages')}</h3>
              <span className="card-count">{log.length}</span>
            </div>
            <div className="agents-log" ref={logRef}>
              {log.map((item, i) => (
                <div key={i} className={`agents-line agents-line-${item.kind}`} style={{'--route': colorOf(agents[item.who] || {})}}>
                  <strong>{nameOf(item.who)}</strong>
                  {item.kind === 'message' && <span className="agents-arrow"><i className="fa-solid fa-arrow-right"></i> {nameOf(item.to)}</span>}
                  {item.kind === 'tool' && <span className="agents-arrow"><i className="fa-solid fa-wrench"></i></span>}
                  {item.kind === 'error' && <span className="agents-arrow"><i className="fa-solid fa-circle-exclamation"></i></span>}
                  <span className="agents-text">{item.text}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {answer && (
          <section className="card">
            <div className="card-head">
              <h3>{tr('Answer')}</h3>
              <button type="button" className="agents-small" onClick={() => navigator.clipboard?.writeText(answer)}><i className="fa-regular fa-copy"></i> {tr('Copy')}</button>
            </div>
            <div className="markdown-body" dangerouslySetInnerHTML={{__html: DOMPurify.sanitize(marked.parse(answer))}} />
          </section>
        )}
      </div>
    </div>
  );
}
