import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n.js';
import AgentChats from './AgentChats.jsx';
import AgentTeam from './AgentTeam.jsx';
import AgentModels from './AgentModels.jsx';

const tabs = [
  {value: 'chats', label: 'Conversations', icon: 'fa-solid fa-comments'},
  {value: 'team', label: 'Team run', icon: 'fa-solid fa-diagram-project'},
  {value: 'models', label: 'Models', icon: 'fa-solid fa-layer-group'}
];

const newRoute = (preset) => ({...preset, id: `route-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, apiKey: '', effort: 'medium'});

export default function AgentsView({apiUrl, projects = [], project = ''}){
  const [tab, setTab] = useState('chats');
  const [routes, setRoutes] = useState([]);
  const [presets, setPresets] = useState([]);
  const [claudeModels, setClaudeModels] = useState([]);
  const [claudeCodeModels, setClaudeCodeModels] = useState([]);
  const [dirty, setDirty] = useState(false);
  const [notice, setNotice] = useState('');
  const routesRef = useRef([]);
  const dirtyRef = useRef(false);
  routesRef.current = routes;
  dirtyRef.current = dirty;

  const loadRoutes = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/team/routes`).then(answer => answer.json());
      setRoutes(res.routes || []);
      setPresets(res.presets || []);
      setClaudeModels(res.claudeModels || []);
      setClaudeCodeModels(res.claudeCodeModels || []);
      setDirty(false);
      setNotice('');
    } catch(e) {
      setNotice(tr('The app server did not answer: {0}', e.message));
    }
  };

  useEffect(() => {
    loadRoutes();
  }, []);

  // cuando el servidor agrega una ruta (el segundo modelo) se suma sin borrar lo que se esta editando
  const mergeRoutes = async () => {
    if(!dirtyRef.current){
      return await loadRoutes();
    }

    try {
      const res = await fetch(`${apiUrl}/api/team/routes`).then(answer => answer.json());
      setRoutes(prev => [...prev, ...(res.routes || []).filter(item => !prev.some(old => old.id === item.id))]);
    } catch(e) {
    }
  };

  // guarda las rutas si quedaron cambios antes de usarlas
  const saveRoutes = async () => {
    if(!dirtyRef.current){
      return true;
    }

    try {
      const req = await fetch(`${apiUrl}/api/team/routes`, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({routes: routesRef.current})});
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || tr('Could not save the models'));
      }

      setRoutes(res.routes || []);
      setDirty(false);
      setNotice('');
      return true;
    } catch(e) {
      setNotice(e.message);
      return false;
    }
  };

  const editRoute = (id, change) => {
    setRoutes(prev => prev.map(item => item.id === id ? {...item, ...change} : item));
    setDirty(true);
  };

  const addRoute = (preset) => {
    setRoutes(prev => [...prev, newRoute(preset)]);
    setDirty(true);
  };

  const removeRoute = (id) => {
    setRoutes(prev => prev.filter(item => item.id !== id));
    setDirty(true);
  };

  return (
    <div className="view-page agents-page">
      <div className="view-head">
        <div>
          <h1>{tr('Agents')}</h1>
          <span>{tr('Several models at once: Sneo on this PC, a second local model, Claude, OpenAI or DeepSeek')}</span>
        </div>

        <div className="view-head-actions agents-tabs" role="tablist">
          {tabs.map(item => (
            <button
              key={item.value}
              type="button"
              role="tab"
              aria-selected={tab === item.value}
              className={tab === item.value ? 'agents-tab active' : 'agents-tab'}
              onClick={() => setTab(item.value)}
            >
              <i className={item.icon}></i> {tr(item.label)}
            </button>
          ))}
        </div>
      </div>

      <div className="agents-panel agents-fill" hidden={tab !== 'chats'}>
        <AgentChats apiUrl={apiUrl} routes={routes} projects={projects} project={project} saveFirst={saveRoutes} />
      </div>

      <div className="agents-panel" hidden={tab !== 'team'}>
        <AgentTeam apiUrl={apiUrl} routes={routes} projects={projects} project={project} saveFirst={saveRoutes} />
      </div>

      <div className="agents-panel" hidden={tab !== 'models'}>
        <AgentModels
          apiUrl={apiUrl}
          routes={routes}
          presets={presets}
          claudeModels={claudeModels}
          claudeCodeModels={claudeCodeModels}
          dirty={dirty}
          notice={notice}
          onAdd={addRoute}
          onEdit={editRoute}
          onRemove={removeRoute}
          onSave={saveRoutes}
          onReload={mergeRoutes}
        />
      </div>
    </div>
  );
}
