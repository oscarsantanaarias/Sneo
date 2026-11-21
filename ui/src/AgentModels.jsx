import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n.js';
import { colorOf, short } from './agentsShared.js';
import { SecondModel } from './TwoModels.jsx';

const efforts = ['low', 'medium', 'high', 'xhigh', 'max'];
const contexts = [8192, 16384, 32768, 65536, 131072];

const gigabytes = value => `${Number(value || 0).toFixed(1)} GB`;

export default function AgentModels({apiUrl, routes, presets, claudeModels, claudeCodeModels = [], dirty, notice, onAdd, onEdit, onRemove, onSave, onReload}){
  const [tests, setTests] = useState({});
  const [modelLists, setModelLists] = useState({});
  const [listNotice, setListNotice] = useState('');
  const [engine, setEngine] = useState({models: [], currentModel: '', ready: false, warming: false});

  // los modelos de esta pc y cual esta cargado, igual que en el resto de la app
  useEffect(() => {
    let alive = true;
    const look = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/engine/files`).then(answer => answer.json());
        if(alive) setEngine({models: res.models || [], currentModel: res.currentModel || '', ready: !!res.ready, warming: !!res.warming});
      } catch(e) {}
    };
    look();
    const timer = setInterval(look, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [apiUrl]);

  const thisPc = (item) => item.kind === 'local' && /^https?:\/\/(127\.0\.0\.1|localhost):8000(\/|$)/i.test(item.baseUrl || '');
  const loadedName = engine.currentModel.split(/[\\/]/).pop();
  const loadedLabel = engine.ready ? loadedName : engine.warming ? `${loadedName} (${tr('loading')})` : '';

  const testRoute = async (id) => {
    if(!await onSave()){
      return;
    }

    setTests(prev => ({...prev, [id]: {busy: true}}));

    try {
      const res = await fetch(`${apiUrl}/api/team/routes/test`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id})}).then(answer => answer.json());
      setTests(prev => ({...prev, [id]: res}));
    } catch(e) {
      setTests(prev => ({...prev, [id]: {ok: false, message: e.message}}));
    }
  };

  const listModels = async (id) => {
    if(!await onSave()){
      return;
    }

    try {
      const res = await fetch(`${apiUrl}/api/team/models`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id})}).then(answer => answer.json());
      setModelLists(prev => ({...prev, [id]: res.models || []}));
      setListNotice(res.message && !res.models?.length ? res.message : '');
    } catch(e) {
      setListNotice(e.message);
    }
  };

  return (
    <div className="agents-grid agents-grid-flip">
      <div className="agents-col agents-wide">
        <section className="card">
          <div className="card-head">
            <h3>{tr('Models the agents use')}</h3>
            <select className="agents-add" value="" onChange={(e) => { const preset = presets[Number(e.target.value)]; if(preset) onAdd(preset); }}>
              <option value="">{tr('Add a model...')}</option>
              {presets.map((item, i) => <option key={item.name} value={i}>{item.name}</option>)}
            </select>
          </div>

          <div className="routes-grid">
            {routes.map(item => (
              <div key={item.id} className="agents-route" style={{'--route': colorOf(item)}}>
                <div className="agents-row">
                  <input className="agents-name" value={item.name} onChange={(e) => onEdit(item.id, {name: e.target.value})} aria-label={tr('Name')} />
                  <span className="agents-kind">{item.kind === 'local' ? tr('this PC') : item.kind === 'claude' ? 'Claude' : item.kind === 'claudecode' ? tr('your plan') : item.kind === 'agy' ? tr('your account') : 'API'}</span>
                  <button type="button" className="icon-btn icon-danger" title={tr('Remove')} onClick={() => onRemove(item.id)}><i className="fa-solid fa-xmark"></i></button>
                </div>

                {item.kind !== 'claude' && item.kind !== 'claudecode' && item.kind !== 'agy' && (
                  <input value={item.baseUrl} placeholder="https://.../v1" onChange={(e) => onEdit(item.id, {baseUrl: e.target.value})} aria-label={tr('Address')} />
                )}

                <div className="agents-row">
                  {thisPc(item) ? (
                    <select value={item.model} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')}>
                      <option value="">{loadedLabel ? tr('The loaded one: {0}', loadedLabel) : tr('The loaded one (none now)')}</option>
                      {engine.models.map(model => <option key={model.path} value={model.name}>{model.name}{model.path === engine.currentModel && engine.ready ? ` · ${tr('loaded')}` : ''}</option>)}
                    </select>
                  ) : item.kind === 'claudecode' ? (
                    <select value={item.model} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')}>
                      {claudeCodeModels.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
                    </select>
                  ) : item.kind === 'claude' ? (
                    <select value={item.model} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')}>
                      {claudeModels.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
                    </select>
                  ) : (
                    <>
                      {/* la lista del navegador solo mostraba los parecidos a lo escrito: con la lista bajada va un select con todos */}
                      {(modelLists[item.id] || []).length ? (
                        <select value={item.model} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')}>
                          {item.kind === 'agy' && <option value="">{tr('Default')}</option>}
                          {item.model && !modelLists[item.id].includes(item.model) && <option value={item.model}>{item.model}</option>}
                          {modelLists[item.id].map(model => <option key={model} value={model}>{model}</option>)}
                        </select>
                      ) : (
                        <input value={item.model} placeholder={item.kind === 'local' ? tr('the model loaded there') : tr('model id')} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')} />
                      )}
                      {item.kind !== 'local' && <button type="button" className="agents-small" onClick={() => listModels(item.id)}>{tr('List')}</button>}
                    </>
                  )}
                </div>

                {item.kind === 'claudecode' && (
                  <div className="agents-row">
                    <span className="field-label">{tr('Effort')}</span>
                    <select className="agents-effort" value={item.effort} title={tr('Effort')} onChange={(e) => onEdit(item.id, {effort: e.target.value})}>
                      {efforts.map(level => <option key={level} value={level}>{level}</option>)}
                    </select>
                  </div>
                )}

                {item.kind === 'claudecode' && (
                  <em className="field-help">{tr('Uses Claude Code installed on this PC with your own Claude account, no API key. It counts against your plan limits.')}</em>
                )}

                {item.kind === 'agy' && (
                  <em className="field-help">{tr('Uses Antigravity installed on this PC with your own Google account, no API key. It counts against your Antigravity limits. Press List to see the models your account has.')}</em>
                )}

                {item.kind !== 'local' && item.kind !== 'claudecode' && item.kind !== 'agy' && (
                  <div className="agents-row">
                    <input type="password" value={item.apiKey} placeholder={tr('API key')} autoComplete="off" onChange={(e) => onEdit(item.id, {apiKey: e.target.value})} aria-label={tr('API key')} />
                    {item.kind === 'claude' && (
                      <select className="agents-effort" value={item.effort} title={tr('Effort')} onChange={(e) => onEdit(item.id, {effort: e.target.value})}>
                        {efforts.map(level => <option key={level} value={level}>{level}</option>)}
                      </select>
                    )}
                  </div>
                )}

                <div className="agents-row">
                  <label className="agents-parallel" title={tr('How many agents this model answers at the same time')}>
                    {tr('At once')}
                    <input type="number" min="1" max="16" value={item.parallel} onChange={(e) => onEdit(item.id, {parallel: Number(e.target.value) || 1})} />
                  </label>
                  <button type="button" className="agents-small" disabled={tests[item.id]?.busy} onClick={() => testRoute(item.id)}>
                    {tests[item.id]?.busy ? tr('Testing...') : tr('Test')}
                  </button>
                  {tests[item.id] && !tests[item.id].busy && (
                    <span className={tests[item.id].ok ? 'agents-ok' : 'agents-bad'} title={tests[item.id].message || tests[item.id].answer || ''}>
                      {tests[item.id].ok ? (tests[item.id].model ? tr('Answers in {0} s with {1}', (tests[item.id].ms / 1000).toFixed(1), short(tests[item.id].model, 40)) : tr('Answers in {0} s', (tests[item.id].ms / 1000).toFixed(1))) : short(tests[item.id].message, 60)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="agents-row agents-actions">
            {(notice || listNotice) && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {notice || listNotice}</div>}
            <span className="rail-spacer"></span>
            {dirty && <button type="button" className="agents-run" onClick={onSave}><i className="fa-solid fa-floppy-disk"></i> {tr('Save')}</button>}
          </div>
          <span className="agents-hint">{tr('Keys stay on this PC, the page only sees ***. Claude, OpenAI, DeepSeek, OpenRouter or any server with a chat API like OpenAI works here.')}</span>
        </section>
      </div>

      <div className="agents-col">
        <SecondModel apiUrl={apiUrl} onReady={onReload} />
      </div>
    </div>
  );
}
