import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n.js';
import { colorOf, short } from './agentsShared.js';
import { SecondModel } from './TwoModels.jsx';

const efforts = ['low', 'medium', 'high', 'xhigh', 'max'];
const contexts = [8192, 16384, 32768, 65536, 131072];

const gigabytes = value => `${Number(value || 0).toFixed(1)} GB`;

export default function AgentModels({apiUrl, routes, presets, claudeModels, dirty, notice, onAdd, onEdit, onRemove, onSave, onReload}){
  const [tests, setTests] = useState({});
  const [modelLists, setModelLists] = useState({});
  const [listNotice, setListNotice] = useState('');

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
                  <span className="agents-kind">{item.kind === 'local' ? tr('this PC') : item.kind === 'claude' ? 'Claude' : 'API'}</span>
                  <button type="button" className="icon-btn icon-danger" title={tr('Remove')} onClick={() => onRemove(item.id)}><i className="fa-solid fa-xmark"></i></button>
                </div>

                {item.kind !== 'claude' && (
                  <input value={item.baseUrl} placeholder="https://.../v1" onChange={(e) => onEdit(item.id, {baseUrl: e.target.value})} aria-label={tr('Address')} />
                )}

                <div className="agents-row">
                  {item.kind === 'claude' ? (
                    <select value={item.model} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')}>
                      {claudeModels.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
                    </select>
                  ) : (
                    <>
                      <input list={`models-${item.id}`} value={item.model} placeholder={item.kind === 'local' ? tr('the model loaded there') : tr('model id')} onChange={(e) => onEdit(item.id, {model: e.target.value})} aria-label={tr('Model')} />
                      <datalist id={`models-${item.id}`}>
                        {(modelLists[item.id] || []).map(model => <option key={model} value={model} />)}
                      </datalist>
                      {item.kind !== 'local' && <button type="button" className="agents-small" onClick={() => listModels(item.id)}>{tr('List')}</button>}
                    </>
                  )}
                </div>

                {item.kind !== 'local' && (
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
                      {tests[item.id].ok ? tr('Answers in {0} s', (tests[item.id].ms / 1000).toFixed(1)) : short(tests[item.id].message, 60)}
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
