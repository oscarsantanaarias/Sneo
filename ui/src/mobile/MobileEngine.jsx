import React from 'react';
import { useEffect, useState } from 'react';
import { tr } from '../i18n.js';
import { SecondModel } from '../TwoModels.jsx';

const gb = value => `${(Number(value || 0) / 1024 ** 3).toFixed(1)} GB`;

export default function MobileEngine({visible}){
  const [monitor, setMonitor] = useState(null);
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');

  const read = async () => {
    const [watch, list] = await Promise.all([
      fetch('/api/engine/monitor').then(answer => answer.json()).catch(() => null),
      fetch('/api/engine/files').then(answer => answer.json()).catch(() => null)
    ]);
    setMonitor(watch);
    setFiles(list);
  };

  useEffect(() => {
    if(!visible){
      return;
    }

    read();
    const timer = setInterval(read, 3000);
    return () => clearInterval(timer);
  }, [visible]);

  const load = async (file) => {
    setBusy(file);
    setNotice('');

    try {
      const res = await fetch('/api/engine/files/load', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path: file})}).then(answer => answer.json());

      if(res.success === false){
        setNotice(res.error || tr('The model could not load'));
      }

      await read();
    } catch(e) {
      setNotice(e.message);
    } finally {
      setBusy('');
    }
  };

  const self = (monitor?.nodes || []).find(item => item.self)?.stats || {};
  const last = (monitor?.requests || [])[0];
  const state = files?.ready ? tr('ready') : files?.warming ? tr('loading') : files?.loaded ? tr('loaded') : tr('off');

  return (
    <div className="m-engine">
      <section className="card">
        <div className="card-head">
          <h3>{tr('Engine')}</h3>
          <span className={`state-pill ${files?.ready ? 'pill-ok' : files?.warming ? 'pill-busy' : 'pill-idle'}`}>
            <span className={files?.warming ? 'state-dot dot-live' : 'state-dot'}></span>{state}
          </span>
        </div>
        {!monitor && !files && <div className="agents-hint">{tr('The engine does not answer, start Sneo on the PC.')}</div>}
        {monitor && (
          <div className="m-stats">
            <div><span>{tr('Model')}</span><strong>{monitor.model || tr('no model')}</strong></div>
            <div><span>{tr('Writing')}</span><strong>{last ? `${Number(last.outputSpeed || 0).toFixed(1)} tok/s` : '—'}</strong></div>
            <div><span>{tr('Reading')}</span><strong>{last ? `${Number(last.promptSpeed || 0).toFixed(0)} tok/s` : '—'}</strong></div>
            <div><span>{tr('Video memory')}</span><strong>{gb(self.heldGpu)}</strong></div>
            <div><span>{tr('Ram')}</span><strong>{gb(self.heldRam)}</strong></div>
            <div><span>{tr('Context')}</span><strong>{Math.round(Number(monitor.context || 0) / 1024)}K</strong></div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h3>{tr('Models')}</h3>
        </div>
        {(files?.models || []).map(item => {
          const current = item.path === files.currentModel;
          return (
            <div key={item.path} className={current ? 'm-model current' : 'm-model'}>
              <div>
                <strong>{item.name}</strong>
                <span>{Math.round(Number(item.size || 0) / 1024 ** 3)} GB</span>
              </div>
              {current
                ? <span className="state-pill pill-ok">{files.ready ? tr('in use') : tr('loading')}</span>
                : <button type="button" className="agents-small" disabled={Boolean(busy) || files.warming} onClick={() => load(item.path)}>
                    {busy === item.path ? <i className="fa-solid fa-spinner fa-spin"></i> : tr('Load')}
                  </button>}
            </div>
          );
        })}
        {notice && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {notice}</div>}
      </section>

      <SecondModel apiUrl="" />
    </div>
  );
}
