import React from 'react';
import { useState, useEffect } from 'react';
import { tr } from './i18n.js';

const gib = 1073741824;

const gb = (bytes) => `${(Number(bytes || 0) / gib).toFixed(1)} GB`;

const eta = (left, speed) => {
  if(!speed) return '';
  const seconds = left / speed;

  return seconds >= 3600
    ? tr('{0} h {1} min left', Math.floor(seconds / 3600), Math.round(seconds % 3600 / 60))
    : seconds >= 60
      ? tr('{0} min left', Math.round(seconds / 60))
      : tr('{0} s left', Math.round(seconds));
};

export default function ModelStore({apiUrl, onClose, onReady}) {
  const [store, setStore] = useState(null);
  const [problem, setProblem] = useState('');
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState('');

  const loadStore = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/catalog`);
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      setStore(res);
      setJob(res.job);
      setProblem('');
    } catch(e) {
      setProblem(e.message);
    }
  };

  useEffect(() => {
    loadStore();
  }, []);

  useEffect(() => {
    if(!job || !['checking', 'downloading'].includes(job.state)) return;

    const timer = setInterval(async () => {
      try {
        const req = await fetch(`${apiUrl}/api/engine/download`);
        const res = await req.json();
        setJob(res);

        if(res.state === 'done'){
          onReady?.();
          loadStore();
        }
      } catch(e) {
        setJob(prev => ({...prev, problem: e.message}));
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [job?.state]);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const download = async (id) => {
    setBusy(id);
    try {
      const req = await fetch(`${apiUrl}/api/engine/download`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      setJob(res);
    } catch(e) {
      setJob(prev => ({...(prev || {}), state: 'error', problem: e.message}));
    } finally {
      setBusy('');
    }
  };

  const pause = async () => {
    const req = await fetch(`${apiUrl}/api/engine/download/stop`, {method: 'POST'});
    setJob(await req.json());
  };

  const running = job && ['checking', 'downloading'].includes(job.state);
  const share = job?.total ? Math.min(100, job.done / job.total * 100) : 0;

  return (
    <div className="modal-bg store-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="store-modal" role="dialog" aria-label={tr('Get models')}>
        <div className="settings-head">
          <div>
            <h2>{tr('Get models')}</h2>
            <span>
              {store
                ? tr('This PC: {0} RAM, {1} VRAM, {2} free in {3}', gb(store.ramTotal), gb(store.vramTotal), gb(store.diskFree), store.folder || tr('no folder yet'))
                : tr('Reading the list from Hugging Face...')}
            </span>
          </div>
          <button type="button" className="icon-btn" title={tr('Close')} onClick={onClose}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>

        {job && job.state !== 'idle' && (
          <div className={job.state === 'error' ? 'store-job failed' : 'store-job'}>
            <div className="store-job-head">
              <strong>{tr(job.label)}</strong>
              <span>
                {job.state === 'checking'
                  ? tr('Checking the model header')
                  : job.state === 'downloading'
                    ? tr('{0} of {1}, {2} MB/s{3}', gb(job.done), gb(job.total), (job.speed / 1048576).toFixed(1), job.speed ? `, ${eta(job.total - job.done, job.speed)}` : '')
                    : job.state === 'done'
                      ? tr('Ready')
                      : job.state === 'paused'
                        ? tr('Paused at {0} of {1}', gb(job.done), gb(job.total))
                        : tr('Stopped')}
              </span>
            </div>
            <div className="stats-meter">
              <div className="stats-meter-fill" style={{width: `${share}%`}}></div>
            </div>
            <div className="store-job-foot">
              <span>{job.problem || tr(job.note) || job.file}</span>
              {running && <button type="button" className="btn btn-ghost" onClick={pause}>{tr('Pause')}</button>}
              {(job.state === 'paused' || job.state === 'error') && job.id && (
                <button type="button" className="btn btn-ghost" disabled={!!busy} onClick={() => download(job.id)}>{tr('Resume')}</button>
              )}
            </div>
          </div>
        )}

        <div className="store-body">
          {problem && <div className="store-problem">{problem}</div>}
          {!store && !problem && <div className="store-empty">{tr('Loading...')}</div>}
          {store?.options.map((item, index) => (
            <React.Fragment key={item.id}>
            {store.options.findIndex(other => other.family === item.family) === index && <h3 className="store-family">{item.family}</h3>}
            <div className={item.recommended ? 'store-row picked' : 'store-row'}>
              <div className="store-main">
                <div className="store-title">
                  <strong>{tr(item.label)}</strong>
                  {item.recommended && <span className="store-tag good">{tr('Best fit for this PC')}</span>}
                  {item.installed && <span className="store-tag">{tr('On disk')}</span>}
                  {item.older && <span className="store-tag warn">{tr('Older upload on disk')}</span>}
                </div>
                <span className="store-runs">{item.runs}</span>
                <span className="store-meta">
                  {tr('{0} in {1}', gb(item.size), item.parts)} {item.parts === 1 ? tr('file') : tr('files')}
                  {item.shared > 0 && !item.installed ? tr(', {0} already on disk from another model', gb(item.shared)) : ''}
                  {` · ${item.repo}`}
                </span>
              </div>
              <div className="store-side">
                {item.installed && item.readerNeeded > 0 && (
                  <button
                    type="button"
                    className="btn btn-ghost"
                    disabled={running || !!busy || !item.diskOk || !store.folder}
                    title={tr('The picture reader lets the model see pictures when Vision is on in Settings')}
                    onClick={() => download(item.id)}
                  >
                    {busy === item.id ? tr('Starting...') : tr('Get picture reader {0}', gb(item.readerNeeded))}
                  </button>
                )}
                {item.installed
                  ? <span className="store-have"><i className="fa-solid fa-check"></i>{tr('Installed')}</span>
                  : (
                    <button
                      type="button"
                      className={item.recommended ? 'btn btn-primary' : 'btn btn-ghost'}
                      disabled={running || !!busy || !item.diskOk || !store.folder}
                      title={!item.diskOk ? tr('Needs {0} free on the disk', gb(item.needed)) : ''}
                      onClick={() => download(item.id)}
                    >
                      {busy === item.id
                        ? tr('Starting...')
                        : !item.diskOk
                          ? tr('Not enough disk')
                          : tr('Download {0}', gb(item.needed + item.readerNeeded))}
                    </button>
                  )}
              </div>
            </div>
            </React.Fragment>
          ))}
        </div>

        <div className="store-foot">
          {tr('Files go to the models folder set in Settings, one subfolder per model. A stopped download resumes where it left off. Sneo checks each model\'s header before downloading and links parts it already has instead of downloading them twice.')}
        </div>
      </div>
    </div>
  );
}
