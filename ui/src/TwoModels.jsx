import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n.js';

const contexts = [8192, 16384, 32768, 65536, 131072];
const gib = 1024 ** 3;

const cardName = card => String(card?.name || '').replace(/^NVIDIA (GeForce )?/, '');
const size = item => `${Math.round(Number(item?.size || 0) / gib)} GB`;
const modelName = item => String(item?.name || '').replace(/-\d{5}-of-\d{5}/i, '').replace(/\.gguf$/i, '');

// dos modelos a la vez: se eligen los dos y cuanto le toca a cada uno, se cargan juntos y para cambiarlos se descargan
export function SecondModel({apiUrl, onReady}){
  const [state, setState] = useState(null);
  const [models, setModels] = useState([]);
  const [gpus, setGpus] = useState([]);
  const [mainGpu, setMainGpu] = useState(0);
  const [mainModel, setMainModel] = useState('');
  const [first, setFirst] = useState('');
  const [secondPick, setSecondPick] = useState('');
  const [gpu, setGpu] = useState(0);
  const [firstGb, setFirstGb] = useState(15);
  const [context, setContext] = useState(32768);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const told = useRef(false);

  const read = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/engine/second`).then(answer => answer.json());
      setState(res);

      if(res.generation === 'ready' && !res.starting && !told.current){
        told.current = true;
        onReady?.();
      }

      if(res.generation !== 'ready'){
        told.current = false;
      }
    } catch(e) {
      setState(null);
    }
  };

  useEffect(() => {
    read();
    fetch(`${apiUrl}/api/engine/files`).then(answer => answer.json()).then(res => {
      setModels(res.models || []);
      setMainModel(res.currentModel || '');
      setFirst(prev => prev || res.currentModel || '');
    }).catch(() => {});
    fetch(`${apiUrl}/api/engine/pool`).then(answer => answer.json()).then(res => {
      const cards = res.gpus || [];
      const main = res.mainGpu >= 0 ? res.mainGpu : 0;
      setGpus(cards);
      setMainGpu(main);
      // con otra tarjeta el segundo modelo va ahi, asi el primero se queda con toda la suya
      const other = cards.find(card => card.ordinal !== main);
      setGpu(other ? other.ordinal : main);
      const total = Math.floor(Number(cards.find(card => card.ordinal === main)?.vram || 0) / gib);
      if(total >= 17) setFirstGb(Math.max(8, total - 9));
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if(state?.running && state.path){
      setSecondPick(state.path);
      if(state.first) setFirst(state.first);
      if(state.gpu !== undefined) setGpu(state.gpu);
    }
  }, [state?.running, state?.path]);

  const moving = Boolean(state?.starting) || state?.generation === 'loading';

  useEffect(() => {
    const timer = setInterval(read, moving ? 2000 : 10000);
    return () => clearInterval(timer);
  }, [moving]);

  const card = gpus.find(item => item.ordinal === mainGpu);
  const cardGb = Math.floor(Number(card?.vram || 0) / gib);
  const shared = gpu >= 0 && gpu === mainGpu;
  const secondGb = Math.max(0, cardGb - 1 - firstGb);
  const fits = !shared || cardGb >= 17;
  const running = Boolean(state?.running) && !moving;

  const start = async () => {
    if(!first || !secondPick){
      setNotice(tr('Pick both models first'));
      return;
    }

    setBusy(true);
    setNotice('');

    try {
      const req = await fetch(`${apiUrl}/api/engine/second/start`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({first: {path: first, gb: shared ? firstGb : 0}, second: {path: secondPick, gpu, gb: shared ? secondGb : 0, context}})
      });
      const res = await req.json().catch(() => ({}));

      if(!req.ok){
        throw new Error(res.message || tr('The two models could not start'));
      }

      await read();
    } catch(e) {
      setNotice(e.message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setNotice('');

    try {
      const res = await fetch(`${apiUrl}/api/engine/second/stop`, {method: 'POST'}).then(answer => answer.json());

      if(res.restored){
        setNotice(tr('Unloaded. The first model goes back to its own settings with the whole card.'));
      }

      await read();
    } catch(e) {
      setNotice(e.message);
    } finally {
      setBusy(false);
    }
  };

  const plan = state?.plan;
  const firstName = modelName(models.find(item => item.path === (state?.first || first))) || tr('the first model');
  const status = !state
    ? tr('The app server did not answer')
    : state.starting
      ? state.step || tr('Starting...')
      : state.generation === 'ready'
        ? tr('Both models are loaded: {0} and {1}', firstName, state.model)
        : state.generation === 'loading'
          ? tr('Loading {0}...', state.model || tr('the model'))
          : tr('Off');

  return (
    <section className="card">
      <div className="card-head">
        <h3>{tr('Two models at once')}</h3>
        <span className={`state-pill ${running ? 'pill-ok' : moving ? 'pill-busy' : 'pill-idle'}`}>
          <span className={moving ? 'state-dot dot-live' : 'state-dot'}></span>
          {running ? tr('ready') : moving ? tr('loading') : tr('off')}
        </span>
      </div>

      <span className="agents-hint agents-lead">{tr('Pick both models and how much video memory each one gets, then load them together. To change them, unload and load again. The ram and what each one keeps fixed in ram are shared out by themselves, the rest is read from the disk.')}</span>

      <label className="agents-field">
        <span>{tr('First model')}</span>
        <select value={first} disabled={running || moving} onChange={(e) => setFirst(e.target.value)}>
          <option value="">{tr('Pick a model...')}</option>
          {models.map(item => <option key={item.path} value={item.path}>{`${modelName(item)} · ${size(item)}${item.shared ? ` · ${tr('from another app')}` : ''}`}</option>)}
        </select>
      </label>

      <label className="agents-field">
        <span>{tr('Second model')}</span>
        <select value={secondPick} disabled={running || moving} onChange={(e) => setSecondPick(e.target.value)}>
          <option value="">{tr('Pick a model...')}</option>
          {models.map(item => <option key={item.path} value={item.path}>{`${modelName(item)} · ${size(item)}${item.shared ? ` · ${tr('from another app')}` : ''}`}</option>)}
        </select>
      </label>

      <div className="agents-row">
        <label className="agents-field">
          <span>{tr('The second model runs on')}</span>
          <select value={gpu} disabled={running || moving} onChange={(e) => setGpu(Number(e.target.value))}>
            {gpus.map(item => (
              <option key={item.ordinal} value={item.ordinal}>
                {`${cardName(item)} · ${Math.round(Number(item.vram || 0) / gib)} GB${item.ordinal === mainGpu ? ` · ${tr('same card as the first')}` : ''}`}
              </option>
            ))}
            <option value={-1}>{tr('Processor only (slow)')}</option>
          </select>
        </label>
        <label className="agents-field agents-count">
          <span>{tr('Context')}</span>
          <select value={context} disabled={running || moving} onChange={(e) => setContext(Number(e.target.value))}>
            {contexts.map(item => <option key={item} value={item}>{`${item / 1024}K`}</option>)}
          </select>
        </label>
      </div>

      {shared && fits && (
        <label className="agents-field">
          <span>{tr('Video memory of {0}', cardName(card))}</span>
          <div className="agents-row">
            <input type="range" min="8" max={Math.max(8, cardGb - 9)} step="1" value={firstGb} disabled={running || moving} onChange={(e) => setFirstGb(Number(e.target.value))} />
          </div>
          <div className="agents-row two-split">
            <strong>{tr('First {0} GB', firstGb)}</strong>
            <span className="rail-spacer"></span>
            <strong>{tr('Second {0} GB', secondGb)}</strong>
          </div>
          <span className="agents-hint">{tr('Each model gets at least 8 GB and 1 GB stays free for Windows.')}</span>
        </label>
      )}

      {shared && !fits && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {tr('That card has {0} GB, two models need at least 8 GB each on it. Pick another card or the processor.', cardGb)}</div>}

      {plan && (running || moving) && (
        <div className="two-plan">
          <div><strong>{tr('First')}</strong><span>{plan.firstVram ? tr('{0} GB video', plan.firstVram) : tr('its whole card')} · {tr('{0} GB fixed in ram', plan.pinFirst)} · {tr('{0} GB ram', plan.ramFirst)}</span></div>
          <div><strong>{tr('Second')}</strong><span>{plan.secondVram ? tr('{0} GB video', plan.secondVram) : gpu < 0 ? tr('processor') : tr('its whole card')} · {tr('{0} GB fixed in ram', plan.pinSecond)} · {tr('{0} GB ram', plan.ramSecond)}</span></div>
        </div>
      )}

      <div className="agents-status">{status}</div>
      {(state?.error || notice) && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {state?.error || notice}</div>}
      {(moving || state?.error) && state?.log?.length > 0 && <pre className="agents-logtail">{state.log.join('\n')}</pre>}

      <div className="agents-row agents-actions">
        <span className="rail-spacer"></span>
        {running || moving
          ? <button type="button" className="agents-small agents-danger" disabled={busy} onClick={stop}><i className="fa-solid fa-eject"></i> {tr('Unload the second model')}</button>
          : <button type="button" className="agents-run" disabled={busy || !first || !secondPick || !fits} onClick={start}><i className="fa-solid fa-layer-group"></i> {tr('Load both models')}</button>}
      </div>
    </section>
  );
}
