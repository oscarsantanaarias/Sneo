import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { tr } from './i18n.js';

const span = (seconds) => {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if(s < 60) return tr('{0} s', s);
  if(s < 3600) return tr('{0} min', Math.round(s / 60));
  return tr('{0} h {1} min', Math.floor(s / 3600), Math.round(s % 3600 / 60));
};

const thousands = (value) => Number(value || 0).toLocaleString();

const jobNames = {
  setup: 'Installing the training tools',
  collect: 'Collecting the code',
  teach: 'Lessons from the teacher',
  test: 'Quick test',
  train: 'Training',
  export: 'Exporting to Sneo'
};

export default function TrainView({apiUrl}) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const [gitUrl, setGitUrl] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [draft, setDraft] = useState(null);
  const logRef = useRef(null);

  const call = async (route, method = 'POST', body = null) => {
    setError('');
    try {
      const req = await fetch(`${apiUrl}/api/train/${route}`, {method, headers: {'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
      const res = await req.json();
      if(!req.ok) throw new Error(res.message || tr('The server answered {0}', req.status));
      await load();
      return res;
    } catch(e) {
      setError(e.message);
      return null;
    }
  };

  const load = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/train/state`);
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      setState(res);
      setDraft(prev => prev || res.settings);
    } catch(e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    const timer = setTimeout(load, state?.job?.running ? 1000 : 5000);
    return () => clearTimeout(timer);
  }, [state]);

  useEffect(() => {
    if(logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [state?.job?.log?.length]);

  if(!state || !draft){
    return <div className="view-page"><div className="view-head"><div><h1>{tr('Train')}</h1></div></div>{error && <div className="view-alert alert-bad">{error}</div>}</div>;
  }

  const {job, stats, estimate, ready, routes, baseModels, settings} = state;
  const busy = job.running;
  const change = (part, value) => setDraft(prev => ({...prev, [part]: typeof value === 'object' && !Array.isArray(value) ? {...prev[part], ...value} : value}));
  const save = async (next = draft) => {
    const res = await call('settings', 'POST', {baseModel: next.baseModel, teacher: next.teacher, train: next.train, export: next.export});
    if(res?.settings) setDraft(res.settings);
  };
  const pick = (part, value) => {
    const next = {...draft, [part]: typeof value === 'object' ? {...draft[part], ...value} : value};
    setDraft(next);
    save(next);
  };
  const teacher = routes.find(item => item.id === draft.teacher.routeId);
  const percent = job.total > 0 ? Math.min(100, job.done / job.total * 100) : 0;
  const lessonsUrl = `${apiUrl || window.location.origin}/api/train/lessons`;
  const example = `curl -X POST ${lessonsUrl} -H "x-sneo: 1" -H "Authorization: Bearer ${showKey ? settings.lessonsKey : '<key>'}" -H "Content-Type: application/json" -d "{\\"lessons\\":[{\\"question\\":\\"...\\",\\"answer\\":\\"...\\"}]}"`;

  return (
    <div className="view-page train-page">
      <div className="view-head">
        <div>
          <h1>{tr('Train')}</h1>
          <span>{tr('Teach a model to write code like you, with lessons from another model, then use it in Sneo.')}</span>
        </div>
        <div className="view-head-actions">
          {!ready.python && <button type="button" className="btn btn-primary" disabled={busy} onClick={() => call('setup')}>{tr('Install the training tools')}</button>}
        </div>
      </div>

      {error && <div className="view-alert alert-bad"><i className="fa-solid fa-triangle-exclamation"></i>{error}</div>}

      {(busy || job.kind) && (
        <section className="card train-job">
          <div className="card-head">
            <h3>{tr(jobNames[job.kind] || job.kind)}{busy ? '' : job.problem ? ` · ${tr(job.problem === 'Stopped' ? 'stopped' : 'failed')}` : ` · ${tr('done')}`}</h3>
            <div className="card-head-right">
              {busy && <span className="card-sub">{job.phase ? tr(job.phase) : ''}</span>}
              {busy && <button type="button" className="btn btn-ghost" onClick={() => call('stop')}>{tr('Stop')}</button>}
            </div>
          </div>
          {job.total > 0 && (
            <div className="train-progress">
              <div className="stats-meter"><div className="stats-meter-fill" style={{width: `${percent}%`}}></div></div>
              <span>
                {tr('{0} of {1}', thousands(job.done), thousands(job.total))}
                {busy && job.left > 0 ? ` · ${tr('{0} left', span(job.left))}` : ''}
                {job.loss !== null && job.loss !== undefined ? ` · ${tr('loss {0}', Number(job.loss).toFixed(3))}` : ''}
                {job.evalLoss !== null && job.evalLoss !== undefined ? ` · ${tr('check {0}', Number(job.evalLoss).toFixed(3))}` : ''}
              </span>
            </div>
          )}
          {!busy && job.problem && job.problem !== 'Stopped' && <p className="card-note train-bad">{job.problem}</p>}
          {!busy && job.result && <p className="card-note">{job.kind === 'export' ? tr('Saved to {0}. It shows in the model menu.', job.result) : tr('Saved to {0}', job.result)}</p>}
          {job.log.length > 0 && <pre ref={logRef} className="train-log">{job.log.join('\n')}</pre>}
        </section>
      )}

      <section className="card">
        <div className="card-head"><h3>{tr('1. What it learns from')}</h3><span className="card-sub">{tr('{0} files · {1} pieces · about {2} tokens', thousands(stats.files), thousands(stats.pieces), thousands(stats.tokens))}</span></div>
        <div className="train-sources">
          {settings.sources.length === 0 && <p className="card-note">{tr('Add the folders or git repositories with your code. Folders like node_modules, build and .git are skipped.')}</p>}
          {settings.sources.map(item => (
            <div key={item.id} className="train-source">
              <i className={item.kind === 'git' ? 'fa-brands fa-git-alt' : 'fa-solid fa-folder'}></i>
              <span>{item.kind === 'git' ? item.url : item.path}</span>
              <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call(`sources/${item.id}`, 'DELETE')}>{tr('Remove')}</button>
            </div>
          ))}
        </div>
        <div className="train-row">
          <button type="button" className="btn" disabled={busy} onClick={() => call('sources/folder')}>{tr('Add a folder...')}</button>
          <input value={gitUrl} placeholder="https://github.com/you/repo" onChange={(e) => setGitUrl(e.target.value)} />
          <button type="button" className="btn" disabled={busy || !gitUrl.trim()} onClick={async () => { if(await call('sources/git', 'POST', {url: gitUrl})) setGitUrl(''); }}>{tr('Add repository')}</button>
          <span className="train-spacer"></span>
          <button type="button" className="btn btn-primary" disabled={busy || !settings.sources.length} onClick={() => call('collect')}>{tr('Collect the code')}</button>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>{tr('2. Model to train')}</h3></div>
        <div className="segmented">
          {baseModels.map(item => (
            <button key={item.id} type="button" className={draft.baseModel === item.id ? 'active' : ''} disabled={busy} onClick={() => pick('baseModel', item.id)}>{item.label}</button>
          ))}
        </div>
        <p className="card-note">{tr('Sneo learns a small add-on for this model (LoRA) on your GPU and then writes it as a GGUF file you can load like any other model. 4B trains fastest and answers fastest; 27B learns more but trains much slower.')}</p>
      </section>

      <section className="card">
        <div className="card-head"><h3>{tr('3. Teacher')}</h3><span className="card-sub">{tr('{0} of {1} pieces have lessons', thousands(stats.taught), thousands(Math.min(stats.pieces, draft.teacher.maxPieces)))}</span></div>
        <div className="train-grid">
          <label className="field">
            <span className="field-label">{tr('Who teaches')}</span>
            <select value={draft.teacher.routeId} disabled={busy} onChange={(e) => pick('teacher', {routeId: e.target.value})}>
              <option value="">{tr('No teacher, only your code')}</option>
              {routes.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select>
            <em className="field-help">{tr('These are your routes from Agents, Models: add Gemini, Claude, OpenAI or another API there with its key, or use Sneo on this PC so your code never leaves it.')}</em>
          </label>
          <label className="field">
            <span className="field-label">{tr('Lessons per piece of code')}</span>
            <div className="segmented">
              {[0, 1, 2, 3].map(n => <button key={n} type="button" className={draft.teacher.lessonsPerPiece === n ? 'active' : ''} disabled={busy} onClick={() => pick('teacher', {lessonsPerPiece: n})}>{n}</button>)}
            </div>
            <em className="field-help">{tr('For each piece the teacher writes the request that gives exactly your code, plus questions and answers that explain it in your style.')}</em>
          </label>
          <label className="field">
            <span className="field-label">{tr('Language of the lessons')}</span>
            <input value={draft.teacher.language} disabled={busy} onChange={(e) => change('teacher', {language: e.target.value})} onBlur={() => save()} />
          </label>
          <label className="field">
            <span className="field-label">{tr('Most pieces to teach')}</span>
            <input type="number" min="1" value={draft.teacher.maxPieces} disabled={busy} onChange={(e) => change('teacher', {maxPieces: Math.max(1, Number(e.target.value) || 1)})} onBlur={() => save()} />
          </label>
        </div>
        <div className="train-row">
          <span className="card-note">{teacher ? tr('About {0} for the lessons that are missing.', span(estimate.teachSeconds)) : ''}</span>
          <span className="train-spacer"></span>
          {stats.taught > 0 && <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call('teacher', 'DELETE')}>{tr('Forget the lessons')}</button>}
          <button type="button" className="btn btn-primary" disabled={busy || !teacher || !stats.pieces} onClick={() => call('teach')}>{tr('Get the lessons')}</button>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>{tr('4. Lessons from other models')}</h3><span className="card-sub">{tr('{0} received', thousands(stats.sent))}</span></div>
        <p className="card-note">{tr('Any program or model on this PC can send lessons here, for example an agent that reviews your code. They are used in the next training.')}</p>
        <div className="train-api">
          <code>{lessonsUrl}</code>
          <code>{tr('Key')}: {showKey ? settings.lessonsKey : '••••••••••••••••'}</code>
          <pre className="train-log">{example}</pre>
        </div>
        <div className="train-row">
          <button type="button" className="btn btn-ghost" onClick={() => setShowKey(!showKey)}>{showKey ? tr('Hide the key') : tr('Show the key')}</button>
          <button type="button" className="btn btn-ghost" onClick={() => navigator.clipboard?.writeText(example.replace('<key>', settings.lessonsKey))}>{tr('Copy the example')}</button>
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call('lessons/key')}>{tr('New key')}</button>
          <span className="train-spacer"></span>
          {stats.sent > 0 && <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call('lessons', 'DELETE')}>{tr('Forget them')}</button>}
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>{tr('5. Train')}</h3><span className="card-sub">{tr('{0} examples · {1} tokens per round', thousands(stats.examples), thousands(stats.tokens))}</span></div>
        <div className="train-grid">
          <label className="field">
            <span className="field-label">{tr('Rounds over the data')}</span>
            <div className="segmented">
              {[1, 2, 3].map(n => <button key={n} type="button" className={draft.train.epochs === n ? 'active' : ''} disabled={busy} onClick={() => pick('train', {epochs: n})}>{n}</button>)}
            </div>
            <em className="field-help">{tr('2 is a good start. More rounds copy your style closer but can make it repeat your code instead of writing new code.')}</em>
          </label>
          <label className="field">
            <span className="field-label">{tr('How much it can change')}</span>
            <div className="segmented">
              {[16, 32, 64].map(n => <button key={n} type="button" className={draft.train.rank === n ? 'active' : ''} disabled={busy} onClick={() => pick('train', {rank: n, alpha: n * 2})}>{n === 16 ? tr('Light') : n === 32 ? tr('Normal') : tr('Strong')}</button>)}
            </div>
          </label>
        </div>
        <div className="train-estimate">
          <i className="fa-regular fa-clock"></i>
          <span>
            {tr('Training takes about {0}', span(estimate.trainSeconds))}
            {estimate.measured ? ` (${tr('measured on this PC: {0} tokens/s', thousands(Math.round(estimate.speed)))})` : ` (${tr('rough guess, the quick test measures this PC')})`}
            {teacher && estimate.teachSeconds > 0 ? ` · ${tr('lessons {0}', span(estimate.teachSeconds))}` : ''}
            {` · ${tr('export about {0}', span(estimate.exportSeconds))}`}
          </span>
        </div>
        <div className="train-row">
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call('free-gpu')}>{tr('Free the GPU')}</button>
          <span className="card-note">{tr('Training needs the GPU: this unloads the model from Sneo until you load it again.')}</span>
          <span className="train-spacer"></span>
          <button type="button" className="btn" disabled={busy || !ready.python || !stats.pieces} onClick={() => call('train', 'POST', {quick: true})}>{tr('Quick test')}</button>
          <button type="button" className="btn btn-primary" disabled={busy || !ready.python || !stats.pieces} onClick={() => call('train', 'POST', {quick: false})}>{tr('Train')}</button>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>{tr('6. Use it in Sneo')}</h3>{settings.lastAdapter && <span className="card-sub">{tr('last training ready to export')}</span>}</div>
        <div className="train-grid">
          <label className="field">
            <span className="field-label">{tr('Name')}</span>
            <input value={draft.export.name} disabled={busy} onChange={(e) => change('export', {name: e.target.value})} onBlur={() => save()} />
          </label>
          <label className="field">
            <span className="field-label">{tr('File type')}</span>
            <div className="segmented">
              {[['q8_0', 'Q8_0'], ['bf16', 'BF16']].map(([value, label]) => <button key={value} type="button" className={draft.export.type === value ? 'active' : ''} disabled={busy} onClick={() => pick('export', {type: value})}>{label}</button>)}
            </div>
            <em className="field-help">{tr('Q8_0 is half the size with the same answers in practice; BF16 keeps every bit.')}</em>
          </label>
          <label className="field">
            <span className="field-label">{tr('GGUF converter')}</span>
            <div className="train-row">
              <span className="card-note">{ready.converter ? settings.export.converter : tr('Pick the folder that has convert_hf_to_gguf.py')}</span>
              <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => call('export/converter')}>{tr('Choose...')}</button>
            </div>
          </label>
        </div>
        <div className="train-row">
          <span className="train-spacer"></span>
          <button type="button" className="btn btn-primary" disabled={busy || !settings.lastAdapter || !ready.converter} onClick={() => call('export')}>{tr('Export to the model menu')}</button>
        </div>
      </section>
    </div>
  );
}
