import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import PoolView from './PoolView.jsx';
import EngineView from './EngineView.jsx';
import AgentsView from './AgentsView.jsx';
import PhoneView from './PhoneView.jsx';
import ModelStore from './ModelStore.jsx';
import { FileIcon } from './fileIcons.jsx';
import CodeEditor from './CodeEditor.jsx';
import { tr, setLanguage } from './i18n.js';

const apiUrl = import.meta.env.DEV
  ? import.meta.env.VITE_API_URL || 'http://127.0.0.1:3001'
  : window.location.origin;

marked.setOptions({breaks: true, gfm: true});

marked.use({
  renderer: {
    code({text, lang}) {
      const language = String(lang || '').split(/\s/)[0].replace(/[^\w+#.-]/g, '') || 'text';
      const safe = String(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `<div class="code-block"><div class="code-head"><span>${language}</span><button type="button" class="code-copy">${tr('Copy')}</button></div><pre><code>${safe}</code></pre></div>`;
    }
  }
});

const depthLevels = [
  {value: 'off', label: 'Off'},
  {value: 'light', label: 'Light'},
  {value: 'normal', label: 'Normal'},
  {value: 'deep', label: 'Deep'},
  {value: 'full', label: 'Full'}
];

const depthCaps = [
  {value: 0, label: 'None'},
  {value: 2048, label: '2K'},
  {value: 8192, label: '8K'},
  {value: 32768, label: '32K'}
];

const contextSizes = [
  {value: 8192, label: '8K'},
  {value: 32768, label: '32K'},
  {value: 65536, label: '64K'},
  {value: 131072, label: '128K'},
  {value: 262144, label: '256K'}
];

const modelTitle = (path, list = []) => {
  const known = list.find(item => item.path === path);
  if(known?.shared) return known.name;
  const name = String(path || '').split(/[\\/]/).pop().replace(/\.gguf$/i, '').replace(/-\d{5}-of-\d{5}$/, '');
  return name.replace(/^Qwen3\.8-Flash-Next-/, '') || name;
};

const modelPlace = (path) => {
  return String(path || '').split(/[\\/]/).slice(-2, -1)[0] || '';
};

const accessModes = [
  {value: 'manual', label: 'Manual', icon: 'fa-solid fa-hand', hint: 'Asks before it changes files or runs commands'},
  {value: 'plan', label: 'Plan', icon: 'fa-solid fa-compass-drafting', hint: 'Reads and plans, changes nothing'},
  {value: 'auto', label: 'Auto', icon: 'fa-solid fa-bolt', hint: 'Works without asking'}
];

const pickMode = (value, fallback = 'manual') => {
  return value === 'ask'
    ? 'manual'
    : value === 'full'
      ? 'auto'
      : accessModes.some(item => item.value === value)
        ? value
        : fallback;
};

const railViews = [
  {value: 'workspace', label: 'Workspace', icon: 'fa-solid fa-code'},
  {value: 'agents', label: 'Agents', icon: 'fa-solid fa-diagram-project'},
  {value: 'pool', label: 'Pool', icon: 'fa-solid fa-network-wired'},
  {value: 'engine', label: 'Engine', icon: 'fa-solid fa-gauge-high'},
  {value: 'phone', label: 'Phone', icon: 'fa-solid fa-mobile-screen'}
];

const compact = (value) => {
  const n = Number(value || 0);

  return n >= 1000000
    ? `${(n / 1000000).toFixed(1)}M`
    : n >= 1000
      ? `${(n / 1000).toFixed(1)}K`
      : String(Math.round(n));
};

const gigabytes = (bytes) => `${(Number(bytes || 0) / 1073741824).toFixed(2)} GB`;

const seconds = (ms) => {
  const n = Number(ms || 0);

  return n >= 60000
    ? `${Math.floor(n / 60000)}m ${Math.round(n % 60000 / 1000)}s`
    : n >= 1000
      ? `${(n / 1000).toFixed(1)} s`
      : `${Math.round(n)} ms`;
};

const savedLanguage = () => {
  try {
    return localStorage.getItem('sneo.language') === 'es' ? 'es' : 'en';
  } catch(e) {
    return 'en';
  }
};

const renderMarkdown = (text) => {
  return {__html: DOMPurify.sanitize(marked.parse(String(text || '')))};
};

function renderFileBadge(file) {
  return (
    <span className="step-badge">
      <FileIcon name={file} />
    </span>
  );
}

const stepKinds = {
  Analyzed: 'read files',
  Explored: 'explored files',
  Searched: 'searched the code',
  Browsed: 'browsed the web',
  Edited: 'edited files',
  Created: 'edited files',
  Deleted: 'edited files',
  Renamed: 'edited files',
  Ran: 'ran commands',
  Executed: 'ran commands',
  Planned: 'made a plan',
  Checked: 'checked the projects',
  Ghidra: 'used Ghidra',
  IDA: 'used IDA'
};

const stepVerbs = {Executed: 'Ran', Step: 'Used'};

const liveVerbs = {
  Analyzed: 'Reading',
  Explored: 'Exploring',
  Searched: 'Searching',
  Browsed: 'Browsing',
  Edited: 'Editing',
  Created: 'Creating',
  Deleted: 'Deleting',
  Renamed: 'Renaming',
  Ran: 'Running',
  Executed: 'Running',
  Planned: 'Planning',
  Checked: 'Checking'
};

const spanText = seconds => seconds < 60
  ? `${seconds}s`
  : seconds % 60
    ? `${Math.floor(seconds / 60)}m ${seconds % 60}s`
    : `${Math.floor(seconds / 60)}m`;

const stepsTitle = steps => {
  const kinds = [];

  for(const st of steps){
    const kind = tr(stepKinds[st.action] || 'used tools');
    if(!kinds.includes(kind)) kinds.push(kind);
  }

  const text = kinds.join(', ');
  return text ? text[0].toUpperCase() + text.slice(1) : '';
};

// el servidor manda "files in <proyecto>" en ingles: se traduce aca con el nombre aparte
const stepLabel = label => /^files in /.test(String(label || '')) ? tr('files in {0}', String(label).slice(9)) : tr(label);

const liveStepText = st => `${tr(liveVerbs[st.action] || 'Using')} ${st.file || stepLabel(st.label) || ''}`.trim();

function ThoughtText({text, isLive}) {
  const [open, setOpen] = useState(isLive);
  const box = useRef(null);
  const stuck = useRef(true);

  useEffect(() => {
    if(isLive && open && stuck.current && box.current){
      box.current.scrollTop = box.current.scrollHeight;
    }
  }, [text, open, isLive]);

  return (
    <div className="work-block">
      <button type="button" className="work-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>{isLive ? tr('Thinking') : tr('Thought process')}</span>
        <i className={`fa-solid fa-chevron-right work-chevron ${open ? 'open' : ''}`}></i>
      </button>

      {open && (
        <div
          className="work-thought"
          ref={box}
          onScroll={(e) => {
            const el = e.currentTarget;
            stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}

function ThoughtTree({steps = [], thoughts = '', tools = [], duration = 0, isLive = false, status = '', onOpenFile = null}) {
  const [open, setOpen] = useState(() => isLive || steps.some(st => st.changes?.length));
  const [listOpen, setListOpen] = useState(true);
  const [toolsOpen, setToolsOpen] = useState(false);
  const [zoomed, setZoomed] = useState(null);
  const shown = steps.filter(st => st.action !== 'Executing');
  const calls = tools || [];

  if(!shown.length && !thoughts && !calls.length && !isLive){
    return null;
  }

  const running = shown.find(st => st.status === 'running');
  const title = isLive
    ? running ? liveStepText(running) : status || tr('Working')
    : duration > 0
      ? tr('Worked for {0}', spanText(duration))
      : shown.length
        ? stepsTitle(shown)
        : tr('Thought process');

  return (
    <div className={`work ${isLive ? 'work-live' : ''}`}>
      {zoomed && <ChangeWindow change={zoomed} onClose={() => setZoomed(null)} />}
      <button type="button" className="work-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={isLive ? 'work-title work-shimmer' : 'work-title'}>{title}</span>
        <i className={`fa-solid fa-chevron-right work-chevron ${open ? 'open' : ''}`}></i>
      </button>

      {open && (
        <div className="work-body">
          {thoughts && <ThoughtText text={thoughts} isLive={isLive} />}

          {shown.length > 0 && (
            <div className="work-block">
              <button type="button" className="work-toggle" onClick={() => setListOpen(!listOpen)} aria-expanded={listOpen}>
                <span>{stepsTitle(shown)}</span>
                <i className={`fa-solid fa-chevron-right work-chevron ${listOpen ? 'open' : ''}`}></i>
              </button>

              {listOpen && (
                <div className="work-steps work-timeline">
                  {shown.map((st, i) => (
                    <div key={i} className={`work-entry work-entry-${st.status || 'done'}`}>
                    <div className={st.status === 'running' ? 'work-step work-step-running' : 'work-step'}>
                      <span className="work-verb">{tr(stepVerbs[st.action] || st.action || 'Used')}</span>
                      {st.file && renderFileBadge(st.file)}
                      {st.file
                        ? (
                          <button
                            type="button"
                            className="work-file"
                            title={tr('Open {0} in the editor', st.file)}
                            onClick={(e) => {
                              e.stopPropagation();
                              if(onOpenFile) onOpenFile(st.file);
                            }}
                          >
                            {st.file}
                          </button>
                        )
                        : <span className={stepKinds[st.action] === 'ran commands' ? 'work-command' : 'work-target'}>{stepLabel(st.label)}</span>}
                      {st.lines && <span className="work-lines">{st.lines}</span>}
                      {(st.added > 0 || st.removed > 0) && (
                        <span className="work-diff">
                          {st.added > 0 && <span className="diff-plus">+{st.added}</span>}
                          {st.removed > 0 && <span className="diff-minus">-{st.removed}</span>}
                        </span>
                      )}
                    </div>
                    {(st.changes || []).map((change, k) => (
                      <div
                        key={k}
                        className="step-change"
                        role="button"
                        tabIndex={0}
                        title={tr('Open the changes')}
                        onClick={() => setZoomed({file: change.file || st.file, rows: change.rows || []})}
                        onKeyDown={(e) => e.key === 'Enter' && setZoomed({file: change.file || st.file, rows: change.rows || []})}
                      >
                        {st.changes.length > 1 && <div className="step-change-file">{change.file}</div>}
                        <SideBySide rows={change.rows || []} />
                      </div>
                    ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {isLive && !running && <div className="work-status work-shimmer">{status || tr('Working')}</div>}

          {calls.length > 0 && (
            <div className="work-block">
              <button type="button" className="work-toggle" onClick={() => setToolsOpen(!toolsOpen)} aria-expanded={toolsOpen}>
                <span>{tr('Tool details ({0})', calls.length)}</span>
                <i className={`fa-solid fa-chevron-right work-chevron ${toolsOpen ? 'open' : ''}`}></i>
              </button>

              {toolsOpen && (
                <div className="work-tools">
                  {calls.map((call, idx) => (
                    <div key={idx} className="tool-call-card">
                      <div className="tool-call-name">
                        <strong>{call.tool}</strong>
                      </div>
                      {call.args && (
                        <pre className="tool-args-pre">{JSON.stringify(call.args, null, 2)}</pre>
                      )}
                      {call.result && (
                        <pre className="tool-result-pre">{typeof call.result === 'object' ? JSON.stringify(call.result, null, 2).slice(0, 500) : String(call.result).slice(0, 500)}</pre>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const agoText = (when) => {
  const seconds = Math.max(0, (Date.now() - new Date(when || 0).getTime()) / 1000);
  return seconds < 60
    ? tr('now')
    : seconds < 3600
      ? `${Math.floor(seconds / 60)}m`
      : seconds < 86400
        ? `${Math.floor(seconds / 3600)}h`
        : seconds < 86400 * 30
          ? `${Math.floor(seconds / 86400)}d`
          : `${Math.floor(seconds / (86400 * 30))}mo`;
};

function ChatMessage({item, onOpenFile, canRunPlan = false, onRunPlan = null}) {
  const [changesOpen, setChangesOpen] = useState(true);
  const isUser = item.role === 'user';

  if(isUser){
    return (
      <div className="message-row message-row-user">
        <div className="user-bubble">
          {item.steer && <span className="steer-tag">{tr('steer')}</span>}
          {item.image && <img className="msg-image" src={item.image} alt={tr('attached')} />}
          {item.content}
        </div>
      </div>
    );
  }

  const edits = [];

  for(const st of item.partial ? [] : item.allSteps || item.steps || []){
    if(!st.file || st.status === 'running' || !(st.added > 0 || st.removed > 0 || st.action === 'Edited')){
      continue;
    }

    const target = st.args?.file || st.args?.path || st.file;
    const found = edits.find(entry => entry.target === target);

    if(found){
      found.added += Number(st.added || 0);
      found.removed += Number(st.removed || 0);
      continue;
    }

    edits.push({
      target,
      name: st.file,
      folder: target.includes('/') || target.includes('\\') ? target.replace(/[\\/][^\\/]*$/, '') : '',
      added: Number(st.added || 0),
      removed: Number(st.removed || 0)
    });
  }

  return (
    <div className="message-row message-row-agent">
      <div className="agent-line">
        <span className="avatar"><i className="fa-solid fa-cube"></i></span>
        <span className="agent-name">{tr('Agent')}</span>
      </div>

      <ThoughtTree
        steps={item.steps || []}
        thoughts={item.thoughts || ''}
        tools={item.tools || []}
        duration={item.duration || 0}
        isLive={false}
        onOpenFile={onOpenFile}
      />

      {item.content && <div className="agent-text markdown-body" dangerouslySetInnerHTML={renderMarkdown(edits.length ? item.content.replace(/\n+Changes:\n(- [^\n]*(\n|$))+\s*$/, '') : item.content)} />}

      {item.mode === 'plan' && canRunPlan && onRunPlan && (
        <div className="plan-run">
          <button type="button" className="plan-run-btn" onClick={onRunPlan}>
            <i className="fa-solid fa-play"></i>
            {tr('Run this plan')}
          </button>
          <span>{tr('Runs once in Auto: it writes files and runs commands without asking.')}</span>
        </div>
      )}

      {edits.length > 0 && (
        <div className="changes-card">
          <button type="button" className="changes-head" onClick={() => setChangesOpen(!changesOpen)} aria-expanded={changesOpen}>
            <span>{tr(edits.length === 1 ? '{0} file changed' : '{0} files changed', edits.length)}</span>
            <span className="diff-plus">+{edits.reduce((sum, entry) => sum + entry.added, 0)}</span>
            <span className="diff-minus">-{edits.reduce((sum, entry) => sum + entry.removed, 0)}</span>
            <i className={`fa-solid fa-chevron-right work-chevron ${changesOpen ? 'open' : ''}`}></i>
          </button>

          {changesOpen && (
            <div className="changes-list">
              {edits.map(entry => (
                <button
                  key={entry.target}
                  type="button"
                  className="changes-row"
                  title={tr('Open {0} beside the chat', entry.name)}
                  onClick={() => onOpenFile && onOpenFile(entry.target)}
                >
                  <FileIcon name={entry.name} />
                  <strong>{entry.name}</strong>
                  <span className="changes-folder">{entry.folder}</span>
                  <span className="changes-diff">
                    <span className="diff-plus">+{entry.added}</span>
                    <span className="diff-minus">-{entry.removed}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function lineDiff(before, after) {
  const a = before.split('\n');
  const b = after.split('\n');
  let head = 0;
  while(head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while(tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const midA = a.slice(head, a.length - tail);
  const midB = b.slice(head, b.length - tail);
  const rows = a.slice(0, head).map(text => ({kind: 'same', text}));

  if(midA.length * midB.length > 4000000){
    for(const text of midA) rows.push({kind: 'del', text});
    for(const text of midB) rows.push({kind: 'add', text});
  } else {
    const width = midB.length + 1;
    const table = new Uint32Array((midA.length + 1) * width);
    for(let i = midA.length - 1; i >= 0; i--){
      for(let j = midB.length - 1; j >= 0; j--){
        table[i * width + j] = midA[i] === midB[j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
      }
    }
    let i = 0;
    let j = 0;
    while(i < midA.length || j < midB.length){
      if(i < midA.length && j < midB.length && midA[i] === midB[j]){
        rows.push({kind: 'same', text: midA[i]});
        i++;
        j++;
      } else if(i < midA.length && (j >= midB.length || table[(i + 1) * width + j] >= table[i * width + j + 1])){
        rows.push({kind: 'del', text: midA[i]});
        i++;
      } else {
        rows.push({kind: 'add', text: midB[j]});
        j++;
      }
    }
  }

  for(const text of a.slice(a.length - tail)) rows.push({kind: 'same', text});
  return rows;
}

function ChangeWindow({change, onClose}) {
  useEffect(() => {
    const key = (e) => {
      if(e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose]);

  return createPortal(
    <div className="change-window" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="change-window-box">
        <div className="change-window-head">
          <span title={change.file}>{change.file}</span>
          <button type="button" className="icon-btn" title={tr('Close')} onClick={onClose}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
        <div className="change-window-body">
          <SideBySide rows={change.rows} />
        </div>
      </div>
    </div>,
    document.body
  );
}

function SideBySide({rows}) {
  const pairs = [];
  let i = 0;
  while(i < rows.length){
    const row = rows[i];
    if(row.kind === 'gap' || row.kind === 'more'){
      pairs.push({note: row.kind === 'gap'
        ? tr(row.count === 1 ? '··· {0} unchanged line' : '··· {0} unchanged lines', row.count)
        : tr('{0} more changed lines not shown', row.count)});
      i++;
      continue;
    }
    if(row.kind === 'same'){
      pairs.push({left: row.text, right: row.text});
      i++;
      continue;
    }
    const gone = [];
    const came = [];
    while(i < rows.length && rows[i].kind === 'del') gone.push(rows[i++].text);
    while(i < rows.length && rows[i].kind === 'add') came.push(rows[i++].text);
    for(let k = 0; k < Math.max(gone.length, came.length); k++) pairs.push({left: gone[k], right: came[k], changed: true});
  }

  const cell = (text, other, side, changed) => {
    if(text === undefined) return <div className="side-cell side-empty"></div>;
    if(!changed) return <div className="side-cell"><code>{text || ' '}</code></div>;
    if(other === undefined || !text) return <div className={`side-cell side-${side}`}><code>{text || ' '}</code></div>;
    let head = 0;
    while(head < text.length && head < other.length && text[head] === other[head]) head++;
    let tail = 0;
    while(tail < text.length - head && tail < other.length - head && text[text.length - 1 - tail] === other[other.length - 1 - tail]) tail++;
    return (
      <div className={`side-cell side-${side}`}>
        <code>{text.slice(0, head)}<mark>{text.slice(head, text.length - tail)}</mark>{text.slice(text.length - tail)}</code>
      </div>
    );
  };

  return (
    <div className="side-diff">
      {pairs.map((pair, index) => pair.note
        ? <div key={index} className="side-note">{pair.note}</div>
        : (
          <div key={index} className="side-row">
            {cell(pair.left, pair.right, 'del', pair.changed)}
            {cell(pair.right, pair.left, 'add', pair.changed)}
          </div>
        ))}
    </div>
  );
}

function ChangesView({before, after}) {
  const rows = lineDiff(before, after);
  const near = rows.map(() => false);
  rows.forEach((row, index) => {
    if(row.kind === 'same') return;
    for(let k = Math.max(0, index - 3); k <= Math.min(rows.length - 1, index + 3); k++) near[k] = true;
  });

  const shown = [];
  let skipped = 0;
  let line = 0;
  rows.forEach((row, index) => {
    if(row.kind !== 'del') line++;
    if(!near[index]){
      skipped++;
      return;
    }
    if(skipped){
      shown.push(<div key={`gap-${index}`} className="diff-gap">{tr(skipped === 1 ? '··· {0} unchanged line' : '··· {0} unchanged lines', skipped)}</div>);
      skipped = 0;
    }
    shown.push(
      <div key={index} className={`diff-line diff-${row.kind}`}>
        <span className="diff-number">{row.kind === 'del' ? '' : line}</span>
        <span className="diff-sign">{row.kind === 'add' ? '+' : row.kind === 'del' ? '-' : ' '}</span>
        <code>{row.text || ' '}</code>
      </div>
    );
  });
  if(skipped) shown.push(<div key="gap-end" className="diff-gap">{tr(skipped === 1 ? '··· {0} unchanged line' : '··· {0} unchanged lines', skipped)}</div>);

  return <div className="diff-view">{shown.length ? shown : <div className="diff-gap">{tr('No differences.')}</div>}</div>;
}

function FileItem({item, level = 0, onOpen, openFile, onContext}) {
  const folder = item.type === 'folder';
  const holds = folder && !!openFile && (openFile.startsWith(item.path + '/') || openFile.startsWith(item.path + '\\'));
  const [open, setOpen] = useState(holds);

  useEffect(() => {
    if(holds) setOpen(true);
  }, [holds]);

  return (
    <div>
      <button
        type="button"
        className={item.path === openFile ? 'tree-item active' : 'tree-item'}
        style={{paddingLeft: `${8 + level * 14}px`}}
        onClick={() => folder ? setOpen(!open) : onOpen(item.path)}
        onContextMenu={(e) => onContext(e, item)}
        aria-expanded={folder ? open : undefined}
      >
        <i className={folder ? `fa-solid fa-chevron-${open ? 'down' : 'right'} tree-chevron` : 'tree-chevron'}></i>
        <FileIcon name={item.name} folder={folder} open={folder && open} />
        <span>{item.name}</span>
      </button>

      {open && item.children?.map(child => (
        <FileItem
          key={child.path}
          item={child}
          level={level + 1}
          onOpen={onOpen}
          openFile={openFile}
          onContext={onContext}
        />
      ))}
    </div>
  );
}

function App(){
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState('');
  const [messages, setMessages] = useState([]);
  const [projects, setProjects] = useState([]);
  const [project, setProject] = useState('');
  const [projectPath, setProjectPath] = useState('');
  const [files, setFiles] = useState([]);
  const [openFile, setOpenFile] = useState('');
  const [fileBefore, setFileBefore] = useState(null);
  const [showChanges, setShowChanges] = useState(false);
  const [fileContent, setFileContent] = useState('');
  const [editorRequest, setEditorRequest] = useState(null);
  const [editorRefresh, setEditorRefresh] = useState(0);
  const [editorMax, setEditorMax] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [liveSteps, setLiveSteps] = useState([]);
  const [liveThoughts, setLiveThoughts] = useState('');
  const [liveStatus, setLiveStatus] = useState('');
  const [prefillLive, setPrefillLive] = useState(null);
  const [lastPrefill, setLastPrefill] = useState(null);
  const [elapsedSecs, setElapsedSecs] = useState(0);

  useEffect(() => {
    if(!loading){
      setElapsedSecs(0);
      return;
    }
    const timer = setInterval(() => {
      setElapsedSecs(s => s + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [loading]);
  const [image, setImage] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState(tr('Checking server...'));
  const [models, setModels] = useState([]);
  const [modelFolder, setModelFolder] = useState('');
  const [loadingModel, setLoadingModel] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [secrets, setSecrets] = useState({providerApiKey: null, imageApiKey: null});
  const [settings, setSettings] = useState({
    model: '',
    accessMode: 'manual',
    language: savedLanguage(),
    contextMessages: 16,
    maxToolLoops: 50,
    reverseTool: 'auto',
    ghidraPort: 5093,
    idaPort: 5073
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [storeOpen, setStoreOpen] = useState(false);
  const [engineFiles, setEngineFiles] = useState({list: [], current: '', reachable: false, ready: false, warming: false, prepare: null});
  const [pickedFile, setPickedFile] = useState('');
  const [fileBusy, setFileBusy] = useState('');
  const [memoryPlan, setMemoryPlan] = useState({self: '', screen: -1, slots: [], vramTotal: 0, ramTotal: 0, vram: 0, ram: 0, savedVram: 0, savedRam: 0, gpus: [], mainGpu: -1, savedMainGpu: -1, secondGpu: true, savedSecondGpu: true, helperGpu: -1, savedHelperGpu: -1, helperGb: 0, savedHelperGb: 0, drafts: 3, savedDrafts: 3, copyReach: 8, savedCopyReach: 8, holdRam: true, savedHoldRam: true, pinShare: 50, savedPinShare: 50, pinGb: 0, savedPinGb: 0, vision: false, savedVision: false, visionGpu: -1, savedVisionGpu: -1, visionReady: false, visionProblem: ''});
  const [recipeBusy, setRecipeBusy] = useState('');
  const [recipePlan, setRecipePlan] = useState(null);
  const [modelDrive, setModelDrive] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const [engineContext, setEngineContext] = useState(0);
  const [savedContext, setSavedContext] = useState(0);
  const [view, setView] = useState(() => {
    try {
      const saved = localStorage.getItem('sneo_view');
      return railViews.some(item => item.value === saved) ? saved : 'workspace';
    } catch(e) {
      return 'workspace';
    }
  });
  const [thinkFallback] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('sneo_think') || '{}') || {};
    } catch(e) {
      return {};
    }
  });
  const [popover, setPopover] = useState(null);
  const [liveUsage, setLiveUsage] = useState(null);
  const [liveSpeed, setLiveSpeed] = useState(null);
  const [brainUse, setBrainUse] = useState(null);
  const engineReads = useRef({at: 0, bytes: 0});
  const [editorView, setEditorView] = useState('code');
  const [sideUrl, setSideUrl] = useState('');
  const [sideLoad, setSideLoad] = useState(0);
  const [sideDrag, setSideDrag] = useState(false);
  const [sideShare, setSideShare] = useState(() => {
    try {
      const saved = Number(localStorage.getItem('sneo-side-share'));
      return saved >= 0.25 && saved <= 0.75 ? saved : 0.5;
    } catch(e) {
      return 0.5;
    }
  });
  const gridRef = useRef(null);
  const [loadedContent, setLoadedContent] = useState('');
  const [contextMenu, setContextMenu] = useState(null);
  const [pending, setPending] = useState([]);
  const [dialog, setDialog] = useState(null);
  const [queuePosition, setQueuePosition] = useState(0);
  const [runMode, setRunMode] = useState('');

  const bottomRef = useRef(null);
  const chatStuck = useRef(true);
  const autoScrolled = useRef(0);
  const messagesRef = useRef([]);
  const frozenSteps = useRef(0);
  const thoughtSkip = useRef(0);
  const thoughtShown = useRef(0);
  const textareaRef = useRef(null);
  const wasWarming = useRef(false);
  const imageInputRef = useRef(null);
  const dialogRef = useRef(null);
  setLanguage(settings.language);
  const isHtmlFile = /\.html?$/i.test(openFile);
  const activeChat = chats.find(item => item.id === activeChatId) || null;
  const thinkDepth = activeChat?.thinkDepth || settings.thinkDepth || thinkFallback.thinkDepth || 'normal';
  const thinkCap = Number(activeChat?.thinkCap ?? settings.thinkCap ?? thinkFallback.thinkCap ?? 0) || 0;
  const chatMode = pickMode(activeChat?.accessMode, pickMode(settings.accessMode));
  const shownMode = loading && runMode ? runMode : chatMode;
  const modelLoading = engineFiles.warming;
  const modelIdle = !engineFiles.ready && !engineFiles.warming;
  const loadPercent = modelLoading && engineFiles.load?.total > 0 ? Math.min(100, Math.floor(engineFiles.load.done / engineFiles.load.total * 100)) : null;
  const stoppingLoad = fileBusy === 'stop' || engineFiles.load?.step === 'Stopping';
  const loadStep = !modelLoading
    ? ''
    : stoppingLoad
      ? tr('Stopping the load...')
      : (engineFiles.load?.step ? tr(engineFiles.load.step) : tr('Loading into memory...')) + (loadPercent !== null ? ` ${loadPercent}%` : '');
  const preparingNow = engineFiles.ready && !!engineFiles.prepare?.task;
  const prepareStep = !preparingNow
    ? ''
    : engineFiles.prepare.task === 'pack'
      ? tr('Packing the experts for faster reads from disk: {0} of {1} layers', engineFiles.prepare.done, engineFiles.prepare.total)
      : engineFiles.prepare.task === 'draft'
        ? engineFiles.prepare.total > 0
          ? tr('Downloading the draft layer that makes writing faster: {0} of {1} GB', Number(engineFiles.prepare.done).toFixed(1), Number(engineFiles.prepare.total).toFixed(1))
          : tr('Downloading the draft layer that makes writing faster...')
        : tr('Building the draft layer that makes writing faster...');

  useEffect(() => {
    if(popover !== 'brain' || !engineFiles.ready){
      setBrainUse(null);
      engineReads.current = {at: 0, bytes: 0};
      return;
    }

    let alive = true;
    let timer = null;

    const tick = async () => {
      try {
        const req = await fetch(`${apiUrl}/api/engine/monitor`);
        const res = await req.json();

        if(alive && req.ok){
          const stats = ((res.nodes || []).find(node => node.self) || {}).stats || {};
          const now = Date.now();
          const read = Number(res.disk?.engineRead || 0);
          const before = engineReads.current;
          const rate = before.at && read >= before.bytes ? (read - before.bytes) / Math.max(0.2, (now - before.at) / 1000) : 0;
          engineReads.current = {at: now, bytes: read};
          setBrainUse({vram: Number(stats.heldGpu || 0), ram: Number(stats.heldRam || 0), disk: Number(res.disk?.fromDisk || 0), rate});
        }
      } catch(e) {
        console.log(e.message);
      }

      if(alive){
        timer = setTimeout(tick, 1000);
      }
    };

    tick();

    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [popover, engineFiles.ready]);
  const usages = messages.filter(item => item.role === 'assistant' && item.usage).map(item => item.usage);
  const lastUsage = (loading && liveUsage) || usages[usages.length - 1] || {};
  const writingSpeed = loading && liveSpeed ? liveSpeed.speed : Number(lastUsage.outputSpeed || 0);
  const ledger = [...usages, ...(loading && liveUsage ? [liveUsage] : [])].reduce((sum, item) => ({
    sent: sum.sent + Number(item.inputTokens || 0),
    received: sum.received + Number(item.outputTokens || 0),
    cached: sum.cached + Number(item.cachedTokens || 0)
  }), {sent: 0, received: 0, cached: 0});
  const windowSize = Number(lastUsage.contextLimit || 0) || 131072;
  const windowUsed = Number(lastUsage.lastPromptTokens || 0);
  const windowPercent = Math.min(100, windowUsed / windowSize * 100);

  const changeView = (next) => {
    setView(next);

    try {
      localStorage.setItem('sneo_view', next);
    } catch(e) {
      console.log(e.message);
    }
  };

  const changeDepth = async (patch) => {
    const next = {thinkDepth, thinkCap, ...patch};

    setChats(prev => prev.map(chat => chat.id === activeChatId ? {...chat, ...next} : chat));
    setSettings(prev => ({...prev, ...next}));

    try {
      localStorage.setItem('sneo_think', JSON.stringify(next));
    } catch(e) {
      console.log(e.message);
    }

    try {
      await fetch(`${apiUrl}/api/settings`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(next)
      });

      if(activeChatId){
        await fetch(`${apiUrl}/api/chats/${activeChatId}`, {
          method: 'PUT',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(next)
        });
      }
    } catch(e) {
      console.log(e.message);
    }
  };

  const changeMode = async (value) => {
    const next = pickMode(value);

    setChats(prev => prev.map(chat => chat.id === activeChatId ? {...chat, accessMode: next} : chat));
    setSettings(prev => ({...prev, accessMode: next}));
    setPopover(null);

    try {
      await fetch(`${apiUrl}/api/settings`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({accessMode: next})
      });

      if(loading && activeChatId){
        await fetch(`${apiUrl}/api/chat/mode`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({chatId: activeChatId, mode: next})
        });
        setRunMode(next);
      }

      if(activeChatId){
        await fetch(`${apiUrl}/api/chats/${activeChatId}`, {
          method: 'PUT',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({accessMode: next})
        });
      }
    } catch(e) {
      console.log(e.message);
    }
  };

  const openDialog = (options) => {
    return new Promise(resolve => {
      dialogRef.current = resolve;
      setDialog({
        value: options.defaultValue || '',
        ...options
      });
    });
  };

  const closeDialog = (value) => {
    const resolve = dialogRef.current;
    dialogRef.current = null;
    setDialog(null);

    if(resolve){
      resolve(value);
    }
  };

  const uiAlert = async (message) => {
    await openDialog({
      type: 'alert',
      title: 'SNEO',
      message
    });
  };

  const uiConfirm = async (message) => {
    return await openDialog({
      type: 'confirm',
      title: 'SNEO',
      message
    });
  };

  const uiPrompt = async (title, defaultValue = '') => {
    return await openDialog({
      type: 'prompt',
      title,
      message: title,
      defaultValue
    });
  };

  useEffect(() => {
    startApp();
  }, []);

  useEffect(() => {
    if(!prefillLive) return;
    const tick = () => setLiveStatus(tr('Reading the prompt · {0} new tokens · {1}', compact(prefillLive.fresh), seconds(Date.now() - prefillLive.started)));
    tick();
    const timer = setInterval(tick, 250);
    return () => clearInterval(timer);
  }, [prefillLive]);

  useEffect(() => {
    checkServer();
    try {
      localStorage.setItem('sneo.language', settings.language === 'es' ? 'es' : 'en');
    } catch(e) {
    }
  }, [settings.language]);

  useEffect(() => {
    const closeMenu = () => setContextMenu(null);
    window.addEventListener('click', closeMenu);
    return () => window.removeEventListener('click', closeMenu);
  }, []);

  useEffect(() => {
    if(!popover){
      return;
    }

    const onDown = (e) => {
      if(e.target.closest?.('.pop-anchor')){
        return;
      }
      setPopover(null);
    };

    const onKey = (e) => {
      if(e.key === 'Escape'){
        setPopover(null);
      }
    };

    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);

    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [popover]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if(!chatStuck.current) return;
    autoScrolled.current = Date.now();
    bottomRef.current?.scrollIntoView({behavior: loading ? 'auto' : 'smooth'});
  }, [messages, loading, liveSteps, liveStatus]);

  useEffect(() => {
    const found = projects.find(item => item.name === project);
    setProjectPath(found?.path || '');

    if(project){
      loadFiles(project);
    } else {
      setFiles([]);
    }
  }, [project, projects]);

  const startApp = async () => {
    await loadSettings();
    await checkServer();
    await loadModels();
    await loadEngineFiles();
    await loadProjects();
    await loadChats();
    await loadPending();
  };

  useEffect(() => {
    if(!engineFiles.reachable || !engineFiles.current) return;
    const name = engineFiles.current.split(/[\\/]/).pop();
    setStatus(engineFiles.ready
      ? `${tr('Ready with')} ${name}`
      : engineFiles.warming
        ? `${tr('Loading')} ${name}...`
        : `${name} ${tr('loads with the first message')}`);
  }, [engineFiles.current, engineFiles.ready, engineFiles.warming, engineFiles.reachable]);

  const checkServer = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/health`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setStatus(res.ready ? `${tr('Ready with')} ${res.model}` : tr('Server OK. Check model.'));

    } catch(e) {
      setStatus(tr('Backend off'));
    }
  };

  const loadSettings = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/settings`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setSettings(res.settings);

    } catch(e) {
      console.log(e.message);
    }
  };

  const saveSettings = async (nextSettings = settings) => {
    try {
      const req = await fetch(`${apiUrl}/api/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...nextSettings,
          allowNetwork: undefined,
          providerApiKey: secrets.providerApiKey ?? undefined,
          imageApiKey: secrets.imageApiKey ?? undefined
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setSettings(res.settings);
      setSecrets({providerApiKey: null, imageApiKey: null});
      await checkServer();

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const closeSettings = () => {
    setSettingsOpen(false);
    setSecrets({providerApiKey: null, imageApiKey: null});
  };

  const loadEngineContext = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/pool`);
      const res = await req.json();
      if(!req.ok) return;
      const size = Number(res.context || 0);
      setEngineContext(size);
      setSavedContext(size);
      const stats = (res.nodes || []).find(node => node.self)?.stats || {};
      const slots = Array.isArray(res.slots) ? res.slots : [];
      const ramTotal = Number(stats.ramTotal || 0) / 1073741824;
      const vramTotal = Number(stats.gpu?.vramTotal || stats.vramTotal || 0) / 1073741824;
      // un slot guardado sin gb ni porcentaje es usar todo lo que se pueda: se muestra el total y no lo que alcanzo hoy con otros programas abiertos
      const limitOf = (kind) => {
        const slot = slots.find(item => item.node === res.self && item.kind === kind);
        const total = kind === 'gpu' ? vramTotal : ramTotal;
        if(slot && !Number(slot.gb) && !Number(slot.percent) && !Number(slot.keep) && total > 0) return Math.floor(total * 10) / 10;
        return Math.round(Number(slot?.limit || 0) / 1073741824 * 10) / 10;
      };
      const vram = limitOf('gpu');
      const ram = limitOf('ram');
      const gpus = Array.isArray(res.gpus) ? res.gpus : [];
      const mainGpu = Number(res.mainGpu ?? -1);
      const secondGpu = res.secondGpu !== false;
      const helperGpu = Number(res.helperGpu ?? -1);
      const helperGb = Number(res.helperGb || 0);
      const drafts = res.speculate === false ? 0 : Number(res.draftLength || 3);
      const copyReach = Number(res.copyReach ?? 8);
      const holdRam = res.holdRam !== false;
      const ramWhole = res.ramWhole === true;
      const pinShare = Number(res.pinShare ?? 50);
      const pinGb = Number(res.pinGb || 0);
      const vision = res.vision === true;
      const visionGpu = Number(res.visionGpu ?? -1);
      setModelDrive(String(stats.drive || ''));
      setMemoryPlan({self: res.self || '', screen: Number(res.selfScreen ?? -1), slots, vramTotal, ramTotal, vram, ram, savedVram: vram, savedRam: ram, gpus, mainGpu, savedMainGpu: mainGpu, secondGpu, savedSecondGpu: secondGpu, helperGpu, savedHelperGpu: helperGpu, helperGb, savedHelperGb: helperGb, drafts, savedDrafts: drafts, copyReach, savedCopyReach: copyReach, holdRam, savedHoldRam: holdRam, ramWhole, savedRamWhole: ramWhole, pinShare, savedPinShare: pinShare, pinGb, savedPinGb: pinGb, vision, savedVision: vision, visionGpu, savedVisionGpu: visionGpu, visionReady: res.visionReady === true, visionProblem: String(res.visionProblem || '')});
    } catch(e) {
      setSavedContext(0);
    }
  };

  const saveEngineContext = async () => {
    const body = {};
    if(engineContext && engineContext !== savedContext) body.context = engineContext;
    const memoryMoved = !!memoryPlan.self && (memoryPlan.vram !== memoryPlan.savedVram || memoryPlan.ram !== memoryPlan.savedRam);
    if(memoryMoved){
      body.slots = memoryPlan.slots.map(item => item.node !== memoryPlan.self || item.kind === 'ssd'
        ? item
        : {...item, gb: item.kind === 'gpu' ? memoryPlan.vram : memoryPlan.ram, percent: 0, keep: 0});
    }
    const gpusMoved = memoryPlan.mainGpu !== memoryPlan.savedMainGpu || memoryPlan.secondGpu !== memoryPlan.savedSecondGpu || memoryPlan.helperGpu !== memoryPlan.savedHelperGpu || memoryPlan.helperGb !== memoryPlan.savedHelperGb;
    if(gpusMoved){
      body.mainGpu = memoryPlan.mainGpu;
      body.secondGpu = memoryPlan.secondGpu;
      body.helperGpu = memoryPlan.helperGpu;
      body.helperGb = memoryPlan.helperGb;
    }
    const draftsMoved = memoryPlan.drafts !== memoryPlan.savedDrafts;
    if(draftsMoved){
      body.speculate = memoryPlan.drafts > 0;
      if(memoryPlan.drafts > 0) body.draftLength = memoryPlan.drafts;
    }
    const copyMoved = memoryPlan.copyReach !== memoryPlan.savedCopyReach;
    if(copyMoved){
      body.copyReach = memoryPlan.copyReach;
    }
    const holdMoved = memoryPlan.holdRam !== memoryPlan.savedHoldRam;
    if(holdMoved){
      body.holdRam = memoryPlan.holdRam;
    }
    const wholeMoved = memoryPlan.ramWhole !== memoryPlan.savedRamWhole;
    if(wholeMoved){
      body.ramWhole = memoryPlan.ramWhole;
    }
    const pinMoved = memoryPlan.pinShare !== memoryPlan.savedPinShare || memoryPlan.pinGb !== memoryPlan.savedPinGb;
    if(pinMoved){
      body.pinShare = memoryPlan.pinShare;
      body.pinGb = memoryPlan.pinGb;
    }
    const visionMoved = memoryPlan.vision !== memoryPlan.savedVision || memoryPlan.visionGpu !== memoryPlan.savedVisionGpu;
    if(visionMoved){
      body.vision = memoryPlan.vision;
      body.visionGpu = memoryPlan.visionGpu;
    }
    if(!Object.keys(body).length) return;

    try {
      const req = await fetch(`${apiUrl}/api/engine/pool`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || res.error || tr('Engine answered {0}', req.status));
      }

      setSavedContext(Number(res.context || engineContext));
      if(memoryMoved) setMemoryPlan(prev => ({...prev, savedVram: prev.vram, savedRam: prev.ram}));
      if(gpusMoved) setMemoryPlan(prev => ({...prev, savedMainGpu: prev.mainGpu, savedSecondGpu: prev.secondGpu, savedHelperGpu: prev.helperGpu, savedHelperGb: prev.helperGb}));
      if(draftsMoved) setMemoryPlan(prev => ({...prev, savedDrafts: prev.drafts}));
      if(copyMoved) setMemoryPlan(prev => ({...prev, savedCopyReach: prev.copyReach}));
      if(holdMoved) setMemoryPlan(prev => ({...prev, savedHoldRam: prev.holdRam}));
      if(wholeMoved) setMemoryPlan(prev => ({...prev, savedRamWhole: prev.ramWhole}));
      if(pinMoved) setMemoryPlan(prev => ({...prev, savedPinShare: prev.pinShare, savedPinGb: prev.pinGb}));
      if(visionMoved) setMemoryPlan(prev => ({...prev, savedVision: prev.vision, savedVisionGpu: prev.visionGpu}));
    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const showEngineFiles = res => {
    const list = Array.isArray(res.models) ? res.models : [];
    const current = res.loaded ? String(res.currentModel || '') : '';
    setEngineFiles({list, current, reachable: true, ready: !!res.ready, warming: !!res.warming, problem: res.problem || '', load: res.load || null, prepare: res.prepare || null, healed: res.healed || null});
    if(res.folder) setModelFolder(res.folder);
    if(current) setSettings(prev => ({...prev, model: current.split(/[\\/]/).pop()}));
    setPickedFile(prev => list.some(item => item.path === prev)
      ? prev
      : current || list[0]?.path || '');
  };

  const loadEngineFiles = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/files`);
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      showEngineFiles(res);
    } catch(e) {
      setEngineFiles({list: [], current: '', reachable: false, ready: false, warming: false, prepare: null});
    }
  };

  const scanEngineFiles = async () => {
    setFileBusy('scan');
    try {
      const req = await fetch(`${apiUrl}/api/engine/files/scan`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dir: modelFolder})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      showEngineFiles(res);
      if(!res.models?.length) await uiAlert(tr('No models found in that folder.'));
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setFileBusy('');
    }
  };

  const browseModelFolder = async () => {
    setFileBusy('pick');
    try {
      const req = await fetch(`${apiUrl}/api/engine/files/pick`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dir: modelFolder})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      if(res.cancelled) return;
      showEngineFiles(res);
      if(!res.models?.length) await uiAlert(tr('No models found in that folder.'));
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setFileBusy('');
    }
  };

  const switchEngineFile = async (path = pickedFile) => {
    if(!path) return;
    setFileBusy('load');
    try {
      const req = await fetch(`${apiUrl}/api/engine/files/load`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      if(!res.success) throw new Error(res.error || tr('The engine could not open that model.'));
      setPickedFile(path);
      await loadEngineFiles();
      await uiAlert(tr('Model loading. The first answer can take about a minute while it fills memory.'));
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setFileBusy('');
    }
  };

  const stopEngineLoad = async () => {
    setFileBusy('stop');
    try {
      await fetch(`${apiUrl}/api/engine/files/stop`, {method: 'POST'});
      // la carga para al terminar el paso en curso asi que se espera a que el motor lo confirme en vez de soltar el boton a los 30 s
      for(let i = 0; i < 240; i++){
        await new Promise(done => setTimeout(done, 500));
        const req = await fetch(`${apiUrl}/api/engine/files`);
        const res = await req.json();
        showEngineFiles(res);
        if(!res.warming) break;
      }
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setFileBusy('');
    }
  };

  const freeEngineMemory = async () => {
    setFileBusy('free');
    try {
      await fetch(`${apiUrl}/api/models/unload`, {method: 'POST'});
      await loadEngineFiles();
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setFileBusy('');
    }
  };

  const recipeCall = async (route, body) => {
    const req = await fetch(`${apiUrl}/api/engine/recipe/${route}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})
    });
    const res = await req.json().catch(() => ({}));
    if(!req.ok) throw new Error(res.message || tr('Engine answered {0}', req.status));
    return res;
  };

  const exportRecipe = async () => {
    setRecipeBusy('export');
    try {
      const res = await recipeCall('export');
      if(res.saved) await uiAlert(tr('Recipe saved to {0}{1}', res.saved, res.heat ? tr(', with the heat map.') : tr('. It has no heat map yet: use the model for a while and export again to include one.')));
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setRecipeBusy('');
    }
  };

  const importRecipe = async () => {
    setRecipeBusy('import');
    try {
      const res = await recipeCall('pick');
      if(!res.cancelled) setRecipePlan(res);
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setRecipeBusy('');
    }
  };

  const applyRecipe = async () => {
    setRecipeBusy('apply');
    try {
      const res = await recipeCall('apply', {recipe: recipePlan.recipe});
      setRecipePlan(null);
      await loadEngineContext();
      await loadEngineFiles();
      await uiAlert(res.notes.join(' '));
    } catch(e) {
      await uiAlert(e.message);
    } finally {
      setRecipeBusy('');
    }
  };

  useEffect(() => {
    const timer = setTimeout(loadEngineFiles, loading ? 1500 : 300);
    return () => clearTimeout(timer);
  }, [loading]);

  useEffect(() => {
    if(!engineFiles.warming && !engineFiles.prepare?.task) return;
    const timer = setTimeout(loadEngineFiles, engineFiles.warming ? 1000 : 3000);
    return () => clearTimeout(timer);
  }, [engineFiles]);

  useEffect(() => {
    if(wasWarming.current && !engineFiles.warming) textareaRef.current?.focus();
    wasWarming.current = engineFiles.warming;
  }, [engineFiles.warming]);

  const openSettings = () => {
    setSettingsOpen(true);
    loadEngineContext();
    loadEngineFiles();
  };

  const scanModelFolder = async () => {
    setScanning(true);

    try {
      await fetch(`${apiUrl}/api/models/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dir: modelFolder })
      });
      await loadModels();
    } catch(e) {
      console.log(e.message);
    } finally {
      setScanning(false);
    }
  };

  const loadModelToEngine = async (modelPath, modelName) => {
    setLoadingModel(true);
    try {
      const req = await fetch(`${apiUrl}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: modelPath, name: modelName })
      });
      const res = await req.json();

      if(res.cargado){
        setSettings(prev => ({...prev, model: modelName}));
        await loadModels();
      }

      return res;
    } catch(e) {
      console.log(e.message);
      return {success: false, error: e.message};
    } finally {
      setLoadingModel(false);
    }
  };

  const loadModels = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/models`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setModels(res.models);
      setSettings(prev => ({
        ...prev,
        model: prev.model || res.currentModel
      }));
    } catch(e) {
      console.log(e.message);
    }
  };

  const loadChats = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/chats`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      if(res.chats.length === 0){
        await createChat();
        return;
      }

      setChats(prev => res.chats.map(chat => {
        const old = prev.find(item => item.id === chat.id);

        if(!old || chat.thinkDepth || !old.thinkDepth){
          return chat;
        }

        return {...chat, thinkDepth: old.thinkDepth, thinkCap: old.thinkCap};
      }));
      selectChat(res.chats[0]);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const loadPending = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/pending`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setPending(res.items);

    } catch(e) {
      console.log(e.message);
    }
  };

  const answerPending = async (id, action) => {
    try {
      const req = await fetch(`${apiUrl}/api/pending/${id}/${action}`, {
        method: 'POST'
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      await loadPending();

      if(project){
        await loadFiles(project);
      }

      setEditorRefresh(count => count + 1);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const loadProjects = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/projects`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      const cleanProjects = res.projects.map(item => typeof item === 'string' ? {
        name: item,
        path: ''
      } : item);

      setProjects(cleanProjects);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const loadFiles = async (projectName) => {
    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(projectName)}/files`);
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setFiles(res.files);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const createChat = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/chats`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: 'New chat'
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      setChats(prev => [res.chat, ...prev]);
      selectChat(res.chat);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const pickProjectFolder = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/projects/pick`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({start: projectPath})
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      if(res.cancelled){
        return;
      }

      await loadProjects();
      changeProject(res.project.name);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const selectChat = (chat) => {
    setActiveChatId(chat.id);
    setMessages(chat.messages || []);
    setProject(chat.project || '');
    setOpenFile('');
    setFileContent('');
  };

  const deleteChat = async (id) => {
    try {
      await fetch(`${apiUrl}/api/chats/${id}`, {method: 'DELETE'});

      if(id === activeChatId){
        setActiveChatId(null);
        setMessages([]);
      }

      await loadChats();
    } catch(e) {
      console.log(e.message);
    }
  };

  const saveChatLocal = async (nextMessages, nextProject = project) => {
    if(!activeChatId){
      return;
    }

    await fetch(`${apiUrl}/api/chats/${activeChatId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        messages: nextMessages,
        project: nextProject
      })
    });
  };

  const findFilePathInTree = (nodes, targetName) => {
    for(const n of nodes || []){
      if(n.type === 'file' && (n.name === targetName || n.path === targetName || n.path.endsWith('/' + targetName) || n.path.endsWith('\\' + targetName))){
        return n.path;
      }
      if(n.children){
        const sub = findFilePathInTree(n.children, targetName);
        if(sub) return sub;
      }
    }
    return null;
  };

  const openProjectFile = async (filePath, line = 0) => {
    const activeProj = project || (projects[0]?.name || '');
    if(!activeProj){
      return;
    }

    const resolvedPath = findFilePathInTree(files, filePath) || filePath;
    setOpenFile(resolvedPath);
    setSideUrl('');
    setEditorRequest({path: resolvedPath, line, nonce: Date.now()});
  };

  const closeOpenFile = () => {
    setOpenFile('');
    setEditorMax(false);
  };

  const createProjectFile = async (basePath = '') => {
    if(!project){
      return;
    }

    const file = await uiPrompt(tr('New file'), basePath ? `${basePath}/new-file.txt` : 'new-file.txt');

    if(!file){
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/file`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          file,
          content: ''
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      await loadFiles(project);
      await openProjectFile(file);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const createProjectFolder = async (basePath = '') => {
    if(!project){
      return;
    }

    const folder = await uiPrompt(tr('New folder'), basePath ? `${basePath}/new-folder` : 'new-folder');

    if(!folder){
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/folder`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          folder
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      await loadFiles(project);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const deleteProjectPath = async (target) => {
    if(!project || !target){
      return;
    }

    if(!await uiConfirm(`${tr('Delete')}: ${target}?`)){
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/path?path=${encodeURIComponent(target)}`, {
        method: 'DELETE'
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      await loadFiles(project);
      setEditorRefresh(count => count + 1);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const renameProjectPath = async (from) => {
    if(!project || !from){
      return;
    }

    const to = await uiPrompt(tr('Rename'), from);

    if(!to || to === from){
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/rename`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          from,
          to
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      await loadFiles(project);
      setEditorRefresh(count => count + 1);
      if(openFile === from) openProjectFile(to);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const openContextMenu = (e, item) => {
    e.preventDefault();
    e.stopPropagation();

    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      item
    });
  };

  const undoLastChange = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/undo`, {
        method: 'POST'
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      if(project){
        await loadFiles(project);
      }

      await loadPending();
      setEditorRefresh(count => count + 1);

      await uiAlert(res.message);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const runSandbox = async () => {
    if(!project){
      await uiAlert(tr('Select project'));
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/sandbox/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          project,
          command: ''
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      const content = `${tr('Sandbox OK')}\n\n${res.result.command}\n\n${res.result.stdout || res.result.stderr || res.result.sandbox}`;
      const nextMessages = [
        ...messages,
        {
          role: 'assistant',
          content
        }
      ];

      setMessages(nextMessages);
      await saveChatLocal(nextMessages);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const runAgents = async () => {
    if(!project){
      await uiAlert(tr('Select project'));
      return;
    }

    const task = await uiPrompt(tr('Agents'), text.trim() || `Review project ${project}`);

    if(!task){
      return;
    }

    try {
      const req = await fetch(`${apiUrl}/api/agents/parallel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          project,
          task
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message);
      }

      const content = `${tr('Subagents OK')}\n\n${res.results.map(item => `## ${item.name}\n${item.answer}`).join('\n\n')}`;
      const nextMessages = [
        ...messages,
        {
          role: 'assistant',
          content
        }
      ];

      setMessages(nextMessages);
      await saveChatLocal(nextMessages);

    } catch(e) {
      await uiAlert(e.message);
    }
  };

  const loadImageFile = (file) => {
    if(!file || !file.type.startsWith('image/')){
      return;
    }

    const reader = new FileReader();
    reader.onload = () => setImage(reader.result);
    reader.readAsDataURL(file);
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items || [];

    for(const item of items){
      if(item.type.startsWith('image/')){
        e.preventDefault();
        loadImageFile(item.getAsFile());
        return;
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];

    if(file){
      loadImageFile(file);
    }
  };

  const stopAgent = async () => {
    try {
      await fetch(`${apiUrl}/api/chat/stop`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({chatId: activeChatId})
      });
    } catch(e) {
      console.log(e.message);
    }
  };

  // lo que se escribe mientras trabaja tambien lleva la foto adjunta y puede ser solo la foto
  const steerMessage = async () => {
    const cleanText = text.trim();
    const sentImage = image;

    if((!cleanText && !sentImage) || engineFiles.warming || preparingNow){
      return;
    }

    setText('');
    setImage(null);
    const settled = liveSteps.filter(st => st.status !== 'running');
    const segment = settled.length || liveThoughts
      ? [{role: 'assistant', content: '', steps: settled, thoughts: liveThoughts, partial: true}]
      : [];
    frozenSteps.current += settled.length;
    thoughtSkip.current = thoughtShown.current;
    chatStuck.current = true;
    setMessages(prev => [...prev, ...segment, {role: 'user', content: cleanText, image: sentImage, steer: true}]);
    setLiveSteps(liveSteps.filter(st => st.status === 'running'));
    setLiveThoughts('');

    try {
      await fetch(`${apiUrl}/api/chat/steer`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({chatId: activeChatId, message: cleanText, image: sentImage})
      });
    } catch(e) {
      console.log(e.message);
    }
  };

  const sendMessage = async (override = null) => {
    const cleanText = String(override?.text ?? text).trim();
    const modeNow = pickMode(override?.mode, chatMode);

    if(engineFiles.warming || !engineFiles.ready || preparingNow){
      return;
    }

    if(loading){
      if(!override){
        await steerMessage();
      }

      return;
    }

    if(!cleanText && !image){
      return;
    }

    const sentImage = override ? null : image;

    const userMessage = {
      role: 'user',
      content: cleanText,
      image: sentImage
    };

    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);

    if(!override){
      setText('');
      setImage(null);
    }

    setLiveUsage(null);
    setLiveSpeed(null);
    setPopover(null);
    setQueuePosition(0);
    setRunMode(modeNow);
    setLoading(true);
    setLiveStatus('');
    setPrefillLive(null);
    frozenSteps.current = 0;
    thoughtSkip.current = 0;
    thoughtShown.current = 0;
    chatStuck.current = true;
    const startedAt = Date.now();
    let usage = null;
    let thoughtText = '';
    let thoughtDelta = false;

    try {
      const req = await fetch(`${apiUrl}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          chatId: activeChatId,
          message: cleanText,
          history: messages.filter(item => !item.partial),
          project,
          reverseTool: settings.reverseTool,
          image: sentImage,
          thinkDepth,
          thinkCap,
          accessMode: modeNow
        })
      });

      if(!req.ok || !req.body){
        const res = await req.json().catch(() => ({}));
        throw new Error(res.message || tr('Could not send the message'));
      }

      const reader = req.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let final = null;

      while(true){
        const {done, value} = await reader.read();

        if(done){
          break;
        }

        buffer += decoder.decode(value, {stream: true});
        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for(const part of parts){
          const line = part.split('\n').find(item => item.startsWith('data: '));

          if(!line){
            continue;
          }

          let event;

          try {
            event = JSON.parse(line.slice(6));
          } catch(err) {
            continue;
          }

          if(event.type === 'step' && event.step){
            setLiveStatus('');
            setLiveSteps(prev => {
              const idx = prev.findIndex(p => p.tool === event.step.tool && p.file === event.step.file && p.status === 'running');
              if(idx >= 0 && event.step.status !== 'running'){
                const copy = [...prev];
                copy[idx] = event.step;
                return copy;
              }
              return [...prev, event.step];
            });
          }

          if(event.type === 'tool' && !event.step && !/^[a-z0-9_]+$/.test(String(event.tool || ''))){
            const note = String(event.tool || '');
            setLiveStatus(note === 'generating response...'
              ? tr('Writing the reply')
              : note === 'thinking (processing results)...'
                ? tr('Reading the results')
                : note.replace(/\.+$/, '').replace(/^./, first => first.toUpperCase()));
          }

          if(event.type === 'thought' && event.thought){
            thoughtText = event.delta && thoughtDelta
              ? `${thoughtText}${event.thought}`
              : thoughtText
                ? `${thoughtText}\n\n${event.thought}`
                : event.thought;
            thoughtDelta = !!event.delta;
            thoughtShown.current = thoughtText.length;
            setLiveStatus(tr('Thinking'));
            setLiveThoughts(thoughtText.slice(thoughtSkip.current).replace(/^\s+/, ''));
          }

          if(event.type === 'prefill'){
            const fresh = Math.max(0, Number(event.tokens || 0) - Number(event.cached || 0));
            if(event.ms === null || event.ms === undefined){
              setPrefillLive({fresh, started: Date.now()});
            } else {
              const ms = Number(event.ms) || 0;
              const speed = ms > 0 ? fresh * 1000 / ms : 0;
              setPrefillLive(null);
              if(fresh >= 256) setLastPrefill({fresh, cached: Number(event.cached || 0), ms, speed});
              setLiveStatus(fresh > 0
                ? tr('Read {0} new tokens in {1} · {2} tok/s', compact(fresh), seconds(ms), compact(speed))
                : tr('The whole prompt came from the cache'));
            }
          }

          if(event.type === 'speed'){
            setLiveSpeed({tokens: Number(event.tokens) || 0, ms: Number(event.ms) || 0, speed: Number(event.speed) || 0});
          }

          if(event.type === 'writing'){
            const tool = String(event.tool || '').replace(/_/g, ' ');
            setLiveStatus(!tool
              ? tr('Writing a tool call · {0} tokens', compact(event.tokens))
              : event.file
                ? tr('Writing {0} for {1} · {2} tokens', tool, event.file, compact(event.tokens))
                : tr('Writing {0} · {1} tokens', tool, compact(event.tokens)));
          }

          if(event.type === 'queued'){
            setQueuePosition(Number(event.position) || 0);
          }

          if(event.type === 'started'){
            setQueuePosition(0);
          }

          if(event.type === 'approval' || event.type === 'approval-done'){
            loadPending();
          }

          if(event.type === 'files'){
            if(project){
              loadFiles(project);
            }

            if(event.project && event.project !== project){
              loadProjects();
            }
          }

          if(event.type === 'usage' && event.usage){
            usage = event.usage;
            setLiveUsage(event.usage);
          }

          if(event.type === 'done'){
            final = event;

            if(event.usage){
              usage = event.usage;
            }
          }

          if(event.type === 'error'){
            throw new Error(event.message);
          }
        }
      }

      if(!final){
        throw new Error(tr('The server cut the answer. Try again.'));
      }

      const allSteps = final.steps || [];
      const shownBefore = messagesRef.current.length >= nextMessages.length ? messagesRef.current : nextMessages;
      const finalMessages = [
        ...shownBefore,
        {
          role: 'assistant',
          content: final.message,
          tools: final.tools || [],
          thoughts: thoughtSkip.current
            ? thoughtText.slice(thoughtSkip.current).replace(/^\s+/, '')
            : final.thoughts || thoughtText,
          steps: allSteps.slice(Math.min(frozenSteps.current, allSteps.length)),
          ...(frozenSteps.current ? {allSteps} : {}),
          duration: Math.max(elapsedSecs, Math.round((Date.now() - startedAt) / 1000)),
          mode: final.mode || modeNow,
          ...(usage ? {usage} : {})
        }
      ];

      setMessages(finalMessages);
      await saveChatLocal(finalMessages);
      await loadChats();
      await loadPending();

      if(project){
        await loadFiles(project);
      }

    } catch(e) {
      const errorMessages = [
        ...(messagesRef.current.length >= nextMessages.length ? messagesRef.current : nextMessages),
        {
          role: 'assistant',
          content: e.message || tr('Could not send the message')
        }
      ];

      setMessages(errorMessages);
      await saveChatLocal(errorMessages);
    } finally {
      setLoading(false);
      setLiveSteps([]);
      setLiveThoughts('');
      setLiveStatus('');
      setPrefillLive(null);
      setLiveUsage(null);
      setLiveSpeed(null);
      setQueuePosition(0);
      setRunMode('');
      loadPending();
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  };

  const changeProject = async (value) => {
    setProject(value);
    await saveChatLocal(messages, value);
  };

  const changeSetting = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const changeLanguage = async (language) => {
    changeSetting('language', language);
    try {
      await fetch(`${apiUrl}/api/settings`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({language})
      });
    } catch(e) {
    }
  };

  const toggleHistory = () => setHistoryOpen(open => !open);

  const chatRow = (chat) => (
    <div key={chat.id} className={chat.id === activeChatId ? 'history-row active' : 'history-row'}>
      <button type="button" className="history-pick" title={chat.title} onClick={() => {
        selectChat(chat);
        setHistoryOpen(false);
      }}>
        <span className="history-title">{!chat.title || chat.title === 'New chat' ? tr('New chat') : chat.title}</span>
        <span className="history-ago">{agoText(chat.updatedAt || chat.createdAt)}</span>
      </button>
      <button type="button" className="history-del" title={tr('Delete chat')} onClick={() => deleteChat(chat.id)}>
        <i className="fa-solid fa-trash"></i>
      </button>
    </div>
  );

  const sortedChats = [...chats].sort((a, b) => new Date(b.updatedAt || b.createdAt || 0) - new Date(a.updatedAt || a.createdAt || 0));
  const chatProjects = [...new Set(sortedChats.map(chat => chat.project).filter(Boolean))];
  const looseChats = sortedChats.filter(chat => !chat.project);

  const historyPanel = historyOpen && (
    <aside className="history-panel" aria-label={tr('Chat history')}>
      <button type="button" className="history-new" onClick={() => {
        createChat();
        setHistoryOpen(false);
      }}>
        <i className="fa-solid fa-plus"></i>
        {tr('New conversation')}
      </button>

      <div className="history-scroll">
        {chatProjects.length > 0 && <div className="history-heading">{tr('Projects')}</div>}
        {chatProjects.map(name => (
          <div key={name} className="history-group">
            <div className="history-project">
              <i className="fa-regular fa-folder"></i>
              <span>{name}</span>
            </div>
            {sortedChats.filter(chat => chat.project === name).map(chatRow)}
          </div>
        ))}

        {looseChats.length > 0 && <div className="history-heading">{tr('Conversations')}</div>}
        {looseChats.map(chatRow)}

        {!sortedChats.length && <div className="history-empty">{tr('No conversations yet.')}</div>}
      </div>
    </aside>
  );

  const chatPanel = (
    <section className="agent-panel">
      {historyPanel}
      <div className="agent-main">
      <div className="panel-title">
        <div className="agent-head">
          <div className="agent-logo">{tr('AI')}</div>
          <div>
          <strong>{tr('Agent')}</strong>
            <span>• {tr('Online')}</span>
          </div>
        </div>
        <span className="agent-status">{status}</span>
        <div className="panel-tools">
          <button type="button" className={historyOpen ? 'active' : ''} title={historyOpen ? tr('Hide chat history') : tr('Show chat history')} onClick={toggleHistory}>
            <i className="fa-solid fa-clock-rotate-left"></i>
          </button>
          <button type="button" title={tr('New conversation')} onClick={createChat}>
            <i className="fa-solid fa-plus"></i>
          </button>
        </div>
      </div>

      <div
        className="messages"
        onScroll={(e) => {
          if(Date.now() - autoScrolled.current < 700) return;
          const el = e.currentTarget;
          chatStuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        onClick={(e) => {
          const link = e.target.closest?.('a[href]');

          if(link){
            e.preventDefault();
            const href = link.getAttribute('href') || '';

            if(/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?(\/|$)/i.test(href)){
              setSideUrl(href);
              setSideLoad(count => count + 1);
            } else if(/^https?:\/\//i.test(href)){
              window.open(href, '_blank', 'noopener');
            }

            return;
          }

          const button = e.target.closest?.('.code-copy');

          if(!button){
            return;
          }

          const code = button.closest('.code-block')?.querySelector('code')?.textContent || '';
          navigator.clipboard?.writeText(code).then(() => {
            button.textContent = tr('Copied');
            setTimeout(() => {
              button.textContent = tr('Copy');
            }, 1200);
          }).catch(() => {});
        }}
      >
        <div className="chat-column">
        {messages.map((item, index) => (
          <ChatMessage
            key={`${item.role}-${index}`}
            item={item}
            onOpenFile={openProjectFile}
            canRunPlan={!loading && index === messages.length - 1}
            onRunPlan={() => sendMessage({text: tr('Run the plan above, step by step.'), mode: 'auto'})}
          />
        ))}

        {loading && (
          <div className="message-row message-row-agent">
            <div className="agent-line">
              <span className="avatar"><i className="fa-solid fa-cube"></i></span>
              <span className="agent-name">{tr('Agent')}</span>
            </div>
            <ThoughtTree
              steps={liveSteps}
              thoughts={liveThoughts}
              tools={[]}
              duration={elapsedSecs}
              isLive={true}
              status={queuePosition > 0
                ? tr('Queued, position {0}', queuePosition)
                : liveStatus}
              onOpenFile={openProjectFile}
            />
          </div>
        )}

        {pending.length > 0 && (
          <div className="pending-list">
            {pending.map(item => (
              <div className={`pending-card pending-${item.type || 'patch'}`} key={item.id}>
                <div className="pending-head">
                  <i className={item.type === 'command'
                    ? 'fa-solid fa-terminal'
                    : item.type === 'action'
                      ? 'fa-solid fa-triangle-exclamation'
                      : 'fa-solid fa-pen-to-square'}></i>
                  <strong>
                    {item.type === 'command'
                      ? tr('Run this command?')
                      : item.type === 'action'
                        ? tr('Allow this change?')
                        : `${tr('Pending')}: ${tr(item.title)}`}
                  </strong>
                </div>
                {item.project && <span>{item.project}</span>}
                {item.type === 'command' && <pre className="pending-command">{item.command}</pre>}
                {item.type === 'command' && item.cwd && <em className="pending-where">{tr('in {0}', item.cwd)}</em>}
                {item.type === 'action' && <pre>{tr(item.title)}{item.details ? `\n${item.details}` : ''}</pre>}
                {item.files?.map(file => (
                  <pre key={file.file}>{file.file}{file.conflict ? tr('\nCONFLICT: {0}', file.reason) : ''}{file.diff ? `\n${file.diff}` : ''}</pre>
                ))}
                <div className="pending-actions">
                  <button type="button" onClick={() => answerPending(item.id, 'approve')}>{tr('Approve')}</button>
                  <button type="button" onClick={() => answerPending(item.id, 'deny')}>{tr('Deny')}</button>
                </div>
              </div>
            ))}
          </div>
        )}

        <div ref={bottomRef}></div>
        </div>
      </div>

      <div className="composer-dock">
      <div className="composer-column">
      {image && (
        <div className="image-preview">
          <img src={image} alt={tr('attached')} />
          <button type="button" onClick={() => setImage(null)} title={tr('Remove image')}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
      )}

      <div
        className={dragOver ? 'composer drag-over' : 'composer'}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <textarea
          ref={textareaRef}
          value={text}
          placeholder={modelIdle
            ? engineFiles.problem
              ? tr('The model did not load: {0}', engineFiles.problem)
              : engineFiles.current ? tr('Load model first') : tr('Pick a model first')
            : modelLoading
              ? engineFiles.healed && engineFiles.healed.seconds < 180
                ? tr('Windows moved {0} MB of the model out of video memory, Sneo loads it again to get the speed back...', engineFiles.healed.mb)
                : tr('{0} · you can write when it is ready', loadStep)
              : preparingNow
                ? tr('{0} · you can write when it finishes', prepareStep)
                : loading
                  ? tr('Type to steer it while it works...')
                  : tr('Ask the agent to read, edit, create, run commands...')}
          disabled={modelLoading || modelIdle || preparingNow}
          rows="2"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
        />

        <div className="composer-bar">
        <button
          type="button"
          className="attach-btn"
          title={tr('Attach image (or paste with Ctrl+V, or drop it here)')}
          onClick={() => imageInputRef.current?.click()}
        >
          <i className="fa-solid fa-image"></i>
        </button>

        <div className="pop-anchor">
          <button
            type="button"
            className={popover === 'context' ? 'stats-chip open' : 'stats-chip'}
            onClick={() => setPopover(popover === 'context' ? null : 'context')}
          >
            <svg className="ctx-ring" viewBox="0 0 32 32">
              <circle cx="16" cy="16" r="12" className="ring-track" />
              <circle
                cx="16"
                cy="16"
                r="12"
                className={windowPercent >= 85 ? 'ring-fill ring-full' : 'ring-fill'}
                strokeDasharray={`${(windowPercent / 100 * 75.4).toFixed(2)} 75.4`}
                transform="rotate(-90 16 16)"
              />
            </svg>
            <span>{windowPercent.toFixed(0)}%</span>
          </button>

          {popover === 'context' && (
            <div className="pop-menu window-menu">
              <div className="context-head">
                <strong className={windowPercent >= 85 ? 'full' : ''}>{windowPercent.toFixed(0)}%</strong>
                <span>{tr('of the context window in use')}</span>
              </div>
              <div className="stats-meter">
                <div className={windowPercent >= 85 ? 'stats-meter-fill full' : 'stats-meter-fill'} style={{width: `${windowPercent}%`}}></div>
              </div>
              <div className="stats-tiles context-tiles">
                <div>
                  <strong>{compact(windowUsed)}</strong>
                  <span>{tr('in use')}</span>
                </div>
                <div>
                  <strong>{compact(windowSize)}</strong>
                  <span>{tr('window size')}</span>
                </div>
                <div>
                  <strong>{compact(Math.max(0, windowSize - windowUsed))}</strong>
                  <span>{tr('free')}</span>
                </div>
              </div>
              <div className="stats-line">
                {windowPercent >= 85
                  ? tr('Almost full: older messages get summarized to make room.')
                  : tr('Older messages get summarized when the window fills up.')}
              </div>
            </div>
          )}
        </div>

        <div className="pop-anchor">
          <button
            type="button"
            className={popover === 'stats' ? 'stats-chip open' : 'stats-chip'}
            onClick={() => setPopover(popover === 'stats' ? null : 'stats')}
          >
            <i className="fa-solid fa-gauge-high"></i>
            <span>{tr('{0} tok/s', writingSpeed.toFixed(1))}</span>
            {lastPrefill && lastPrefill.fresh > 0 && (
              <span className="stats-prefill" title={tr('Prompt: {0} new tokens read in {1}', compact(lastPrefill.fresh), seconds(lastPrefill.ms))}>
                <i className="fa-solid fa-book-open"></i>
                {tr('{0} tok/s', compact(lastPrefill.speed))}
              </span>
            )}
          </button>

          {popover === 'stats' && (
            <div className="pop-menu stats-menu">
              <div className="stats-tiles">
                <div>
                  <strong>{writingSpeed.toFixed(1)}</strong>
                  <span>{tr('tok/s writing')}</span>
                </div>
                <div>
                  <strong>{lastPrefill && lastPrefill.fresh > 0 ? compact(lastPrefill.speed) : Number(lastUsage.promptSpeed || 0).toFixed(0)}</strong>
                  <span>{tr('tok/s reading')}</span>
                </div>
                <div>
                  <strong>{seconds(lastUsage.firstTokenMs)}</strong>
                  <span>{tr('to the first word')}</span>
                </div>
              </div>

              {loading && liveSpeed && (
                <div className="stats-line">{tr('Writing now: {0} tokens in {1}', compact(liveSpeed.tokens), seconds(liveSpeed.ms))}</div>
              )}

              {lastPrefill && (
                <div className="stats-line">
                  {lastPrefill.fresh > 0
                    ? tr('Last prompt: {0} new tokens read in {1}, {2} reused from the cache', compact(lastPrefill.fresh), seconds(lastPrefill.ms), compact(lastPrefill.cached))
                    : tr('Last prompt: all {0} tokens reused from the cache', compact(lastPrefill.cached))}
                </div>
              )}

              <div className="pop-section">{tr('Last reply')}</div>
              <div className="time-bar">
                <span className="time-think" style={{flexGrow: Number(lastUsage.thinkMs || 0)}}></span>
                <span className="time-write" style={{flexGrow: Number(lastUsage.writeMs || 0)}}></span>
                <span className="time-tool" style={{flexGrow: Number(lastUsage.toolMs || 0)}}></span>
              </div>
              <div className="time-legend">
                <span><i className="time-think"></i>{tr('Thinking {0}', seconds(lastUsage.thinkMs))}</span>
                <span><i className="time-write"></i>{tr('Writing {0}', seconds(lastUsage.writeMs))}</span>
                <span><i className="time-tool"></i>{tr('Tools {0}', seconds(lastUsage.toolMs))}</span>
              </div>
              <div className="stats-line">{tr('{0} model calls, {1} tool calls', Number(lastUsage.rounds || 0), Number(lastUsage.toolCalls || 0))}</div>

              <div className="pop-section">{tr('This chat')}</div>
              <div className="stats-line">
                {tr('{0} sent, {1} received, {2} reused from cache', compact(ledger.sent), compact(ledger.received), compact(ledger.cached))}
              </div>
              {loading && <div className="stats-live">{tr('live, updates after each model call')}</div>}
            </div>
          )}
        </div>

        <div className="pop-anchor">
          <button
            type="button"
            className={popover === 'brain' ? 'brain-chip open' : 'brain-chip'}
            onClick={() => {
              if(popover !== 'brain') loadEngineFiles();
              setPopover(popover === 'brain' ? null : 'brain');
            }}
          >
            <span className="brain-name">
              {fileBusy === 'load'
                ? tr('Loading...')
                : modelTitle(engineFiles.current, engineFiles.list) || tr('No model')}
            </span>
            <span className="brain-effort">{depthLevels.find(item => item.value === thinkDepth)?.label || tr('Normal')}</span>
            {engineFiles.current && !engineFiles.ready && (
              <span className={engineFiles.warming ? 'brain-state warming' : engineFiles.problem ? 'brain-state failed' : 'brain-state'} title={engineFiles.problem || undefined}>{!engineFiles.warming && engineFiles.problem ? tr('failed') : engineFiles.warming ? stoppingLoad ? tr('stopping') : loadPercent !== null ? tr('loading {0}%', loadPercent) : tr('loading') : tr('idle')}</span>
            )}
            {engineFiles.ready && engineFiles.prepare?.task && <span className="brain-state warming">{tr('preparing')}</span>}
            <i className="fa-solid fa-chevron-up depth-caret"></i>
          </button>

          {popover === 'brain' && (
            <div className="pop-menu brain-menu">
              <div className="pop-section">{tr('Model')}</div>
              <div className="brain-list">
                {engineFiles.list.map(item => (
                  <button
                    key={item.path}
                    type="button"
                    className={item.path === engineFiles.current ? 'brain-item active' : 'brain-item'}
                    disabled={!!fileBusy || loading || engineFiles.ready || engineFiles.warming}
                    title={engineFiles.ready || engineFiles.warming ? tr('Free the memory first to switch models') : ''}
                    onClick={() => switchEngineFile(item.path)}
                  >
                    <span className="brain-item-text">
                      <strong>{modelTitle(item.path, engineFiles.list)}</strong>
                      <em>{item.shared ? tr('From another app') : modelPlace(item.path)} {tr('· {0} GB', (item.size / 1073741824).toFixed(1))}</em>
                    </span>
                    {item.path === engineFiles.current && <i className="fa-solid fa-check"></i>}
                  </button>
                ))}
                {!!engineFiles.list.length && (engineFiles.ready || engineFiles.warming) && (
                  <div className="brain-empty">{tr('Free the memory to switch to another model.')}</div>
                )}
                {!engineFiles.list.length && (
                  <div className="brain-empty">
                    {engineFiles.reachable ? tr('No models in the folder set in Settings.') : tr('The engine is not answering.')}
                  </div>
                )}
              </div>

              <button type="button" className="brain-folder" disabled={!!fileBusy || loading} onClick={browseModelFolder}>
                <i className="fa-solid fa-folder-open"></i>
                <span>{fileBusy === 'pick' ? tr('Choosing a folder...') : tr('Change folder...')}</span>
                <em>{modelFolder}</em>
              </button>

              <button
                type="button"
                className="brain-folder"
                onClick={() => {
                  setPopover(null);
                  setStoreOpen(true);
                }}
              >
                <i className="fa-solid fa-cloud-arrow-down"></i>
                <span>{tr('Get models...')}</span>
                <em>{tr('Download a model that suits this PC')}</em>
              </button>

              <div className="brain-row">
                <span>{tr('Effort')} <em>{depthLevels.find(item => item.value === thinkDepth)?.label || tr('Normal')}</em></span>
                <div className="effort-steps">
                  {depthLevels.map((item, index) => (
                    <button
                      key={item.value}
                      type="button"
                      title={tr(item.label)}
                      className={item.value === thinkDepth
                        ? 'effort-step on knob'
                        : index < depthLevels.findIndex(level => level.value === thinkDepth)
                          ? 'effort-step on'
                          : 'effort-step'}
                      onClick={() => changeDepth({thinkDepth: item.value})}
                    ></button>
                  ))}
                </div>
              </div>

              <div className="brain-row">
                <span>{tr('Thinking cap')}</span>
                <div className="brain-caps">
                  {depthCaps.map(item => (
                    <button
                      key={item.value}
                      type="button"
                      className={thinkCap === item.value ? 'pop-cap active' : 'pop-cap'}
                      onClick={() => changeDepth({thinkCap: item.value})}
                    >
                      {tr(item.label)}
                    </button>
                  ))}
                </div>
              </div>

              {(engineFiles.prepare?.task || engineFiles.prepare?.problem || engineFiles.prepare?.note) && (
                <div className={engineFiles.prepare.problem && !engineFiles.prepare.task ? 'brain-prepare failed' : 'brain-prepare'}>
                  {engineFiles.prepare.task === 'pack'
                    ? tr('Packing the experts for faster reads from disk: {0} of {1} layers', engineFiles.prepare.done, engineFiles.prepare.total)
                    : engineFiles.prepare.task === 'draft'
                      ? engineFiles.prepare.total > 0
                        ? tr('Downloading the draft layer: {0} of {1} GB', Number(engineFiles.prepare.done).toFixed(1), Number(engineFiles.prepare.total).toFixed(1))
                        : tr('Downloading the draft layer...')
                      : engineFiles.prepare.task
                        ? tr('Building the draft layer...')
                        : engineFiles.prepare.problem || tr(engineFiles.prepare.note)}
                  {engineFiles.prepare.task && engineFiles.prepare.total > 0 && (
                    <div className="stats-meter">
                      <div className="stats-meter-fill" style={{width: `${Math.min(100, engineFiles.prepare.done / engineFiles.prepare.total * 100)}%`}}></div>
                    </div>
                  )}
                </div>
              )}

              {engineFiles.healed && (
                <div className="brain-prepare">
                  {tr('Windows moved {0} MB of the model out of video memory {1} ago; Sneo loaded it again so writing stays fast.', engineFiles.healed.mb, seconds(engineFiles.healed.seconds * 1000))}
                </div>
              )}

              {engineFiles.ready && brainUse && (
                <div className="brain-use">
                  <span title={tr('Exactly {0} bytes in video memory', brainUse.vram.toLocaleString())}>
                    <i className="fa-solid fa-layer-group"></i>{tr('VRAM')} <strong>{gigabytes(brainUse.vram)}</strong>
                  </span>
                  <span title={tr('Exactly {0} bytes of RAM', brainUse.ram.toLocaleString())}>
                    <i className="fa-solid fa-memory"></i>{tr('RAM')} <strong>{gigabytes(brainUse.ram)}</strong>
                  </span>
                  <span title={brainUse.disk > 0 ? tr('{0} bytes of the model stay on the disk and are read when needed', brainUse.disk.toLocaleString()) : tr('The whole model is in VRAM and RAM')}>
                    <i className="fa-solid fa-hard-drive"></i>{tr('Disk')} <strong>{brainUse.disk > 0 ? gigabytes(brainUse.disk) : tr('nothing')}</strong>
                    {brainUse.rate >= 1048576 && <em>{tr('reading {0} MB/s', (brainUse.rate / 1048576).toFixed(0))}</em>}
                  </span>
                </div>
              )}

              <div className="brain-row">
                <span className="brain-memory">
                  {engineFiles.ready
                    ? tr('In memory')
                    : engineFiles.warming
                      ? loadStep
                      : engineFiles.problem
                        ? tr('The model did not load: {0}', engineFiles.problem)
                        : tr('Not in memory, load it to start')}
                </span>
                {engineFiles.ready && (
                  <button type="button" className="brain-action" disabled={!!fileBusy || loading} onClick={freeEngineMemory}>
                    {fileBusy === 'free' ? tr('Freeing...') : tr('Free memory')}
                  </button>
                )}
                {!engineFiles.ready && !engineFiles.warming && engineFiles.current && (
                  <button type="button" className="brain-action" disabled={!!fileBusy} onClick={() => switchEngineFile(engineFiles.current)}>
                    {fileBusy === 'load' ? tr('Loading...') : tr('Load now')}
                  </button>
                )}
                {engineFiles.warming && (
                  <button type="button" className="brain-action" disabled={stoppingLoad} onClick={stopEngineLoad}>
                    {stoppingLoad ? tr('Stopping...') : tr('Stop loading')}
                  </button>
                )}
              </div>
              {engineFiles.warming && loadPercent !== null && !stoppingLoad && (
                <div className="stats-meter brain-load-meter">
                  <div className="stats-meter-fill" style={{width: `${loadPercent}%`}}></div>
                </div>
              )}

            </div>
          )}
        </div>

        <span className="composer-spacer"></span>

        <div className="pop-anchor">
          <button
            type="button"
            className={`mode-chip mode-${shownMode}${popover === 'mode' ? ' open' : ''}`}
            onClick={() => setPopover(popover === 'mode' ? null : 'mode')}
            title={accessModes.find(item => item.value === shownMode)?.hint}
          >
            <span className="mode-dot"></span>
            {accessModes.find(item => item.value === shownMode)?.label}
            {loading && runMode && runMode !== chatMode && <span className="mode-once">{tr('this run')}</span>}
            <i className="fa-solid fa-chevron-up depth-caret"></i>
          </button>

          {popover === 'mode' && (
            <div className="pop-menu mode-menu pop-right">
              <div className="pop-section">{tr('Permissions')}</div>
              {accessModes.map(item => (
                <button
                  key={item.value}
                  type="button"
                  className={chatMode === item.value ? `mode-item mode-${item.value} active` : `mode-item mode-${item.value}`}
                  onClick={() => changeMode(item.value)}
                >
                  <i className={item.icon}></i>
                  <span className="mode-item-text">
                    <strong>{tr(item.label)}</strong>
                    <em>{tr(item.hint)}</em>
                  </span>
                  {chatMode === item.value && <i className="fa-solid fa-check mode-item-check"></i>}
                </button>
              ))}
              <div className="mode-foot">{tr('Read-only commands never ask. Applies to the next message.')}</div>
            </div>
          )}
        </div>

        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          style={{display: 'none'}}
          onChange={(e) => {
            const file = e.target.files?.[0];

            if(!file){
              return;
            }

            const reader = new FileReader();
            reader.onload = () => setImage(reader.result);
            reader.readAsDataURL(file);
            e.target.value = '';
          }}
        />

        {loading ? (
          <div className="composer-actions">
            <button type="button" className="stop-btn" onClick={stopAgent} title={tr('Stop')}>
              <i className="fa-solid fa-stop"></i>
            </button>
            <button type="button" className="steer-btn" onClick={steerMessage} disabled={modelLoading || preparingNow || (!text.trim() && !image)} title={preparingNow ? tr('Sneo is preparing this model, wait until it finishes') : tr('Send while it works')}>
              <i className="fa-solid fa-arrow-up"></i>
            </button>
          </div>
        ) : modelIdle ? (
          <button
            type="button"
            className="load-btn"
            disabled={!!fileBusy || !engineFiles.reachable}
            onClick={() => engineFiles.current ? switchEngineFile(engineFiles.current) : setPopover('brain')}
            title={engineFiles.current ? tr('Load model first') : tr('Pick a model first')}
          >
            <i className="fa-solid fa-microchip"></i>
            <span>{fileBusy === 'load' ? tr('Loading...') : engineFiles.current ? tr('Load model') : tr('Pick a model')}</span>
          </button>
        ) : (
          <button
            type="button"
            className="send-btn"
            onClick={() => sendMessage()}
            disabled={modelLoading || preparingNow || (!text.trim() && !image)}
            title={modelLoading
              ? tr('The model is still loading')
              : preparingNow
                ? tr('Sneo is preparing this model, wait until it finishes')
                : tr('Send')}
          >
            <i className="fa-solid fa-arrow-up"></i>
          </button>
        )}
        </div>
      </div>
      </div>
      </div>
      </div>
    </section>
  );

  const dragSide = (e) => {
    const rect = gridRef.current?.getBoundingClientRect();

    if(!rect){
      return;
    }

    setSideShare(Math.min(0.75, Math.max(0.25, (rect.right - e.clientX) / (rect.width - 294))));
  };

  const sidePanel = (sideUrl || openFile) && (
    <>
      <div
        className="side-grip"
        title={tr('Drag to resize')}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setSideDrag(true);
        }}
        onPointerMove={(e) => sideDrag && dragSide(e)}
        onPointerUp={(e) => {
          e.currentTarget.releasePointerCapture(e.pointerId);
          setSideDrag(false);

          try {
            localStorage.setItem('sneo-side-share', String(sideShare));
          } catch(err) {
            console.log(err.message);
          }
        }}
      ></div>

      {sideUrl ? (
        <section className="editor-panel" aria-label={tr('Preview')}>
          <div className="editor-tabs">
            <button type="button" className="icon-btn" title={tr('Reload')} onClick={() => setSideLoad(count => count + 1)}>
              <i className="fa-solid fa-rotate-right"></i>
            </button>
            <input
              key={sideUrl}
              type="text"
              className="side-address"
              defaultValue={sideUrl}
              aria-label={tr('Address')}
              onKeyDown={(e) => {
                const next = e.currentTarget.value.trim();

                if(e.key === 'Enter' && /^https?:\/\//i.test(next)){
                  setSideUrl(next);
                  setSideLoad(count => count + 1);
                }
              }}
            />
            <button type="button" className="icon-btn" title={tr('Open in the browser')} onClick={() => window.open(sideUrl, '_blank', 'noopener')}>
              <i className="fa-solid fa-arrow-up-right-from-square"></i>
            </button>
            <button type="button" className="icon-btn" title={tr('Close')} onClick={() => setSideUrl('')}>
              <i className="fa-solid fa-xmark"></i>
            </button>
          </div>

          <iframe
            key={sideLoad}
            title={tr('preview')}
            className="preview-frame"
            sandbox="allow-scripts allow-forms allow-modals allow-same-origin allow-popups"
            src={sideUrl}
          />
        </section>
      ) : (
        <CodeEditor
          apiUrl={apiUrl}
          project={project || projects[0]?.name || ''}
          files={files}
          request={editorRequest}
          refreshKey={editorRefresh}
          onSaved={() => loadFiles(project)}
          onEmpty={closeOpenFile}
          onClose={closeOpenFile}
          maximized={editorMax}
          onMaximize={() => setEditorMax(value => !value)}
          alert={uiAlert}
          confirm={uiConfirm}
        />
      )}
    </>
  );

  return (
    <main className="app-shell">
      <nav className="side-rail">
        <div className="rail-logo" title={tr('SNEO')}>
          <i className="fa-solid fa-cube"></i>
        </div>

        {railViews.map(item => (
          <button
            key={item.value}
            type="button"
            className={view === item.value ? 'rail-btn active' : 'rail-btn'}
            title={tr(item.label)}
            onClick={() => changeView(item.value)}
          >
            <i className={item.icon}></i>
            <span>{tr(item.label)}</span>
          </button>
        ))}

        <span className="rail-spacer"></span>

        <button type="button" className="rail-btn" title={tr('Settings')} onClick={openSettings}>
          <i className="fa-solid fa-gear"></i>
          <span>{tr('Settings')}</span>
        </button>
      </nav>

      <div className="app-body">
      {view === 'pool' && (
        <PoolView
          apiUrl={apiUrl}
          models={models}
          currentModel={settings.model}
          modelFolder={modelFolder}
          scanning={scanning}
          loadingModel={loadingModel}
          onFolderChange={setModelFolder}
          onScan={scanModelFolder}
          onLoad={loadModelToEngine}
        />
      )}
      {view === 'engine' && <EngineView apiUrl={apiUrl} />}
      {view === 'phone' && <PhoneView apiUrl={apiUrl} />}
      <div className="agents-host" style={{display: view === 'agents' ? 'flex' : 'none'}}>
        <AgentsView apiUrl={apiUrl} projects={projects} project={project} />
      </div>

      <div className="workspace" style={{display: view === 'workspace' ? 'flex' : 'none'}}>

      <div
        ref={gridRef}
        className={(sideDrag ? 'main-grid side-dragging' : 'main-grid') + (editorMax && openFile && !sideUrl ? ' editor-max' : '')}
        style={{gridTemplateColumns: editorMax && openFile && !sideUrl
          ? '288px 0px 0px minmax(0, 1fr)'
          : sideUrl || openFile
            ? `288px minmax(0, ${1 - sideShare}fr) 6px minmax(0, ${sideShare}fr)`
            : '288px minmax(0, 1fr)'}}
      >
        <aside className="left-panel">
          <div className="project-card">
            <div className="section-row">
              <span>{tr('PROJECT')}</span>
            </div>

          <select
            value={project}
            onChange={(e) => changeProject(e.target.value)}
          >
            <option value="">{tr('Select project')}</option>
            {projects.map(item => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>

          <div className="current-project">
            <i className="fa-solid fa-folder"></i>
            <div>
              <strong>{project || tr('Select project')}</strong>
              <span>{projectPath || tr('Open a folder or select a project.')}</span>
            </div>
          </div>

          <button type="button" className="open-folder-btn" onClick={pickProjectFolder}>
            <i className="fa-regular fa-folder-open"></i>
            {tr('Open project folder')}
          </button>

          <div className="project-actions">
            <button type="button" onClick={undoLastChange}>{tr('Undo')}</button>
            <button type="button" onClick={runSandbox} disabled={!project}>{tr('Sandbox')}</button>
            <button type="button" onClick={runAgents} disabled={!project}>{tr('Agents')}</button>
          </div>
        </div>

        <div className="explorer-card">
          <div className="section-row">
            <span>{tr('EXPLORER')}</span>
            <div className="explorer-actions">
              <button type="button" onClick={() => createProjectFile()}>
                <i className="fa-solid fa-file-circle-plus"></i>
              </button>
              <button type="button" onClick={() => createProjectFolder()}>
                <i className="fa-solid fa-folder-plus"></i>
              </button>
              <button type="button" onClick={() => project && loadFiles(project)}>
                <i className="fa-solid fa-rotate"></i>
              </button>
            </div>
          </div>

          <div className="tree">
            {project ? (
              files.map(item => (
                <FileItem
                  key={item.path}
                  item={item}
                  onOpen={openProjectFile}
                  openFile={openFile}
                  onContext={openContextMenu}
                />
              ))
            ) : (
              <p>{tr('Open a folder or select a project.')}</p>
            )}
          </div>
        </div>

      </aside>

      {chatPanel}
      {sidePanel}
      </div>
      </div>
      </div>

      {storeOpen && <ModelStore apiUrl={apiUrl} onClose={() => setStoreOpen(false)} onReady={loadEngineFiles} />}

      {settingsOpen && (
        <div className="modal-bg">
          <div className="settings-modal" role="dialog" aria-label={tr('Settings')}>
            <div className="settings-head">
              <div>
                <h2>{tr('Settings')}</h2>
                <span>{tr('Changes are kept when you press Save')}</span>
              </div>
              <button type="button" className="icon-btn" title={tr('Close')} onClick={closeSettings}>
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>

            <div className="settings-body">
              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-language"></i>{tr('Language')}</h3>
                  <p>{tr('The whole app and the agent replies use this language.')}</p>
                </div>

                <div className="settings-grid">
                  <div className="field">
                    <span className="field-label">{tr('App language')}</span>
                    <div className="segmented">
                      <button type="button" className={settings.language === 'es' ? '' : 'active'} onClick={() => changeLanguage('en')}>English</button>
                      <button type="button" className={settings.language === 'es' ? 'active' : ''} onClick={() => changeLanguage('es')}>Español</button>
                    </div>
                    <em className="field-help">{tr('Changes right away, without pressing Save.')}</em>
                  </div>
                </div>
              </section>

              <div className="settings-note">
                <i className="fa-solid fa-network-wired"></i>
                <span>{tr('Models, machines and memory limits are set in the Pool view.')}</span>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => {
                    closeSettings();
                    changeView('pool');
                  }}
                >
                  {tr('Open Pool')}
                  <i className="fa-solid fa-arrow-right"></i>
                </button>
              </div>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-microchip"></i>{tr('Engine')}</h3>
                  <p>{tr('Which model runs and how much of the conversation it keeps.')}</p>
                </div>

                <div className="settings-grid">
                  <div className="field field-wide">
                    <span className="field-label">{tr('Models folder')}</span>
                    <div className="field-inline">
                      <input
                        type="text"
                        value={modelFolder}
                        disabled={!engineFiles.reachable || !!fileBusy}
                        onChange={e => setModelFolder(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && scanEngineFiles()}
                      />
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!engineFiles.reachable || !!fileBusy}
                        onClick={browseModelFolder}
                      >
                        {fileBusy === 'pick' ? tr('Choosing...') : tr('Browse')}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!engineFiles.reachable || !!fileBusy || !modelFolder.trim()}
                        onClick={scanEngineFiles}
                      >
                        {fileBusy === 'scan' ? tr('Scanning...') : tr('Scan')}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!engineFiles.reachable}
                        onClick={() => setStoreOpen(true)}
                      >
                        {tr('Get models')}
                      </button>
                    </div>
                    <em className={modelDrive === 'hdd' ? 'field-help field-warn' : 'field-help'}>
                      {modelDrive === 'hdd'
                        ? tr('This folder is on a hard disk (HDD). The experts that do not fit in video memory and RAM are read from the model files here, so answers will be much slower. Move the models to an SSD.')
                        : tr('Experts that do not fit in video memory and RAM are read straight from the model files in this folder, so keep it on your fastest SSD.')}
                    </em>
                  </div>

                  <div className="field field-wide">
                    <span className="field-label">{tr('Model')}</span>
                    <div className="field-inline">
                      <select
                        value={pickedFile}
                        disabled={!engineFiles.list.length || !!fileBusy}
                        onChange={e => setPickedFile(e.target.value)}
                      >
                        {!engineFiles.list.length && <option value="">{tr('No models found')}</option>}
                        {engineFiles.list.map(item => (
                          <option key={item.path} value={item.path}>
                            {tr('{0} ({1} GB)', item.name, (item.size / 1073741824).toFixed(1))}{item.path === engineFiles.current ? tr(', loaded') : ''}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className="btn btn-primary"
                        disabled={!pickedFile || pickedFile === engineFiles.current || !!fileBusy || engineFiles.ready || engineFiles.warming}
                        onClick={() => switchEngineFile()}
                      >
                        {fileBusy === 'load' ? tr('Loading...') : tr('Load')}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!!fileBusy || (!engineFiles.ready && !engineFiles.warming)}
                        onClick={freeEngineMemory}
                      >
                        {fileBusy === 'free' ? tr('Freeing memory...') : tr('Unload')}
                      </button>
                    </div>
                    <em className="field-help">
                      {!engineFiles.reachable
                        ? tr('The engine is not answering, start it to pick a model.')
                        : engineFiles.ready || engineFiles.warming
                          ? tr('{0} {1}. Press Unload to free the memory before loading another model.', engineFiles.warming ? tr('Loading') : tr('Running'), (engineFiles.current || '').split(/[\\/]/).pop())
                          : engineFiles.current
                          ? tr('Running {0}. The engine opens the last model you loaded when it starts.', engineFiles.current.split(/[\\/]/).pop())
                          : tr('No model is loaded yet. Pick one and press Load.')}
                    </em>
                  </div>

                  <div className="field field-wide">
                    <span className="field-label">{tr('Context window')}</span>
                    <div className="segmented">
                      {contextSizes.map(item => (
                        <button
                          key={item.value}
                          type="button"
                          className={engineContext === item.value ? 'active' : ''}
                          disabled={!savedContext}
                          onClick={() => setEngineContext(item.value)}
                        >
                          {tr(item.label)}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">
                      {!savedContext
                        ? tr('The engine is not answering, start it to change this.')
                        : engineContext === 8192
                          ? tr('8K only fits short chats: the agent instructions and tools alone take several thousand tokens, so agent tasks need 32K or more.')
                          : engineContext !== savedContext
                            ? tr('The model reloads with the new size when you press Save. More context takes more video memory and leaves less room for experts.')
                            : tr('Longest conversation the model keeps. 64K fits most chats, 128K helps agents that read many files.')}
                    </em>
                  </div>

                  {memoryPlan.gpus.length <= 1 && (
                  <div className="field">
                    <span className="field-label">{tr('Video memory')}</span>
                    <div className="field-inline">
                      <input
                        type="number"
                        step="0.5"
                        min="1"
                        max={memoryPlan.vramTotal.toFixed(1)}
                        value={memoryPlan.vram}
                        disabled={!memoryPlan.self}
                        onChange={(e) => setMemoryPlan(prev => ({...prev, vram: Math.min(prev.vramTotal, Math.max(0, Number(e.target.value) || 0))}))}
                      />
                      <span className="memory-unit">{tr('of {0} GB', memoryPlan.vramTotal.toFixed(1))}</span>
                    </div>
                    <div className="segmented screen-presets">
                      <button
                        type="button"
                        className={memoryPlan.vram >= Math.round(memoryPlan.vramTotal * 10) / 10 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, vram: Math.round(prev.vramTotal * 10) / 10}))}
                      >
                        {tr('Use all, no screen on this GPU')}
                      </button>
                      <button
                        type="button"
                        className={Math.abs(memoryPlan.vram - (Math.round(memoryPlan.vramTotal * 10) / 10 - 1)) < 0.05 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, vram: Math.max(1, Math.round(prev.vramTotal * 10) / 10 - 1)}))}
                      >
                        {tr('Leave 1 GB for the screen')}
                      </button>
                    </div>
                    <em className="field-help">
                      {memoryPlan.screen === 0
                        ? tr('No screen is running on this GPU, so it can give all its memory to the model.')
                        : memoryPlan.screen === 1
                          ? tr('This GPU shows your screen; leave about 1 GB so Windows stays smooth.')
                          : tr('More video memory holds more experts.')} {tr('The engine keeps a small margin of its own so it never runs out.')}
                    </em>
                  </div>
                  )}

                  {memoryPlan.gpus.length > 1 && (() => {
                    const gib = 1073741824;
                    const level = (gpu) => {
                      const [major, minor] = String(gpu.compute || '0.0').split('.').map(Number);
                      return (major || 0) * 10 + (minor || 0);
                    };
                    const mainOrdinal = memoryPlan.mainGpu >= 0 ? memoryPlan.mainGpu : Number(memoryPlan.gpus[0]?.ordinal ?? 0);
                    const autoHelper = memoryPlan.gpus.find(gpu => gpu.ordinal !== mainOrdinal && level(gpu) >= 61);
                    const helperOrdinal = !memoryPlan.secondGpu
                      ? -1
                      : memoryPlan.helperGpu >= 0 && memoryPlan.helperGpu !== mainOrdinal
                        ? memoryPlan.helperGpu
                        : Number(autoHelper?.ordinal ?? -1);
                    const pick = (gpu, role) => setMemoryPlan(prev => {
                      if(role === 'main'){
                        if(gpu.ordinal === mainOrdinal) return prev;
                        return {...prev, mainGpu: gpu.ordinal, vram: Math.min(prev.vram, Math.round(Number(gpu.vram || 0) / gib * 10) / 10), helperGpu: gpu.ordinal === helperOrdinal ? mainOrdinal : prev.helperGpu};
                      }
                      if(gpu.ordinal === mainOrdinal) return prev;
                      if(role === 'helper') return {...prev, secondGpu: true, helperGpu: gpu.ordinal};
                      return gpu.ordinal === helperOrdinal ? {...prev, secondGpu: false} : prev;
                    });
                    return (
                      <div className="field field-wide">
                        <span className="field-label">{tr('GPUs')}</span>
                        <div className="gpu-rows">
                          {memoryPlan.gpus.map(gpu => {
                            const total = Math.round(Number(gpu.vram || 0) / gib * 10) / 10;
                            const role = gpu.ordinal === mainOrdinal ? 'main' : gpu.ordinal === helperOrdinal ? 'helper' : 'off';
                            return (
                              <div key={gpu.ordinal} className="gpu-row">
                                <div className="gpu-row-name" title={gpu.name}>
                                  <i className="fa-solid fa-microchip"></i>
                                  <span>{gpu.name}</span>
                                  <em>{tr('{0} GB', total.toFixed(1))}</em>
                                </div>
                                <div className="segmented gpu-role">
                                  <button type="button" className={role === 'main' ? 'active' : ''} onClick={() => pick(gpu, 'main')}>{tr('Main')}</button>
                                  <button type="button" className={role === 'helper' ? 'active' : ''} disabled={role === 'main' || level(gpu) < 61} onClick={() => pick(gpu, 'helper')}>{tr('Helps with experts')}</button>
                                  <button type="button" className={role === 'off' ? 'active' : ''} disabled={role === 'main'} onClick={() => pick(gpu, 'off')}>{tr('Off')}</button>
                                </div>
                                <div className="field-inline">
                                  <input
                                    type="number"
                                    step="0.5"
                                    min="0"
                                    max={total.toFixed(1)}
                                    disabled={role === 'off' || !memoryPlan.self}
                                    value={role === 'main' ? Math.min(memoryPlan.vram, total) : role === 'helper' ? memoryPlan.helperGb : 0}
                                    onChange={(e) => {
                                      const gb = Math.min(total, Math.max(0, Number(e.target.value) || 0));
                                      setMemoryPlan(prev => role === 'main' ? {...prev, vram: gb} : {...prev, helperGb: gb});
                                    }}
                                  />
                                  <span className="memory-unit">{tr('of {0} GB', total.toFixed(1))}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        <em className="field-help">
                          {tr('The main GPU runs every layer and reads the long prompts, so it works best in the x16 slot')}{memoryPlan.screen === 1 ? tr('; it shows your screen, so leave about 1 GB for Windows') : ''}{tr('. The helper keeps the experts that do not fit in the main GPU and computes them beside the CPU; 0 GB lets it take all it can. Off leaves a GPU free for games or other apps. The model reloads when you press Save.')}
                        </em>
                      </div>
                    );
                  })()}

                  <div className="field">
                    <span className="field-label">{tr('RAM')}</span>
                    <div className="field-inline">
                      <input
                        type="number"
                        step="0.5"
                        min="2"
                        max={memoryPlan.ramTotal.toFixed(1)}
                        value={memoryPlan.ram}
                        disabled={!memoryPlan.self}
                        onChange={(e) => setMemoryPlan(prev => ({...prev, ram: Math.min(prev.ramTotal, Math.max(0, Number(e.target.value) || 0))}))}
                      />
                      <span className="memory-unit">{tr('of {0} GB', memoryPlan.ramTotal.toFixed(1))}</span>
                    </div>
                    <div className="segmented screen-presets">
                      <button
                        type="button"
                        className={memoryPlan.ram >= Math.floor(memoryPlan.ramTotal * 10) / 10 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, ram: Math.floor(prev.ramTotal * 10) / 10}))}
                      >
                        {tr('Use all it can')}
                      </button>
                      <button
                        type="button"
                        className={Math.abs(memoryPlan.ram - (Math.floor(memoryPlan.ramTotal * 10) / 10 - 6)) < 0.05 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, ram: Math.max(2, Math.floor(prev.ramTotal * 10) / 10 - 6)}))}
                      >
                        {tr('Leave 6 GB for apps')}
                      </button>
                    </div>
                    <em className="field-help">
                      {!memoryPlan.ramTotal
                        ? tr('The engine is not answering.')
                        : memoryPlan.ram >= Math.floor(memoryPlan.ramTotal * 10) / 10
                          ? tr('Sneo fills the RAM with experts and leaves about 0.5 GB free for Windows. Pick Leave 6 GB for apps if you run other heavy programs at the same time.')
                          : tr('Leaves {0} GB for Windows and your apps.', Math.max(0, memoryPlan.ramTotal - memoryPlan.ram).toFixed(1))} {tr('A loaded model reloads with the new limits when you press Save.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('When other apps need RAM')}</span>
                    <div className="segmented">
                      <button
                        type="button"
                        className={memoryPlan.holdRam ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, holdRam: true}))}
                      >
                        {tr('Hold the experts')}
                      </button>
                      <button
                        type="button"
                        className={memoryPlan.holdRam ? '' : 'active'}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, holdRam: false}))}
                      >
                        {tr('Give RAM back')}
                      </button>
                    </div>
                    <em className="field-help">
                      {memoryPlan.holdRam
                        ? tr('Sneo keeps the experts it loaded in RAM even when another program asks for memory. Answers stay fast when RAM is almost full, other programs may get slower.')
                        : tr('Sneo gives RAM back to Windows when another program needs it. Other programs stay smooth, answers slow down while RAM is full.')} {tr('Changes when you press Save, without reloading the model.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Model in RAM')}</span>
                    <div className="segmented">
                      <button
                        type="button"
                        className={memoryPlan.ramWhole ? '' : 'active'}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, ramWhole: false}))}
                      >
                        {tr('What the GPU needs')}
                      </button>
                      <button
                        type="button"
                        className={memoryPlan.ramWhole ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, ramWhole: true}))}
                      >
                        {tr('Whole model')}
                      </button>
                    </div>
                    <em className="field-help">
                      {memoryPlan.ramWhole
                        ? tr('Sneo also reads the rest of the model file into RAM after loading and holds it there, so long prompts never wait on the disk. It uses about as much RAM as the model file.')
                        : tr('Sneo keeps in RAM only what the graphics card needs and reads the rest of the model file from the disk when it is used. It uses far less RAM; Windows keeps the file in its cache when RAM is free.')} {tr('The model reloads when you press Save.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Pinned RAM')}</span>
                    <div className="segmented">
                      {[[50, 'Half'], [75, 'Three quarters'], [90, 'Most']].map(([share, label]) => (
                        <button
                          key={share}
                          type="button"
                          className={memoryPlan.pinGb <= 0 && memoryPlan.pinShare === share ? 'active' : ''}
                          disabled={!memoryPlan.self}
                          onClick={() => setMemoryPlan(prev => ({...prev, pinShare: share, pinGb: 0}))}
                        >
                          {tr(label)}
                        </button>
                      ))}
                      <button
                        type="button"
                        className={memoryPlan.pinGb > 0 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, pinGb: prev.pinGb > 0 ? prev.pinGb : Math.max(1, Math.floor(prev.ramTotal * prev.pinShare / 100 - 1))}))}
                      >
                        {tr('Exact')}
                      </button>
                    </div>
                    {memoryPlan.pinGb > 0 && (
                      <div className="field-inline">
                        <input
                          type="number"
                          step="1"
                          min="1"
                          max={Math.floor(memoryPlan.ramTotal)}
                          value={memoryPlan.pinGb}
                          disabled={!memoryPlan.self}
                          onChange={(e) => setMemoryPlan(prev => ({...prev, pinGb: Math.min(Math.floor(prev.ramTotal), Math.max(1, Math.round(Number(e.target.value) || 1)))}))}
                        />
                        <span className="memory-unit">{tr('GB of {0} GB', memoryPlan.ramTotal.toFixed(1))}</span>
                      </div>
                    )}
                    <em className="field-help">
                      {tr('Up to {0} GB of RAM is pinned for experts, never more than the RAM free when the model loads. Pinned experts reach the graphics card fastest, but Windows cannot use that RAM for other programs while the model is loaded. Half is the safe choice; more helps models bigger than your graphics memory when you do not run other heavy programs.', (memoryPlan.pinGb > 0 ? memoryPlan.pinGb : Math.max(0, memoryPlan.ramTotal * memoryPlan.pinShare / 100 - 1)).toFixed(0))} {tr('The model reloads when you press Save.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Vision')}</span>
                    <div className="segmented">
                      <button
                        type="button"
                        className={memoryPlan.vision ? '' : 'active'}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, vision: false}))}
                      >
                        {tr('Off')}
                      </button>
                      <button
                        type="button"
                        className={memoryPlan.vision && memoryPlan.visionGpu < 0 ? 'active' : ''}
                        disabled={!memoryPlan.self}
                        onClick={() => setMemoryPlan(prev => ({...prev, vision: true, visionGpu: -1}))}
                      >
                        {memoryPlan.gpus.length > 1 ? tr('Auto') : tr('On')}
                      </button>
                      {memoryPlan.gpus.length > 1 && memoryPlan.gpus.map(gpu => (
                        <button
                          key={gpu.ordinal}
                          type="button"
                          title={gpu.name}
                          className={memoryPlan.vision && memoryPlan.visionGpu === gpu.ordinal ? 'active' : ''}
                          disabled={!memoryPlan.self}
                          onClick={() => setMemoryPlan(prev => ({...prev, vision: true, visionGpu: gpu.ordinal}))}
                        >
                          {String(gpu.name || '').replace(/^NVIDIA (GeForce )?/, '')}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">
                      {!memoryPlan.vision
                        ? tr('Pictures you attach are not read.')
                        : memoryPlan.visionProblem
                          ? memoryPlan.visionProblem
                          : tr('The model looks at the pictures you attach with its own picture reader, the mmproj file next to the model. The reader takes about 1.1 GB of graphics memory on the card that runs it: on a second card it costs nothing while the model writes, on the main card it leaves fewer experts there, about 4% slower writing on an RTX 3090.{0}', memoryPlan.gpus.length > 1 && memoryPlan.visionGpu < 0 ? tr(' Auto uses the second card when there is one.') : '')} {tr('The model reloads when you press Save.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Drafts per step')}</span>
                    <div className="field-inline">
                      <input
                        id="drafts-per-step"
                        type="number"
                        min="0"
                        max="16"
                        step="1"
                        value={memoryPlan.drafts}
                        disabled={!memoryPlan.self}
                        onChange={(e) => setMemoryPlan(prev => ({...prev, drafts: Math.min(16, Math.max(0, Math.round(Number(e.target.value) || 0)))}))}
                      />
                      <span className="memory-unit">{memoryPlan.drafts ? tr('tokens guessed ahead') : tr('off')}</span>
                    </div>
                    <em className="field-help">
                      {memoryPlan.drafts
                        ? tr('A small draft model guesses {0} tokens ahead and the big one checks them in one pass.', memoryPlan.drafts)
                        : tr('Every token comes from the big model alone.')} {tr('Recommended: 3, at most 4. Each extra guess is right less often and every guess costs time to check. 0 turns it off. The model reloads when you press Save.')}
                    </em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Copy from the prompt')}</span>
                    <div className="segmented">
                      {[0, 8, 16, 32].map(count => (
                        <button
                          key={count}
                          type="button"
                          className={memoryPlan.copyReach === count ? 'active' : ''}
                          disabled={!memoryPlan.self}
                          onClick={() => setMemoryPlan(prev => ({...prev, copyReach: count}))}
                        >
                          {count ? tr('Up to {0}', count) : tr('Off')}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">
                      {memoryPlan.copyReach
                        ? tr('When the answer repeats text it has already seen, like code lines it quotes or a file it rewrites, Sneo proposes up to {0} tokens at once and checks them in one pass. It starts with 8 and doubles while the copy keeps matching.', memoryPlan.copyReach)
                        : tr('Sneo never proposes text copied from the prompt.')} {tr('The model reloads when you press Save.')}
                    </em>
                  </div>
                </div>
              </section>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-scroll"></i>{tr('Recipes')}</h3>
                  <p>{tr('Share the settings that run a model well on your hardware, or use someone else\'s.')}</p>
                </div>

                <div className="settings-grid">
                  <div className="field field-wide">
                    <div className="field-inline recipe-actions">
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!engineFiles.reachable || !!recipeBusy || !engineFiles.current}
                        onClick={exportRecipe}
                      >
                        <i className="fa-solid fa-file-export"></i>
                        {recipeBusy === 'export' ? tr('Saving...') : tr('Export recipe')}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        disabled={!engineFiles.reachable || !!recipeBusy}
                        onClick={importRecipe}
                      >
                        <i className="fa-solid fa-file-import"></i>
                        {recipeBusy === 'import' ? tr('Opening...') : tr('Import recipe')}
                      </button>
                    </div>
                    <em className="field-help">
                      {tr('A recipe keeps the model, your memory limits, context, drafts, routing and the heat map of the experts you use most. Machine names and addresses are left out. Importing adapts the memory limits to the PC that opens it.')}
                    </em>
                  </div>

                  {recipePlan && (
                    <div className="field field-wide recipe-plan">
                      <div className="recipe-plan-head">
                        <strong>{recipePlan.name}</strong>
                        {recipePlan.created && <span>{tr('made {0}', new Date(recipePlan.created).toLocaleDateString())}</span>}
                      </div>
                      <div className="recipe-rows">
                        <div className="recipe-row recipe-row-head">
                          <span></span>
                          <span>{tr('Recipe')}</span>
                          <span>{tr('This PC')}</span>
                        </div>
                        {recipePlan.rows.map(row => (
                          <div className="recipe-row" key={row.label}>
                            <span className="recipe-label">{row.label}</span>
                            <span>{row.recipe}</span>
                            <span>
                              {row.here}
                              {row.note && <em>{row.note}</em>}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="field-inline recipe-actions">
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={!!recipeBusy}
                          onClick={applyRecipe}
                        >
                          {recipeBusy === 'apply' ? tr('Applying...') : recipePlan.loads ? tr('Apply and reload') : tr('Apply')}
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          disabled={!!recipeBusy}
                          onClick={() => setRecipePlan(null)}
                        >
                          {tr('Cancel')}
                        </button>
                      </div>
                      <em className="field-help">
                        {recipePlan.loads
                          ? tr('The model unloads, the recipe is applied right away and the model loads again with it.')
                          : tr('The recipe is applied right away.')}
                      </em>
                    </div>
                  )}
                </div>
              </section>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-robot"></i>{tr('Agent')}</h3>
                  <p>{tr('How the agent works inside your projects.')}</p>
                </div>

                <div className="settings-grid">
                  <div className="field">
                    <span className="field-label">{tr('Default mode')}</span>
                    <div className="segmented">
                      {accessModes.map(item => (
                        <button
                          key={item.value}
                          type="button"
                          className={pickMode(settings.accessMode) === item.value ? 'active' : ''}
                          onClick={() => changeSetting('accessMode', item.value)}
                        >
                          {tr(item.label)}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">{tr('For new chats. Manual asks before changes, Plan only reads, Auto works without asking.')}</em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Default depth')}</span>
                    <div className="segmented">
                      {depthLevels.map(item => (
                        <button
                          key={item.value}
                          type="button"
                          className={(settings.thinkDepth || 'normal') === item.value ? 'active' : ''}
                          onClick={() => changeSetting('thinkDepth', item.value)}
                        >
                          {tr(item.label)}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">{tr('How much the model thinks in new chats.')}</em>
                  </div>

                  <div className="field">
                    <span className="field-label">{tr('Default cap')}</span>
                    <div className="segmented">
                      {depthCaps.map(item => (
                        <button
                          key={item.value}
                          type="button"
                          className={Number(settings.thinkCap || 0) === item.value ? 'active' : ''}
                          onClick={() => changeSetting('thinkCap', item.value)}
                        >
                          {tr(item.label)}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">{tr('Most thinking tokens per reply.')}</em>
                  </div>
                </div>
              </section>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-bug"></i>{tr('Reverse engineering')}</h3>
                  <p>{tr('Which disassembler bridge the agent talks to.')}</p>
                </div>

                <div className="settings-grid">
                  <div className="field field-wide">
                    <span className="field-label">{tr('Tool')}</span>
                    <div className="segmented">
                      {[{value: 'auto', label: 'Auto'}, {value: 'ghidra', label: 'Ghidra'}, {value: 'ida', label: 'IDA'}].map(item => (
                        <button
                          key={item.value}
                          type="button"
                          className={(settings.reverseTool ?? 'auto') === item.value ? 'active' : ''}
                          onClick={() => changeSetting('reverseTool', item.value)}
                        >
                          {tr(item.label)}
                        </button>
                      ))}
                    </div>
                    <em className="field-help">{tr('Auto uses whichever bridge answers.')}</em>
                  </div>

                  <label className="field">
                    <span className="field-label">{tr('Ghidra port')}</span>
                    <input
                      type="number"
                      min="1"
                      max="65535"
                      value={settings.ghidraPort ?? 5093}
                      onChange={(e) => changeSetting('ghidraPort', e.target.value)}
                    />
                  </label>

                  <label className="field">
                    <span className="field-label">{tr('IDA port')}</span>
                    <input
                      type="number"
                      min="1"
                      max="65535"
                      value={settings.idaPort ?? 5073}
                      onChange={(e) => changeSetting('idaPort', e.target.value)}
                    />
                  </label>
                </div>
              </section>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-plug"></i>{tr('Connection')}</h3>
                  <p>{tr('Where SNEO sends the model requests.')}</p>
                </div>

                <div className="settings-grid">
                  <label className="field">
                    <span className="field-label">{tr('Engine URL')}</span>
                    <input
                      value={settings.providerBaseUrl || ''}
                      placeholder={tr('http://localhost:8000/v1')}
                      onChange={(e) => changeSetting('providerBaseUrl', e.target.value)}
                    />
                    <em className="field-help">{tr('OpenAI style endpoint. Empty uses the native chat API on port 11434.')}</em>
                  </label>

                  <div className="field">
                    <span className="field-label">{tr('API key')}</span>
                    <div className="field-inline">
                      <input
                        type="password"
                        autoComplete="off"
                        value={secrets.providerApiKey ?? ''}
                        placeholder={settings.providerApiKey ? tr('Saved, type to replace') : tr('Not needed for Sneo')}
                        onChange={(e) => setSecrets(prev => ({...prev, providerApiKey: e.target.value}))}
                      />
                      {settings.providerApiKey && secrets.providerApiKey !== '' && (
                        <button type="button" className="btn btn-ghost" onClick={() => setSecrets(prev => ({...prev, providerApiKey: ''}))}>{tr('Remove')}</button>
                      )}
                    </div>
                    <em className="field-help">
                      {secrets.providerApiKey === '' && settings.providerApiKey
                        ? tr('The saved key will be removed on Save.')
                        : settings.providerApiKey
                          ? tr('A key is saved.')
                          : tr('Only for hosted providers.')}
                    </em>
                  </div>
                </div>
              </section>

              <section className="settings-section">
                <div className="settings-section-head">
                  <h3><i className="fa-solid fa-sliders"></i>{tr('Advanced')}</h3>
                  <p>{tr('Leave these empty to use the model defaults.')}</p>
                </div>

                <div className="settings-grid">
                  <label className="field">
                    <span className="field-label">{tr('Context window')}</span>
                    <input
                      type="number"
                      min="2048"
                      step="1024"
                      value={settings.numCtx ?? ''}
                      placeholder={tr('Model default')}
                      onChange={(e) => changeSetting('numCtx', e.target.value)}
                    />
                    <em className="field-help">{tr('Tokens the model can see at once.')}</em>
                  </label>

                  <label className="field">
                    <span className="field-label">{tr('Temperature')}</span>
                    <input
                      type="number"
                      min="0"
                      max="2"
                      step="0.05"
                      value={settings.temperature ?? ''}
                      placeholder={tr('Model default')}
                      onChange={(e) => changeSetting('temperature', e.target.value)}
                    />
                    <em className="field-help">{tr('0 is steady, 2 is very loose.')}</em>
                  </label>

                  {settings.imageBaseUrl !== undefined && (
                    <>
                      <label className="field">
                        <span className="field-label">{tr('Image endpoint')}</span>
                        <input
                          value={settings.imageBaseUrl || ''}
                          placeholder={tr('Uses the engine URL')}
                          onChange={(e) => changeSetting('imageBaseUrl', e.target.value)}
                        />
                        <em className="field-help">{tr('OpenAI style images endpoint.')}</em>
                      </label>

                      <label className="field">
                        <span className="field-label">{tr('Image model')}</span>
                        <input
                          value={settings.imageModel || ''}
                          placeholder={tr('dall-e-3')}
                          onChange={(e) => changeSetting('imageModel', e.target.value)}
                        />
                      </label>

                      <div className="field">
                        <span className="field-label">{tr('Image API key')}</span>
                        <div className="field-inline">
                          <input
                            type="password"
                            autoComplete="off"
                            value={secrets.imageApiKey ?? ''}
                            placeholder={settings.imageApiKey ? tr('Saved, type to replace') : tr('Uses the engine key')}
                            onChange={(e) => setSecrets(prev => ({...prev, imageApiKey: e.target.value}))}
                          />
                          {settings.imageApiKey && secrets.imageApiKey !== '' && (
                            <button type="button" className="btn btn-ghost" onClick={() => setSecrets(prev => ({...prev, imageApiKey: ''}))}>{tr('Remove')}</button>
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </section>
            </div>

            <div className="settings-foot">
              <span className="composer-spacer"></span>
              <button type="button" className="btn" onClick={closeSettings}>{tr('Cancel')}</button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={async () => {
                  await saveSettings(settings);
                  await saveEngineContext();
                  setSettingsOpen(false);
                }}
              >
                {tr('Save')}
              </button>
            </div>
          </div>
        </div>
      )}

      {dialog && (
        <div className="modal-bg">
          <div className="ui-dialog">
            <div className="modal-head">
              <h2>{tr(dialog.title)}</h2>
              <button type="button" onClick={() => closeDialog(dialog.type === 'confirm' ? false : null)}>
                {tr('X')}
              </button>
            </div>

            <p>{dialog.message}</p>

            {dialog.type === 'prompt' && (
              <input
                autoFocus
                value={dialog.value}
                onChange={(e) => setDialog(prev => ({
                  ...prev,
                  value: e.target.value
                }))}
                onKeyDown={(e) => {
                  if(e.key === 'Enter'){
                    closeDialog(dialog.value);
                  }
                }}
              />
            )}

            <div className="modal-actions">
              {dialog.type !== 'alert' && (
                <button type="button" onClick={() => closeDialog(dialog.type === 'confirm' ? false : null)}>
                  {tr('Deny')}
                </button>
              )}

              <button
                type="button"
                className="primary"
                onClick={() => closeDialog(dialog.type === 'prompt' ? dialog.value : true)}
              >
                {dialog.type === 'confirm' ? tr('Approve') : tr('OK')}
              </button>
            </div>
          </div>
        </div>
      )}

      {contextMenu && (
        <div
          className="context-menu"
          style={{left: contextMenu.x, top: contextMenu.y}}
          onClick={(e) => e.stopPropagation()}
        >
          <button type="button" onClick={() => createProjectFile(contextMenu.item.type === 'folder' ? contextMenu.item.path : '')}>
            {tr('New file')}
          </button>
          <button type="button" onClick={() => createProjectFolder(contextMenu.item.type === 'folder' ? contextMenu.item.path : '')}>
            {tr('New folder')}
          </button>
          <button type="button" onClick={() => renameProjectPath(contextMenu.item.path)}>
            {tr('Rename')}
          </button>
          <button type="button" className="danger" onClick={() => deleteProjectPath(contextMenu.item.path)}>
            {tr('Delete')}
          </button>
        </div>
      )}
    </main>
  );
}

export default App;
