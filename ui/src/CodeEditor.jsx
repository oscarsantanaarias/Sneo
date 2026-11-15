import {useEffect, useRef, useState} from 'react';
import {tr} from './i18n.js';
import * as monaco from 'monaco-editor';
import editorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import jsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import cssWorker from 'monaco-editor/esm/vs/language/css/css.worker?worker';
import htmlWorker from 'monaco-editor/esm/vs/language/html/html.worker?worker';
import tsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker?worker';

self.MonacoEnvironment = {
  getWorker(_, label) {
    if(label === 'json') return new jsonWorker();
    if(['css', 'scss', 'less'].includes(label)) return new cssWorker();
    if(['html', 'handlebars', 'razor'].includes(label)) return new htmlWorker();
    if(['typescript', 'javascript'].includes(label)) return new tsWorker();
    return new editorWorker();
  }
};

monaco.editor.defineTheme('sneo-dark', {
  base: 'vs-dark',
  inherit: true,
  rules: [],
  colors: {
    'editor.background': '#131316',
    'editor.lineHighlightBackground': '#1a1a1e',
    'editor.lineHighlightBorder': '#1a1a1e',
    'editorLineNumber.foreground': '#6b6b74',
    'editorLineNumber.activeForeground': '#eae8e4',
    'editorCursor.foreground': '#f0a63c',
    'editor.selectionBackground': '#f0a63c40',
    'editor.inactiveSelectionBackground': '#f0a63c22',
    'editorIndentGuide.background1': '#26262c',
    'editorIndentGuide.activeBackground1': '#34343d',
    'editorWidget.background': '#1a1a1e',
    'editorWidget.border': '#34343d',
    'editorSuggestWidget.background': '#1a1a1e',
    'editorSuggestWidget.selectedBackground': '#2b2b32',
    'input.background': '#212127',
    'focusBorder': '#f0a63c66',
    'minimap.background': '#131316',
    'scrollbarSlider.background': '#2b2b3299',
    'scrollbarSlider.hoverBackground': '#34343d'
  }
});

const baseName = (path) => path.split(/[\\/]/).pop();

const languageOf = (path) => {
  const name = baseName(path).toLowerCase();
  const dot = name.lastIndexOf('.');
  const ext = dot >= 0 ? name.slice(dot) : '';
  for(const lang of monaco.languages.getLanguages()){
    if(lang.filenames?.some(item => item.toLowerCase() === name)) return lang.id;
    if(ext && lang.extensions?.some(item => item.toLowerCase() === ext)) return lang.id;
  }
  return 'plaintext';
};

const languageName = (id) => {
  const found = monaco.languages.getLanguages().find(lang => lang.id === id);
  return found?.aliases?.[0] || id;
};

const flatten = (items, out = []) => {
  for(const item of items || []){
    if(item.type === 'folder') flatten(item.children, out);
    else out.push(item.path);
  }
  return out;
};

const fuzzy = (text, query) => {
  const hay = text.toLowerCase();
  let at = 0, score = 0;
  for(const ch of query.toLowerCase()){
    const found = hay.indexOf(ch, at);
    if(found < 0) return -1;
    score += found - at;
    at = found + 1;
  }
  return score + (hay.length - at) * 0.01;
};

export default function CodeEditor({apiUrl, project, files, request, refreshKey, onSaved, onEmpty, onClose, maximized, onMaximize, alert, confirm}){
  const holder = useRef(null);
  const diffHolder = useRef(null);
  const editor = useRef(null);
  const diffEditor = useRef(null);
  const entries = useRef(new Map());
  const activeRef = useRef('');
  const tabsRef = useRef([]);
  const commands = useRef({});
  const [tabs, setTabs] = useState([]);
  const [active, setActive] = useState('');
  const [view, setView] = useState('code');
  const [changed, setChanged] = useState({});
  const [cursor, setCursor] = useState({line: 1, column: 1, picked: 0});
  const [wrap, setWrap] = useState(false);
  const [palette, setPalette] = useState(null);
  const [preview, setPreview] = useState('');

  const url = (path, tail = 'file') => `${apiUrl}/api/projects/${encodeURIComponent(project)}/${tail}?path=${encodeURIComponent(path)}`;

  const markChanged = (path) => {
    const entry = entries.current.get(path);
    if(!entry) return;
    const dirty = entry.model.getAlternativeVersionId() !== entry.saved;
    setChanged(prev => prev[path] === dirty ? prev : {...prev, [path]: dirty});
  };

  const fetchBefore = async (path, content) => {
    try {
      const req = await fetch(url(path, 'changes'));
      if(!req.ok) return null;
      const before = (await req.json()).before ?? null;
      return before !== null && before !== content ? before : null;
    } catch(e) {
      return null;
    }
  };

  const showDiff = (entry) => {
    if(!diffHolder.current) return;
    if(!diffEditor.current){
      diffEditor.current = monaco.editor.createDiffEditor(diffHolder.current, {
        theme: 'sneo-dark',
        automaticLayout: true,
        renderSideBySide: true,
        fontSize: 13,
        minimap: {enabled: false},
        scrollBeyondLastLine: false
      });
    }
    const old = diffEditor.current.getModel()?.original;
    const original = monaco.editor.createModel(entry.before ?? '', entry.model.getLanguageId());
    diffEditor.current.setModel({original, modified: entry.model});
    if(old) old.dispose();
  };

  const activate = (path) => {
    const entry = entries.current.get(path);
    if(!entry || !editor.current) return;
    const current = entries.current.get(activeRef.current);
    if(current && current !== entry) current.state = editor.current.saveViewState();
    activeRef.current = path;
    setActive(path);
    setView(entry.view);
    editor.current.setModel(entry.model);
    if(entry.state) editor.current.restoreViewState(entry.state);
    if(entry.view === 'preview') setPreview(entry.model.getValue());
    if(entry.view === 'changes') setTimeout(() => showDiff(entry), 0);
    else setTimeout(() => editor.current?.focus(), 0);
    const position = editor.current.getPosition();
    setCursor({line: position?.lineNumber || 1, column: position?.column || 1, picked: 0});
  };

  const openPath = async (path, line = 0) => {
    if(!path || !project) return;
    if(entries.current.has(path)){
      activate(path);
      if(line > 0) reveal(line);
      return;
    }
    try {
      const req = await fetch(url(path));
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      const content = res.content ?? '';
      const model = monaco.editor.createModel(content, languageOf(path));
      const entry = {model, saved: model.getAlternativeVersionId(), before: await fetchBefore(path, content), view: /\.html?$/i.test(path) ? 'preview' : 'code', state: null};
      model.onDidChangeContent(() => markChanged(path));
      entries.current.set(path, entry);
      tabsRef.current = [...tabsRef.current, path];
      setTabs(tabsRef.current);
      activate(path);
      if(line > 0) reveal(line);
    } catch(e) {
      await alert(e.message);
    }
  };

  const reveal = (line) => {
    if(!editor.current) return;
    editor.current.revealLineInCenter(line);
    editor.current.setPosition({lineNumber: line, column: 1});
    editor.current.focus();
  };

  const save = async (path = activeRef.current) => {
    const entry = entries.current.get(path);
    if(!entry || !project) return;
    const version = entry.model.getAlternativeVersionId();
    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/file`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({file: path, content: entry.model.getValue()})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      entry.saved = version;
      markChanged(path);
      onSaved?.(path);
    } catch(e) {
      await alert(e.message);
    }
  };

  const close = async (path = activeRef.current) => {
    const entry = entries.current.get(path);
    if(!entry) return;
    if(entry.model.getAlternativeVersionId() !== entry.saved && !await confirm(tr('Close {0} without saving your changes?', baseName(path)))) return;
    const index = tabsRef.current.indexOf(path);
    const rest = tabsRef.current.filter(item => item !== path);
    tabsRef.current = rest;
    setTabs(rest);
    setChanged(prev => {
      const next = {...prev};
      delete next[path];
      return next;
    });
    if(activeRef.current === path){
      const next = rest[Math.min(index, rest.length - 1)];
      if(next) activate(next);
      else {
        activeRef.current = '';
        setActive('');
        editor.current?.setModel(null);
      }
    }
    const shown = diffEditor.current?.getModel();
    if(shown?.modified === entry.model){
      diffEditor.current.setModel(null);
      shown.original?.dispose();
    }
    entry.model.dispose();
    entries.current.delete(path);
    if(!rest.length) onEmpty?.();
  };

  const switchView = (next) => {
    const entry = entries.current.get(activeRef.current);
    if(!entry) return;
    entry.view = next;
    setView(next);
    if(next === 'preview') setPreview(entry.model.getValue());
    if(next === 'changes') setTimeout(() => showDiff(entry), 0);
    if(next === 'code') setTimeout(() => editor.current?.focus(), 0);
  };

  const step = (offset) => {
    const list = tabsRef.current;
    if(list.length < 2) return;
    const index = list.indexOf(activeRef.current);
    activate(list[(index + offset + list.length) % list.length]);
  };

  commands.current = {save, close, step, files: () => setPalette({kind: 'files', query: '', results: [], index: 0}), search: () => setPalette({kind: 'search', query: editor.current?.getModel()?.getValueInRange(editor.current.getSelection()) || '', results: [], index: 0})};

  useEffect(() => {
    const instance = monaco.editor.create(holder.current, {
      theme: 'sneo-dark',
      automaticLayout: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
      fontLigatures: true,
      minimap: {enabled: true},
      bracketPairColorization: {enabled: true},
      guides: {bracketPairs: true, indentation: true},
      smoothScrolling: true,
      cursorBlinking: 'smooth',
      cursorSmoothCaretAnimation: 'on',
      renderWhitespace: 'selection',
      scrollBeyondLastLine: false,
      stickyScroll: {enabled: true},
      model: null
    });
    editor.current = instance;
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => commands.current.save());
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyP, () => commands.current.files());
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyF, () => commands.current.search());
    instance.addCommand(monaco.KeyMod.Alt | monaco.KeyCode.KeyZ, () => setWrap(value => !value));
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.PageDown, () => commands.current.step(1));
    instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.PageUp, () => commands.current.step(-1));
    instance.onDidChangeCursorSelection((e) => {
      const model = instance.getModel();
      const picked = model ? model.getValueInRange(e.selection).length : 0;
      setCursor({line: e.selection.positionLineNumber, column: e.selection.positionColumn, picked});
    });
    const onKey = (e) => {
      if(e.defaultPrevented || !(e.ctrlKey || e.metaKey) || !activeRef.current) return;
      const key = e.key.toLowerCase();
      const run = key === 'w'
        ? commands.current.close
        : key === 's'
          ? commands.current.save
          : key === 'p'
            ? commands.current.files
            : key === 'f' && e.shiftKey
              ? commands.current.search
              : null;
      if(!run) return;
      e.preventDefault();
      run();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      for(const entry of entries.current.values()) entry.model.dispose();
      entries.current.clear();
      diffEditor.current?.getModel()?.original?.dispose();
      diffEditor.current?.dispose();
      instance.dispose();
    };
  }, []);

  useEffect(() => {
    editor.current?.updateOptions({wordWrap: wrap ? 'on' : 'off'});
  }, [wrap]);

  useEffect(() => {
    if(request?.path) openPath(request.path, request.line || 0);
  }, [request?.nonce]);

  useEffect(() => {
    for(const entry of entries.current.values()) entry.model.dispose();
    entries.current.clear();
    tabsRef.current = [];
    activeRef.current = '';
    setTabs([]);
    setActive('');
    setChanged({});
    editor.current?.setModel(null);
  }, [project]);

  useEffect(() => {
    if(!refreshKey) return;
    const refresh = async () => {
      for(const path of [...tabsRef.current]){
        const entry = entries.current.get(path);
        if(!entry || entry.model.getAlternativeVersionId() !== entry.saved) continue;
        try {
          const req = await fetch(url(path));
          if(!req.ok){
            await close(path);
            continue;
          }
          const content = (await req.json()).content ?? '';
          if(content !== entry.model.getValue()){
            entry.model.setValue(content);
            entry.saved = entry.model.getAlternativeVersionId();
            markChanged(path);
          }
          entry.before = await fetchBefore(path, content);
          if(path === activeRef.current && entry.view === 'changes') showDiff(entry);
          if(path === activeRef.current && entry.view === 'preview') setPreview(content);
        } catch(e) {
          continue;
        }
      }
    };
    refresh();
  }, [refreshKey]);

  const runSearch = async (query) => {
    if(!query.trim() || !project) return;
    try {
      const req = await fetch(`${apiUrl}/api/projects/${encodeURIComponent(project)}/search?q=${encodeURIComponent(query)}`);
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      setPalette(prev => prev && {...prev, results: res.matches || [], index: 0});
    } catch(e) {
      await alert(e.message);
    }
  };

  const fileList = palette?.kind === 'files'
    ? flatten(files)
      .map(path => ({path, score: palette.query ? fuzzy(path, palette.query) : 0}))
      .filter(item => item.score >= 0)
      .sort((a, b) => a.score - b.score)
      .slice(0, 60)
    : [];
  const items = palette?.kind === 'files'
    ? fileList
    : palette?.results || [];

  const pick = (item) => {
    setPalette(null);
    if(!item) return;
    if(palette.kind === 'files') openPath(item.path);
    else openPath(item.file, item.line);
  };

  const entry = entries.current.get(active);
  const language = entry ? entry.model.getLanguageId() : '';
  const options = entry ? entry.model.getOptions() : null;
  const dirtyCount = Object.values(changed).filter(Boolean).length;

  return (
    <section className="editor-panel code-studio" aria-label={active || tr('Editor')}>
      <div className="editor-tabs studio-tabs">
        <div className="studio-tab-strip">
          {tabs.map(path => (
            <div
              key={path}
              className={path === active ? 'studio-tab active' : 'studio-tab'}
              title={path}
              onClick={() => activate(path)}
              onMouseDown={(e) => {
                if(e.button === 1){
                  e.preventDefault();
                  close(path);
                }
              }}
            >
              <span className="studio-tab-name">{baseName(path)}</span>
              <button
                type="button"
                className={changed[path] ? 'studio-tab-close dirty' : 'studio-tab-close'}
                title={changed[path] ? tr('Unsaved changes') : tr('Close')}
                onClick={(e) => {
                  e.stopPropagation();
                  close(path);
                }}
              >
                <i className={changed[path] ? 'fa-solid fa-circle' : 'fa-solid fa-xmark'}></i>
              </button>
            </div>
          ))}
        </div>
        <span className="editor-spacer"></span>
        {entry && /\.html?$/i.test(active) && (
          <button type="button" className={view === 'preview' ? 'active-mini' : ''} onClick={() => switchView(view === 'preview' ? 'code' : 'preview')}>
            {view === 'preview' ? tr('Code') : tr('Preview')}
          </button>
        )}
        {entry && entry.before !== null && (
          <button type="button" className={view === 'changes' ? 'active-mini' : ''} onClick={() => switchView(view === 'changes' ? 'code' : 'changes')}>
            {view === 'changes' ? tr('Edit file') : tr('Show changes')}
          </button>
        )}
        <button type="button" className="icon-btn" title={tr('Go to file (Ctrl+P)')} onClick={() => commands.current.files()}>
          <i className="fa-solid fa-magnifying-glass"></i>
        </button>
        <button type="button" className="icon-btn" title={tr('Search in files (Ctrl+Shift+F)')} onClick={() => commands.current.search()}>
          <i className="fa-solid fa-file-lines"></i>
        </button>
        <button type="button" className={wrap ? 'icon-btn active-mini' : 'icon-btn'} title={tr('Word wrap (Alt+Z)')} onClick={() => setWrap(!wrap)}>
          <i className="fa-solid fa-paragraph"></i>
        </button>
        <button type="button" className="icon-btn" title={maximized ? tr('Restore') : tr('Maximize')} onClick={onMaximize}>
          <i className={maximized ? 'fa-solid fa-down-left-and-up-right-to-center' : 'fa-solid fa-up-right-and-down-left-from-center'}></i>
        </button>
        <button type="button" onClick={() => save()} disabled={!changed[active]}>{tr('Save')}</button>
        <button
          type="button"
          className="icon-btn"
          title={tr('Close the editor')}
          onClick={async () => {
            if(dirtyCount > 0 && !(await confirm(tr('{0} files have unsaved changes. Close the editor anyway?', dirtyCount)))) return;
            onClose?.();
          }}
        >
          <i className="fa-solid fa-xmark"></i>
        </button>
      </div>

      <div className="studio-body">
        <div ref={holder} className="studio-editor" style={{display: view === 'code' ? 'block' : 'none'}}></div>
        <div ref={diffHolder} className="studio-editor" style={{display: view === 'changes' ? 'block' : 'none'}}></div>
        {view === 'preview' && (
          <iframe title={tr('preview')} className="preview-frame" sandbox="allow-scripts allow-forms allow-modals" srcDoc={preview} />
        )}

        {palette && (
          <div className="studio-palette" onMouseDown={(e) => e.target === e.currentTarget && setPalette(null)}>
            <div className="studio-palette-box">
              <input
                autoFocus
                className="studio-palette-input"
                placeholder={palette.kind === 'files' ? tr('Go to file by name') : tr('Search in all files, press Enter')}
                value={palette.query}
                onChange={(e) => setPalette({...palette, query: e.target.value, index: 0, results: palette.kind === 'files' ? [] : palette.results})}
                onKeyDown={(e) => {
                  if(e.key === 'Escape'){
                    setPalette(null);
                    editor.current?.focus();
                    return;
                  }
                  if(e.key === 'ArrowDown'){
                    e.preventDefault();
                    setPalette({...palette, index: Math.min(palette.index + 1, Math.max(0, items.length - 1))});
                    return;
                  }
                  if(e.key === 'ArrowUp'){
                    e.preventDefault();
                    setPalette({...palette, index: Math.max(palette.index - 1, 0)});
                    return;
                  }
                  if(e.key !== 'Enter') return;
                  e.preventDefault();
                  if(palette.kind === 'search' && !palette.results.length){
                    runSearch(palette.query);
                    return;
                  }
                  pick(items[palette.index]);
                }}
              />
              <div className="studio-palette-list">
                {items.map((item, index) => (
                  <button
                    key={palette.kind === 'files' ? item.path : `${item.file}:${item.line}:${index}`}
                    type="button"
                    className={index === palette.index ? 'studio-palette-item active' : 'studio-palette-item'}
                    onMouseEnter={() => setPalette({...palette, index})}
                    onClick={() => pick(item)}
                  >
                    {palette.kind === 'files' ? (
                      <>
                        <span className="studio-palette-main">{baseName(item.path)}</span>
                        <span className="studio-palette-side">{item.path}</span>
                      </>
                    ) : (
                      <>
                        <span className="studio-palette-main">{item.text.trim() || ' '}</span>
                        <span className="studio-palette-side">{item.file}:{item.line}</span>
                      </>
                    )}
                  </button>
                ))}
                {palette.kind === 'search' && palette.query && !items.length && (
                  <div className="studio-palette-empty">{tr('Press Enter to search')}</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="studio-status">
        <span>{active || tr('No file')}</span>
        <span className="editor-spacer"></span>
        {dirtyCount > 0 && <span>{tr('{0} unsaved', dirtyCount)}</span>}
        {entry && <span>{tr('Ln {0}, Col {1}', cursor.line, cursor.column)}{cursor.picked ? tr(' ({0} selected)', cursor.picked) : ''}</span>}
        {options && <span>{options.insertSpaces ? tr('Spaces') : tr('Tab Size')}: {options.tabSize}</span>}
        {entry && <span>{tr('UTF-8')}</span>}
        {entry && <span>{entry.model.getEOL() === '\r\n' ? tr('CRLF') : tr('LF')}</span>}
        {entry && <span>{languageName(language)}</span>}
      </div>
    </section>
  );
}
