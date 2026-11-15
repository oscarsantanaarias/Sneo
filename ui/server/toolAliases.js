import { toolSchemas } from './toolSchemas.js';

const known = new Set([...toolSchemas.map(item => item.function.name), 'plugin_list', 'plugin_reload', 'parallel_agents', 'orchestrate']);

const editNames = new Set(['edit', 'edit_file', 'file_edit', 'str_replace', 'search_replace', 'search_and_replace', 'replace', 'replace_text', 'replace_string', 'modify_file', 'update_file', 'apply_edit', 'multi_edit', 'multiedit', 'edit_files', 'patch_file']);
const writeNames = new Set(['write', 'create_file', 'generate_file', 'save_file', 'new_file', 'overwrite_file', 'file_write', 'create', 'write_to_file', 'put_file']);
const readNames = new Set(['read', 'view', 'view_file', 'open_file', 'cat', 'file_read', 'read_lines', 'read_file_range', 'get_file']);
const shellNames = new Set(['shell', 'bash', 'exec', 'execute', 'execute_command', 'run_shell', 'run_terminal_command', 'run_terminal', 'terminal', 'powershell', 'cmd', 'run', 'command', 'run_bash']);
const grepNames = new Set(['grep_files', 'search', 'rg', 'ripgrep', 'search_code', 'find_in_files', 'code_search', 'grep_search', 'search_text', 'text_search']);
const listNames = new Set(['ls', 'list_dir', 'list_directory', 'list', 'dir', 'tree', 'list_folder']);
const globNames = new Set(['find_files', 'file_search', 'find', 'glob_files']);
const lookNames = new Set(['review_console_logs', 'review_network_logs', 'console_logs', 'network_logs', 'screenshot', 'take_screenshot', 'browser_check', 'check_page', 'view_page', 'open_browser']);
const patchNames = new Set(['apply_patch', 'patch', 'apply_diff', 'diff']);
const plainNames = {delete_file: 'delete_path', remove_file: 'delete_path', rm: 'delete_path', delete: 'delete_path', mkdir: 'create_folder', make_dir: 'create_folder', create_directory: 'create_folder', create_dir: 'create_folder', move_file: 'rename_path', rename_file: 'rename_path', mv: 'rename_path', move: 'rename_path', rename: 'rename_path'};

const fileKeys = ['file', 'file_path', 'filepath', 'filePath', 'path', 'filename', 'file_name', 'target_file', 'targetFile', 'target'];
const oldKeys = ['search', 'old_string', 'oldString', 'old_str', 'old', 'old_text', 'oldText', 'find', 'context', 'original', 'before'];
const newKeys = ['replace', 'new_string', 'newString', 'new_str', 'new', 'new_text', 'newText', 'replacement', 'replace_with', 'updated', 'after'];
const bodyKeys = ['content', 'file_text', 'fileText', 'text', 'code', 'contents', 'data', 'body'];

export const toolHelp = 'File tools here: read_file {file}, read_range {file, start, end}, write_file {file, content} for a whole file, replace_in_file {file, search, replace} for a part (search = exact current text), propose_patch {changes: [{file, search, replace}]} for several parts, list_files, glob {pattern}, grep {query, pattern}, run_command {command}, visual_check {file, question}.';

const pick = (args, keys, empty = false) => {
  for(const key of keys){
    const value = args[key];
    if(typeof value === 'string' && (empty || value !== '')) return value;
  }
  return undefined;
};

const keyOf = (name) => String(name || '').trim().replace(/^functions?[.:]/i, '').replace(/[\s-]+/g, '_').toLowerCase();

const base = (args) => args.project ? {project: args.project} : {};

const readCall = (args, file) => {
  const range = Array.isArray(args.view_range) ? args.view_range : null;
  const start = Number(args.start ?? args.offset ?? args.line ?? args.start_line ?? (range ? range[0] : NaN));
  const end = Number(args.end ?? args.end_line ?? (range ? range[1] : NaN));
  const limit = Number(args.limit ?? args.lines ?? NaN);

  if(!Number.isFinite(start) && !Number.isFinite(end)){
    return {name: 'read_file', args: {...base(args), file}};
  }

  const first = Math.max(1, Number.isFinite(start) ? Math.floor(start) : 1);
  const last = Number.isFinite(end) && end >= first
    ? Math.floor(end)
    : Number.isFinite(limit) && limit > 0
      ? first + Math.floor(limit) - 1
      : first + 399;
  return {name: 'read_range', args: {...base(args), file, start: first, end: last}};
};

const editCall = (args, file) => {
  const list = Array.isArray(args.edits) ? args.edits : Array.isArray(args.replacements) ? args.replacements : Array.isArray(args.changes) ? args.changes : null;

  if(list){
    const changes = list
      .filter(item => item && typeof item === 'object')
      .map(item => {
        const target = pick(item, fileKeys) || file;
        const search = pick(item, oldKeys);
        const body = pick(item, bodyKeys, true);
        return search === undefined && body !== undefined
          ? {file: target, content: body}
          : {file: target, search, replace: pick(item, newKeys, true) ?? '', all: item.replace_all === true || item.all === true};
      });
    const bad = changes.find(item => !item.file || (item.content === undefined && !item.search));
    if(!changes.length || bad) return {error: 'Each edit needs the file and the exact old text.', hint: toolHelp};
    return {name: 'propose_patch', args: {...base(args), changes}};
  }

  const search = pick(args, oldKeys);
  const replace = pick(args, newKeys, true);
  const body = pick(args, bodyKeys, true);

  if(!file){
    return {error: 'Say which file to change.', hint: toolHelp};
  }

  if(search !== undefined){
    return {name: 'replace_in_file', args: {...base(args), file, search, replace: replace ?? '', all: args.replace_all === true || args.all === true}};
  }

  if(body !== undefined){
    return {name: 'write_file', args: {...base(args), file, content: body}};
  }

  return {error: 'Give the exact old text and the new text, or the whole new content.', hint: toolHelp};
};

// convierte un parche (formato *** Begin Patch o diff unificado) en cambios reales
export const patchPlan = (patch) => {
  const lines = String(patch || '').replace(/\r\n?/g, '\n').split('\n');
  const plan = {writes: [], deletes: [], moves: [], changes: []};
  const clean = (name) => name.trim().split('\t')[0].replace(/^[ab]\//, '');
  let file = '';
  let mode = '';
  let before = [];
  let after = [];
  let fresh = [];

  const flush = () => {
    while(before.length && after.length && before[before.length - 1] === '' && after[after.length - 1] === ''){
      before.pop();
      after.pop();
    }
    if(mode === 'update' && file && before.length) plan.changes.push({file, search: before.join('\n'), replace: after.join('\n')});
    if(mode === 'update' && file && !before.length && after.length) plan.changes.push({file, search: '', replace: after.join('\n')});
    before = [];
    after = [];
  };

  const close = () => {
    flush();
    if(mode === 'add' && file) plan.writes.push({file, content: fresh.join('\n')});
    fresh = [];
  };

  for(let i = 0; i < lines.length; i++){
    const line = lines[i];
    let m;

    if(/^\*\*\* (Begin|End) Patch/.test(line)){
      close();
      mode = '';
      continue;
    }

    if((m = line.match(/^\*\*\* (Update|Add|Delete) File: (.+)$/))){
      close();
      file = clean(m[2]);
      mode = m[1].toLowerCase();
      if(mode === 'delete') plan.deletes.push(file);
      continue;
    }

    if((m = line.match(/^\*\*\* Move to: (.+)$/))){
      plan.moves.push({from: file, to: clean(m[1])});
      continue;
    }

    if(/^diff --git /.test(line) || /^index [0-9a-f]+\.\.[0-9a-f]+/.test(line) || /^(new|deleted) file mode/.test(line)) continue;

    if(line.startsWith('--- ') && (lines[i + 1] || '').startsWith('+++ ')){
      close();
      const from = lines[i].slice(4).trim().split('\t')[0];
      const to = lines[i + 1].slice(4).trim().split('\t')[0];
      i++;
      if(to === '/dev/null'){
        file = clean(from);
        mode = 'delete';
        plan.deletes.push(file);
        continue;
      }
      file = clean(to);
      mode = from === '/dev/null' ? 'add' : 'update';
      continue;
    }

    if(mode === 'add'){
      if(line.startsWith('+')) fresh.push(line.slice(1));
      continue;
    }

    if(mode !== 'update') continue;

    if(line.startsWith('@@')){
      flush();
      continue;
    }

    if(line.startsWith('\\ No newline') || line === '*** End of File') continue;

    if(line.startsWith('-')){
      before.push(line.slice(1));
      continue;
    }

    if(line.startsWith('+')){
      after.push(line.slice(1));
      continue;
    }

    const same = line.startsWith(' ') ? line.slice(1) : line;
    before.push(same);
    after.push(same);
  }

  close();
  return plan;
};

const patchCalls = (args) => {
  const text = pick(args, ['input', 'patch', 'diff', 'content', 'text', 'changes']);

  if(!text){
    return {error: 'apply_patch needs the patch text.', hint: toolHelp};
  }

  const plan = patchPlan(text);
  const blind = plan.changes.find(item => !item.search);

  if(blind){
    return {error: `A hunk for ${blind.file} has no context or removed lines, so there is nothing to anchor it to.`, hint: toolHelp};
  }

  const calls = [
    ...plan.writes.map(item => ({name: 'write_file', args: {...base(args), file: item.file, content: item.content}})),
    ...(plan.changes.length ? [{name: 'propose_patch', args: {...base(args), title: 'Patch', changes: plan.changes}}] : []),
    ...plan.moves.map(item => ({name: 'rename_path', args: {...base(args), from: item.from, to: item.to}})),
    ...plan.deletes.map(item => ({name: 'delete_path', args: {...base(args), target: item}}))
  ];

  if(!calls.length){
    return {error: 'The patch has no file changes I can read. Use the *** Begin Patch / *** Update File: format or a unified diff.', hint: toolHelp};
  }

  return calls.length === 1 ? calls[0] : {calls};
};

const fixKnown = (name, args, lastFile) => {
  if(name === 'write_file' || name === 'replace_in_file'){
    const file = pick(args, fileKeys);
    if(name === 'write_file' && args.content === undefined && pick(args, bodyKeys, true) === undefined && pick(args, oldKeys) !== undefined) return editCall(args, file || lastFile);
    if(name === 'write_file') return {name, args: {...base(args), file: file || lastFile, content: pick(args, bodyKeys, true) ?? ''}};
    return editCall(args, file || lastFile);
  }

  if(name === 'read_file' || name === 'read_range'){
    return readCall(args, pick(args, fileKeys) || lastFile);
  }

  if(name === 'grep' && args.query === undefined){
    const query = pick(args, ['pattern', 'regex', 'search', 'text', 'q']);
    const glob = pick(args, ['glob', 'include', 'file_pattern', 'files']);
    return {name, args: {...base(args), query, ...(glob ? {pattern: glob} : {})}};
  }

  if(name === 'run_command' && typeof args.command !== 'string'){
    const command = Array.isArray(args.command) ? args.command.join(' ') : pick(args, ['cmd', 'script', 'input', 'code']);
    return {name, args: {...base(args), command}};
  }

  if(name === 'propose_patch' && Array.isArray(args.changes)){
    return editCall(args, lastFile);
  }

  return {name, args};
};

// traduce nombres y argumentos de otros agentes a las herramientas de aqui
export const resolveCall = (name, rawArgs, lastFile = '') => {
  const args = rawArgs && typeof rawArgs === 'object' && !Array.isArray(rawArgs) ? {...rawArgs} : {};
  const key = keyOf(name);
  const file = pick(args, fileKeys) || lastFile;

  if(known.has(name)) return fixKnown(name, args, lastFile);
  if(known.has(key)) return fixKnown(key, args, lastFile);

  if(patchNames.has(key)) return patchCalls(args);

  if(editNames.has(key)){
    const mode = String(args.mode || '').toLowerCase();
    if(/^(overwrite|create|write|replace_all|full|rewrite)$/.test(mode) && pick(args, oldKeys) === undefined) return {name: 'write_file', args: {...base(args), file, content: pick(args, bodyKeys, true) ?? ''}};
    return editCall(args, file);
  }

  if(writeNames.has(key)){
    if(!file) return {error: 'Say which file to write.', hint: toolHelp};
    return {name: 'write_file', args: {...base(args), file, content: pick(args, bodyKeys, true) ?? ''}};
  }

  if(readNames.has(key)) return readCall(args, file);

  if(shellNames.has(key)) return fixKnown('run_command', args, lastFile);

  if(grepNames.has(key)) return fixKnown('grep', args, lastFile);

  if(listNames.has(key)) return {name: 'list_files', args: base(args)};

  if(globNames.has(key)){
    const pattern = pick(args, ['pattern', 'glob', 'query', 'name']);
    return pattern ? {name: 'glob', args: {...base(args), pattern}} : {name: 'list_files', args: base(args)};
  }

  if(lookNames.has(key)){
    const page = pick(args, fileKeys) || lastFile;
    const url = pick(args, ['url']);
    if(!page && !url) return {error: 'Say which page to look at.', hint: 'Use visual_check {file} with the HTML file, or {url}. It returns a screenshot, the console errors and failed requests.'};
    return {name: 'visual_check', args: {...base(args), ...(url ? {url} : {file: page}), question: pick(args, ['question']) || 'List every JavaScript console error and every failed network request, then describe what looks wrong.'}};
  }

  if(plainNames[key]){
    const real = plainNames[key];
    if(real === 'delete_path') return {name: real, args: {...base(args), target: pick(args, ['target', ...fileKeys])}};
    if(real === 'create_folder') return {name: real, args: {...base(args), folder: pick(args, ['folder', 'dir', 'directory', ...fileKeys])}};
    return {name: real, args: {...base(args), from: pick(args, ['from', 'source', 'src', 'old_path', ...fileKeys]), to: pick(args, ['to', 'destination', 'dest', 'new_path', 'new_name'])}};
  }

  return {name, args, unknown: true};
};

export const isKnownTool = (name) => known.has(name);

export const writesFiles = (name) => {
  const key = keyOf(name);
  return ['write_file', 'replace_in_file', 'propose_patch'].includes(key) || editNames.has(key) || writeNames.has(key) || patchNames.has(key);
};

export const unknownTool = (name) => ({
  error: `There is no tool named "${name}" here, nothing was run.`,
  hint: `${toolHelp} Every tool: ${[...known].join(', ')}.`
});
