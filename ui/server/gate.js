import { existsSync } from 'fs';
import os from 'os';
import path from 'path';

export const accessModes = ['manual', 'plan', 'auto'];

export const normalizeMode = (value, fallback = 'manual') => {
  if(value === 'ask'){
    return 'manual';
  }

  if(value === 'full'){
    return 'auto';
  }

  return accessModes.includes(value)
    ? value
    : accessModes.includes(fallback)
      ? fallback
      : 'manual';
};

const historyView = {
  short: ['-D', '-E', '-F', '-P', '-R', '-W', '-a', '-b', '-c', '-i', '-m', '-p', '-q', '-r', '-s', '-t', '-u', '-w', '-z'],
  joined: ['-B', '-C', '-M', '-U'],
  valued: ['-G', '-L', '-S', '-n'],
  counts: true,
  long: ['--abbrev', '--abbrev-commit', '--all', '--all-match', '--ancestry-path', '--author-date-order', '--binary', '--boundary', '--branches', '--cached', '--check', '--cherry-mark', '--cherry-pick', '--children', '--color', '--color-moved', '--color-words', '--compact-summary', '--cumulative', '--date', '--date-order', '--decorate', '--diff-algorithm', '--dirstat', '--encoding', '--exit-code', '--extended-regexp', '--find-copies', '--find-copies-harder', '--find-renames', '--first-parent', '--fixed-strings', '--follow', '--format', '--full-diff', '--full-history', '--full-index', '--function-context', '--graph', '--histogram', '--ignore-all-space', '--ignore-blank-lines', '--ignore-cr-at-eol', '--ignore-space-at-eol', '--ignore-space-change', '--invert-grep', '--left-only', '--left-right', '--merge-base', '--merges', '--minimal', '--name-only', '--name-status', '--no-abbrev-commit', '--no-color', '--no-decorate', '--no-ext-diff', '--no-index', '--no-mailmap', '--no-merges', '--no-notes', '--no-patch', '--no-renames', '--no-textconv', '--no-walk', '--notes', '--numstat', '--oneline', '--parents', '--patch', '--patch-with-stat', '--patience', '--perl-regexp', '--pretty', '--quiet', '--raw', '--regexp-ignore-case', '--relative-date', '--remotes', '--reverse', '--right-only', '--shortstat', '--simplify-by-decoration', '--simplify-merges', '--source', '--staged', '--stat', '--summary', '--tags', '--text', '--topo-order', '--use-mailmap', '--word-diff', '--word-diff-regex'],
  longValued: ['--after', '--author', '--before', '--committer', '--diff-filter', '--grep', '--max-count', '--since', '--skip', '--stat-width', '--unified', '--until']
};

const listingFlags = ['--contains', '--list', '--merged', '--no-contains', '--no-merged', '--points-at', '-l'];
const onlyListing = (used, count) => count === 0 || listingFlags.some(flag => used.has(flag));
const configWrites = ['edit', 'remove-section', 'rename-section', 'set', 'unset'];

const gitReads = {
  blame: {
    short: ['-b', '-c', '-e', '-f', '-l', '-n', '-p', '-s', '-t', '-w'],
    joined: ['-C', '-M'],
    valued: ['-L'],
    long: ['--abbrev', '--color-by-age', '--color-lines', '--date', '--first-parent', '--incremental', '--line-porcelain', '--porcelain', '--reverse', '--root', '--show-email', '--show-name', '--show-number', '--show-stats']
  },
  branch: {
    short: ['-a', '-i', '-l', '-r', '-v'],
    long: ['--abbrev', '--all', '--color', '--column', '--contains', '--format', '--ignore-case', '--list', '--merged', '--no-abbrev', '--no-color', '--no-column', '--no-contains', '--no-merged', '--points-at', '--remotes', '--show-current', '--sort', '--verbose'],
    check: onlyListing
  },
  'cat-file': {short: ['-e', '-p', '-s', '-t']},
  config: {
    short: ['-l', '-z'],
    long: ['--bool', '--get', '--get-all', '--get-regexp', '--global', '--includes', '--int', '--list', '--local', '--name-only', '--no-includes', '--null', '--path', '--show-origin', '--show-scope', '--system', '--worktree'],
    longValued: ['--type'],
    operand: (word, count) => count > 1 || !configWrites.includes(word.toLowerCase()),
    check: (used, count) => count <= 1 || (count === 2 && ['--get', '--get-all', '--get-regexp'].some(flag => used.has(flag)))
  },
  'count-objects': {short: ['-H', '-v'], long: ['--human-readable', '--verbose'], most: 0},
  describe: {
    long: ['--abbrev', '--all', '--always', '--broken', '--candidates', '--contains', '--dirty', '--exact-match', '--first-parent', '--long', '--tags'],
    longValued: ['--exclude', '--match']
  },
  diff: historyView,
  grep: {
    short: ['-E', '-F', '-G', '-H', '-I', '-L', '-P', '-W', '-a', '-c', '-h', '-i', '-l', '-n', '-o', '-p', '-q', '-r', '-v', '-w', '-z'],
    valued: ['-A', '-B', '-C', '-e', '-f', '-m'],
    counts: true,
    long: ['--all-match', '--and', '--basic-regexp', '--break', '--cached', '--column', '--count', '--exclude-standard', '--extended-regexp', '--files-with-matches', '--files-without-match', '--fixed-strings', '--full-name', '--function-context', '--heading', '--ignore-case', '--invert-match', '--line-number', '--name-only', '--no-exclude-standard', '--no-index', '--no-recursive', '--not', '--null', '--only-matching', '--or', '--perl-regexp', '--quiet', '--recursive', '--show-function', '--text', '--untracked', '--word-regexp'],
    longValued: ['--max-depth', '--threads']
  },
  log: historyView,
  'ls-files': {
    short: ['-c', '-d', '-f', '-i', '-k', '-m', '-o', '-s', '-t', '-u', '-v', '-z'],
    valued: ['-X', '-x'],
    long: ['--abbrev', '--cached', '--deduplicate', '--deleted', '--directory', '--eol', '--error-unmatch', '--exclude-standard', '--format', '--full-name', '--ignored', '--killed', '--modified', '--no-empty-directory', '--others', '--recurse-submodules', '--stage', '--unmerged'],
    longValued: ['--exclude', '--exclude-from']
  },
  'ls-tree': {short: ['-d', '-l', '-r', '-t', '-z'], long: ['--abbrev', '--format', '--full-name', '--full-tree', '--long', '--name-only', '--name-status', '--object-only']},
  reflog: {valued: ['-n'], counts: true, long: ['--all', '--date', '--oneline', '--relative-date'], leading: ['show']},
  remote: {short: ['-v'], long: ['--verbose'], most: 0},
  'rev-list': historyView,
  'rev-parse': {
    short: ['-q'],
    long: ['--abbrev-ref', '--absolute-git-dir', '--all', '--branches', '--git-common-dir', '--git-dir', '--is-bare-repository', '--is-inside-git-dir', '--is-inside-work-tree', '--is-shallow-repository', '--quiet', '--remotes', '--short', '--show-cdup', '--show-object-format', '--show-prefix', '--show-toplevel', '--symbolic', '--symbolic-full-name', '--tags', '--verify']
  },
  shortlog: {
    short: ['-c', '-e', '-n', '-s'],
    joined: ['-w'],
    counts: true,
    long: ['--all', '--committer', '--email', '--group', '--merges', '--no-merges', '--numbered', '--summary'],
    longValued: ['--after', '--before', '--since', '--until']
  },
  show: historyView,
  'show-ref': {short: ['-d', '-q', '-s'], long: ['--abbrev', '--dereference', '--hash', '--head', '--heads', '--quiet', '--tags', '--verify']},
  stash: {short: ['-p'], long: ['--name-only', '--patch', '--stat'], leading: ['list', 'show'], check: (used, count) => count > 0},
  status: {
    short: ['-b', '-s', '-v', '-z'],
    joined: ['-u'],
    long: ['--ahead-behind', '--branch', '--column', '--find-renames', '--ignore-submodules', '--ignored', '--long', '--no-ahead-behind', '--no-column', '--no-renames', '--null', '--porcelain', '--renames', '--short', '--show-stash', '--untracked-files', '--verbose']
  },
  tag: {
    short: ['-i', '-l'],
    joined: ['-n'],
    long: ['--color', '--column', '--contains', '--format', '--ignore-case', '--list', '--merged', '--no-color', '--no-column', '--no-contains', '--no-merged', '--points-at', '--sort'],
    check: onlyListing
  },
  worktree: {long: ['--porcelain', '--verbose'], leading: ['list']}
};

const npmList = {
  short: ['-a', '-g', '-l', '-p'],
  long: ['--all', '--dev', '--global', '--json', '--link', '--long', '--parseable', '--prod', '--production', '--workspaces'],
  longValued: ['--depth', '--include', '--omit', '--workspace']
};

const npmReads = {
  config: {short: ['-l'], long: ['--global', '--json', '--long'], leading: ['get', 'list', 'ls'], check: (used, count) => count > 0},
  info: {long: ['--json']},
  la: {short: ['-a', '-g'], long: ['--all', '--global', '--json'], longValued: ['--depth']},
  list: npmList,
  ll: {short: ['-a', '-g'], long: ['--all', '--global', '--json'], longValued: ['--depth']},
  ls: npmList,
  outdated: {short: ['-g', '-l'], long: ['--all', '--global', '--json', '--long', '--parseable'], longValued: ['--depth']},
  prefix: {short: ['-g'], long: ['--global'], most: 0},
  root: {short: ['-g'], long: ['--global'], most: 0},
  show: {long: ['--json']},
  view: {long: ['--json']}
};

const pipReads = {
  check: {most: 0},
  freeze: {short: ['-l'], long: ['--all', '--local', '--user']},
  list: {short: ['-l', '-o', '-u', '-v'], long: ['--local', '--not-required', '--outdated', '--uptodate', '--user', '--verbose'], longValued: ['--format']},
  show: {short: ['-f', '-v'], long: ['--files', '--verbose']}
};

const grepLike = {
  short: ['-E', '-F', '-G', '-H', '-I', '-L', '-P', '-R', '-T', '-U', '-Z', '-a', '-b', '-c', '-h', '-i', '-l', '-n', '-o', '-q', '-r', '-s', '-v', '-w', '-x', '-z'],
  valued: ['-A', '-B', '-C', '-D', '-d', '-e', '-f', '-m'],
  counts: true,
  long: ['--basic-regexp', '--byte-offset', '--color', '--colour', '--count', '--dereference-recursive', '--extended-regexp', '--files-with-matches', '--files-without-match', '--fixed-strings', '--ignore-case', '--initial-tab', '--invert-match', '--line-buffered', '--line-number', '--line-regexp', '--no-filename', '--no-group-separator', '--no-ignore-case', '--no-messages', '--null', '--null-data', '--only-matching', '--perl-regexp', '--quiet', '--recursive', '--silent', '--text', '--with-filename', '--word-regexp'],
  longValued: ['--after-context', '--before-context', '--binary-files', '--context', '--exclude', '--exclude-dir', '--file', '--include', '--max-count', '--regexp']
};

const plainGrep = {
  short: ['-E', '-F', '-G', '-H', '-L', '-a', '-b', '-c', '-h', '-i', '-l', '-n', '-o', '-q', '-r', '-s', '-v', '-w', '-x'],
  valued: ['-A', '-B', '-C', '-e', '-f', '-m'],
  counts: true
};

const ends = {
  short: ['-q', '-v', '-z'],
  valued: ['-c', '-n'],
  counts: true,
  long: ['--quiet', '--silent', '--verbose', '--zero-terminated'],
  longValued: ['--bytes', '--lines']
};

const viewers = {
  attrib: {slash: ['d', 'l', 's'], operand: word => !/^[+-]/.test(word)},
  basename: {short: ['-a', '-z'], valued: ['-s']},
  cat: {short: ['-A', '-E', '-T', '-b', '-e', '-n', '-s', '-t', '-u', '-v'], long: ['--number', '--number-nonblank', '--show-all', '--show-ends', '--show-nonprinting', '--show-tabs', '--squeeze-blank']},
  cd: {slash: ['d']},
  chdir: {slash: ['d']},
  cut: {short: ['-n', '-s', '-z'], valued: ['-b', '-c', '-d', '-f']},
  date: {slash: ['t'], need: '/t', most: 0},
  df: {short: ['-H', '-P', '-T', '-a', '-h', '-i', '-k', '-l', '-m', '-v'], long: ['--all', '--human-readable', '--inodes', '--local', '--print-type', '--si', '--total']},
  dir: {slash: '*'},
  dirname: {short: ['-z']},
  du: {
    short: ['-H', '-L', '-P', '-S', '-a', '-b', '-c', '-h', '-k', '-l', '-m', '-s', '-x'],
    valued: ['-B', '-d', '-t'],
    long: ['--all', '--apparent-size', '--bytes', '--human-readable', '--one-file-system', '--si', '--summarize', '--total'],
    longValued: ['--exclude', '--max-depth']
  },
  echo: {slash: '*'},
  egrep: plainGrep,
  fc: {slash: '*'},
  fgrep: plainGrep,
  find: {slash: ['c', 'i', 'n', 'off', 'offline', 'v']},
  findstr: {slash: '*'},
  grep: grepLike,
  head: ends,
  hostname: {slash: [], most: 0},
  ipconfig: {slash: ['all', 'displaydns'], most: 0},
  ls: {any: true},
  netstat: {short: ['-a', '-b', '-e', '-f', '-n', '-o', '-q', '-r', '-s', '-t', '-x', '-y'], valued: ['-p'], most: 0},
  pwd: {short: ['-L', '-P'], most: 0},
  realpath: {short: ['-L', '-P', '-e', '-m', '-q', '-s', '-z']},
  rg: {
    short: ['-0', '-F', '-H', '-I', '-L', '-N', '-P', '-S', '-U', '-V', '-a', '-b', '-c', '-h', '-i', '-l', '-n', '-o', '-p', '-q', '-s', '-u', '-v', '-w', '-x'],
    valued: ['-A', '-B', '-C', '-E', '-M', '-T', '-d', '-e', '-f', '-g', '-j', '-m', '-r', '-t'],
    long: ['--case-sensitive', '--column', '--count', '--count-matches', '--files', '--files-with-matches', '--files-without-match', '--fixed-strings', '--follow', '--heading', '--hidden', '--ignore-case', '--invert-match', '--json', '--line-number', '--line-regexp', '--multiline', '--no-filename', '--no-heading', '--no-ignore', '--no-line-number', '--null', '--only-matching', '--quiet', '--smart-case', '--stats', '--text', '--trim', '--type-list', '--vimgrep', '--with-filename', '--word-regexp'],
    longValued: ['--after-context', '--before-context', '--color', '--context', '--glob', '--iglob', '--max-columns', '--max-count', '--max-depth', '--max-filesize', '--sort', '--sortr', '--type', '--type-not']
  },
  sort: {
    slash: ['+*', 'l', 'm', 'r', 'rec'],
    slashValued: ['l', 'm', 'rec'],
    short: ['-M', '-V', '-b', '-d', '-f', '-g', '-h', '-i', '-n', '-r', '-s', '-u', '-z'],
    valued: ['-k', '-t'],
    long: ['--dictionary-order', '--general-numeric-sort', '--human-numeric-sort', '--ignore-case', '--ignore-leading-blanks', '--month-sort', '--numeric-sort', '--reverse', '--stable', '--unique', '--version-sort', '--zero-terminated'],
    longValued: ['--field-separator', '--key']
  },
  systeminfo: {slash: '*'},
  tail: ends,
  tasklist: {slash: '*'},
  time: {slash: ['t'], need: '/t', most: 0},
  tree: {slash: '*'},
  type: {slash: []},
  uname: {short: ['-a', '-i', '-m', '-n', '-o', '-p', '-r', '-s', '-v'], most: 0},
  uniq: {short: ['-D', '-c', '-d', '-i', '-u', '-z'], valued: ['-f', '-s', '-w'], most: 1},
  ver: {slash: [], most: 0},
  vol: {slash: []},
  wc: {short: ['-L', '-c', '-l', '-m', '-w'], long: ['--bytes', '--chars', '--lines', '--max-line-length', '--words']},
  where: {slash: '*'},
  which: {short: ['-a']},
  whoami: {slash: '*'}
};

const versionTools = ['node', 'python', 'python3', 'py', 'java', 'javac', 'go', 'cargo', 'rustc', 'dotnet', 'deno', 'bun', 'yarn', 'pnpm', 'tsc', 'gcc', 'g++', 'clang', 'cmake', 'make', 'perl', 'ruby', 'php', 'pip', 'pip3', 'npx'];
const versionWords = ['--version', '-v', '-V', 'version', '-version', '--info', '--list-sdks', '--list-runtimes'];
const shellBuiltins = ['dir', 'type', 'echo', 'cd', 'chdir', 'ver', 'vol', 'date', 'time'];

const slashKnown = (rule, part) => {
  const name = part.split(':')[0];
  return rule.slash === '*' || rule.slash.some(item => item.endsWith('*') ? part.startsWith(item.slice(0, -1)) : item === name);
};

const fitsRule = (rule, args) => {
  const used = new Set();
  let count = 0;
  let ended = false;

  for(let i = 0; i < args.length; i++){
    const arg = args[i];

    if(!ended && rule.slash && arg.length > 1 && arg.startsWith('/')){
      const parts = arg.slice(1).toLowerCase().split('/');

      if(parts.some(part => !part || !slashKnown(rule, part))){
        return false;
      }

      parts.forEach(part => used.add('/' + part.split(':')[0]));
      const last = parts[parts.length - 1];

      if(rule.slash !== '*' && rule.slashValued?.includes(last)){
        i++;

        if(i >= args.length){
          return false;
        }
      }

      continue;
    }

    if(!ended && arg.length > 1 && arg.startsWith('-')){
      if(arg === '--'){
        ended = true;
        continue;
      }

      if(rule.any){
        continue;
      }

      if(arg.startsWith('--')){
        const name = arg.split('=')[0];
        const valued = rule.longValued?.includes(name);

        if(!valued && !rule.long?.includes(name)){
          return false;
        }

        used.add(name);

        if(valued && !arg.includes('=')){
          i++;

          if(i >= args.length){
            return false;
          }
        }

        continue;
      }

      if(rule.counts && /^-\d+$/.test(arg)){
        continue;
      }

      for(let j = 1; j < arg.length; j++){
        const flag = '-' + arg[j];
        used.add(flag);

        if(rule.short?.includes(flag)){
          continue;
        }

        if(rule.joined?.includes(flag)){
          break;
        }

        if(!rule.valued?.includes(flag)){
          return false;
        }

        if(j === arg.length - 1){
          i++;

          if(i >= args.length){
            return false;
          }
        }

        break;
      }

      continue;
    }

    if(rule.leading && count === 0 && !rule.leading.includes(arg.toLowerCase())){
      return false;
    }

    count++;

    if(count > (rule.most ?? Infinity)){
      return false;
    }

    if(rule.operand && !rule.operand(arg, count)){
      return false;
    }
  }

  if(rule.need && !used.has(rule.need)){
    return false;
  }

  return rule.check ? rule.check(used, count) : true;
};

export const splitCommand = (source) => {
  const segments = [];
  let words = [];
  let word = '';
  let started = false;
  let quoted = false;

  const closeWord = () => {
    if(started){
      words.push(word);
    }

    word = '';
    started = false;
  };

  const closeSegment = () => {
    closeWord();
    segments.push(words);
    words = [];
  };

  for(let i = 0; i < source.length; i++){
    const char = source[i];

    if(char === '\n' || char === '\r'){
      return null;
    }

    if(quoted){
      if(char === '"'){
        quoted = false;
        continue;
      }

      if(char === '%'){
        return null;
      }

      word += char;
      continue;
    }

    if(char === '"'){
      quoted = true;
      started = true;
      continue;
    }

    if(char === ' ' || char === '\t'){
      closeWord();
      continue;
    }

    if(char === '&' || char === '|' || char === ';'){
      if(char !== ';' && source[i + 1] === char){
        i++;
      }

      closeSegment();
      continue;
    }

    if(char === '>'){
      if(word === '1' || word === '2'){
        word = '';
        started = false;
      } else {
        closeWord();
      }

      if(source[i + 1] === '>'){
        i++;
      }

      let next = i + 1;

      while(source[next] === ' ' || source[next] === '\t'){
        next++;
      }

      if(source[next] === '&' && /[12]/.test(source[next + 1] || '')){
        i = next + 1;
        continue;
      }

      const target = /^[\w.:\\/-]+/.exec(source.slice(next))?.[0] || '';

      if(target.toLowerCase() !== 'nul'){
        return null;
      }

      i = next + target.length - 1;
      continue;
    }

    if('<()^%`$'.includes(char)){
      return null;
    }

    word += char;
    started = true;
  }

  if(quoted){
    return null;
  }

  closeSegment();
  return segments;
};

const segmentReadOnly = (given, cwd) => {
  const glued = /^([a-z]+)(\/.+)$/i.exec(given[0] || '');
  const words = glued && viewers[glued[1].toLowerCase()]?.slash
    ? [glued[1], glued[2], ...given.slice(1)]
    : given;

  if(!words.length || !words[0]){
    return false;
  }

  if(/[\\/:]/.test(words[0])){
    return false;
  }

  const name = words[0].toLowerCase().replace(/\.(exe|cmd|com)$/, '');
  const args = words.slice(1);

  // cambiar de carpeta deja que el siguiente comando encuentre un programa del proyecto, eso ya no corre sin preguntar
  if(['cd', 'chdir', 'pushd', 'popd'].includes(name)){
    return false;
  }

  const runnable = String(process.env.PATHEXT || '.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC').toLowerCase().split(';').filter(Boolean);

  if(cwd && !shellBuiltins.includes(name) && ['', ...runnable].some(ext => existsSync(path.join(cwd, `${name}${ext}`)))){
    return false;
  }

  if(name === 'git'){
    let at = 0;

    while(at < args.length){
      if(args[at] === '--no-pager' || args[at] === '--no-optional-locks'){
        at++;
        continue;
      }

      if(args[at] === '-C'){
        at += 2;
        continue;
      }

      break;
    }

    const sub = String(args[at] || '').toLowerCase();

    if(sub === '--version' || sub === 'version'){
      return args.length === at + 1;
    }

    return Object.hasOwn(gitReads, sub) && fitsRule(gitReads[sub], args.slice(at + 1));
  }

  if(name === 'npm'){
    if(args.length === 1 && ['-v', '--version'].includes(args[0])){
      return true;
    }

    const sub = String(args[0] || '').toLowerCase();
    return Object.hasOwn(npmReads, sub) && fitsRule(npmReads[sub], args.slice(1));
  }

  if((name === 'pip' || name === 'pip3') && Object.hasOwn(pipReads, String(args[0] || '').toLowerCase())){
    return fitsRule(pipReads[args[0].toLowerCase()], args.slice(1));
  }

  if(versionTools.includes(name)){
    return args.length === 1 && versionWords.includes(args[0]);
  }

  return Object.hasOwn(viewers, name) && fitsRule(viewers[name], args);
};

export const isReadOnlyCommand = (command, cwd = '') => {
  const source = String(command ?? '').trim();

  if(!source || source.length > 1000){
    return false;
  }

  const segments = splitCommand(source);

  if(!segments || !segments.length){
    return false;
  }

  return segments.every(words => segmentReadOnly(words, cwd));
};

const freeTools = new Set([
  'list_projects', 'add_project', 'list_files', 'read_file', 'read_range', 'glob', 'grep', 'search_files',
  'visual_check', 're_index', 're_search', 're_map', 'pc_search', 'web_search', 'web_fetch',
  'sandbox_read', 'sandbox_stop', 'terminal_read', 'terminal_stop', 'terminal_list', 'git_status', 'git_diff',
  'context_index', 'context_get', 'context_pick', 'context_read', 'codebase_search', 'codebase_index',
  'plan_create', 'plan_update', 'plan_list', 'memory_add', 'memory_search', 'memory_list',
  'parallel_agents', 'orchestrate', 'plugin_list', 'vector_remember', 'vector_recall', 'vector_forget',
  'read_document', 'deep_research', 'speak', 'generate_image', 'note_add', 'note_list',
  'task_add', 'task_list', 'task_done', 'compare_models'
]);

const commandTools = new Set(['run_command', 'sandbox_run', 'terminal_start', 'sandbox_start']);
const fileTools = new Set(['write_file', 'replace_in_file', 'propose_patch']);
const reverseChanges = /^(RENAME\w*|PATCH\w*|MAKEFUNC|RETYPE|DEFSTRUCT|SETFIELD|MKPTRS|SETCONV|EXPORT|QUIT|SHUTDOWN|EXIT)\b/i;

export const toolKind = (name) => {
  if(freeTools.has(name)){
    return 'free';
  }

  if(commandTools.has(name)){
    return 'command';
  }

  if(fileTools.has(name)){
    return 'file';
  }

  if(name === 'ghidra' || name === 'ida'){
    return 'reverse';
  }

  return 'change';
};

const projectReads = new Set(['list_files', 'read_file', 'read_range', 'glob', 'grep', 'search_files', 'read_document']);

const outsideProject = (value, cwd) => {
  const text = String(value || '').trim();

  if(!/^([a-z]:[\\/]|\\\\|\/|~)/i.test(text)){
    return false;
  }

  if(!cwd){
    return true;
  }

  const rel = path.relative(path.resolve(cwd), path.resolve(text.replace(/^~/, os.homedir())));
  return rel.startsWith('..') || path.isAbsolute(rel);
};

// un texto del proyecto o de una pagina podria pedir leer algo de la pc y mandarlo afuera: eso pregunta antes, salvo en modo auto
const leavesProject = (name, args, cwd) => name === 'web_fetch'
  || (name === 'pc_search' && Boolean(String(args?.content || '').trim()))
  || (projectReads.has(name) && [args?.project, args?.file].some(value => outsideProject(value, cwd)));

export const decideTool = (mode, name, args = {}, cwd = '') => {
  const kind = toolKind(name);

  if(kind === 'free' && mode !== 'auto' && leavesProject(name, args, cwd)){
    return 'ask';
  }

  if(kind === 'free'){
    return 'run';
  }

  if(kind === 'reverse' && !reverseChanges.test(String(args?.command || '').trim())){
    return 'run';
  }

  if(kind === 'command' && isReadOnlyCommand(args?.command, cwd)){
    return 'run';
  }

  return mode === 'auto'
    ? 'run'
    : mode === 'plan'
      ? 'refuse'
      : 'ask';
};
