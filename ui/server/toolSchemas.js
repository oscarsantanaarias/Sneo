const S = 'string';
const N = 'number';
const B = 'boolean';

const def = (name, description, props, required) => ({
  type: 'function',
  function: {
    name,
    description,
    parameters: {
      type: 'object',
      properties: Object.fromEntries(Object.entries(props).map(([key, type]) => [key, Array.isArray(type) ? type[0] : {type}])),
      required: required || Object.keys(props)
    }
  }
});

const defs = [
  def('list_projects', 'List the registered projects with name and path', {}),
  def('add_project', 'Register a PC folder as a project by its absolute path, so you can read/grep/edit inside. Use it when the user gives a path like C:\\something\\folder that is not a project yet. Then use its name (or the path) in the other tools', {projectPath: S, name: S}, ['projectPath']),
  def('list_files', 'List the project file tree', {project: S}, []),
  def('read_file', 'Read the full content of a project file', {project: S, file: S}, ['file']),
  def('read_range', 'Read a range of lines from a file', {project: S, file: S, start: N, end: N}, ['file', 'start', 'end']),
  def('glob', 'Find files by glob pattern, e.g. src/**/*.js', {project: S, pattern: S}, ['pattern']),
  def('grep', 'Search text inside files, with an optional glob pattern', {project: S, query: S, pattern: S}, ['query']),
  def('search_files', 'Find files whose name or content matches the text', {project: S, query: S}, ['query']),
  def('write_file', 'Create a NEW file, or replace a whole file when most of it changes. To change some lines of an existing file use replace_in_file instead: rewriting a whole file to change a few lines takes minutes', {project: S, file: S, content: S}, ['file', 'content']),
  def('replace_in_file', 'Change part of an existing file: search = the exact current lines to change (with a line around them if needed so it matches once), replace = the new lines. The tool for any edit of a few lines; all: true replaces every match', {project: S, file: S, search: S, replace: S, all: B}, ['file', 'search', 'replace']),
  def('propose_patch', 'Apply several search/replace changes across several files (applies directly in Auto mode)', {project: S, title: S, changes: [{type: 'array', items: {type: 'object', properties: {file: {type: S}, search: {type: S}, replace: {type: S}}, required: ['file', 'search', 'replace']}}]}, ['changes']),
  def('delete_path', 'Delete a project file or folder', {project: S, target: S}, ['target']),
  def('create_folder', 'Create a folder in the project', {project: S, folder: S}, ['folder']),
  def('rename_path', 'Rename or move a file/folder', {project: S, from: S, to: S}, ['from', 'to']),
  def('run_command', 'Run a terminal command inside the project (npm.cmd install X, pip install X, node script.js, etc). 5 min timeout. Do NOT use dev servers (npm run dev/start) here, they hang', {project: S, command: S}, ['command']),
  def('visual_check', 'Render an HTML page in a real browser and return how it LOOKS: screenshot, JS console errors, elements overflowing the screen, and design critique from a vision model. Use it AFTER editing a web page to confirm it looks good and fix what is broken. file = project file (e.g. index.html) or url = http://localhost:PORT', {project: S, file: S, url: S, question: S}, []),
  def('ghidra', 'Query Ghidra for reverse engineering via the TCP server (default port 5093). Commands: HELP, CURRENT, FUNC <addr>, DECOMP <addr>, XREF <addr>, XFROM <addr>, STR <ascii text>, WSTR <utf16 text>, BYTES <hex bytes>, DECOMP_XREF <addr>, TRACE <addr>, NEARSTR <addr>, READPTR <addr> [count], WRITERS <addr>, READERS <addr>, CALLERS <addr>, FINDPTR <addr>, STRUCT <addr> [n], VTABLE <funcaddr>, PCODE <addr>, RETYPE <func> <var> <type>, DEFSTRUCT <name> [size], SETFIELD <struct> <off> <type> [field], VTDUMP <addr> [count], MKPTRS <addr> [count], RTTI <addr|Clase|Ns::Clase>, SETCONV <func> <convention>, PATCH <addr> <hex bytes>, MAKEFUNC <addr>, RENAME_FUNC <addr> <name>, RENAME_SYMBOL <addr> <name>, RENAME_LABEL <addr> <name>, RENAME_LOCAL <func_addr> <old_name> <new_name>, IMPORTXREF <name>. Addresses in hex without 0x, e.g. FUNC 01b25f2a', {command: S, port: N}, ['command']),
  def('ida', 'Query IDA Pro through ida_bridge.py on port 5073. Commands: PING, NAME <addr>, RENAME <addr> <name>, DECOMPILE <addr>, DISASM <addr> [count], FIND <text>, XREFS <addr>, STR <text>, EXPORT [path], QUIT. Use this when the user says IDA or asks about the IDA database.', {command: S, port: N}, ['command']),
  def('re_index', 'Decompile a list of Ghidra functions and index them by meaning (embeddings) for semantic search. Pass the addresses you gathered from XREF/TRACE. E.g. addresses ["01b25f2a","01c33a10"]', {addresses: [{type: 'array', items: {type: S}}], port: N}, ['addresses']),
  def('re_search', 'Search the already-indexed decompiled functions by MEANING. E.g.: "where the license is validated", "function that decrypts the packet". Returns addresses + pseudo-C snippet ordered by relevance. re_index first', {query: S, k: N}, ['query']),
  def('re_map', 'Automatically map a subsystem: decompile the root function, extract the functions it calls, decompile them and follow the chain N levels (all auto-indexed). Returns the call graph. Use it for "reverse the whole subsystem of X". address = root function, depth = levels (1-3)', {address: S, depth: N, port: N}, ['address']),
  def('pc_search', 'Search files across the user PC (Desktop/Downloads/Documents by default). query = text in the FILENAME (optional). content = text INSIDE the file (optional, searches text-file contents regardless of name). Give at least one. E.g. to find a .txt containing "AQUI QWEM30" pass content:"AQUI QWEM30". root optional to narrow the base folder. Returns absolute paths', {query: S, content: S, root: S}, []),
  def('web_search', 'Search the internet (DuckDuckGo). Use it for docs, error messages, references for structs/APIs/formats, current library versions. Returns title, url and snippet', {query: S, count: N}, ['query']),
  def('web_fetch', 'Download a web page and return its plain text. Use it after web_search to read a specific doc or article', {url: S}, ['url']),
  def('sandbox_run', 'Copy the project to a sandbox and run the command there without touching the real project. Do NOT use dev servers here (they hang); use npm.cmd run build to verify it compiles', {project: S, command: S}, ['command']),
  def('sandbox_start', 'Start a long-running process in a sandbox and return an id', {project: S, command: S, timeoutMs: N}, ['command']),
  def('sandbox_read', 'Read the accumulated output of a started sandbox', {id: S}),
  def('sandbox_stop', 'Stop a started sandbox', {id: S}),
  def('terminal_start', 'Start a persistent terminal session in the project (use for dev servers/long processes)', {project: S, command: S}, ['command']),
  def('terminal_read', 'Read the output of a terminal session', {id: S}),
  def('terminal_stop', 'Stop a terminal session', {id: S}),
  def('terminal_list', 'List the active terminal sessions', {}),
  def('git_status', 'Show the project git status', {project: S}, []),
  def('git_diff', 'Show the project git diff', {project: S, staged: B}, []),
  def('git_commit', 'Run git add -A and commit with the given message', {project: S, message: S}, ['message']),
  def('context_index', 'Index the project for context searches', {project: S}, []),
  def('context_get', 'Return the project context index', {project: S}, []),
  def('context_pick', 'Pick the most relevant files for a query', {project: S, query: S}, ['query']),
  def('context_read', 'Read the content of a list of files', {project: S, files: [{type: 'array', items: {type: S}}]}, ['files']),
  def('codebase_search', 'Search the project by MEANING (embeddings), not exact words. Finds code relevant to a query even if it uses other names. Use it BEFORE editing when you do not know which file something is in. Returns files + line range + snippet', {project: S, query: S, k: N}, ['query']),
  def('codebase_index', 'Reindex the project for semantic search. Run once or when the project changed a lot; codebase_search auto-indexes on first use', {project: S}, []),
  def('plan_create', 'Create a step plan for a big task', {title: S, steps: [{type: 'array', items: {type: S}}]}, ['title', 'steps']),
  def('plan_update', 'Update the status or text of a plan step', {planId: S, stepId: S, status: S, text: S}, ['planId', 'stepId']),
  def('plan_list', 'List the saved plans', {}),
  def('memory_add', 'Save a durable fact or user preference', {text: S, tags: [{type: 'array', items: {type: S}}]}, ['text']),
  def('memory_search', 'Search the saved memory', {query: S}),
  def('memory_list', 'List all saved memory', {}),
  def('parallel_agents', 'Launch up to 5 subagents in parallel with different roles to review/explore the project', {project: S, task: S, agents: [{type: 'array', items: {type: 'object', properties: {name: {type: S}, role: {type: S}}}}]}, ['task']),
  def('orchestrate', 'Multi-agent orchestrator for BIG tasks: a planner (big model) splits the task into up to 15 subtasks, worker subagents (small model) run in parallel waves, then a synthesis merges everything. Use for large audits, building big features, reverse-engineering swarms, or wide research. mode = audit | build | reverse | research | auto.', {project: S, task: S, mode: S, maxAgents: {type: 'number'}}, ['task']),
  def('plugin_list', 'List installed plugins (external MCP servers) and the extra tools they expose', {}),
  def('plugin_reload', 'Reconnect the plugins declared in data/plugins.json (after adding or editing a server)', {}),
  def('vector_remember', 'Save a piece of text into long-term semantic (vector) memory so it can be recalled later by meaning', {text: S, tags: [{type: 'array', items: {type: S}}]}, ['text']),
  def('vector_recall', 'Search the semantic (vector) memory by meaning and return the most similar saved items', {query: S, k: N}, ['query']),
  def('vector_forget', 'Remove an item from the vector memory by id', {id: S}, ['id']),
  def('read_document', 'Extract text from a document (PDF, DOCX, or plain text) given its ABSOLUTE path', {file: S}, ['file']),
  def('deep_research', 'Deep research: split the question into web searches, fetch pages, and synthesize a sourced answer', {query: S}, ['query']),
  def('speak', 'Speak text out loud through the computer speakers (Windows text-to-speech)', {text: S}, ['text']),
  def('generate_image', 'Generate an image from a text prompt (needs an image backend configured in settings: imageBaseUrl/imageApiKey)', {prompt: S, size: S}, ['prompt']),
  def('note_add', 'Save a note', {text: S, title: S}, ['text']),
  def('note_list', 'List saved notes', {}),
  def('task_add', 'Add a task/todo', {text: S}, ['text']),
  def('task_list', 'List tasks/todos', {}),
  def('task_done', 'Mark a task as done by id', {id: S}, ['id']),
  def('compare_models', 'Run the same prompt through 2-3 models and return every answer side by side', {prompt: S, models: [{type: 'array', items: {type: S}}]}, ['prompt', 'models'])
];

export const toolSchemas = defs;

const subAgentNames = ['list_files', 'read_file', 'read_range', 'search_files', 'glob', 'grep', 'ghidra', 'ida', 're_index', 're_search', 're_map', 'pc_search', 'web_search', 'web_fetch', 'sandbox_run', 'git_status', 'git_diff', 'context_index', 'context_get', 'context_pick', 'context_read', 'codebase_search', 'memory_search', 'memory_list'];

export const subAgentToolSchemas = defs.filter(item => subAgentNames.includes(item.function.name));
