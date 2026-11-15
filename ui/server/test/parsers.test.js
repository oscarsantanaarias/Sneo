import './setup.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { codeBlocksOf, detectBuildIntent, parseDirectGhidraRequest, parseToolCall } from '../agents.js';

const prose = [
  'I used Ghidra DECOMP on FUN_0179aca0 and it reads the packet header before the loop.',
  'Ghidra\'s DECOMP of FUN_00401000 shows a loop over the player list.',
  'Ghidra: the function at 0x401000 reads the config file and keeps it in a global.',
  'To create the file I would call write_file(file="a.txt", content="x") but you asked me not to touch it.',
  'Here is an example of the protocol:\n```json\n{"tool": "read_file", "args": {"file": "a.txt"}}\n```\nThat is how a call looks.',
  'Your package.json should look like this:\n{"name": "app", "version": "1.0.0", "scripts": {"build": "vite build"}}',
  'The read_file and list_files tools are both available, and grep can search inside files.',
  'I ran `git status` and the tree is clean. Next I would use run_command to build it.',
  'Done. The ida bridge answered PING and the DECOMPILE output matches the strings.',
  '<p>Use the &lt;function&gt; tag in your template.</p> nothing else here',
  'ida.decompile is a method name I saw in the docs, not something to call now.'
];

test('prose never becomes a tool call', () => {
  for(const text of prose){
    assert.equal(parseToolCall(text), null, text);
  }
});

test('explicit tool calls are still understood', () => {
  assert.equal(parseToolCall('{"tool":"read_file","args":{"file":"a.txt"}}')?.tool, 'read_file');
  assert.equal(parseToolCall('```json\n{"tool":"list_files","args":{}}\n```')?.tool, 'list_files');
  assert.equal(parseToolCall('<tool_call>{"name":"list_files","arguments":{}}</tool_call>')?.tool, 'list_files');
  assert.equal(parseToolCall('<function=read_file><parameter=file>a.txt</parameter></function>')?.args.file, 'a.txt');
  assert.deepEqual(parseToolCall('ghidra DECOMP 0179aca0'), {tool: 'ghidra', args: {command: 'DECOMP 0179aca0'}});
  assert.equal(parseToolCall('{"tool":"not_a_tool","args":{}}'), null);
});

test('ordinary words never trigger the Ghidra shortcut', () => {
  assert.equal(parseDirectGhidraRequest('I already read the facade docs from last decade'), null);
  assert.equal(parseDirectGhidraRequest('the background is #1e293b, make it lighter'), null);
  assert.equal(parseDirectGhidraRequest('decompile FUN_00401000')?.args.command, 'DECOMP 00401000');
});

test('build intent needs a change request and a place to write', () => {
  assert.equal(detectBuildIntent('In that project create a page for my bakery', 'site')?.project, 'site');
  assert.ok(detectBuildIntent('make an index.html with a contact form', 'site'));
  assert.ok(detectBuildIntent('add a settings screen', 'site'));
  assert.ok(detectBuildIntent('can you add a settings screen?', 'site'));
  assert.ok(detectBuildIntent('agrega una pantalla de ajustes', 'site'));
  assert.ok(detectBuildIntent('arregla el boton de enviar', 'site'));
  assert.equal(detectBuildIntent('create a page for my bakery in the folder bakery', '')?.project, 'bakery');
  assert.deepEqual(detectBuildIntent('create a landing in the folder web', 'site'), {project: 'site', folder: 'web'});
  assert.equal(detectBuildIntent('How do I create a page in React?', 'site'), null);
  assert.equal(detectBuildIntent('what does this function do?', 'site'), null);
  assert.equal(detectBuildIntent('explain how flexbox works', 'site'), null);
  assert.equal(detectBuildIntent('Rename the variables of FUN_00401000 in ghidra and fix it', 'site'), null);
  assert.equal(detectBuildIntent('create a page for my bakery', ''), null);
});

test('decompiled code and shell snippets are not files', () => {
  const text = [
    'Decompiled:',
    '```c\nundefined4 FUN_00401000(int param_1)\n{\n  return *(undefined4 *)(param_1 + 0x10);\n}\n```',
    'Run it with:',
    '```bash\nnpm install && npm run build && npm run preview --host\n```',
    'And the page:',
    '```html\n<!doctype html>\n<html><body><h1>Real page content here</h1></body></html>\n```'
  ].join('\n');
  const blocks = codeBlocksOf(text);

  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].info, 'html');
});
