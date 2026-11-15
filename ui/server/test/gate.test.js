import { testRoot } from './setup.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { decideTool, isReadOnlyCommand, normalizeMode, toolKind } from '../gate.js';

const lookOnly = [
  'dir',
  'dir /b /o:n',
  'dir /a:d docs',
  'dir/b/s',
  'type config.json',
  'findstr /n /c:"render" *.ts',
  'findstr "x|y & z" readme.txt',
  'where git',
  'tree /a',
  'echo ready',
  'ver',
  'time /t',
  'tasklist /fi "status eq running"',
  'type log.txt | findstr error',
  'dir 2>nul',
  'dir 2>&1 | findstr assets',
  'git status',
  'git status -s',
  'git status -unormal',
  'git diff --name-only',
  'git diff --staged main',
  'git log --graph --oneline -20',
  'git log -n 3 --grep "fix"',
  'git show HEAD --name-status',
  'git branch -v',
  'git branch --show-current',
  'git tag -l',
  'git -C docs log -1',
  'git --no-pager diff',
  'git config --get-regexp alias',
  'git config core.editor',
  'git stash show -p',
  'git blame -L 10,20 app.js',
  'git ls-files -m',
  'git rev-parse --show-toplevel',
  'git describe --tags',
  'git log -1 && git status -s',
  'cargo --version',
  'node -v',
  'npm --version',
  'npm ls --depth 1',
  'npm.cmd outdated',
  'npm config list',
  'pip show requests',
  'ls -lh',
  'cat -n main.py',
  'head -c 100 data.bin',
  'tail -n 50 server.log',
  'wc -w notes.md',
  'grep -ril "fixme" lib',
  'rg -n "useState" src',
  'ipconfig /displaydns',
  'attrib /s *.tmp',
  'uniq -c counts.txt',
  'sort -r names.txt',
  'du -sh build'
];

const changing = [
  'cd docs && dir /b',
  'cd sub & git log',
  'del notes.txt',
  'erase *.log',
  'rd /s /q dist',
  'rmdir tmp',
  'npm install react',
  'npm run start',
  'npm version minor',
  'npm config set color false',
  'node build.js',
  'npx eslint --fix .',
  'git commit -am wip',
  'git push origin main',
  'git checkout -b next',
  'git branch feature-x',
  'git branch -D stale',
  'git branch -m old new',
  'git tag -a v2 -m release',
  'git stash',
  'git config core.editor vim',
  'git config edit',
  'git diff --output=patch.txt',
  'git -c alias.x=!rm log',
  'git log --format=%an',
  'git reflog expire --all',
  'git worktree add ../w',
  'git grep -O notepad todo',
  'git constructor',
  'dir > list.txt',
  'dir >> list.txt',
  'dir>list.txt',
  'echo hi > a.txt',
  'type a.txt | tee copy.txt',
  'dir & del a.txt',
  'dir && rd /s /q docs',
  'git status; rm -rf build',
  'git log || shutdown /r',
  'dir ^& del a',
  'dir & ',
  'powershell -c "ls"',
  'powershell -EncodedCommand ZABpAHIA',
  'pwsh -c dir',
  'cmd /c del x',
  '(dir)',
  'type %APPDATA%\\keys.txt',
  'findstr "%HOME%" a.txt',
  'tail -f server.log',
  'sort /o sorted.txt names.txt',
  'sort -o sorted.txt names.txt',
  'uniq names.txt unique.txt',
  'time 12:00',
  'start calc',
  'curl https://example.org',
  'certutil -decode a b',
  '"open quote',
  'findstr "x" < input.txt',
  'dir\nrd x',
  'C:\\bin\\findstr.exe x',
  '.\\tree.exe',
  'attrib -h secret.txt',
  'ipconfig /flushdns',
  'taskkill /pid 1234',
  'format d:',
  'set PATH=x',
  'find . -name "*.js" -delete',
  'rg --pre cat x',
  '',
  'x'.repeat(1200)
];

test('read-only commands are recognized', () => {
  for(const command of lookOnly){
    assert.equal(isReadOnlyCommand(command), true, command);
  }
});

test('anything that can change, chain, redirect or run code is not read-only', () => {
  for(const command of changing){
    assert.equal(isReadOnlyCommand(command), false, command);
  }
});

test('a program planted in the project folder is not trusted', () => {
  const folder = path.join(testRoot, 'shadow');
  fs.mkdirSync(folder, {recursive: true});
  fs.writeFileSync(path.join(folder, 'git.bat'), '@echo off');
  fs.writeFileSync(path.join(folder, 'node.vbs'), 'WScript.Echo 1');

  assert.equal(isReadOnlyCommand('git status', folder), false);
  assert.equal(isReadOnlyCommand('node -v', folder), false);
  assert.equal(isReadOnlyCommand('dir', folder), true);
});

test('reads outside the project and web pages ask first unless the mode is auto', () => {
  const project = path.join(testRoot, 'game');
  assert.equal(decideTool('manual', 'web_fetch', {url: 'https://example.com/?d=secret'}, project), 'ask');
  assert.equal(decideTool('manual', 'read_document', {file: path.join(os.homedir(), '.ssh', 'id_ed25519')}, project), 'ask');
  assert.equal(decideTool('manual', 'read_file', {project: 'C:\\', file: 'secrets.txt'}, project), 'ask');
  assert.equal(decideTool('manual', 'pc_search', {content: 'PRIVATE KEY'}, project), 'ask');
  assert.equal(decideTool('manual', 'read_file', {project: 'game', file: 'index.html'}, project), 'run');
  assert.equal(decideTool('manual', 'read_document', {file: path.join(project, 'notes.pdf')}, project), 'run');
  assert.equal(decideTool('manual', 'pc_search', {query: 'report.docx'}, project), 'run');
  assert.equal(decideTool('auto', 'web_fetch', {url: 'https://example.com'}, project), 'run');
});

test('old access values are migrated', () => {
  assert.equal(normalizeMode('ask'), 'manual');
  assert.equal(normalizeMode('full'), 'auto');
  assert.equal(normalizeMode('plan'), 'plan');
  assert.equal(normalizeMode('weird'), 'manual');
  assert.equal(normalizeMode(undefined, 'auto'), 'auto');
});

test('tool decisions follow the permission mode', () => {
  assert.equal(decideTool('plan', 'read_file', {file: 'a'}), 'run');
  assert.equal(decideTool('plan', 'write_file', {file: 'a'}), 'refuse');
  assert.equal(decideTool('manual', 'write_file', {file: 'a'}), 'ask');
  assert.equal(decideTool('auto', 'write_file', {file: 'a'}), 'run');

  assert.equal(decideTool('plan', 'run_command', {command: 'git status'}), 'run');
  assert.equal(decideTool('manual', 'run_command', {command: 'git status'}), 'run');
  assert.equal(decideTool('plan', 'run_command', {command: 'npm install'}), 'refuse');
  assert.equal(decideTool('manual', 'run_command', {command: 'npm install'}), 'ask');
  assert.equal(decideTool('auto', 'run_command', {command: 'npm install'}), 'run');

  assert.equal(decideTool('manual', 'terminal_start', {command: 'npm run dev'}), 'ask');
  assert.equal(decideTool('plan', 'terminal_start', {command: 'npm run dev'}), 'refuse');
  assert.equal(decideTool('manual', 'sandbox_start', {command: 'node server.js'}), 'ask');
  assert.equal(decideTool('manual', 'sandbox_run', {command: 'dir'}), 'run');
  assert.equal(decideTool('manual', 'terminal_start', {}), 'ask');

  assert.equal(decideTool('manual', 'delete_path', {target: 'x'}), 'ask');
  assert.equal(decideTool('plan', 'git_commit', {message: 'x'}), 'refuse');
  assert.equal(decideTool('manual', 'some_plugin_tool', {}), 'ask');

  assert.equal(decideTool('plan', 'ghidra', {command: 'DECOMP 00401000'}), 'run');
  assert.equal(decideTool('plan', 'ghidra', {command: 'RENAME_FUNC 00401000 Init'}), 'refuse');
  assert.equal(decideTool('manual', 'ida', {command: 'RENAME 401000 Init'}), 'ask');
  assert.equal(decideTool('manual', 'ghidra', {command: 'PATCH 00401000 90 90'}), 'ask');
});

test('tool kinds', () => {
  assert.equal(toolKind('grep'), 'free');
  assert.equal(toolKind('run_command'), 'command');
  assert.equal(toolKind('replace_in_file'), 'file');
  assert.equal(toolKind('ida'), 'reverse');
  assert.equal(toolKind('rename_path'), 'change');
});
