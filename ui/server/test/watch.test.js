import test from 'node:test';
import assert from 'node:assert/strict';
import { LoopWatch, watchNotes, silentNote } from '../watch.js';

test('searches that keep finding nothing are redirected, then stopped', () => {
  const guard = new LoopWatch({mode: 'auto'});

  assert.equal(guard.tool('grep', {query: 'a'}, []), null);
  assert.equal(guard.tool('glob', {pattern: 'b'}, []), null);
  assert.ok(guard.tool('search_files', {query: 'c'}, [])?.redirect);
  assert.equal(guard.tool('grep', {query: 'd'}, [])?.stop, 'searches-failed');
  assert.equal(guard.reason, 'searches-failed');
});

test('a search that finds something resets the empty streak', () => {
  const guard = new LoopWatch({mode: 'auto'});

  guard.tool('grep', {query: 'a'}, []);
  guard.tool('grep', {query: 'b'}, []);
  guard.tool('grep', {query: 'c'}, [{file: 'x.js', line: 1}]);
  guard.tool('grep', {query: 'd'}, []);
  assert.equal(guard.tool('grep', {query: 'e'}, []), null);
});

test('too many searches in one turn stop the run, sooner in plan mode', () => {
  const auto = new LoopWatch({mode: 'auto'});
  const plan = new LoopWatch({mode: 'plan'});
  let autoStop = null;
  let planStop = null;

  for(let i = 0; i < 40 && !autoStop; i++){
    autoStop = auto.tool('grep', {query: `q${i}`}, [{file: 'a'}])?.stop || null;
  }

  for(let i = 0; i < 40 && !planStop; i++){
    planStop = plan.tool('grep', {query: `q${i}`}, [{file: 'a'}])?.stop || null;
  }

  assert.equal(autoStop, 'searches');
  assert.equal(auto.searches, 30);
  assert.equal(planStop, 'searches');
  assert.equal(plan.searches, 15);
});

test('the same failing call is redirected once and then stopped', () => {
  const guard = new LoopWatch({mode: 'auto'});
  const failure = {error: 'ENOENT: no such file'};

  assert.equal(guard.tool('read_file', {file: 'a.js'}, failure), null);
  assert.ok(guard.tool('read_file', {file: 'a.js'}, failure)?.redirect);
  assert.equal(guard.tool('read_file', {file: 'a.js'}, failure)?.stop, 'repeat');
});

test('different failures do not count as a repeat', () => {
  const guard = new LoopWatch({mode: 'auto'});

  guard.tool('read_file', {file: 'a.js'}, {error: 'ENOENT'});
  guard.tool('read_file', {file: 'b.js'}, {error: 'ENOENT'});
  assert.equal(guard.tool('read_file', {file: 'c.js'}, {error: 'ENOENT'}), null);
});

test('a plan rewritten again and again is stopped in plan mode only', () => {
  const plan = new LoopWatch({mode: 'plan'});
  const auto = new LoopWatch({mode: 'auto'});
  const text = 'Plan:\n1. Read app.js\n2. Change the header\n3. Build';

  assert.equal(plan.text(text), null);
  assert.equal(plan.text(text), null);
  assert.equal(plan.text(text), null);
  assert.equal(plan.text(text)?.stop, 'rewrites');

  for(let i = 0; i < 6; i++){
    assert.equal(auto.text(text), null);
  }
});

test('plan_create called over and over counts as rewriting', () => {
  const guard = new LoopWatch({mode: 'auto'});

  guard.tool('plan_create', {title: 'a'}, {id: 1});
  guard.tool('plan_create', {title: 'a'}, {id: 2});
  guard.tool('plan_create', {title: 'a'}, {id: 3});
  assert.equal(guard.tool('plan_create', {title: 'a'}, {id: 4})?.stop, 'rewrites');
});

test('a context above 93% of the window stops the run', () => {
  const guard = new LoopWatch({mode: 'auto', window: 1000});

  assert.equal(guard.context(900), null);
  assert.equal(guard.context(940)?.stop, 'context');
  assert.equal(guard.context(10)?.stop, 'context');
});

test('every stop has a plain sentence and an empty end is never silent', () => {
  for(const reason of ['searches-failed', 'searches', 'rewrites', 'repeat', 'context']){
    assert.ok(watchNotes[reason].length > 40, reason);
  }

  assert.ok(silentNote(false).length > 20);
  assert.ok(silentNote(true).length > 20);
  assert.notEqual(silentNote(false), silentNote(true));
});
