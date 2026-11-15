import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateRatio, clipText, estimateTokens, fitMessages, outputLimit, promptBudget, summaryHeader } from '../fit.js';

const longText = (label, size) => `${label} `.repeat(Math.ceil(size / (label.length + 1))).slice(0, size);

const conversation = () => {
  const system = {role: 'system', content: longText('system rules', 6000)};
  const list = [system];

  for(let i = 0; i < 8; i++){
    list.push({role: 'user', content: longText(`question ${i}`, 9000)});
    list.push({role: 'assistant', content: '', tool_calls: [{id: `call_${i}`, type: 'function', function: {name: 'read_file', arguments: `{"file":"f${i}.js"}`}}]});
    list.push({role: 'tool', tool_call_id: `call_${i}`, content: longText(`file body ${i}`, 12000)});
    list.push({role: 'assistant', content: longText(`answer ${i}`, 3000)});
  }

  return list;
};

test('nothing changes when the conversation fits', async () => {
  const messages = conversation().slice(0, 5);
  const fitted = await fitMessages({messages, window: 131072, ratio: 3, summarize: async () => 'unused'});

  assert.equal(fitted.compacted, false);
  assert.equal(fitted.messages, messages);
});

test('the oldest part becomes one summary after an identical system prompt', async () => {
  const messages = conversation();
  const pinned = {role: 'user', content: 'CURRENT: add a footer to index.html'};
  messages.push(pinned);
  const window = 32768;
  const asked = [];
  const fitted = await fitMessages({
    messages,
    window,
    ratio: 3,
    pinned,
    summarize: async (text) => {
      asked.push(text);
      return 'The user asked about eight files and the agent read them.';
    }
  });

  assert.equal(fitted.compacted, true);
  assert.equal(fitted.messages[0], messages[0]);
  assert.equal(fitted.messages[0].content, messages[0].content);
  assert.ok(fitted.messages[1].content.startsWith(summaryHeader));
  assert.ok(fitted.messages[1].content.includes('eight files'));
  assert.equal(fitted.messages.filter(item => String(item.content).includes(summaryHeader)).length, 1);
  assert.ok(fitted.after <= promptBudget(window));
  assert.ok(fitted.after < fitted.before);
  assert.ok(asked.length >= 1);

  const open = new Set();

  for(const item of fitted.messages){
    for(const call of item.tool_calls || []){
      open.add(call.id);
    }

    if(item.role === 'tool'){
      assert.ok(open.has(item.tool_call_id), `tool result ${item.tool_call_id} lost its call`);
      open.delete(item.tool_call_id);
    }
  }

  assert.equal(open.size, 0);
  assert.equal(fitted.messages[fitted.messages.length - 1].content.includes('CURRENT: add a footer'), true);
});

test('the current request survives word for word when it is summarized', async () => {
  const pinned = {role: 'user', content: 'CURRENT: rename the export button to Download'};
  const messages = [{role: 'system', content: 'rules'}, pinned];

  for(let i = 0; i < 6; i++){
    messages.push({role: 'assistant', content: '', tool_calls: [{id: `c${i}`, type: 'function', function: {name: 'grep', arguments: '{"query":"export"}'}}]});
    messages.push({role: 'tool', tool_call_id: `c${i}`, content: longText(`hits ${i}`, 9000)});
  }

  messages.push({role: 'assistant', content: 'Found it in toolbar.js.'});

  const fitted = await fitMessages({messages, window: 16384, ratio: 3, pinned, summarize: async () => 'Searched for the export button.'});

  assert.equal(fitted.compacted, true);
  assert.ok(fitted.region.includes(pinned));
  assert.ok(fitted.messages[1].content.includes('CURRENT: rename the export button to Download'));
  assert.notEqual(fitted.messages[2].role, 'tool');
});

test('a failing summarizer still produces a summary', async () => {
  const messages = conversation();
  messages.push({role: 'user', content: 'go on'});
  const fitted = await fitMessages({messages, window: 32768, ratio: 3, summarize: async () => { throw new Error('model down'); }});

  assert.equal(fitted.compacted, true);
  assert.ok(fitted.messages[1].content.includes('The user wrote: question 0'));
});

test('tool output is capped relative to the window', () => {
  const small = outputLimit(8192, 3);
  const large = outputLimit(131072, 3);
  const text = longText('row', 200000);
  const clipped = clipText(text, small);

  assert.ok(small < large);
  assert.ok(clipped.length < small + 300);
  assert.ok(clipped.startsWith('row row'));
  assert.ok(clipped.includes('output cut'));
});

test('the chars per token ratio follows what the engine reports', () => {
  assert.equal(calibrateRatio(0, 3000, 1000), 3);
  assert.ok(calibrateRatio(3, 2000, 1000) < 3);
  assert.equal(calibrateRatio(2.5, 100, 0), 2.5);
  assert.equal(calibrateRatio(2.5, 100000, 10), 2.5);
  assert.ok(estimateTokens([{role: 'user', content: 'x'.repeat(3000)}], null, 2) > estimateTokens([{role: 'user', content: 'x'.repeat(3000)}], null, 4));
});

test('when the system prompt alone does not fit, nothing is summarized', async () => {
  const messages = [{role: 'system', content: longText('rules', 40000)}, {role: 'user', content: 'old'}, {role: 'assistant', content: 'older'}, {role: 'user', content: 'now'}];
  let called = 0;
  const fitted = await fitMessages({messages, window: 8192, ratio: 3, summarize: async () => { called++; return 'x'; }});

  assert.equal(fitted.compacted, false);
  assert.equal(fitted.hopeless, true);
  assert.equal(fitted.messages, messages);
  assert.equal(called, 0);
});
