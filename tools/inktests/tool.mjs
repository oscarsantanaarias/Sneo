const port = process.argv[2] || '8010';
const tools = [
  {type: 'function', function: {name: 'read_file', description: 'Read a file of the project', parameters: {type: 'object', properties: {file: {type: 'string', description: 'Path of the file'}, startLine: {type: 'integer'}}, required: ['file']}}},
  {type: 'function', function: {name: 'terminal_list', description: 'List the running terminals', parameters: {type: 'object', properties: {}}}}
];
const asks = [
  {name: 'tool-read', messages: [{role: 'system', content: 'You are a coding agent. Use the tools.'}, {role: 'user', content: 'Open src/main.js from line 10 and tell me what it does.'}], thinking: true, max: 900},
  {name: 'tool-empty', messages: [{role: 'user', content: 'Which terminals are running right now? Use the tool.'}], thinking: false, max: 400}
];
for(const ask of asks){
  const began = Date.now();
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: ask.messages, tools, max_tokens: ask.max, temperature: 0, chat_template_kwargs: {enable_thinking: ask.thinking}})
  });
  const data = await response.json();
  const choice = data.choices?.[0] || {};
  const message = choice.message || {};
  console.log(`--- ${ask.name} ${((Date.now() - began) / 1000).toFixed(1)} s, finish ${choice.finish_reason}, usage ${JSON.stringify(data.usage || data.error || {})}`);
  if(message.reasoning_content) console.log(`[thinking] ${message.reasoning_content.slice(0, 300)}`);
  console.log(`content: ${String(message.content ?? '').slice(0, 300)}`);
  console.log(`tool_calls: ${JSON.stringify(message.tool_calls || null)}`);
}
