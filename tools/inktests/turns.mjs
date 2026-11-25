const port = process.argv[2] || '8010';
const tools = [{type: 'function', function: {name: 'read_file', description: 'Read a file of the project', parameters: {type: 'object', properties: {file: {type: 'string'}}, required: ['file']}}}];
const filler = Array.from({length: 120}, (_, i) => `Rule ${i + 1}: keep the code style consistent and explain each change briefly.`).join('\n');
const messages = [{role: 'system', content: `You are a coding agent.\n${filler}`}, {role: 'user', content: 'Open src/main.js and tell me what it does.'}];
for(let turn = 0; turn < 3; turn++){
  const began = Date.now();
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages, tools, max_tokens: 300, temperature: 0})
  });
  const data = await response.json();
  const message = data.choices?.[0]?.message || {};
  console.log(`turn ${turn} ${((Date.now() - began) / 1000).toFixed(1)} s, usage ${JSON.stringify(data.usage)}`);
  messages.push({role: 'assistant', content: message.content || '', reasoning_content: message.reasoning_content, tool_calls: message.tool_calls});
  if(message.tool_calls?.length){
    for(const c of message.tool_calls) messages.push({role: 'tool', tool_call_id: c.id, content: 'const app = express();\napp.listen(3000);'});
  } else messages.push({role: 'user', content: 'Thanks, now what port does it use?'});
}
