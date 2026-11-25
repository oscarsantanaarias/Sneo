import fs from 'fs';
const port = process.argv[2] || '8010';
const file = new URL('./ref/multicase.json', import.meta.url);
const kase = JSON.parse(fs.readFileSync(file, 'utf8'));
for(const m of kase.messages){
  for(const c of m.tool_calls || []) c.function.arguments = JSON.stringify(c.function.arguments);
}
const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
  method: 'POST',
  headers: {'content-type': 'application/json'},
  body: JSON.stringify({messages: kase.messages, tools: kase.tools, max_tokens: 300, temperature: 0})
});
const data = await response.json();
const choice = data.choices?.[0] || {};
console.log(`finish ${choice.finish_reason}, usage ${JSON.stringify(data.usage || data.error || {})}`);
console.log(`content: ${String(choice.message?.content ?? '').slice(0, 300)}`);
console.log(`tool_calls: ${JSON.stringify(choice.message?.tool_calls || null)}`);
