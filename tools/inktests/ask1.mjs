const port = process.argv[2] || '8010';
const began = Date.now();
const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
  method: 'POST',
  headers: {'content-type': 'application/json'},
  body: JSON.stringify({messages: [{role: 'user', content: 'What is the capital of France? Answer with one word.'}], max_tokens: 30, temperature: 0, chat_template_kwargs: {enable_thinking: false}})
});
const data = await response.json();
const message = data.choices?.[0]?.message || {};
console.log(`--- ${((Date.now() - began) / 1000).toFixed(1)} s, usage ${JSON.stringify(data.usage || data.error || {})}`);
if(message.reasoning_content) console.log(`[thinking] ${message.reasoning_content.slice(0, 300)}`);
console.log(`content: ${JSON.stringify(message.content ?? data)}`);
