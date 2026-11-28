const port = process.argv[2] || '8010';
const colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'silver', 'brown'];
const lines = [];
for(let i = 1; i <= 420; i++) lines.push(`Box ${i} is stored on shelf ${(i * 7) % 31} and is painted ${colors[(i * 13 + 5) % colors.length]}.`);
const needle = 263;
const expected = colors[(needle * 13 + 5) % colors.length];
const asks = [
  {name: 'short', messages: [{role: 'user', content: 'What is the capital of France? Answer with one word.'}], thinking: false, max: 160},
  {name: 'haiku', messages: [{role: 'user', content: 'Write a haiku about the sea.'}], thinking: true, max: 900},
  {name: 'needle', messages: [{role: 'user', content: `${lines.join('\n')}\n\nWhat color is box ${needle}? Answer with one word.`}], thinking: false, max: 200, expected},
  {name: 'code', messages: [{role: 'user', content: 'Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.'}], thinking: false, max: 700}
];
for(const ask of asks){
  const began = Date.now();
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: ask.messages, max_tokens: ask.max, temperature: 0, chat_template_kwargs: {enable_thinking: ask.thinking}})
  });
  const data = await response.json();
  const message = data.choices?.[0]?.message || {};
  console.log(`--- ${ask.name} ${((Date.now() - began) / 1000).toFixed(1)} s, usage ${JSON.stringify(data.usage || data.error || {})}${ask.expected ? `, expected ${ask.expected}` : ''}`);
  if(message.reasoning_content) console.log(`[thinking] ${message.reasoning_content.slice(0, 400)}`);
  console.log(String(message.content ?? JSON.stringify(data)).slice(0, 800));
}
