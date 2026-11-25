const port = process.argv[2] || '8010';
const colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'silver', 'brown'];
const lines = [];
for(let i = 1; i <= 1300; i++) lines.push(`Box ${i} is stored on shelf ${(i * 7) % 31} and is painted ${colors[(i * 13 + 5) % colors.length]}.`);
for(const needle of [37, 640, 1290]){
  const expected = colors[(needle * 13 + 5) % colors.length];
  const began = Date.now();
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: [{role: 'user', content: `${lines.join('\n')}\n\nWhat color is box ${needle}? Answer with one word.`}], max_tokens: 600, temperature: 0, reasoning_effort: 'low'})
  });
  const data = await response.json();
  const message = data.choices?.[0]?.message || {};
  const got = String(message.content ?? '').trim();
  console.log(`box ${needle} expected ${expected} got ${JSON.stringify(got.slice(0, 80))} ${got.toLowerCase().includes(expected) ? 'OK' : 'BAD'} ${((Date.now() - began) / 1000).toFixed(1)} s usage ${JSON.stringify(data.usage)}`);
}
