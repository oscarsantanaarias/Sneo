const port = process.argv[2] || '8010';
const tools = [{type: 'function', function: {name: 'read_file', description: 'Read a file of the project', parameters: {type: 'object', properties: {file: {type: 'string'}}, required: ['file']}}}];
const filler = Array.from({length: 150}, (_, i) => `Rule ${i + 1}: keep the code style consistent, item ${i * 7 % 13}, and explain each change briefly.`).join('\n');
const system = {role: 'system', content: `You are a coding agent.\n${filler}`};
const asks = ['What is 17 times 23? Answer with the number only.', 'Name three primary colors, comma separated.', 'What is 17 times 23? Answer with the number only.', 'Name three primary colors, comma separated.'];
const outs = [];
for(const ask of asks){
  const began = Date.now();
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: [system, {role: 'user', content: ask}], tools, max_tokens: 160, temperature: 0, reasoning_effort: 'low'})
  });
  const data = await response.json();
  const message = data.choices?.[0]?.message || {};
  const text = `${message.reasoning_content || ''}|${message.content || ''}`;
  outs.push(text);
  console.log(`${((Date.now() - began) / 1000).toFixed(1)} s cached ${data.usage?.prompt_tokens_details?.cached_tokens} of ${data.usage?.prompt_tokens}: ${JSON.stringify((message.content || '').slice(0, 60))}`);
}
console.log(`repeat 1 same: ${outs[0] === outs[2]}, repeat 2 same: ${outs[1] === outs[3]}`);
if(outs[0] !== outs[2]) console.log(`A1 ${JSON.stringify(outs[0].slice(0, 200))}\nA2 ${JSON.stringify(outs[2].slice(0, 200))}`);
if(outs[1] !== outs[3]) console.log(`B1 ${JSON.stringify(outs[1].slice(0, 200))}\nB2 ${JSON.stringify(outs[3].slice(0, 200))}`);
