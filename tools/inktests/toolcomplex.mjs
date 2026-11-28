const port = process.argv[2] || '8010';
const tools = [
  {type: 'function', function: {name: 'write_file', description: 'Create or overwrite a file with the given content', parameters: {type: 'object', properties: {path: {type: 'string'}, content: {type: 'string', description: 'Full file content'}}, required: ['path', 'content']}}},
  {type: 'function', function: {name: 'run_tests', description: 'Run test files with options', parameters: {type: 'object', properties: {files: {type: 'array', items: {type: 'string'}}, options: {type: 'object', properties: {verbose: {type: 'boolean'}, timeout: {type: 'integer'}}}}, required: ['files']}}}
];
const asks = [
  'Create hello.py with a function greet(name) that returns the string f"Hello, {name}!" and a main block that prints greet("World"). Use the tool.',
  'Run the tests tests/test_a.py and tests/test_b.py verbosely with a timeout of 30 seconds. Use the tool.'
];
for(const content of asks){
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: [{role: 'system', content: 'You are a coding agent. Use the tools.'}, {role: 'user', content}], tools, max_tokens: 900, temperature: 0})
  });
  const data = await response.json();
  const choice = data.choices?.[0] || {};
  const calls = choice.message?.tool_calls || [];
  console.log(`finish ${choice.finish_reason}, ${calls.length} calls`);
  for(const c of calls){
    let parsed = null;
    try { parsed = JSON.parse(c.function.arguments); } catch(e) { console.log(`  BAD JSON: ${c.function.arguments.slice(0, 200)}`); continue; }
    console.log(`  ${c.function.name} ${JSON.stringify(parsed).slice(0, 400)}`);
  }
  if(!calls.length) console.log(`  content: ${String(choice.message?.content ?? '').slice(0, 300)}`);
}
