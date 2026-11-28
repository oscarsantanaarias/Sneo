const port = process.argv[2] || '8010';
const content = 'Repeat: read_file, terminal_list, foo_bar_baz: {"a": 1, "b": [2, 3]} __init__ x_y  two  spaces\n\nnew lines\ttab 12345 3.14 CamelCaseWord HTTPServer don\'t I\'LL émigré naïve 日本語 😀 a//b ../src/main.js';
const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
  method: 'POST',
  headers: {'content-type': 'application/json'},
  body: JSON.stringify({messages: [{role: 'user', content}], max_tokens: 4, temperature: 0, reasoning_effort: 'none'})
});
const data = await response.json();
console.log(JSON.stringify(data.usage));
