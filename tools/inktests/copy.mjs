const port = process.argv[2] || '8010';
const lines = [
  'read_file, terminal_list, foo_bar_baz: {"a": 1, "b": [2, 3]}',
  'get_user_by_id(user_id=42, include_deleted=False)',
  '{"name":"write_file","args":{"path":"src/app/main_window.py","line_count":17}}',
  'Box 252 is stored on shelf 28 and is painted blue; box 253 is on shelf 4, painted purple.',
  'MAX_RETRY_COUNT = 5; DEFAULT_TIMEOUT_MS = 2500; __all__ = ["parse_args", "run_main"]',
  'zq_vex lorn_pid 7731 kiv_ab_ux: [m, n, o_p] -> tr_88'
];
let exact = 0;
for(const line of lines){
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({messages: [{role: 'user', content: `Repeat this line exactly and output nothing else:\n${line}`}], max_tokens: 400, temperature: 0, reasoning_effort: 'none'})
  });
  const data = await response.json();
  const got = String(data.choices?.[0]?.message?.content ?? '').trim().replace(/^`+|`+$/g, '').trim();
  const ok = got === line;
  if(ok) exact++;
  console.log(`${ok ? 'OK  ' : 'BAD '} ${got.slice(0, 160)}`);
}
console.log(`exact ${exact} of ${lines.length}`);
