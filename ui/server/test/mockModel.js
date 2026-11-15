import http from 'http';

export const startModel = async () => {
  const state = {replies: [], requests: [], closed: 0};

  const server = http.createServer((req, res) => {
    let raw = '';

    req.on('data', chunk => {
      raw += chunk;
    });

    req.on('end', async () => {
      if(req.method === 'GET'){
        res.writeHead(200, {'Content-Type': 'application/json'});
        return res.end(JSON.stringify({data: [{id: 'mock-model'}]}));
      }

      const body = JSON.parse(raw || '{}');
      const index = state.requests.length;
      let ended = false;

      state.requests.push(body);
      res.on('close', () => {
        if(!ended){
          state.closed++;
        }
      });

      const next = state.replies.length > 1 ? state.replies.shift() : state.replies[0];
      const reply = typeof next === 'function'
        ? await next(body, index)
        : next || {content: 'Done.'};

      if(reply.wait){
        await reply.wait;
      }

      if(res.destroyed){
        return;
      }

      const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
      const text = String(reply.content || '');

      res.writeHead(200, {'Content-Type': 'text/event-stream'});

      for(const notice of reply.notices || []) send(notice);

      for(let i = 0; i < text.length; i += 40){
        send({choices: [{index: 0, delta: {content: text.slice(i, i + 40)}}]});
      }

      (reply.calls || []).forEach((call, i) => {
        const args = JSON.stringify(call.args || {});
        send({choices: [{index: 0, delta: {tool_calls: [{index: i, id: call.id || `call_${index}_${i}`, type: 'function', function: {name: call.name, arguments: args.slice(0, 5)}}]}}]});
        send({choices: [{index: 0, delta: {tool_calls: [{index: i, function: {arguments: args.slice(5)}}]}}]});
      });

      send({choices: [{index: 0, delta: {}, finish_reason: reply.finish || (reply.calls?.length ? 'tool_calls' : 'stop')}]});
      send({choices: [], usage: {prompt_tokens: reply.promptTokens || Math.ceil(raw.length / 3), completion_tokens: 12}});
      res.write('data: [DONE]\n\n');
      ended = true;
      res.end();
    });
  });

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

  return {
    url: `http://127.0.0.1:${server.address().port}/v1`,
    state,
    reset: (replies = []) => {
      state.replies = replies;
      state.requests = [];
      state.closed = 0;
    },
    close: () => new Promise(resolve => {
      server.closeAllConnections?.();
      server.close(resolve);
    })
  };
};
