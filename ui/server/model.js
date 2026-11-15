import { runnerUrl, defaultModel } from './config.js';
import { cleanText, readManualModels, readSettings } from './storage.js';

const effortLevels = {off: 'none', light: 'low', normal: 'medium', deep: 'high', full: 'max'};

export const getCurrentModel = async () => {
  const settings = await readSettings();
  return cleanText(settings.model) || defaultModel;
};

let engineSeen = {at: 0, model: '', context: 0};

export const forgetEngine = () => {
  engineSeen = {at: 0, model: '', context: 0};
};

export const readEngine = async () => {
  if(Date.now() - engineSeen.at < 3000) return engineSeen;
  const settings = await readSettings().catch(() => ({}));
  const base = cleanText(settings.providerBaseUrl).replace(/\/+$/, '').replace(/\/v1$/, '');
  const seen = {at: Date.now(), model: '', context: 0, vision: false, visionProblem: ''};
  if(base){
    try {
      const [models, pool] = await Promise.all([
        fetch(`${base}/api/models`, {signal: AbortSignal.timeout(2000)}).then(r => r.json()),
        fetch(`${base}/api/pool`, {signal: AbortSignal.timeout(2000)}).then(r => r.json())
      ]);
      seen.model = models.loaded ? cleanText(models.currentModel).split(/[\\/]/).pop() : '';
      seen.context = Number(pool.context) || 0;
      seen.vision = pool.vision === true;
      seen.visionProblem = cleanText(pool.visionProblem);
    } catch(e) {
      seen.model = '';
    }
  }
  engineSeen = seen;
  return seen;
};

export const contextWindow = async (settings) => {
  const asked = Number(settings.numCtx) || 0;
  const engine = (await readEngine()).context;
  return engine
    ? Math.min(asked || engine, engine)
    : asked || (settings.providerBaseUrl ? 131072 : 8192);
};

const getProvider = async () => {
  const settings = await readSettings();
  const baseUrl = cleanText(settings.providerBaseUrl);

  if(!baseUrl){
    return null;
  }

  return {baseUrl: baseUrl.replace(/\/+$/, ''), apiKey: cleanText(settings.providerApiKey)};
};

const trackUsage = (run, call) => {
  if(!run?.usage){
    return;
  }

  const usage = run.usage;
  usage.rounds++;
  usage.inputTokens += call.promptTokens;
  usage.outputTokens += call.outputTokens;
  usage.cachedTokens += call.cachedTokens;
  usage.lastPromptTokens = call.promptTokens;
  usage.lastPromptExact = call.exactPrompt === true;
  usage.thinkMs += call.thinkMs;
  usage.writeMs += call.writeMs;
  usage.firstTokenMs = Math.round(usage.firstTokenMs + (call.firstTokenMs - usage.firstTokenMs) / usage.rounds);
  usage.outputSpeed = call.outputSpeed > 0 ? Math.round(call.outputSpeed * 10) / 10 : usage.outputSpeed;
  usage.promptSpeed = call.promptSpeed > 0 ? Math.round(call.promptSpeed * 10) / 10 : usage.promptSpeed;
  usage.contextLimit = call.contextLimit || usage.contextLimit;

  run.onEvent?.({type: 'usage', usage: {...usage}});
};

const openaiChatRequest = async (currentModel, messages, tools, provider, run, side = false, plain = false) => {
  const settings = await readSettings().catch(() => ({}));
  const contextLimit = await contextWindow(settings);
  const room = Math.min(16384, Math.floor(contextLimit / 2));
  const body = {
    model: currentModel,
    messages: messages.map(({images, _claude, ...item}) => images?.length
      ? {...item, content: [{type: 'text', text: String(item.content ?? '')}, ...images.map(url => ({type: 'image_url', image_url: {url}}))]}
      : item),
    stream: true,
    stream_options: {include_usage: true}
  };

  // una api externa solo recibe los campos estandar, lo propio del motor de esta pc va solo al motor
  if(plain){
    if(/api\.openai\.com/i.test(provider.baseUrl)){
      body.max_completion_tokens = room;
    } else {
      body.max_tokens = room;
    }
  } else {
    body.sneo_side = side;
    body.max_tokens = room;
    body.temperature = typeof settings.temperature === 'number'
      ? settings.temperature
      : run?.thinkDepth === 'off'
        ? 0.7
        : 0.6;

    if(typeof settings.temperature !== 'number' && run?.thinkDepth === 'off'){
      body.top_p = 0.8;
    }
  }

  if(tools?.length){
    body.tools = tools;
  }

  if(!plain && effortLevels[run?.thinkDepth]){
    body.reasoning_effort = effortLevels[run.thinkDepth];
    body.chat_template_kwargs = {enable_thinking: run.thinkDepth !== 'off'};
  }

  if(!plain && run?.thinkDepth !== 'off'){
    body.thinking_budget = run?.thinkCap > 0
      ? Math.min(run.thinkCap, Math.floor(room * 0.75))
      : Math.floor(room / 2);
  }

  const startedAt = Date.now();
  const req = await fetch(`${provider.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(provider.apiKey ? {Authorization: `Bearer ${provider.apiKey}`} : {})
    },
    body: JSON.stringify(body),
    signal: run?.signal
  });

  if(!req.ok){
    const raw = await req.text().catch(() => '');
    let info = null;

    try {
      info = JSON.parse(raw);
    } catch(e) {
      info = null;
    }

    const reason = info?.error?.message || (typeof info?.error === 'string' ? info.error : '') || info?.message || raw.slice(0, 300);
    throw new Error(reason || `Provider answered ${req.status}`);
  }

  const call = {content: '', reasoning: '', toolCalls: [], finishReason: null, promptTokens: 0, outputTokens: 0, cachedTokens: 0, thinkMs: 0, writeMs: 0, firstTokenMs: 0, outputSpeed: 0, promptSpeed: 0, contextLimit};
  let usage = null;
  let timings = null;
  let lastAt = 0;
  let deltas = 0;
  let pendingThought = '';

  if(!/event-stream/i.test(req.headers.get('content-type') || '')){
    const res = await req.json();
    const choice = res.choices?.[0] || {};
    const message = choice.message || {};
    call.content = message.content || '';
    call.reasoning = message.reasoning_content || message.reasoning || '';
    call.toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    call.finishReason = choice.finish_reason || null;
    call.firstTokenMs = Date.now() - startedAt;
    usage = res.usage || null;
    timings = res.timings || null;

    if(call.reasoning){
      run?.onEvent?.({type: 'thought', thought: call.reasoning});
    }
  } else {
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    let toldAt = 0;

    for await (const chunk of req.body){
      buffer += decoder.decode(chunk, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for(const line of lines){
        const data = line.trim();

        if(!data.startsWith('data:')){
          continue;
        }

        const payload = data.slice(5).trim();

        if(payload === '[DONE]'){
          finished = true;
          break;
        }

        let event;

        try {
          event = JSON.parse(payload);
        } catch(e) {
          continue;
        }

        if(event.error){
          throw new Error(event.error.message || String(event.error));
        }

        usage = event.usage || usage;
        timings = event.timings || timings;

        if(event.prefill){
          run?.onEvent?.({type: 'prefill', tokens: Number(event.prefill.tokens) || 0, cached: Number(event.prefill.cached) || 0, ms: event.prefill.ms === undefined ? null : Number(event.prefill.ms)});
        }

        if(event.speed){
          const tokens = Number(event.speed.tokens) || 0;
          const ms = Number(event.speed.ms) || 0;
          run?.onEvent?.({type: 'speed', tokens, ms, speed: ms > 0 ? tokens * 1000 / ms : 0});
        }

        if(event.writing){
          const head = String(event.writing.head || '');
          const tool = (head.match(/"name"\s*:\s*"([^"\\]{1,80})"/) || head.match(/<function=([\w.-]{1,80})>/) || [])[1] || '';
          const file = (head.match(/"(?:file|file_path|path|filename|target)"\s*:\s*"([^"\\]{1,200})"/) || head.match(/<parameter=(?:file|file_path|path|filename)>\s*([^\n<]{1,200})/) || [])[1] || '';
          run?.onEvent?.({type: 'writing', tool, file: file.trim(), tokens: Number(event.writing.tokens) || 0});
        }

        const choice = event.choices?.[0];

        if(!choice){
          continue;
        }

        call.finishReason = choice.finish_reason || call.finishReason;

        const delta = choice.delta || {};
        const thought = delta.reasoning_content || delta.reasoning || '';

        if(!thought && !delta.content && !delta.tool_calls?.length){
          continue;
        }

        const now = Date.now();

        if(!lastAt){
          call.firstTokenMs = now - startedAt;
          lastAt = now;
        }

        if(thought){
          call.thinkMs += now - lastAt;
        } else {
          call.writeMs += now - lastAt;
        }

        lastAt = now;
        deltas++;

        if(thought){
          call.reasoning += thought;
          pendingThought += thought;
          const cut = pendingThought.lastIndexOf('\n');

          if(cut >= 0 || pendingThought.length > 400){
            const end = cut >= 0 ? cut + 1 : pendingThought.length;
            run?.onEvent?.({type: 'thought', thought: pendingThought.slice(0, end), delta: true});
            pendingThought = pendingThought.slice(end);
          }
        } else if(pendingThought){
          run?.onEvent?.({type: 'thought', thought: pendingThought, delta: true});
          pendingThought = '';
        }

        if(delta.content){
          call.content += delta.content;
        }

        for(const part of delta.tool_calls || []){
          const index = Number.isInteger(part.index) ? part.index : call.toolCalls.length;
          const slot = call.toolCalls[index] || (call.toolCalls[index] = {id: '', type: 'function', function: {name: '', arguments: ''}});
          const args = part.function?.arguments;

          if(part.id){
            slot.id = part.id;
          }

          if(part.function?.name && !slot.function.name){
            slot.function.name = part.function.name;
          }

          if(args){
            slot.function.arguments += typeof args === 'string' ? args : JSON.stringify(args);
          }

          slot.pieces = (slot.pieces || 0) + 1;
        }

        // mientras escribe una herramienta larga la app muestra cual y cuanto lleva
        if(delta.tool_calls?.length && now - toldAt > 700){
          toldAt = now;
          const slot = call.toolCalls[call.toolCalls.length - 1];
          const file = (String(slot?.function.arguments || '').match(/"(?:file|file_path|path|filename|target)"\s*:\s*"([^"\\]{1,200})"/) || [])[1] || '';
          run?.onEvent?.({type: 'writing', tool: slot?.function.name || '', file, tokens: slot?.pieces || 0});
        }
      }

      if(finished){
        break;
      }
    }

    if(pendingThought){
      run?.onEvent?.({type: 'thought', thought: pendingThought, delta: true});
    }
  }

  const genMs = lastAt ? lastAt - startedAt - call.firstTokenMs : 0;
  call.promptTokens = usage?.prompt_tokens || 0;
  call.exactPrompt = Number(usage?.prompt_tokens) > 0;
  call.outputTokens = usage?.completion_tokens || deltas;
  call.cachedTokens = usage?.prompt_tokens_details?.cached_tokens || timings?.cache_n || 0;
  call.outputSpeed = timings?.predicted_per_second || (genMs > 0 ? call.outputTokens / (genMs / 1000) : 0);
  call.promptSpeed = timings?.prompt_per_second || (call.promptTokens && call.firstTokenMs > 0 ? (call.promptTokens - call.cachedTokens) / (call.firstTokenMs / 1000) : 0);
  trackUsage(run, call);

  return {
    content: call.content,
    reasoning: call.reasoning,
    tool_calls: call.toolCalls.filter(Boolean).map(({pieces, ...item}, i) => ({...item, id: item.id || `call_${startedAt}_${i}`})),
    finishReason: call.finishReason
  };
};

// una ruta del equipo de agentes: el motor local con sus campos, o una api externa con lo estandar
export const askPlain = async (route, messages, tools, run) => {
  if(!route.baseUrl){
    throw new Error(`${route.name} has no address`);
  }

  const provider = {baseUrl: route.baseUrl.replace(/\/+$/, ''), apiKey: route.apiKey || ''};
  return await openaiChatRequest(route.model, messages, tools, provider, run, true, route.kind !== 'local');
};

export const getRunnerModels = async () => {
  try {
    const runnerReq = await fetch(`${runnerUrl}/api/tags`);
    if(!runnerReq.ok) return [];
    const runnerRes = await runnerReq.json();
    return runnerRes.models || [];
  } catch(e) {
    return [];
  }
};

export const getProviderModels = async (provider) => {
  if(!provider?.baseUrl) return [];
  try {
    const pUrl = provider.baseUrl.endsWith('/v1') ? `${provider.baseUrl}/models` : `${provider.baseUrl}/v1/models`;
    const req = await fetch(pUrl, {
      headers: provider.apiKey ? {Authorization: `Bearer ${provider.apiKey}`} : {}
    });
    if(!req.ok) return [];
    const res = await req.json();
    const dataList = res.data || res.models || [];
    return dataList.map(item => ({
      name: item.id || item.name,
      size: item.size || null,
      modified_at: item.created ? new Date(item.created * 1000).toISOString() : null
    }));
  } catch(e) {
    return [];
  }
};

export const getAllModels = async () => {
  const provider = await getProvider();
  const providerModels = await getProviderModels(provider);
  const runnerModels = await getRunnerModels();
  const manualModels = (await readManualModels()).map(name => ({
    name,
    size: null,
    modified_at: null,
    manual: true
  }));

  const all = [...providerModels, ...runnerModels, ...manualModels];
  const seen = new Set();

  return all.filter(item => {
    if(!item.name || seen.has(item.name)){
      return false;
    }
    seen.add(item.name);
    return true;
  });
};

const chatRequest = async (currentModel, messages, tools, run, ctxOverride) => {
  const settings = await readSettings().catch(() => ({}));
  const ctx = ctxOverride || Number(settings.numCtx) || 8192;
  const body = {
    model: currentModel,
    messages,
    stream: false,
    keep_alive: '10m',
    options: {
      num_ctx: ctx,
      temperature: typeof settings.temperature === 'number' ? settings.temperature : 0.35,
      top_p: 0.9,
      repeat_penalty: 1.1
    }
  };

  if(!run?.thinkDepth || run.thinkDepth === 'off'){
    body.think = false;
  }

  if(tools?.length){
    body.tools = tools;
  }

  let runnerReq;
  let lastErr;
  const signal = run?.signal;
  const requestSignal = signal
    ? AbortSignal.any([signal, AbortSignal.timeout(240000)])
    : AbortSignal.timeout(240000);

  for(let attempt = 0; attempt < 4; attempt++){
    if(signal?.aborted){
      throw Object.assign(new Error('aborted'), {name: 'AbortError'});
    }

    try {
      runnerReq = await fetch(`${runnerUrl}/api/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
        signal: requestSignal
      });
      break;
    } catch(e) {
      if(e.name === 'AbortError' || e.name === 'TimeoutError' || requestSignal.aborted){
        throw e;
      }

      lastErr = e;
      await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
    }
  }

  if(!runnerReq){
    throw new Error(`The engine did not answer (${lastErr?.message || 'fetch failed'}). Check that master.exe is running and look at master.log, then try again.`);
  }

  const runnerRes = await runnerReq.json();

  if(!runnerReq.ok){
    throw new Error(runnerRes.error || 'The model answered with an error');
  }

  const message = runnerRes.message || {};
  const content = message.content || '';
  const reasoning = message.thinking || '';
  const evalMs = (runnerRes.eval_duration || 0) / 1e6;
  const promptMs = (runnerRes.prompt_eval_duration || 0) / 1e6;
  const thinkShare = reasoning ? reasoning.length / (reasoning.length + content.length) : 0;

  if(reasoning){
    run?.onEvent?.({type: 'thought', thought: reasoning});
  }

  trackUsage(run, {
    promptTokens: runnerRes.prompt_eval_count || 0,
    outputTokens: runnerRes.eval_count || 0,
    cachedTokens: 0,
    thinkMs: Math.round(evalMs * thinkShare),
    writeMs: Math.round(evalMs * (1 - thinkShare)),
    firstTokenMs: Math.round((runnerRes.load_duration || 0) / 1e6 + promptMs),
    outputSpeed: evalMs > 0 ? (runnerRes.eval_count || 0) / (evalMs / 1000) : 0,
    promptSpeed: promptMs > 0 ? (runnerRes.prompt_eval_count || 0) / (promptMs / 1000) : 0,
    contextLimit: ctx
  });

  return {
    content,
    reasoning,
    tool_calls: Array.isArray(message.tool_calls) ? message.tool_calls.map((item, i) => ({...item, id: item.id || `call_${Date.now()}_${i}`})) : [],
    finishReason: runnerRes.done_reason === 'length' ? 'length' : 'stop'
  };
};

export const askModel = async (messages, run, modelOverride) => {
  const currentModel = cleanText(modelOverride) || await getCurrentModel();
  const provider = await getProvider();

  if(provider){
    const message = await openaiChatRequest(currentModel, messages, null, provider, run, true);
    return message.content || '';
  }

  try {
    const message = await chatRequest(currentModel, messages, null, run);
    return message.content || '';
  } catch(e) {
    if(e.name === 'AbortError') throw e;
    const msg = String(e.message || '');
    if(/cuda|illegal memory|out of memory|oom|0xc0000409/i.test(msg)){
      console.warn('The runner ran out of memory, retrying with num_ctx=4096');
      const rescued = await chatRequest(currentModel, messages, null, run, 4096);
      return rescued.content || '';
    }
    throw e;
  }
};

export const askVision = async (prompt, imageBase64) => {
  const seen = await readEngine();
  if(!seen.vision) throw new Error('Vision is off in Settings');
  if(seen.visionProblem) throw new Error(seen.visionProblem);
  const settings = await readSettings().catch(() => ({}));
  const base = cleanText(settings.providerBaseUrl).replace(/\/+$/, '');
  const req = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model: await getCurrentModel(),
      messages: [{role: 'user', content: [{type: 'text', text: prompt}, {type: 'image_url', image_url: {url: `data:image/png;base64,${imageBase64}`}}]}],
      sneo_side: true,
      stream: false,
      max_tokens: 2048,
      chat_template_kwargs: {enable_thinking: false}
    })
  });

  const res = await req.json().catch(() => ({}));

  if(!req.ok){
    throw new Error(res.error?.message || `the engine answered ${req.status}`);
  }

  return res.choices?.[0]?.message?.content || '';
};

export const getEmbedding = async (text) => {
  const req = await fetch(`${runnerUrl}/api/embeddings`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model: process.env.EMBED_MODEL || 'nomic-embed-text',
      prompt: String(text || '').slice(0, 8000)
    })
  });

  const res = await req.json();

  if(!req.ok || !Array.isArray(res.embedding)){
    throw new Error(res.error || 'Error generando embedding. Esta bajado nomic-embed-text?');
  }

  return res.embedding;
};

export const askModelChat = async (messages, tools, run, modelOverride) => {
  const currentModel = cleanText(modelOverride) || await getCurrentModel();
  const provider = await getProvider();

  if(provider){
    return await openaiChatRequest(currentModel, messages, tools, provider, run);
  }

  try {
    const res = await chatRequest(currentModel, messages, tools, run);

    if(!res.content.trim() && !res.tool_calls.length && tools?.length){
      return await chatRequest(currentModel, messages, null, run);
    }

    return res;
  } catch(e) {
    if(e.name === 'AbortError'){
      throw e;
    }

    const msg = String(e.message || '');
    if(/cuda|illegal memory|out of memory|oom|0xc0000409/i.test(msg)){
      console.warn('The runner ran out of memory, retrying with num_ctx=4096');
      try {
        return await chatRequest(currentModel, messages, tools, run, 4096);
      } catch(retryErr) {
        return await chatRequest(currentModel, messages, null, run, 4096);
      }
    }

    if(/tool|glm|not declared|schema|invalid|function/i.test(msg)){
      return await chatRequest(currentModel, messages, null, run);
    }

    throw e;
  }
};
