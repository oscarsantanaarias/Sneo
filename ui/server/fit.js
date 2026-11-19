export const defaultRatio = 3;

export const countChars = (messages, tools = null) => {
  let chars = tools?.length ? JSON.stringify(tools).length : 0;

  for(const item of messages || []){
    chars += String(item?.content ?? '').length + 12 + (item?.images?.length || 0) * 2400;

    for(const call of item?.tool_calls || []){
      chars += String(call.function?.name || '').length + String(call.function?.arguments || '').length + 24;
    }
  }

  return chars;
};

export const estimateTokens = (messages, tools = null, ratio = defaultRatio) => {
  return Math.ceil(countChars(messages, tools) / (Number(ratio) > 0 ? Number(ratio) : defaultRatio));
};

export const calibrateRatio = (previous, chars, promptTokens) => {
  const measured = chars / promptTokens;

  if(!(chars > 0) || !(promptTokens > 0) || measured < 1 || measured > 8){
    return Number(previous) > 0 ? Number(previous) : defaultRatio;
  }

  const next = Number(previous) > 0
    ? Number(previous) * 0.4 + measured * 0.6
    : measured;

  return Math.round(Math.min(6, Math.max(1.2, next)) * 100) / 100;
};

export const promptBudget = (window) => {
  const size = Number(window) > 0 ? Number(window) : 8192;
  return size - Math.min(16384, Math.floor(size / 4));
};

export const outputLimit = (window, ratio = defaultRatio) => {
  const size = Number(window) > 0 ? Number(window) : 8192;
  return Math.max(2000, Math.min(60000, Math.floor(size * 0.08 * (Number(ratio) > 0 ? Number(ratio) : defaultRatio))));
};

export const clipText = (text, limit) => {
  const source = String(text ?? '');

  if(source.length <= limit){
    return source;
  }

  const head = Math.floor(limit * 0.7);
  const tail = Math.max(0, limit - head);

  return `${source.slice(0, head)}\n\n[output cut: ${source.length} characters in total, ${source.length - head - tail} were left out of the middle. Ask for a narrower range or query to see more]\n\n${source.slice(-tail)}`;
};

// el razonamiento del modelo no es parte de lo que dijo: fuera los bloques de pensar, con o sin etiqueta, y lo que quedo antes de un cierre suelto
export const withoutThinking = (text) => {
  let out = String(text ?? '').replace(/<think(?::[\w-]+)?>[\s\S]*?<\/think(?::[\w-]+)?>/gi, '');
  const stray = [...out.matchAll(/<\/think(?::[\w-]+)?>/gi)].pop();

  if(stray){
    out = out.slice(stray.index + stray[0].length);
  }

  return out.replace(/<think(?::[\w-]+)?>[\s\S]*$/i, '').trim();
};

const recordHeader = "WHAT EACH TOOL CALL REALLY RETURNED (from the app's log, it wins over anything written above):";

// lo que paso con cada herramienta sale del registro de la app y no de como lo cuente el modelo: un resumen llego a inventar errores que nunca hubo
const toolRecord = (region) => {
  const results = new Map(region.filter(item => item.role === 'tool').map(item => [item.tool_call_id, String(item.content ?? '')]));
  const rows = [];

  for(const item of region){
    const text = String(item.content ?? '');
    const at = text.indexOf(recordHeader);

    if(at >= 0){
      for(const line of text.slice(at + recordHeader.length).split('\n').slice(1)){
        if(!line.startsWith('- ')){
          break;
        }

        rows.push(line);
      }
    }

    for(const call of item.tool_calls || []){
      const result = results.get(call.id);

      if(result === undefined){
        continue;
      }

      const failed = result.match(/^\s*\{"error":"((?:[^"\\]|\\.){0,200})/);
      const args = String(call.function?.arguments || '');
      const what = (args.match(/"(?:file|target|folder|command|pattern|query|url)"\s*:\s*"((?:[^"\\]|\\.){0,120})/) || [])[1] || '';
      const outcome = failed
        ? `failed, ${failed[1]}`
        : /"action":"no-change"/.test(result.slice(0, 300))
          ? 'ran but changed nothing'
          : 'ok';
      rows.push(`- ${call.function?.name}${what ? ` ${what}` : ''}: ${outcome}`);
    }
  }

  return rows.slice(-40).join('\n');
};

const describe = (item) => {
  const calls = (item.tool_calls || []).map(call => `${call.function?.name}(${String(call.function?.arguments || '').slice(0, 400)})`).join(', ');

  if(item.role === 'tool'){
    return `TOOL RESULT: ${clipText(item.content, 1500)}`;
  }

  return `${String(item.role).toUpperCase()}: ${clipText(item.role === 'assistant' ? withoutThinking(item.content) : item.content, 4000)}${calls ? `\nCALLED: ${calls}` : ''}`;
};

const plainSummary = (region) => {
  const asks = region.filter(item => item.role === 'user').map(item => `- The user wrote: ${String(item.content || '').replace(/\s+/g, ' ').slice(0, 200)}`);
  const tools = [...new Set(region.flatMap(item => (item.tool_calls || []).map(call => call.function?.name)).filter(Boolean))];

  return `${asks.slice(-15).join('\n')}${tools.length ? `\n- Tools already used: ${tools.join(', ')}` : ''}`;
};

export const summaryHeader = 'SUMMARY OF THE EARLIER PART OF THIS CONVERSATION (older messages were condensed to fit the context window):';

export const placeSummary = (messages, summary, pinned = null, region = []) => {
  const system = messages[0];
  const rest = messages.slice(1);
  const lost = pinned && region.includes(pinned);
  const block = `${summaryHeader}\n${summary}${lost ? `\n\nTHE USER'S CURRENT REQUEST (verbatim):\n${pinned.content}` : ''}`;

  if(rest[0]?.role === 'user'){
    const carrier = {...rest[0], content: `${block}\n\n${rest[0].content}`};
    return {messages: [system, carrier, ...rest.slice(1)], carrier, mergedWith: rest[0]};
  }

  const carrier = {role: 'user', content: block};
  return {messages: [system, carrier, ...rest], carrier, mergedWith: null};
};

export const fitMessages = async ({messages, tools = null, window, ratio = defaultRatio, summarize = null, pinned = null}) => {
  const budget = promptBudget(window);
  const before = estimateTokens(messages, tools, ratio);

  if(before <= budget){
    return {messages, compacted: false, before, after: before};
  }

  const target = Math.floor(budget * 0.55);
  const fixed = estimateTokens([messages[0]], tools, ratio) + 1200;

  if(fixed >= budget){
    return {messages, compacted: false, hopeless: true, before, after: before};
  }

  if(messages.length < 3){
    return {messages, compacted: false, before, after: before};
  }

  const open = new Set();
  let tail = estimateTokens(messages.slice(1), null, ratio);
  let cut = -1;

  for(let i = 1; i < messages.length - 1; i++){
    const item = messages[i];

    for(const call of item.tool_calls || []){
      open.add(call.id);
    }

    if(item.role === 'tool'){
      open.delete(item.tool_call_id);
    }

    tail -= estimateTokens([item], null, ratio);

    if(open.size || !['user', 'assistant'].includes(messages[i + 1].role)){
      continue;
    }

    cut = i + 1;

    if(fixed + tail <= target){
      break;
    }
  }

  if(cut < 2){
    return {messages, compacted: false, before, after: before};
  }

  const region = messages.slice(1, cut);
  const transcript = region.map(describe).join('\n\n');
  const piece = Math.max(4000, Math.floor(budget * 0.6 * (Number(ratio) > 0 ? Number(ratio) : defaultRatio)));
  let summary = '';

  try {
    for(let at = 0; at < transcript.length; at += piece){
      const chunk = transcript.slice(at, at + piece);
      const text = summary ? `Summary so far:\n${summary}\n\nContinue the summary with this part:\n${chunk}` : chunk;
      summary = withoutThinking(summarize ? await summarize(text) : '') || summary;
    }
  } catch(e) {
    if(e?.name === 'AbortError'){
      throw e;
    }

    summary = '';
  }

  if(!summary){
    summary = plainSummary(region);
  }

  const record = toolRecord(region);

  if(record){
    summary = `${summary}\n\n${recordHeader}\n${record}`;
  }

  const placed = placeSummary([messages[0], ...messages.slice(cut)], summary, pinned, region);
  const after = estimateTokens(placed.messages, tools, ratio);

  return {
    messages: placed.messages,
    compacted: true,
    summary,
    region,
    carrier: placed.carrier,
    mergedWith: placed.mergedWith,
    before,
    after
  };
};
