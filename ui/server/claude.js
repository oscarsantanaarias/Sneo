import Anthropic from '@anthropic-ai/sdk';

const clients = new Map();

const clientFor = (apiKey) => {
  if(!clients.has(apiKey)){
    clients.set(apiKey, new Anthropic({apiKey}));
  }

  return clients.get(apiKey);
};

// los agentes hablan en el formato de chat comun: aqui los mensajes pasan al formato de claude y la respuesta vuelve igual
// los bloques que claude devolvio se guardan en _claude y se le devuelven tal cual en la vuelta siguiente
const toClaude = (messages) => {
  const system = [];
  const turns = [];

  for(const item of messages){
    if(item.role === 'system'){
      system.push(String(item.content ?? ''));
      continue;
    }

    if(item.role === 'tool'){
      const block = {type: 'tool_result', tool_use_id: item.tool_call_id, content: String(item.content ?? '')};
      const last = turns[turns.length - 1];

      if(last && last.role === 'user' && Array.isArray(last.content) && last.content.every(part => part.type === 'tool_result')){
        last.content.push(block);
      } else {
        turns.push({role: 'user', content: [block]});
      }

      continue;
    }

    if(item.role === 'assistant'){
      if(Array.isArray(item._claude)){
        turns.push({role: 'assistant', content: item._claude});
        continue;
      }

      const content = [];

      if(item.content){
        content.push({type: 'text', text: String(item.content)});
      }

      for(const call of item.tool_calls || []){
        let input = {};

        try {
          input = JSON.parse(call.function?.arguments || '{}');
        } catch(e) {
          input = {};
        }

        content.push({type: 'tool_use', id: call.id, name: call.function?.name || '', input});
      }

      if(content.length){
        turns.push({role: 'assistant', content});
      }

      continue;
    }

    const text = typeof item.content === 'string'
      ? item.content
      : Array.isArray(item.content)
        ? item.content.map(part => part.text || '').join('\n')
        : String(item.content ?? '');
    turns.push({role: 'user', content: text});
  }

  return {system: system.join('\n\n'), turns};
};

const toTools = (tools) => (tools || []).map(item => ({
  name: item.function?.name || item.name,
  description: item.function?.description || item.description || '',
  input_schema: item.function?.parameters || item.parameters || {type: 'object', properties: {}}
}));

export const claudeModels = [
  {id: 'claude-opus-5-5', name: 'Claude Opus 5.5'},
  {id: 'claude-sonnet-5-5', name: 'Claude Sonnet 5.5'},
  {id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5'}
];

export const askClaude = async (route, messages, tools, run) => {
  if(!route.apiKey){
    throw new Error(`${route.name || 'Claude'} has no API key, add it in Agents > Routes`);
  }

  const {system, turns} = toClaude(messages);
  const model = route.model || 'claude-opus-5-5';
  // si claude declina una peticion la api la repite en otro modelo dentro de la misma llamada, haiku no tiene esa vuelta
  const fallback = /^claude-(opus|sonnet)/.test(model);
  const body = {
    model,
    max_tokens: 16000,
    messages: turns,
    thinking: {type: 'adaptive', display: 'summarized'},
    output_config: {effort: route.effort || 'medium'},
    ...(system ? {system} : {}),
    ...(tools?.length ? {tools: toTools(tools)} : {}),
    ...(fallback ? {betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default'} : {})
  };
  const startedAt = Date.now();
  let response;

  try {
    response = await clientFor(route.apiKey).beta.messages.create(body, {signal: run?.signal});
  } catch(e) {
    if(e instanceof Anthropic.AuthenticationError){
      throw new Error(`${route.name || 'Claude'} rejected the API key`);
    }

    if(e instanceof Anthropic.RateLimitError){
      throw new Error(`${route.name || 'Claude'} is limiting this key right now, try again in a moment`);
    }

    if(e instanceof Anthropic.APIError){
      throw new Error(`${route.name || 'Claude'} answered ${e.status}: ${e.message}`);
    }

    throw e;
  }

  const text = response.content.filter(block => block.type === 'text').map(block => block.text).join('');
  const reasoning = response.content.filter(block => block.type === 'thinking').map(block => block.thinking || '').join('\n');
  const calls = response.content
    .filter(block => block.type === 'tool_use')
    .map(block => ({id: block.id, type: 'function', function: {name: block.name, arguments: JSON.stringify(block.input ?? {})}}));
  const ms = Date.now() - startedAt;
  const usage = {
    promptTokens: response.usage?.input_tokens || 0,
    outputTokens: response.usage?.output_tokens || 0,
    cachedTokens: response.usage?.cache_read_input_tokens || 0,
    ms
  };

  if(response.stop_reason === 'refusal'){
    return {content: text || `${route.name || 'Claude'} declined this request.`, reasoning, tool_calls: [], finishReason: 'stop', raw: response.content, usage};
  }

  // una llamada cortada por max_tokens puede traer argumentos incompletos, no se ejecuta
  const cut = response.stop_reason === 'max_tokens';

  return {content: text, reasoning, tool_calls: cut ? [] : calls, finishReason: cut ? 'length' : 'stop', raw: response.content, usage};
};
