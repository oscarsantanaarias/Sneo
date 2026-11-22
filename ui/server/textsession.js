import crypto from 'crypto';

// una conversacion viva por panel de agente: claude code y antigravity recuerdan lo dicho y en cada turno solo reciben lo nuevo
// antes cada turno abria una conversacion nueva con todo el historial y la cache no servia porque el texto cambiaba al final
const kept = new Map();

const textOf = (item) => typeof item?.content === 'string'
  ? item.content
  : Array.isArray(item?.content)
    ? item.content.map(part => part.text || '').join('\n')
    : '';

// el panel guarda las respuestas cortadas en 16000 letras: se compara solo el principio
const replyHash = (text) => crypto.createHash('sha1').update(String(text || '').replace(/\s+/g, ' ').trim().slice(0, 4000)).digest('hex');
const systemHash = (text) => crypto.createHash('sha1').update(String(text || '')).digest('hex');

const turnOf = (item) => {
  const text = textOf(item);

  if(!text.trim()){
    return '';
  }

  return item.role === 'assistant'
    ? `Assistant: ${text}`
    : item.role === 'tool'
      ? `Tool result: ${text}`
      : `User: ${text}`;
};

// lo que el modelo todavia no vio: todo lo que viene despues de su ultima respuesta
export const sessionPart = (key, route, messages) => {
  const system = systemHash((messages || []).filter(item => item.role === 'system').map(textOf).join('\n\n'));
  const record = kept.get(key);

  if(!record || record.route !== route.id || record.model !== (route.model || '') || record.system !== system){
    return {record: null, system};
  }

  const rest = (messages || []).filter(item => item.role !== 'system');
  const at = rest.findLastIndex(item => item.role === 'assistant' && replyHash(textOf(item)) === record.reply);
  const fresh = at < 0 ? [] : rest.slice(at + 1).map(turnOf).filter(Boolean);

  if(!fresh.length){
    return {record: null, system};
  }

  const prompt = fresh.length === 1 ? fresh[0].replace(/^User: /, '') : `${fresh.join('\n\n')}\n\nAnswer the last user message.`;
  return {record, prompt, system};
};

export const keepSession = (key, route, system, data) => {
  kept.set(key, {...data, route: route.id, model: route.model || '', system, reply: replyHash(data.reply), at: Date.now()});
};

export const dropSession = (key) => {
  const record = kept.get(key) || null;
  kept.delete(key);
  return record;
};
