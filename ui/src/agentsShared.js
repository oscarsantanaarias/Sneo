import { useEffect, useState } from 'react';
import { tr } from './i18n.js';

// el color de cada agente dice de que proveedor viene
export const colorOf = (item = {}) => {
  const name = `${item.name || ''} ${item.route || ''} ${item.baseUrl || ''}`.toLowerCase();

  if(item.kind === 'claude' || name.includes('claude')) return '#d97757';
  if(name.includes('deepseek')) return '#4d6bfe';
  if(name.includes('openrouter')) return '#9b8cf5';
  if(name.includes('openai') || name.includes('gpt')) return '#10a37f';
  if(name.includes(':8001') || name.includes('second')) return '#7fd1b9';
  if(item.kind === 'local' || name.includes('sneo')) return '#f0a63c';
  return '#8b8b96';
};

export const short = (text, most) => String(text || '').length > most ? `${String(text).slice(0, most - 1)}…` : String(text || '');

const thisPc = /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/i;

// el nombre corto de un modelo: sin carpeta, sin la parte 00001-of-00004 y sin .gguf
export const modelName = (text) => String(text || '').split(/[\\/]/).pop().replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '');

// que modelos de esta pc estan cargados y cuales son: un agente en un modelo que no esta cargado no puede escribir
export const useLoaded = (apiUrl) => {
  const [loaded, setLoaded] = useState({main: false, second: false, mainName: '', secondName: '', checked: false});

  useEffect(() => {
    let alive = true;
    const read = async () => {
      const [main, second] = await Promise.all([
        fetch(`${apiUrl}/api/engine/files`).then(answer => answer.json()).catch(() => null),
        fetch(`${apiUrl}/api/engine/second`).then(answer => answer.json()).catch(() => null)
      ]);
      const listed = (main?.models || []).find(item => item.path === main?.currentModel);
      const mainReady = Boolean(main?.ready);
      const secondReady = second?.generation === 'ready' && !second?.starting;

      if(alive){
        setLoaded({
          main: mainReady,
          second: secondReady,
          mainName: mainReady ? modelName(listed?.name || main?.currentModel) : '',
          secondName: secondReady ? modelName(second?.model || second?.path) : '',
          checked: true
        });
      }
    };

    read();
    const timer = setInterval(read, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [apiUrl]);

  // las apis de afuera siempre estan, los motores de esta pc solo cuando su modelo esta cargado
  const readyOf = (route) => {
    if(!route) return false;
    if(route.kind !== 'local' || !thisPc.test(route.baseUrl || '')) return true;
    return /:8001(\/|$)/.test(route.baseUrl) ? loaded.second : loaded.main;
  };

  // lo que se lee en el selector: en esta pc el modelo que esta cargado y en las apis su nombre y modelo
  const labelOf = (route) => {
    if(!route) return '';
    if(route.kind === 'local' && thisPc.test(route.baseUrl || '')){
      return /:8001(\/|$)/.test(route.baseUrl)
        ? `${tr('Second model')} · ${loaded.secondName || tr('off')}`
        : `${tr('This PC')} · ${loaded.mainName || tr('nothing loaded')}`;
    }
    return route.model && route.kind !== 'claude' ? `${route.name} · ${route.model}` : route.name;
  };

  return {loaded, readyOf, labelOf};
};

// lee una respuesta de eventos del servidor y entrega cada evento ya parseado
export const readEvents = async (response, onEvent) => {
  if(!response.ok || !String(response.headers.get('content-type') || '').includes('text/event-stream')){
    const data = await response.json().catch(() => ({}));
    throw new Error(data.message || `The app server answered ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while(true){
    const {done, value} = await reader.read();

    if(done){
      break;
    }

    buffer += decoder.decode(value, {stream: true});
    const parts = buffer.split('\n\n');
    buffer = parts.pop();

    for(const part of parts){
      const line = part.split('\n').find(item => item.startsWith('data: '));

      if(!line){
        continue;
      }

      let event = null;

      try {
        event = JSON.parse(line.slice(6));
      } catch(e) {
        continue;
      }

      onEvent(event);
    }
  }
};
