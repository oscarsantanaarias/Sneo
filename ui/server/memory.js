import { cleanText, makeId, readMemory, saveMemory } from './storage.js';

export const memoryAdd = async ({text, tags}) => {
  const data = await readMemory();
  const item = {
    id: makeId(),
    text: cleanText(text),
    tags: Array.isArray(tags) ? tags.map(cleanText).filter(Boolean) : [],
    createdAt: new Date().toISOString()
  };

  if(!item.text){
    throw new Error('memory text required');
  }

  data.items.unshift(item);
  data.items = data.items.slice(0, 500);
  await saveMemory(data);

  return item;
};

export const memorySearch = async ({query}) => {
  const data = await readMemory();
  const cleanQuery = cleanText(query).toLowerCase();

  if(!cleanQuery){
    return data.items.slice(0, 50);
  }

  return data.items.filter(item => {
    const text = `${item.text} ${(item.tags || []).join(' ')}`.toLowerCase();
    return text.includes(cleanQuery);
  }).slice(0, 50);
};

export const memoryList = async () => {
  const data = await readMemory();
  return data.items.slice(0, 100);
};
