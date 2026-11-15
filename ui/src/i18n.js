import spanish from './lang/es.json';

let current = 'en';

export const setLanguage = (language) => {
  current = language === 'es' ? 'es' : 'en';
};

export const tr = (text, ...values) => {
  if(typeof text !== 'string') return text;
  const said = current === 'es' && typeof spanish[text] === 'string'
    ? spanish[text]
    : text;
  if(!values.length) return said;
  return said.replace(/\{(\d+)\}/g, (whole, at) => {
    const value = values[Number(at)];
    return value === undefined || value === null || value === false || value === true
      ? ''
      : String(value);
  });
};
