import { cleanText } from './storage.js';

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36';

const decodeEntities = (text) => {
  return String(text || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&nbsp;/g, ' ');
};

const stripTags = (html) => decodeEntities(String(html || '').replace(/<[^>]*>/g, ''));

export const toolWebSearch = async ({query, count}) => {
  const cleanQuery = cleanText(query);

  if(!cleanQuery){
    throw new Error('query requerido');
  }

  const req = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(cleanQuery)}`, {
    headers: {'User-Agent': UA, 'Accept': 'text/html'}
  });

  if(!req.ok){
    throw new Error(`Buscador respondio ${req.status}`);
  }

  const html = await req.text();
  const results = [];
  const linkRe = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
  const snippetRe = /<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
  const snippets = [];
  let sm;

  while((sm = snippetRe.exec(html))){
    snippets.push(stripTags(sm[1]).trim());
  }

  let m;
  let i = 0;
  const top = Number(count) || 6;

  while((m = linkRe.exec(html)) && results.length < top){
    let url = decodeEntities(m[1]);
    const uddg = url.match(/[?&]uddg=([^&]+)/);

    if(uddg){
      url = decodeURIComponent(uddg[1]);
    }

    results.push({
      title: stripTags(m[2]).trim(),
      url,
      snippet: snippets[i] || ''
    });

    i++;
  }

  return {query: cleanQuery, results};
};

export const toolWebFetch = async ({url}) => {
  const cleanUrl = cleanText(url);

  if(!/^https?:\/\//i.test(cleanUrl)){
    throw new Error('url requerida (http/https)');
  }

  const req = await fetch(cleanUrl, {headers: {'User-Agent': UA}});

  if(!req.ok){
    throw new Error(`La pagina respondio ${req.status}`);
  }

  const html = await req.text();
  const body = html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ');

  return {url: cleanUrl, text: decodeEntities(body).trim().slice(0, 8000)};
};
