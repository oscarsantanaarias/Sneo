import React from 'react';

const byName = {
  'package.json': {icon: 'fa-brands fa-node-js', color: '#8cc84b'},
  'package-lock.json': {icon: 'fa-brands fa-node-js', color: '#6b9f3a'},
  '.gitignore': {icon: 'fa-brands fa-git-alt', color: '#f05133'},
  '.gitattributes': {icon: 'fa-brands fa-git-alt', color: '#f05133'},
  '.gitmodules': {icon: 'fa-brands fa-git-alt', color: '#f05133'},
  'dockerfile': {icon: 'fa-brands fa-docker', color: '#2496ed'},
  'docker-compose.yml': {icon: 'fa-brands fa-docker', color: '#2496ed'},
  'docker-compose.yaml': {icon: 'fa-brands fa-docker', color: '#2496ed'},
  '.dockerignore': {icon: 'fa-brands fa-docker', color: '#4a7fa8'},
  '.env': {icon: 'fa-solid fa-gear', color: '#e5c07b'},
  'cmakelists.txt': {icon: 'fa-solid fa-gear', color: '#9ba7b4'},
  'makefile': {icon: 'fa-solid fa-terminal', color: '#d9a24a'},
  'license': {icon: 'fa-solid fa-file-lines', color: '#d0b344'}
};

const byExt = {
  jsx: {icon: 'fa-brands fa-react', color: '#61dafb'},
  tsx: {icon: 'fa-brands fa-react', color: '#3d9fd6'},
  js: {icon: 'fa-brands fa-js', color: '#f1dd35'},
  mjs: {icon: 'fa-brands fa-js', color: '#f1dd35'},
  cjs: {icon: 'fa-brands fa-js', color: '#f1dd35'},
  ts: {text: 'TS', color: '#3a8ee6'},
  html: {icon: 'fa-brands fa-html5', color: '#e8622c'},
  htm: {icon: 'fa-brands fa-html5', color: '#e8622c'},
  css: {icon: 'fa-brands fa-css3-alt', color: '#4a9fe8'},
  scss: {icon: 'fa-brands fa-sass', color: '#d0679d'},
  sass: {icon: 'fa-brands fa-sass', color: '#d0679d'},
  less: {icon: 'fa-brands fa-less', color: '#5a7fc0'},
  py: {icon: 'fa-brands fa-python', color: '#4f9ad8'},
  md: {icon: 'fa-brands fa-markdown', color: '#5aa7c9'},
  mdx: {icon: 'fa-brands fa-markdown', color: '#f0a63c'},
  java: {icon: 'fa-brands fa-java', color: '#e76f00'},
  rs: {icon: 'fa-brands fa-rust', color: '#dea584'},
  php: {icon: 'fa-brands fa-php', color: '#8f9ad0'},
  vue: {icon: 'fa-brands fa-vuejs', color: '#41b883'},
  go: {icon: 'fa-brands fa-golang', color: '#00add8'},
  swift: {icon: 'fa-brands fa-swift', color: '#f05138'},
  json: {text: '{}', color: '#e6c44c'},
  jsonc: {text: '{}', color: '#e6c44c'},
  c: {icon: 'fa-solid fa-code', color: '#5f9bd6'},
  cpp: {icon: 'fa-solid fa-code', color: '#f0577f'},
  cc: {icon: 'fa-solid fa-code', color: '#f0577f'},
  cxx: {icon: 'fa-solid fa-code', color: '#f0577f'},
  h: {icon: 'fa-solid fa-code', color: '#a77ad6'},
  hpp: {icon: 'fa-solid fa-code', color: '#c27ad6'},
  cs: {icon: 'fa-solid fa-code', color: '#58a65c'},
  cu: {icon: 'fa-solid fa-code', color: '#76b900'},
  lua: {icon: 'fa-solid fa-file-code', color: '#5a8fe0'},
  rb: {icon: 'fa-solid fa-file-code', color: '#e0534c'},
  kt: {icon: 'fa-solid fa-file-code', color: '#a97bff'},
  asm: {icon: 'fa-solid fa-file-code', color: '#9a8f6a'},
  xml: {icon: 'fa-solid fa-file-code', color: '#e37933'},
  svg: {icon: 'fa-solid fa-file-image', color: '#ffb13b'},
  png: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  jpg: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  jpeg: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  gif: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  webp: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  bmp: {icon: 'fa-solid fa-file-image', color: '#b07fd6'},
  ico: {icon: 'fa-solid fa-file-image', color: '#d6b84a'},
  dds: {icon: 'fa-solid fa-file-image', color: '#8f9ad0'},
  tga: {icon: 'fa-solid fa-file-image', color: '#8f9ad0'},
  txt: {icon: 'fa-solid fa-file-lines', color: '#a3a3ad'},
  log: {icon: 'fa-solid fa-file-lines', color: '#7f8a96'},
  csv: {icon: 'fa-solid fa-file-csv', color: '#5fb86a'},
  pdf: {icon: 'fa-solid fa-file-pdf', color: '#e5534b'},
  zip: {icon: 'fa-solid fa-file-zipper', color: '#d4a72c'},
  rar: {icon: 'fa-solid fa-file-zipper', color: '#d4a72c'},
  '7z': {icon: 'fa-solid fa-file-zipper', color: '#d4a72c'},
  gz: {icon: 'fa-solid fa-file-zipper', color: '#d4a72c'},
  yml: {icon: 'fa-solid fa-gear', color: '#d65f5f'},
  yaml: {icon: 'fa-solid fa-gear', color: '#d65f5f'},
  toml: {icon: 'fa-solid fa-gear', color: '#9c4221'},
  ini: {icon: 'fa-solid fa-gear', color: '#9ba7b4'},
  cfg: {icon: 'fa-solid fa-gear', color: '#9ba7b4'},
  conf: {icon: 'fa-solid fa-gear', color: '#9ba7b4'},
  config: {icon: 'fa-solid fa-gear', color: '#9ba7b4'},
  env: {icon: 'fa-solid fa-gear', color: '#e5c07b'},
  sql: {icon: 'fa-solid fa-database', color: '#f29111'},
  db: {icon: 'fa-solid fa-database', color: '#c48a3a'},
  sqlite: {icon: 'fa-solid fa-database', color: '#4aa0d6'},
  bat: {icon: 'fa-solid fa-terminal', color: '#c1f12e'},
  cmd: {icon: 'fa-solid fa-terminal', color: '#c1f12e'},
  ps1: {icon: 'fa-solid fa-terminal', color: '#5391fe'},
  sh: {icon: 'fa-solid fa-terminal', color: '#89e051'},
  bash: {icon: 'fa-solid fa-terminal', color: '#89e051'},
  exe: {icon: 'fa-solid fa-gears', color: '#9a99a2'},
  dll: {icon: 'fa-solid fa-gears', color: '#7d7c86'},
  gguf: {icon: 'fa-solid fa-cube', color: '#f0a63c'},
  wav: {icon: 'fa-solid fa-file-audio', color: '#e07bb0'},
  mp3: {icon: 'fa-solid fa-file-audio', color: '#e07bb0'},
  ogg: {icon: 'fa-solid fa-file-audio', color: '#e07bb0'},
  mp4: {icon: 'fa-solid fa-file-video', color: '#e0705a'},
  webm: {icon: 'fa-solid fa-file-video', color: '#e0705a'}
};

const plainFile = {icon: 'fa-solid fa-file', color: '#6b6b74'};

export const fileKind = (fileName = '') => {
  const name = String(fileName).split(/[\\/]/).pop().toLowerCase();

  if(byName[name]){
    return byName[name];
  }

  if(name.startsWith('.env')){
    return byName['.env'];
  }

  if(name.startsWith('dockerfile')){
    return byName.dockerfile;
  }

  const dot = name.lastIndexOf('.');

  if(dot < 0){
    return plainFile;
  }

  return byExt[name.slice(dot + 1)] || plainFile;
};

export function FileIcon({name = '', folder = false, open = false, className = ''}) {
  if(folder){
    return <i className={`${open ? 'fa-solid fa-folder-open' : 'fa-solid fa-folder'} file-icon ${className}`} style={{color: 'var(--accent)'}}></i>;
  }

  const found = fileKind(name);

  if(found.text){
    return <span className={`file-icon file-icon-text ${className}`} style={{color: found.color}}>{found.text}</span>;
  }

  return <i className={`${found.icon} file-icon ${className}`} style={{color: found.color}}></i>;
}
