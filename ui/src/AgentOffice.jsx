import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { tr } from './i18n.js';
import { colorOf, short } from './agentsShared.js';

// la oficina dibujada por codigo: cada agente en su escritorio y caminando al de otro cuando le pasa trabajo
const TW = 32;
const TH = 16;
const WALL = 64;
const WALK = 2.6;

// los personajes son propios, uno distinto por agente
const people = [
  {skin: '#f2c7a0', hair: '#2ec4b6', style: 'spiky', shirt: '#f08a24', pants: '#3b3f58', extra: ['phones']},
  {robot: true, body: '#9aa7b8', dark: '#5d6a7c', eye: '#5cf2ff', light: '#ff5c7a'},
  {skin: '#8d5a3b', hair: '#6c3fb8', style: 'long', shirt: '#f5c542', pants: '#2c3e50', extra: []},
  {skin: '#e9b48a', hair: '#1b1b22', style: 'bun', shirt: '#f4f4f0', pants: '#4a4a5c', extra: ['scarf']},
  {skin: '#c98b62', hair: '#6b3e1f', style: 'beanie', cap: '#3f9b4b', shirt: '#3d6fd1', pants: '#2b2b33', extra: ['beard']},
  {hooded: true, hood: '#2a2a3a', eye: '#9dff6a', shirt: '#2a2a3a', pants: '#1c1c26'},
  {skin: '#f6d2b5', hair: '#d9612b', style: 'ponytail', shirt: '#2f8f6a', pants: '#3a3350', extra: ['glasses']},
  {skin: '#6b4026', hair: '#18120e', style: 'afro', shirt: '#ef6fa0', pants: '#2e3a4f', extra: ['chain']},
  {skin: '#f0c09a', hair: '#7a4b2a', style: 'cap', cap: '#d6333f', shirt: '#4b7bc4', pants: '#4b7bc4', extra: ['overalls']},
  {skin: '#d8a27c', hair: '#e8e8ee', style: 'short', shirt: '#7b4fc9', pants: '#30303c', extra: ['earrings']},
  {robot: true, box: true, body: '#f2933a', dark: '#b85f17', eye: '#ffffff', light: '#5cf2ff'},
  {skin: '#b77b55', hair: '#2b1d14', style: 'bald', shirt: '#8c6239', pants: '#3a3026', extra: ['glasses', 'beard']}
];

const rect = (ctx, color, x, y, w, h) => {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), w, h);
};

// un muneco de pixeles parado en (fx, fy), de espaldas sentado o de frente caminando
const drawPerson = (ctx, look, fx, fy, front, pose, t, busy) => {
  ctx.save();
  ctx.translate(Math.round(fx), Math.round(fy));
  ctx.scale(1.5, 1.5);
  drawBody(ctx, look, 0, 0, front, pose, t, busy);
  ctx.restore();
};

const drawBody = (ctx, look, fx, fy, front, pose, t, busy) => {
  const sit = pose === 'sit';
  const x = Math.round(fx - 6);
  const y = Math.round(fy - 25 - (sit ? 5 : 0));
  const step = pose === 'walk' ? Math.floor(t * 8) % 2 : 0;
  const bob = pose === 'walk' ? step : busy ? 0 : Math.floor(t * 1.2) % 2 === 0 ? 0 : 1;
  const type = sit && busy ? Math.floor(t * 9) % 2 : 0;

  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.beginPath();
  ctx.ellipse(fx, fy, 7, 3, 0, 0, Math.PI * 2);
  ctx.fill();

  if(look.robot){
    if(!sit){
      rect(ctx, look.dark, x + 3, y + 19 + step, 2, 6 - step);
      rect(ctx, look.dark, x + 7, y + 20 - step, 2, 5 + step);
    }
    rect(ctx, look.body, x + 2, y + 11 + bob, 8, 8);
    rect(ctx, look.dark, x + 2, y + 17 + bob, 8, 2);
    rect(ctx, look.dark, x, y + 12 + bob - type, 2, 5);
    rect(ctx, look.dark, x + 10, y + 12 + bob - (type ? 0 : 1), 2, 5);
    if(look.box){
      rect(ctx, look.body, x + 1, y + 2 + bob, 10, 9);
      rect(ctx, look.dark, x + 1, y + 10 + bob, 10, 1);
    } else {
      rect(ctx, look.body, x + 2, y + 3 + bob, 8, 8);
      rect(ctx, look.dark, x + 5, y + bob, 2, 3);
      rect(ctx, Math.floor(t * 2) % 2 ? look.light : look.dark, x + 5, y - 1 + bob, 2, 1);
    }
    if(front){
      rect(ctx, '#1d2430', x + 3, y + 5 + bob, 6, 4);
      rect(ctx, look.eye, x + 4, y + 6 + bob, 1, 2);
      rect(ctx, look.eye, x + 7, y + 6 + bob, 1, 2);
    } else {
      rect(ctx, look.dark, x + 4, y + 5 + bob, 4, 1);
      rect(ctx, look.dark, x + 4, y + 7 + bob, 4, 1);
    }
    return;
  }

  const pants = look.pants || '#333';
  if(!sit){
    rect(ctx, pants, x + 3, y + 18 + step, 3, 6 - step);
    rect(ctx, pants, x + 7, y + 19 - step, 3, 5 + step);
    rect(ctx, '#1e1a18', x + 3, y + 24, 3, 1);
    rect(ctx, '#1e1a18', x + 7, y + 24, 3, 1);
  }

  const shirt = look.shirt;
  rect(ctx, shirt, x + 2, y + 10 + bob, 8, 9);
  if(look.extra?.includes('overalls')){
    rect(ctx, '#3a5f9e', x + 3, y + 13 + bob, 6, 6);
    rect(ctx, '#3a5f9e', x + 3, y + 10 + bob, 1, 3);
    rect(ctx, '#3a5f9e', x + 8, y + 10 + bob, 1, 3);
  }
  const skin = look.skin || '#e0b090';
  rect(ctx, shirt, x, y + 11 + bob - type, 2, 5);
  rect(ctx, shirt, x + 10, y + 11 + bob - (type ? 0 : 1), 2, 5);
  rect(ctx, skin, x, y + 16 + bob - type, 2, 1);
  rect(ctx, skin, x + 10, y + 16 + bob - (type ? 0 : 1), 2, 1);
  if(look.extra?.includes('scarf')) rect(ctx, '#d6333f', x + 2, y + 10 + bob, 8, 2);
  if(look.extra?.includes('chain') && front) rect(ctx, '#f2c94c', x + 4, y + 12 + bob, 4, 1);

  if(look.hooded){
    rect(ctx, look.hood, x + 2, y + 2 + bob, 8, 9);
    rect(ctx, look.hood, x + 1, y + 4 + bob, 10, 6);
    if(front){
      rect(ctx, '#0b0b12', x + 3, y + 4 + bob, 6, 6);
      rect(ctx, look.eye, x + 4, y + 6 + bob, 1, 1);
      rect(ctx, look.eye, x + 7, y + 6 + bob, 1, 1);
    }
    return;
  }

  const hy = y + 3 + bob;
  rect(ctx, skin, x + 3, hy, 6, 7);
  if(front){
    rect(ctx, '#1a1418', x + 4, hy + 3, 1, 1);
    rect(ctx, '#1a1418', x + 7, hy + 3, 1, 1);
    if(look.extra?.includes('glasses')){
      rect(ctx, '#1a1418', x + 3, hy + 2, 6, 1);
      rect(ctx, '#9fd3ff', x + 4, hy + 3, 1, 1);
      rect(ctx, '#9fd3ff', x + 7, hy + 3, 1, 1);
    }
    if(look.extra?.includes('beard')) rect(ctx, look.hair === '#2b1d14' ? '#4a3a30' : look.hair, x + 3, hy + 5, 6, 2);
    if(look.extra?.includes('earrings')){
      rect(ctx, '#f2c94c', x + 2, hy + 4, 1, 1);
      rect(ctx, '#f2c94c', x + 9, hy + 4, 1, 1);
    }
  }

  const hair = look.hair;
  switch(look.style){
    case 'spiky':
      rect(ctx, hair, x + 2, hy - 1, 8, 3);
      rect(ctx, hair, x + 3, hy - 3, 1, 2);
      rect(ctx, hair, x + 6, hy - 3, 1, 2);
      rect(ctx, hair, x + 8, hy - 2, 1, 1);
      if(!front) rect(ctx, hair, x + 3, hy + 2, 6, 3);
      break;
    case 'long':
      rect(ctx, hair, x + 2, hy - 1, 8, 3);
      rect(ctx, hair, x + 2, hy, 1, 9);
      rect(ctx, hair, x + 9, hy, 1, 9);
      if(!front) rect(ctx, hair, x + 2, hy, 8, 10);
      break;
    case 'bun':
      rect(ctx, hair, x + 3, hy - 1, 6, 3);
      rect(ctx, hair, x + 5, hy - 4, 3, 3);
      if(!front) rect(ctx, hair, x + 3, hy, 6, 5);
      break;
    case 'ponytail':
      rect(ctx, hair, x + 3, hy - 1, 6, 3);
      rect(ctx, hair, x + 9, hy + 1, 2, 2);
      rect(ctx, hair, x + 10, hy + 3, 1, 5);
      if(!front) rect(ctx, hair, x + 3, hy, 6, 5);
      break;
    case 'afro':
      rect(ctx, hair, x + 1, hy - 3, 10, 5);
      rect(ctx, hair, x + 1, hy + 1, 2, 4);
      rect(ctx, hair, x + 9, hy + 1, 2, 4);
      if(!front) rect(ctx, hair, x + 1, hy, 10, 7);
      break;
    case 'beanie':
      rect(ctx, look.cap, x + 2, hy - 2, 8, 4);
      rect(ctx, '#ffffff', x + 2, hy + 1, 8, 1);
      rect(ctx, look.cap, x + 5, hy - 3, 2, 1);
      if(!front) rect(ctx, hair, x + 3, hy + 2, 6, 3);
      break;
    case 'cap':
      rect(ctx, look.cap, x + 2, hy - 2, 8, 3);
      rect(ctx, look.cap, front ? x + 1 : x + 7, hy + 1, 4, 1);
      if(!front) rect(ctx, hair, x + 3, hy + 2, 6, 3);
      break;
    case 'short':
      rect(ctx, hair, x + 3, hy - 1, 6, 2);
      rect(ctx, hair, x + 2, hy, 1, 3);
      rect(ctx, hair, x + 9, hy, 1, 3);
      if(!front) rect(ctx, hair, x + 3, hy, 6, 4);
      break;
    case 'bald':
      rect(ctx, hair, x + 2, hy + 2, 1, 3);
      rect(ctx, hair, x + 9, hy + 2, 1, 3);
      if(!front) rect(ctx, hair, x + 3, hy + 4, 6, 2);
      break;
    default:
      rect(ctx, hair, x + 3, hy - 1, 6, 2);
  }

  if(look.extra?.includes('phones')){
    rect(ctx, '#2b2b33', x + 2, hy - 2, 8, 1);
    rect(ctx, '#2b2b33', x + 1, hy + 2, 2, 3);
    rect(ctx, '#2b2b33', x + 9, hy + 2, 2, 3);
  }
};

const roundRect = (ctx, x, y, w, h, r) => {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
};

// el globo con texto se dibuja sobre la imagen escalada para que la letra quede nitida
const bubble = (ctx, cx, bottom, text, kind, accent, scale) => {
  const font = Math.round(11 * scale);
  ctx.font = `${kind === 'thought' ? 'italic ' : ''}${font}px system-ui, sans-serif`;
  const most = Math.round(220 * scale);
  const words = String(text).split(/\s+/).filter(Boolean);
  const shown = [];
  let line = '';
  let used = 0;
  for(const word of words){
    const next = line ? `${line} ${word}` : word;
    if(ctx.measureText(next).width > most && line){
      shown.push(line);
      line = word;
      if(shown.length === 3) break;
    } else {
      line = next;
    }
    used++;
  }
  if(shown.length < 3 && line) shown.push(line);
  if(used < words.length) shown[shown.length - 1] = `${shown[shown.length - 1]}…`;
  const pad = Math.round(6 * scale);
  const lh = Math.round(14 * scale);
  const w = Math.min(most, Math.max(...shown.map(item => ctx.measureText(item).width))) + pad * 2;
  const h = shown.length * lh + pad * 2 - Math.round(3 * scale);
  const x = Math.round(cx - w / 2);
  const y = Math.round(bottom - h);

  ctx.fillStyle = kind === 'thought' ? 'rgba(40,36,48,0.86)' : kind === 'talk' ? '#fffaf0' : kind === 'error' ? '#ffe3e3' : '#f4efe6';
  ctx.strokeStyle = kind === 'talk' ? accent : kind === 'error' ? '#d64545' : 'rgba(0,0,0,0.35)';
  ctx.lineWidth = Math.max(1, scale);
  roundRect(ctx, x, y, w, h, Math.round(6 * scale));
  ctx.fill();
  ctx.stroke();
  if(kind !== 'thought'){
    ctx.beginPath();
    ctx.moveTo(cx - 5 * scale, y + h - 1);
    ctx.lineTo(cx, y + h + 6 * scale);
    ctx.lineTo(cx + 5 * scale, y + h - 1);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.arc(cx, y + h + 4 * scale, 2.5 * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = kind === 'thought' ? '#d9d2e6' : '#2a2420';
  ctx.textBaseline = 'top';
  shown.forEach((item, i) => ctx.fillText(item, x + pad, y + pad + i * lh));
  return y;
};

const layoutOf = (count) => {
  const n = Math.max(1, count);
  const cols = n <= 2 ? n : n <= 4 ? 2 : n <= 9 ? 3 : 4;
  const rows = Math.ceil(n / cols);
  const W = Math.max(9, Math.ceil(1.6 + cols * 4.4 + 0.6));
  const H = Math.max(7, Math.ceil(1.4 + rows * 3.8 + 0.4));
  const desks = [];
  for(let i = 0; i < n; i++){
    const c = i % cols;
    const r = Math.floor(i / cols);
    const sx = 1.6 + c * 4.4 + (W - (1.6 + cols * 4.4 + 0.6)) / 2;
    const sy = 1.2 + r * 3.8;
    desks.push({sx, sy, home: {x: sx + 1.25, y: sy + 0.55}});
  }
  const ox = H * TW / 2 + 12;
  const oy = WALL + 40;
  return {W, H, desks, ox, oy, bw: (W + H) * TW / 2 + 24, bh: oy + (W + H) * TH / 2 + 18};
};

export default function AgentOffice({agents, routes, steps, live, speeds, chats, visits, readyOf, labelOf, onSend, onStop}){
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const bufferRef = useRef(null);
  const stateRef = useRef({});
  const walkers = useRef(new Map());
  const seen = useRef(new Set());
  const spots = useRef([]);
  const [size, setSize] = useState({w: 800, h: 500});
  const [picked, setPicked] = useState('');
  const [draft, setDraft] = useState('');

  stateRef.current = {agents, routes, steps, live, chats, picked, readyOf};

  useEffect(() => {
    const box = wrapRef.current;
    if(!box) return;
    const watch = new ResizeObserver(() => setSize({w: box.clientWidth, h: box.clientHeight}));
    watch.observe(box);
    setSize({w: box.clientWidth, h: box.clientHeight});
    return () => watch.disconnect();
  }, []);

  // cada encargo nuevo hace caminar al que lo manda hasta el escritorio del otro
  useEffect(() => {
    const layout = layoutOf(agents.length);
    for(const visit of visits || []){
      if(seen.current.has(visit.id)) continue;
      seen.current.add(visit.id);
      const fromIndex = agents.findIndex(item => item.id === visit.from);
      const toIndex = agents.findIndex(item => item.id === visit.to);
      if(fromIndex < 0 || toIndex < 0) continue;
      const trip = {to: visit.to, text: visit.text, kind: visit.kind};
      const busy = walkers.current.get(visit.from);
      if(busy){
        busy.queue.push(trip);
        continue;
      }
      walkers.current.set(visit.from, startTrip(layout, fromIndex, toIndex, trip));
    }
  }, [visits, agents]);

  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    const draw = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      paint(now / 1000, dt);
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [size]);

  const startTrip = (layout, fromIndex, toIndex, trip) => {
    const mine = layout.desks[fromIndex];
    const home = mine.home;
    const target = layout.desks[toIndex];
    const front = (desk) => desk.sy + 2.75;
    const stand = {x: target.home.x + 0.95, y: target.home.y - 0.65};
    const path = [
      {x: home.x, y: home.y},
      {x: mine.sx + 2.6, y: home.y},
      {x: mine.sx + 2.6, y: front(mine)},
      {x: target.sx + 2.6, y: front(mine)},
      {x: target.sx + 2.6, y: front(target)},
      {x: target.sx + 2.6, y: stand.y},
      stand
    ];
    return {...trip, path, at: 0, phase: 'go', talk: 0, queue: [], from: fromIndex};
  };

  const moveWalker = (walker, dt) => {
    if(walker.phase === 'talk'){
      walker.talk -= dt;
      if(walker.talk <= 0){
        walker.phase = 'back';
        walker.path = [...walker.path].reverse();
        walker.at = 0;
      }
      return;
    }
    let left = WALK * dt;
    while(left > 0 && walker.at < walker.path.length - 1){
      const a = walker.path[Math.floor(walker.at)];
      const b = walker.path[Math.floor(walker.at) + 1];
      const length = Math.hypot(b.x - a.x, b.y - a.y) || 0.0001;
      const part = walker.at - Math.floor(walker.at);
      const room = (1 - part) * length;
      if(left >= room){
        walker.at = Math.floor(walker.at) + 1;
        left -= room;
      } else {
        walker.at += left / length;
        left = 0;
      }
    }
    if(walker.at >= walker.path.length - 1){
      if(walker.phase === 'go'){
        walker.phase = 'talk';
        walker.talk = 3.2;
      } else {
        walker.phase = 'home';
      }
    }
  };

  const walkerSpot = (walker) => {
    const i = Math.min(walker.path.length - 1, Math.floor(walker.at));
    const a = walker.path[i];
    const b = walker.path[Math.min(walker.path.length - 1, i + 1)];
    const part = walker.at - i;
    return {x: a.x + (b.x - a.x) * part, y: a.y + (b.y - a.y) * part, dx: b.x - a.x, dy: b.y - a.y};
  };

  const paint = (t, dt) => {
    const canvas = canvasRef.current;
    if(!canvas) return;
    const {agents: list, routes: allRoutes, steps: stepMap, live: liveMap, chats: chatMap, picked: chosen, readyOf: isReady} = stateRef.current;
    const layout = layoutOf(list.length);
    const {W, H, desks, ox, oy, bw, bh} = layout;

    if(!bufferRef.current) bufferRef.current = document.createElement('canvas');
    const buffer = bufferRef.current;
    if(buffer.width !== bw || buffer.height !== bh){
      buffer.width = bw;
      buffer.height = bh;
    }
    const ctx = buffer.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);

    const P = (gx, gy, z = 0) => [ox + (gx - gy) * TW / 2, oy + (gx + gy) * TH / 2 - z];
    const poly = (color, ...points) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      points.forEach(([px, py], i) => i ? ctx.lineTo(px, py) : ctx.moveTo(px, py));
      ctx.closePath();
      ctx.fill();
    };
    const box = (x0, y0, x1, y1, z0, h, top, left, right) => {
      poly(left, P(x0, y1, z0 + h), P(x1, y1, z0 + h), P(x1, y1, z0), P(x0, y1, z0));
      poly(right, P(x1, y0, z0 + h), P(x1, y1, z0 + h), P(x1, y1, z0), P(x1, y0, z0));
      poly(top, P(x0, y0, z0 + h), P(x1, y0, z0 + h), P(x1, y1, z0 + h), P(x0, y1, z0 + h));
    };

    // paredes, ventanas y piso
    poly('#2d3156', P(0, 0, 0), P(W, 0, 0), P(W, 0, WALL), P(0, 0, WALL));
    poly('#262a4a', P(0, 0, 0), P(0, H, 0), P(0, H, WALL), P(0, 0, WALL));
    poly('#5b3a24', P(0, 0, 0), P(W, 0, 0), P(W, 0, 12), P(0, 0, 12));
    poly('#4e3220', P(0, 0, 0), P(0, H, 0), P(0, H, 12), P(0, 0, 12));
    poly('#1e2140', P(0, 0, WALL), P(W, 0, WALL), P(W, 0, WALL - 3), P(0, 0, WALL - 3));
    for(let gy = 0; gy < H; gy++){
      for(let gx = 0; gx < W; gx++){
        poly((gx + gy) % 2 ? '#c99257' : '#b9824a', P(gx, gy), P(gx + 1, gy), P(gx + 1, gy + 1), P(gx, gy + 1));
      }
    }
    const windows = [];
    for(let wy = 1.5; wy + 1.6 < H - 0.5; wy += 3.5) windows.push(wy);
    for(const wy of windows){
      poly('#ffe2a0', P(0, wy, 22), P(0, wy + 1.4, 22), P(0, wy + 1.4, 52), P(0, wy, 52));
      poly('#f1c46e', P(0, wy + 0.68, 22), P(0, wy + 0.72, 22), P(0, wy + 0.72, 52), P(0, wy + 0.68, 52));
      poly('#f1c46e', P(0, wy, 37), P(0, wy + 1.4, 37), P(0, wy + 1.4, 38), P(0, wy, 38));
      poly('rgba(255,214,140,0.16)', P(0, wy), P(0, wy + 1.4), P(3.4, wy + 3.2), P(3.4, wy + 1.8));
    }

    const items = [];
    // estantes con libros contra la pared del fondo
    const shelves = [[1, 3.2], [W - 3.6, W - 1.4]];
    for(const [x0, x1] of shelves){
      items.push({depth: x0 + 0.1, draw: () => {
        box(x0, 0.05, x1, 0.6, 0, 46, '#7a4a2a', '#8a5530', '#6a3f22');
        const colors = ['#c0392b', '#2e86de', '#27ae60', '#f39c12', '#8e44ad', '#d35400', '#16a085'];
        for(let shelf = 0; shelf < 3; shelf++){
          const z = 6 + shelf * 13;
          poly('#5a3519', P(x0, 0.6, z), P(x1, 0.6, z), P(x1, 0.6, z + 1), P(x0, 0.6, z + 1));
          for(let b = 0; b < 9; b++){
            const bx = x0 + 0.12 + b * (x1 - x0 - 0.24) / 9;
            const tall = 8 + ((b * 7 + shelf * 3) % 4);
            poly(colors[(b + shelf * 2) % colors.length], P(bx, 0.6, z + 1), P(bx + 0.18, 0.6, z + 1), P(bx + 0.18, 0.6, z + tall), P(bx, 0.6, z + tall));
          }
        }
      }});
    }
    // la pizarra con un grafico
    const cx = W / 2;
    items.push({depth: 0, draw: () => {
      poly('#6a3f22', P(cx - 1.7, 0.02, 20), P(cx + 1.7, 0.02, 20), P(cx + 1.7, 0.02, 52), P(cx - 1.7, 0.02, 52));
      poly('#efe9dc', P(cx - 1.55, 0.03, 22), P(cx + 1.55, 0.03, 22), P(cx + 1.55, 0.03, 50), P(cx - 1.55, 0.03, 50));
      const bars = [10, 16, 12, 21, 18];
      bars.forEach((tall, i) => {
        const bx = cx - 1.3 + i * 0.32;
        poly(i === 3 ? '#e67e22' : '#5d8bd6', P(bx, 0.04, 25), P(bx + 0.2, 0.04, 25), P(bx + 0.2, 0.04, 25 + tall), P(bx, 0.04, 25 + tall));
      });
      ctx.fillStyle = '#d9534f';
      const [pcx, pcy] = P(cx + 0.95, 0.04, 36);
      ctx.beginPath();
      ctx.moveTo(pcx, pcy);
      ctx.arc(pcx, pcy, 6, 0, Math.PI * 1.3);
      ctx.fill();
      ctx.fillStyle = '#f0ad4e';
      ctx.beginPath();
      ctx.moveTo(pcx, pcy);
      ctx.arc(pcx, pcy, 6, Math.PI * 1.3, Math.PI * 2);
      ctx.fill();
    }});
    // plantas en las esquinas
    for(const [px, py] of [[W - 0.9, 0.9], [0.9, H - 0.9], [W - 0.9, H - 0.9]]){
      items.push({depth: px + py, draw: () => {
        box(px - 0.3, py - 0.3, px + 0.3, py + 0.3, 0, 9, '#b5562c', '#c4643a', '#a24b25');
        const [lx, ly] = P(px, py, 9);
        [['#2f7d32', 0, -14, 9], ['#3f9a43', -5, -8, 7], ['#3f9a43', 5, -9, 7], ['#57b85a', 0, -20, 6]].forEach(([color, dx, dy, r]) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.ellipse(lx + dx, ly + dy, r, r * 0.85, 0, 0, Math.PI * 2);
          ctx.fill();
        });
      }});
    }

    // escritorios, sillas y agentes
    const nowSpots = [];
    list.forEach((agent, i) => {
      const desk = desks[i];
      if(!desk) return;
      const {sx, sy, home} = desk;
      const busy = Boolean(stepMap[agent.id]);
      const walker = walkers.current.get(agent.id);
      const look = people[i % people.length];
      const route = allRoutes.find(item => item.id === agent.route) || {};
      const accent = colorOf(route);

      const dy = sy + 1.0;
      items.push({depth: sx + 1 + dy + 0.45, draw: () => {
        box(sx, dy, sx + 2, dy + 0.9, 0, 12, '#a8693b', '#8a5530', '#6d4024');
        box(sx + 0.85, dy + 0.12, sx + 1.55, dy + 0.3, 12, 1, '#3a3a44', '#2a2a32', '#24242c');
        box(sx + 0.15, dy + 0.15, sx + 0.75, dy + 0.75, 12, 1, '#555', '#444', '#3a3a3a');
        box(sx + 0.38, dy + 0.38, sx + 0.52, dy + 0.52, 13, 3, '#555', '#444', '#3a3a3a');
        box(sx + 0.1, dy + 0.3, sx + 0.8, dy + 0.6, 16, 10, '#d9d4c7', '#bdb6a6', '#a9a292');
        const glow = busy ? (Math.floor(t * 6) % 3 ? '#7cf29a' : '#5fd47e') : isReady(route) ? '#3c6b4d' : '#2a2f36';
        poly(glow, P(sx + 0.8, dy + 0.36, 18), P(sx + 0.8, dy + 0.56, 18), P(sx + 0.8, dy + 0.56, 24), P(sx + 0.8, dy + 0.36, 24));
        if(busy){
          ctx.fillStyle = 'rgba(124,242,154,0.18)';
          const [gx, gy] = P(sx + 1.05, dy + 0.1, 22);
          ctx.beginPath();
          ctx.ellipse(gx, gy, 12, 7, 0, 0, Math.PI * 2);
          ctx.fill();
        }
        box(sx + 1.7, dy + 0.5, sx + 1.8, dy + 0.6, 12, 8, '#444', '#333', '#2a2a2a');
        const [lx, ly] = P(sx + 1.75, dy + 0.55, 21);
        ctx.fillStyle = busy ? '#ffd76a' : '#e8c26a';
        ctx.beginPath();
        ctx.ellipse(lx, ly, 4, 2.4, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = busy ? 'rgba(255,215,106,0.22)' : 'rgba(255,215,106,0.12)';
        ctx.beginPath();
        ctx.ellipse(lx, ly + 10, 14, 7, 0, 0, Math.PI * 2);
        ctx.fill();
      }});

      items.push({depth: home.x + home.y - 0.2, draw: () => {
        box(home.x - 0.32, home.y - 0.42, home.x + 0.32, home.y - 0.3, 7, 13, '#2a2a34', accent, '#30303a');
        box(home.x - 0.3, home.y - 0.3, home.x + 0.3, home.y + 0.25, 0, 7, accent, '#3a3a46', '#30303a');
      }});

      if(!walker || walker.phase === 'home'){
        items.push({depth: home.x + home.y, draw: () => {
          const [fx, fy] = P(home.x, home.y);
          if(chosen === agent.id){
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(fx, fy, 10, 4.5, 0, 0, Math.PI * 2);
            ctx.stroke();
          }
          drawPerson(ctx, look, fx, fy, true, 'sit', t + i * 0.37, busy);
          nowSpots.push({id: agent.id, x: fx, y: fy - 40, i, accent, busy, sitting: true});
        }});
      }

      if(walker && walker.phase !== 'home'){
        moveWalker(walker, dt);
        const spot = walkerSpot(walker);
        const front = walker.phase === 'talk' ? false : spot.dx + spot.dy >= 0;
        items.push({depth: spot.x + spot.y + 0.1, draw: () => {
          const [fx, fy] = P(spot.x, spot.y);
          drawPerson(ctx, look, fx, fy, front, walker.phase === 'talk' ? 'stand' : 'walk', t + i, false);
          nowSpots.push({id: agent.id, x: fx, y: fy - 40, i, accent, busy: false, walker});
        }});
        if(walker.phase === 'home'){
          const next = walker.queue.shift();
          if(next){
            const toIndex = list.findIndex(item => item.id === next.to);
            if(toIndex >= 0){
              const trip = startTrip(layout, i, toIndex, next);
              trip.queue = walker.queue;
              walkers.current.set(agent.id, trip);
            } else {
              walkers.current.delete(agent.id);
            }
          } else {
            walkers.current.delete(agent.id);
          }
        }
      }
    });

    items.sort((a, b) => a.depth - b.depth).forEach(item => item.draw());

    // la imagen escalada y encima los nombres y globos con letra nitida
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(200, size.w);
    const ch = Math.max(200, size.h);
    if(canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)){
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const main = canvas.getContext('2d');
    main.setTransform(1, 0, 0, 1, 0, 0);
    main.clearRect(0, 0, canvas.width, canvas.height);
    const fit = Math.min(canvas.width / bw, canvas.height / bh);
    const zoom = fit >= 1 ? Math.floor(fit * 8) / 8 : fit;
    const offX = Math.round((canvas.width - bw * zoom) / 2);
    const offY = Math.round((canvas.height - bh * zoom) / 2);
    main.imageSmoothingEnabled = false;
    main.drawImage(buffer, offX, offY, bw * zoom, bh * zoom);

    const scale = Math.max(0.8, Math.min(1.4, zoom / 2)) * dpr;
    spots.current = nowSpots.map(spot => ({...spot, sx: (offX + spot.x * zoom) / dpr, sy: (offY + spot.y * zoom) / dpr, r: zoom / dpr}));

    nowSpots.forEach(spot => {
      const agent = list[spot.i];
      const x = offX + spot.x * zoom;
      const y = offY + spot.y * zoom;
      main.font = `600 ${Math.round(10 * scale)}px system-ui, sans-serif`;
      const name = short(agent.name, 18);
      const tw = main.measureText(name).width + 10 * scale;
      const ny = offY + (spot.y + (spot.sitting ? 62 : 46)) * zoom;
      main.fillStyle = 'rgba(20,18,26,0.78)';
      roundRect(main, x - tw / 2, ny, tw, 15 * scale, 7 * scale);
      main.fill();
      main.fillStyle = spot.accent;
      main.textBaseline = 'top';
      main.fillText(name, x - tw / 2 + 5 * scale, ny + 2.5 * scale);

      const rows = liveMap[agent.id] || [];
      const lastTool = [...rows].reverse().find(row => row.kind === 'tool');
      const thought = [...rows].reverse().find(row => row.kind === 'thought');
      const messages = chatMap[agent.id] || [];
      const failed = messages[messages.length - 1]?.role === 'error';
      let top = y - 6 * scale;
      if(spot.walker?.phase === 'talk' || (spot.walker && spot.walker.phase !== 'home')){
        const said = spot.walker.kind === 'report' ? `${tr('Report')}: ${spot.walker.text}` : spot.walker.text;
        top = bubble(main, x, top, short(said.replace(/\s+/g, ' '), 140), 'talk', spot.accent, scale);
        return;
      }
      const listening = [...walkers.current.values()].some(item => item.phase === 'talk' && item.to === agent.id);
      if(spot.busy && !listening){
        top = bubble(main, x, top, lastTool ? short(lastTool.text, 60) : stepMap[agent.id] || tr('Thinking...'), 'status', spot.accent, scale);
        if(thought) bubble(main, x, top - 8 * scale, short(thought.text.replace(/\s+/g, ' ').slice(-160), 110), 'thought', spot.accent, scale);
        return;
      }
      if(failed){
        bubble(main, x, top, short(messages[messages.length - 1].content, 60), 'error', spot.accent, scale);
      }
    });
  };

  const hitAt = (event) => {
    const box = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - box.left;
    const y = event.clientY - box.top;
    return spots.current.find(spot => Math.abs(spot.sx - x) < 13 * spot.r && y > spot.sy - 10 * spot.r && y < spot.sy + 44 * spot.r);
  };

  const pick = (event) => {
    const hit = hitAt(event);
    setPicked(hit ? hit.id : '');
  };

  const hover = (event) => {
    canvasRef.current.style.cursor = hitAt(event) ? 'pointer' : 'default';
  };

  const agent = agents.find(item => item.id === picked);
  const route = agent ? routes.find(item => item.id === agent.route) : null;
  const rows = agent ? live[agent.id] || [] : [];
  const messages = agent ? chats[agent.id] || [] : [];
  const lastAnswer = [...messages].reverse().find(item => item.role === 'assistant');
  const lastThought = [...rows].reverse().find(row => row.kind === 'thought');
  const busy = agent ? Boolean(steps[agent.id]) : false;
  const ready = agent ? readyOf(route) : false;

  const send = () => {
    if(!agent || !draft.trim()) return;
    onSend(agent.id, draft.trim());
    setDraft('');
  };

  return (
    <div className="office">
      <div className="office-room" ref={wrapRef}>
        <canvas ref={canvasRef} style={{width: '100%', height: '100%'}} onClick={pick} onMouseMove={hover} />
        {!picked && <div className="office-hint">{tr('Click an agent to see what it thinks and talk to it')}</div>}
      </div>

      {agent && (
        <aside className="office-side card" style={{'--agent': colorOf(route || {})}}>
          <div className="office-side-head">
            <strong>{agent.name}</strong>
            <button type="button" className="icon-btn" title={tr('Close')} onClick={() => setPicked('')}><i className="fa-solid fa-xmark"></i></button>
          </div>
          <span className="office-side-sub">{agent.focus || tr('Agent')} · {route ? labelOf(route) : tr('No model')}</span>
          <span className="office-side-state">{busy ? steps[agent.id] : ready ? tr('Ready') : tr('No model loaded')}{speeds[agent.id]?.outputSpeed && busy ? ` · ${speeds[agent.id].outputSpeed} tok/s` : ''}</span>

          {busy && lastThought && (
            <div className="office-block">
              <span className="office-label">{tr('Thinking')}</span>
              <div className="office-thought">{lastThought.text.slice(-1200)}</div>
            </div>
          )}

          {busy && rows.some(row => row.kind === 'tool') && (
            <div className="office-block">
              <span className="office-label">{tr('Doing')}</span>
              {rows.filter(row => row.kind === 'tool').slice(-8).map((row, i) => <div key={i} className="office-tool"><i className="fa-solid fa-wrench"></i> {row.text}</div>)}
            </div>
          )}

          {lastAnswer && (
            <div className="office-block office-answer">
              <span className="office-label">{tr('Last answer')}</span>
              <div className="office-text">{short(lastAnswer.content, 1500)}</div>
              {lastAnswer.handoffs?.length ? <div className="office-tool"><i className="fa-solid fa-share"></i> {tr('Sent to {0}', lastAnswer.handoffs.join(', '))}</div> : null}
            </div>
          )}

          <div className="office-input">
            <input value={draft} placeholder={ready ? tr('Message {0}...', agent.name) : tr('Load its model first')} disabled={!ready} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if(e.key === 'Enter'){ e.preventDefault(); send(); } }} />
            {busy
              ? <button type="button" className="chat-send" title={tr('Stop')} onClick={() => onStop(agent.id)}><i className="fa-solid fa-stop"></i></button>
              : <button type="button" className="chat-send" title={tr('Send')} disabled={!ready || !draft.trim()} onClick={send}><i className="fa-solid fa-arrow-up"></i></button>}
          </div>
        </aside>
      )}
    </div>
  );
}
