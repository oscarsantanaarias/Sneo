import React from 'react';
import { useEffect, useRef, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { tr } from '../i18n.js';
import { readEvents } from '../agentsShared.js';

const modes = [
  {value: 'manual', label: 'Ask first'},
  {value: 'plan', label: 'Plan'},
  {value: 'auto', label: 'Auto'}
];

const html = (text) => ({__html: DOMPurify.sanitize(marked.parse(String(text || '')))});
const emptyLive = {status: '', thoughts: '', steps: [], speed: 0, queue: 0};

export default function MobileChat({projects = []}){
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState('');
  const [messages, setMessages] = useState([]);
  const [project, setProject] = useState('');
  const [mode, setMode] = useState('manual');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [live, setLive] = useState(emptyLive);
  const [pending, setPending] = useState([]);
  const [drawer, setDrawer] = useState(false);
  const [notice, setNotice] = useState('');
  const listRef = useRef(null);

  const loadChats = async () => {
    const res = await fetch('/api/chats').then(answer => answer.json()).catch(() => ({}));
    const list = Array.isArray(res.chats) ? res.chats : [];
    setChats(list);
    return list;
  };

  const loadPending = async () => {
    const res = await fetch('/api/pending').then(answer => answer.json()).catch(() => ({}));
    setPending(Array.isArray(res.items) ? res.items : []);
  };

  const openChat = (chat) => {
    setChatId(chat.id);
    setMessages(Array.isArray(chat.messages) ? chat.messages : []);
    setProject(chat.project || '');

    if(chat.accessMode){
      setMode(chat.accessMode);
    }

    setDrawer(false);
  };

  const newChat = async () => {
    const res = await fetch('/api/chats', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({title: tr('Chat from the phone')})}).then(answer => answer.json()).catch(() => ({}));

    if(res.chat){
      setChats(prev => [res.chat, ...prev]);
      openChat({...res.chat, messages: []});
    }
  };

  useEffect(() => {
    (async () => {
      const settings = await fetch('/api/settings').then(answer => answer.json()).catch(() => ({}));

      if(settings.settings?.accessMode){
        setMode(settings.settings.accessMode);
      }

      const list = await loadChats();

      if(list[0]){
        openChat(list[0]);
      } else {
        await newChat();
      }

      loadPending();
    })();

    const timer = setInterval(loadPending, 8000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({top: listRef.current.scrollHeight, behavior: 'smooth'});
  }, [messages.length, live.status, live.steps.length, pending.length]);

  const save = async (id, list, where) => {
    await fetch(`/api/chats/${encodeURIComponent(id)}`, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({messages: list, project: where, accessMode: mode})}).catch(() => {});
  };

  const answer = async (id, verdict) => {
    const req = await fetch(`/api/pending/${encodeURIComponent(id)}/${verdict}`, {method: 'POST'}).catch(() => null);
    const res = req ? await req.json().catch(() => ({})) : {};

    if(req && !req.ok){
      setNotice(res.message || tr('That request could not be answered'));
    }

    loadPending();
  };

  const stop = async () => {
    await fetch('/api/chat/stop', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({chatId})}).catch(() => {});
  };

  const send = async () => {
    const clean = text.trim();

    if(!clean || !chatId){
      return;
    }

    if(loading){
      setText('');
      await fetch('/api/chat/steer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({chatId, message: clean})}).catch(() => {});
      setLive(prev => ({...prev, status: tr('Sent, the agent reads it at its next step')}));
      return;
    }

    const before = messages.filter(item => !item.partial);
    const next = [...before, {role: 'user', content: clean}];
    const startedAt = Date.now();
    let final = null;
    let thoughts = '';
    let fresh = false;

    setMessages(next);
    setText('');
    setLoading(true);
    setNotice('');
    setLive({...emptyLive, status: tr('Sending...')});

    try {
      const req = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({chatId, message: clean, history: before, project, accessMode: mode})
      });

      await readEvents(req, event => {
        if(event.type === 'queued'){
          setLive(prev => ({...prev, queue: Number(event.position) || 0, status: tr('Waiting for the model, {0} ahead', event.position)}));
        }

        if(event.type === 'started'){
          setLive(prev => ({...prev, queue: 0, status: tr('Thinking')}));
        }

        if(event.type === 'thought' && event.thought){
          thoughts = event.delta && fresh ? `${thoughts}${event.thought}` : thoughts ? `${thoughts}\n\n${event.thought}` : event.thought;
          fresh = Boolean(event.delta);
          setLive(prev => ({...prev, thoughts, status: tr('Thinking')}));
        }

        if(event.type === 'step' && event.step){
          setLive(prev => {
            const steps = [...prev.steps];
            const at = steps.findIndex(item => item.tool === event.step.tool && item.file === event.step.file && item.status === 'running');

            if(at >= 0 && event.step.status !== 'running'){
              steps[at] = event.step;
            } else {
              steps.push(event.step);
            }

            return {...prev, steps: steps.slice(-30), status: ''};
          });
        }

        if(event.type === 'tool' && !event.step && event.tool && !/^[a-z0-9_]+$/.test(String(event.tool))){
          setLive(prev => ({...prev, status: String(event.tool).replace(/\.+$/, '').replace(/^./, first => first.toUpperCase())}));
        }

        if(event.type === 'speed'){
          setLive(prev => ({...prev, speed: Number(event.speed) || 0, status: tr('Writing the reply')}));
        }

        if(event.type === 'writing'){
          setLive(prev => ({...prev, status: tr('Writing {0}', String(event.tool || tr('a tool call')).replace(/_/g, ' '))}));
        }

        if(event.type === 'approval' || event.type === 'approval-done'){
          loadPending();
        }

        if(event.type === 'done'){
          final = event;
        }

        if(event.type === 'error'){
          throw new Error(event.message);
        }
      });

      if(!final){
        throw new Error(tr('The server cut the answer. Try again.'));
      }

      const done = [...next, {role: 'assistant', content: final.message, steps: final.steps || [], thoughts, duration: Math.round((Date.now() - startedAt) / 1000), mode}];
      setMessages(done);
      await save(chatId, done, project);
      loadChats();
    } catch(e) {
      setNotice(e.message);
      await save(chatId, next, project);
    } finally {
      setLoading(false);
      setLive(emptyLive);
      loadPending();
    }
  };

  const changeProject = (value) => {
    setProject(value);

    if(chatId){
      save(chatId, messages.filter(item => !item.partial), value);
    }
  };

  const title = chats.find(item => item.id === chatId)?.title || tr('Chat');

  return (
    <div className="m-chat">
      <div className="m-bar">
        <button type="button" className="icon-btn" title={tr('Chats')} onClick={() => setDrawer(prev => !prev)}><i className="fa-solid fa-bars"></i></button>
        <strong className="m-title">{title}</strong>
        <button type="button" className="icon-btn" title={tr('New chat')} onClick={newChat}><i className="fa-solid fa-pen-to-square"></i></button>
      </div>

      <div className="m-bar m-options">
        <select value={project} onChange={(e) => changeProject(e.target.value)} aria-label={tr('Project')}>
          <option value="">{tr('No project')}</option>
          {projects.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value)} aria-label={tr('What the agent may do')}>
          {modes.map(item => <option key={item.value} value={item.value}>{tr(item.label)}</option>)}
        </select>
      </div>

      {drawer && (
        <div className="m-drawer">
          {chats.map(item => (
            <button key={item.id} type="button" className={item.id === chatId ? 'm-chat-item active' : 'm-chat-item'} onClick={() => openChat(item)}>
              <strong>{item.title}</strong>
              <span>{item.project || tr('No project')}</span>
            </button>
          ))}
        </div>
      )}

      <div className="m-list" ref={listRef}>
        {messages.filter(item => item.role === 'user' || item.role === 'assistant').map((item, i) => (
          item.role === 'user'
            ? <div key={i} className="m-bubble m-user">{item.content}</div>
            : (
              <div key={i} className="m-bubble m-agent">
                {item.thoughts && (
                  <details className="m-thoughts">
                    <summary>{tr('Thoughts')}</summary>
                    <div>{item.thoughts}</div>
                  </details>
                )}
                {item.steps?.length > 0 && (
                  <details className="m-steps">
                    <summary><i className="fa-solid fa-wrench"></i> {tr('{0} steps', item.steps.length)}</summary>
                    {item.steps.map((step, s) => <div key={s} className="m-step">{step.tool}{step.file ? ` · ${step.file}` : ''}</div>)}
                  </details>
                )}
                <div className="markdown-body" dangerouslySetInnerHTML={html(item.content)} />
                {item.duration ? <span className="m-meta">{item.duration} s</span> : null}
              </div>
            )
        ))}

        {pending.map(item => (
          <div key={item.id} className="m-approval">
            <strong><i className="fa-solid fa-shield-halved"></i> {item.title}</strong>
            {item.command && <pre>{item.command}</pre>}
            {item.details && <pre>{item.details}</pre>}
            {(item.files || []).map(file => (
              <details key={file.file}>
                <summary>{file.file}</summary>
                <pre>{String(file.diff || '').slice(0, 4000)}</pre>
              </details>
            ))}
            <div className="agents-row">
              <button type="button" className="agents-small agents-danger" onClick={() => answer(item.id, 'deny')}>{tr('Deny')}</button>
              <span className="rail-spacer"></span>
              <button type="button" className="agents-run" onClick={() => answer(item.id, 'approve')}><i className="fa-solid fa-check"></i> {tr('Approve')}</button>
            </div>
          </div>
        ))}

        {loading && (
          <div className="m-live">
            {live.steps.length > 0 && (
              <div className="m-live-steps">
                {live.steps.slice(-4).map((step, i) => (
                  <span key={i} className={`m-live-step m-${step.status || 'running'}`}>
                    <i className={step.status === 'running' ? 'fa-solid fa-spinner fa-spin' : step.status === 'error' ? 'fa-solid fa-xmark' : 'fa-solid fa-check'}></i> {step.tool}{step.file ? ` · ${step.file}` : ''}
                  </span>
                ))}
              </div>
            )}
            {live.thoughts && <div className="m-live-thoughts">{live.thoughts.slice(-400)}</div>}
            <div className="m-live-status">
              <span className="chat-typing"><span></span><span></span><span></span></span>
              {live.status || tr('Working')}
              {live.speed > 0 && <em>{live.speed.toFixed(0)} tok/s</em>}
            </div>
          </div>
        )}

        {notice && <div className="chat-bubble chat-error"><i className="fa-solid fa-circle-exclamation"></i> {notice}</div>}
      </div>

      <div className="m-input">
        <textarea
          rows="1"
          value={text}
          placeholder={loading ? tr('Add something while it works...') : tr('Message Sneo...')}
          onChange={(e) => setText(e.target.value)}
        />
        {loading && <button type="button" className="m-send m-stop" title={tr('Stop')} onClick={stop}><i className="fa-solid fa-stop"></i></button>}
        <button type="button" className="m-send" title={tr('Send')} disabled={!text.trim()} onClick={send}><i className="fa-solid fa-arrow-up"></i></button>
      </div>
    </div>
  );
}
