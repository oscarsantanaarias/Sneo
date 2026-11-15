import React from 'react';
import { useEffect, useState } from 'react';
import { tr, setLanguage } from '../i18n.js';
import AgentChats from '../AgentChats.jsx';
import MobileChat from './MobileChat.jsx';
import MobileEngine from './MobileEngine.jsx';

const tabs = [
  {value: 'chat', label: 'Chat', icon: 'fa-solid fa-comment'},
  {value: 'agents', label: 'Agents', icon: 'fa-solid fa-diagram-project'},
  {value: 'engine', label: 'Engine', icon: 'fa-solid fa-gauge-high'}
];

const guessName = () => /iphone/i.test(navigator.userAgent)
  ? 'iPhone'
  : /ipad/i.test(navigator.userAgent)
    ? 'iPad'
    : /android/i.test(navigator.userAgent)
      ? 'Android phone'
      : 'Phone';

function Pair({onDone}){
  const [code, setCode] = useState(() => (window.location.hash.match(/code=([A-Za-z0-9]+)/) || [])[1] || '');
  const [name, setName] = useState(guessName);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');

  const pair = async () => {
    setBusy(true);
    setNotice('');

    try {
      const req = await fetch('/phone/pair', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({code, name})});
      const res = await req.json().catch(() => ({}));

      if(!req.ok){
        throw new Error(res.message || tr('This phone could not be paired'));
      }

      window.history.replaceState(null, '', '/');
      onDone();
    } catch(e) {
      setNotice(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="m-pair">
      <div className="m-logo"><i className="fa-solid fa-cube"></i></div>
      <h1>{tr('Use Sneo from this phone')}</h1>
      <p>{tr('On the PC open Phone in Sneo and scan the code, or type the code here.')}</p>
      <input
        className="m-code"
        value={code}
        maxLength={12}
        autoCapitalize="characters"
        autoComplete="one-time-code"
        placeholder="ABCD2345"
        onChange={(e) => setCode(e.target.value.toUpperCase())}
        aria-label={tr('Pairing code')}
      />
      <input value={name} maxLength={40} onChange={(e) => setName(e.target.value)} placeholder={tr('Name of this phone')} aria-label={tr('Name of this phone')} />
      <button type="button" className="agents-run m-wide" disabled={busy || code.replace(/[^A-Za-z0-9]/g, '').length < 8} onClick={pair}>
        <i className={busy ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-link'}></i> {tr('Pair this phone')}
      </button>
      {notice && <div className="agents-notice"><i className="fa-solid fa-triangle-exclamation"></i> {notice}</div>}
    </div>
  );
}

export default function MobileApp(){
  const [paired, setPaired] = useState(null);
  const [name, setName] = useState('');
  const [tab, setTab] = useState('chat');
  const [projects, setProjects] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [language, setLang] = useState('en');
  const [menu, setMenu] = useState(false);

  const check = async () => {
    try {
      const res = await fetch('/phone/me').then(answer => answer.json());
      setPaired(res.paired);
      setName(res.name || '');

      if(!res.paired){
        return;
      }

      const settings = await fetch('/api/settings').then(answer => answer.json()).catch(() => ({}));
      setLanguage(settings.settings?.language);
      setLang(settings.settings?.language === 'es' ? 'es' : 'en');
      fetch('/api/projects').then(answer => answer.json()).then(data => setProjects((data.projects || []).map(item => typeof item === 'string' ? {name: item} : item))).catch(() => {});
      fetch('/api/team/routes').then(answer => answer.json()).then(data => setRoutes(data.routes || [])).catch(() => {});
    } catch(e) {
      setPaired(false);
    }
  };

  useEffect(() => {
    check();
  }, []);

  const forget = async () => {
    await fetch('/phone/forget', {method: 'POST'}).catch(() => {});
    setMenu(false);
    setPaired(false);
  };

  if(paired === null){
    return <div className="m-loading"><i className="fa-solid fa-spinner fa-spin"></i></div>;
  }

  if(!paired){
    return <Pair onDone={check} />;
  }

  return (
    <div className="m-app" data-language={language}>
      <header className="m-top">
        <div className="m-brand"><span className="m-mark"><i className="fa-solid fa-cube"></i></span> Sneo</div>
        <button type="button" className="icon-btn" title={tr('This phone')} onClick={() => setMenu(prev => !prev)}><i className="fa-solid fa-ellipsis-vertical"></i></button>
        {menu && (
          <div className="m-menu">
            <span>{tr('Paired as {0}', name || tr('this phone'))}</span>
            <button type="button" className="agents-small agents-danger" onClick={forget}><i className="fa-solid fa-link-slash"></i> {tr('Unpair this phone')}</button>
          </div>
        )}
      </header>

      <main className="m-main">
        <div className="m-panel" hidden={tab !== 'chat'}><MobileChat projects={projects} /></div>
        <div className="m-panel m-scroll" hidden={tab !== 'agents'}><AgentChats apiUrl="" routes={routes} projects={projects} project="" /></div>
        <div className="m-panel m-scroll" hidden={tab !== 'engine'}><MobileEngine visible={tab === 'engine'} /></div>
      </main>

      <nav className="m-tabs">
        {tabs.map(item => (
          <button key={item.value} type="button" className={tab === item.value ? 'm-tab active' : 'm-tab'} onClick={() => setTab(item.value)}>
            <i className={item.icon}></i>
            <span>{tr(item.label)}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
