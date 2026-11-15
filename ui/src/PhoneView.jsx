import React from 'react';
import { useEffect, useState } from 'react';
import { tr } from './i18n.js';

const when = (value) => {
  const time = Date.parse(value || '');
  return Number.isFinite(time) ? new Date(time).toLocaleString() : '';
};

export default function PhoneView({apiUrl}){
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [now, setNow] = useState(Date.now());

  const read = async () => {
    try {
      setState(await fetch(`${apiUrl}/api/phone`).then(answer => answer.json()));
    } catch(e) {
      setNotice(e.message);
    }
  };

  useEffect(() => {
    read();
    const timer = setInterval(() => {
      setNow(Date.now());
    }, 1000);
    const refresh = setInterval(read, 15000);
    return () => {
      clearInterval(timer);
      clearInterval(refresh);
    };
  }, []);

  const call = async (method, route, body = null) => {
    setBusy(true);
    setNotice('');

    try {
      const res = await fetch(`${apiUrl}/api/phone${route}`, {method, headers: {'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined}).then(answer => answer.json());
      setState(res);

      if(res.problem){
        setNotice(res.problem);
      }
    } catch(e) {
      setNotice(e.message);
    } finally {
      setBusy(false);
    }
  };

  const left = state?.code ? Math.max(0, Math.round((state.code.expiresAt - now) / 1000)) : 0;
  const expired = state?.code && left === 0;

  return (
    <div className="view-page">
      <div className="view-head">
        <div>
          <h1>{tr('Phone')}</h1>
          <span>{tr('Chat with Sneo, answer its questions and load models from your phone')}</span>
        </div>
        <div className="view-head-actions">
          <span className={`state-pill ${state?.listening ? 'pill-ok' : 'pill-idle'}`}>
            <span className="state-dot"></span>
            {state?.listening ? tr('on') : tr('off')}
          </span>
        </div>
      </div>

      {notice && <div className="view-alert alert-bad"><i className="fa-solid fa-triangle-exclamation"></i>{notice}</div>}

      <div className="agents-grid agents-grid-flip">
        <div className="agents-col">
          <section className="card">
            <div className="card-head">
              <h3>{tr('Phone access')}</h3>
              <label className="phone-switch">
                <input type="checkbox" checked={Boolean(state?.enabled)} disabled={busy || !state} onChange={(e) => call('POST', '', {enabled: e.target.checked})} />
                <span>{state?.enabled ? tr('On') : tr('Off')}</span>
              </label>
            </div>

            {!state?.enabled && (
              <span className="agents-hint">{tr('Turn it on to open Sneo from a phone on the same Wi-Fi. Windows may ask once to let Sneo use the network: allow it for private networks.')}</span>
            )}

            {state?.enabled && state.listening && state.firewall === false && (
              <div className="view-alert alert-bad phone-firewall">
                <i className="fa-solid fa-shield-halved"></i>
                <span>{tr('Windows Firewall blocks the phone: it opens the address but the page never loads.')}</span>
                <button type="button" className="agents-small" disabled={busy} onClick={() => call('POST', '/firewall')}>
                  {tr('Allow on private networks')}
                </button>
              </div>
            )}

            {state?.enabled && state.listening && (
              <div className="phone-pair">
                {state.qr && !expired ? <img className="phone-qr" src={state.qr} alt={tr('Code to scan with the phone camera')} /> : <div className="phone-qr phone-qr-empty"><i className="fa-solid fa-qrcode"></i></div>}
                <div className="phone-steps">
                  <ol>
                    <li>{tr('Connect the phone to the same Wi-Fi as this PC.')}</li>
                    <li>{tr('Scan the code with the phone camera, or open the address and type the code.')}</li>
                    <li>{tr('On the phone, use Add to Home screen to open Sneo like an app.')}</li>
                  </ol>
                  {state.code && !expired && (
                    <div className="phone-code">
                      <strong>{state.code.code.replace(/(.{4})/, '$1 ')}</strong>
                      <span>{tr('expires in {0}:{1}', Math.floor(left / 60), String(left % 60).padStart(2, '0'))}</span>
                    </div>
                  )}
                  <button type="button" className="agents-small" disabled={busy} onClick={() => call('POST', '/code')}>
                    <i className="fa-solid fa-rotate"></i> {state.code && !expired ? tr('New code') : tr('Make a code')}
                  </button>
                  {(state.links || []).map(link => <code key={link} className="phone-link">{link}</code>)}
                  {!state.links?.length && <span className="agents-notice">{tr('No home network found on this PC.')}</span>}
                </div>
              </div>
            )}
          </section>
        </div>

        <div className="agents-col">
          <section className="card">
            <div className="card-head">
              <h3>{tr('Paired phones')}</h3>
              <span className="card-count">{state?.devices?.length || 0}</span>
            </div>
            {!state?.devices?.length && <span className="agents-hint">{tr('No phone yet.')}</span>}
            {(state?.devices || []).map(item => (
              <div key={item.id} className="phone-device">
                <i className="fa-solid fa-mobile-screen"></i>
                <div>
                  <strong>{item.name}</strong>
                  <span>{tr('last used {0}', when(item.lastSeen))}</span>
                </div>
                <button type="button" className="icon-btn icon-danger" title={tr('Remove this phone')} disabled={busy} onClick={() => call('DELETE', `/devices/${encodeURIComponent(item.id)}`)}>
                  <i className="fa-solid fa-xmark"></i>
                </button>
              </div>
            ))}
            <span className="agents-hint">{tr('Each phone keeps its own key. Removing it here locks it out right away. Away from home, install Tailscale on the PC and the phone and open the Tailscale address.')}</span>
          </section>
        </div>
      </div>
    </div>
  );
}
