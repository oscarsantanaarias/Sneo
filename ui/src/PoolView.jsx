import React from 'react';
import { useState, useEffect, useRef } from 'react';
import { tr } from './i18n.js';

const gib = 1073741824;
const kindColors = {gpu: '#c98420', ram: '#22a6b0', ssd: '#9377ea'};
const kindIcons = {gpu: 'fa-solid fa-microchip', ram: 'fa-solid fa-memory', ssd: 'fa-solid fa-hard-drive'};
const driveLabels = {nvme: 'NVMe', ssd: 'SSD', hdd: 'HDD', usb: 'USB'};
const slowDrives = {
  hdd: 'Spinning hard drive: experts read from it come in at a fraction of SSD speed. Keep the model on an SSD or NVMe drive.',
  usb: 'USB drive: expert reads are limited by the USB link. Keep the model on an internal SSD or NVMe drive.'
};
const contextSizes = [
  {label: '32K', value: 32768},
  {label: '64K', value: 65536},
  {label: '128K', value: 131072},
  {label: '256K', value: 262144}
];

const gb = (bytes) => `${(Number(bytes || 0) / gib).toFixed(1)} GB`;

const specsOf = (item) => [
  item.gpu || tr('no GPU'),
  item.gpu ? `${Math.round(Number(item.vramTotal || 0) / gib)} GB` : '',
  tr('{0} GB RAM', Math.round(Number(item.ramTotal || 0) / gib))
].filter(Boolean).join(' · ');

const ago = (seconds) => {
  const value = Number(seconds || 0);

  return value < 60
    ? tr('{0}s ago', value)
    : value < 3600
      ? tr('{0} min ago', Math.floor(value / 60))
      : tr('{0} h ago', Math.floor(value / 3600));
};

const amountModes = [
  {value: 'gb', label: 'GB', title: 'Use up to this many GB'},
  {value: 'percent', label: '%', title: 'Use this share of the device'},
  {value: 'keep', label: 'Keep', title: 'Keep this many GB free, use the rest'}
];

const readSlots = (list) => {
  const seen = {};

  return (list || []).map(item => {
    const base = `${item.node}|${item.kind}`;
    seen[base] = (seen[base] || 0) + 1;
    const row = {id: `${base}|${seen[base]}`, node: item.node, kind: item.kind, unit: 'gb', value: ''};

    if(item.keep > 0){
      return {...row, unit: 'keep', value: item.keep};
    }

    if(item.gb > 0){
      return {...row, value: item.gb};
    }

    if(item.percent > 0 && item.limit > 0){
      return {...row, value: Math.round(item.limit / gib * 10) / 10};
    }

    if(item.percent > 0){
      return {...row, unit: 'percent', value: item.percent};
    }

    return row;
  });
};

function PoolView({apiUrl, models = [], currentModel = '', modelFolder = '', scanning = false, loadingModel = false, onFolderChange, onScan, onLoad}) {
  const [pool, setPool] = useState(null);
  const [remote, setRemote] = useState(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [slots, setSlots] = useState([]);
  const [context, setContext] = useState(131072);
  const [frontLayers, setFrontLayers] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [dragIndex, setDragIndex] = useState(-1);
  const [grabbed, setGrabbed] = useState('');
  const [newSlot, setNewSlot] = useState({node: '', kind: 'gpu'});
  const [busy, setBusy] = useState('');
  const [copied, setCopied] = useState('');
  const [pick, setPick] = useState('');
  const [network, setNetwork] = useState({acceptRequests: true, doorPort: '9097', apiListen: 'local', allowNetwork: false});
  const [networkDirty, setNetworkDirty] = useState(false);
  const [sneo, setSneo] = useState(null);
  const [firewall, setFirewall] = useState(null);
  const [found, setFound] = useState(null);
  const [joinHost, setJoinHost] = useState('');
  const [helper, setHelper] = useState({running: false});
  const [helperProblem, setHelperProblem] = useState('');
  const dirtyRef = useRef(false);
  const networkDirtyRef = useRef(false);
  const dragRef = useRef(-1);

  dirtyRef.current = dirty;
  networkDirtyRef.current = networkDirty;

  const readSneo = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/network`);

      if(!req.ok){
        setSneo(null);
        return;
      }

      const res = await req.json();
      setSneo(res);

      if(!networkDirtyRef.current){
        setNetwork(prev => ({...prev, allowNetwork: res.allowNetwork === true}));
      }

    } catch(e) {
      setSneo(null);
    }
  };

  useEffect(() => {
    readSneo();
  }, [apiUrl]);

  const readFirewall = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/firewall`);
      const res = await req.json();
      setFirewall(res);
      return res.ready;
    } catch(e) {
      setFirewall(null);
      return false;
    }
  };

  useEffect(() => {
    readFirewall();
  }, [apiUrl]);

  const readHelper = async () => {
    try {
      const req = await fetch(`${apiUrl}/api/engine/worker`);
      setHelper(await req.json());
    } catch(e) {
      setHelper({running: false});
    }
  };

  useEffect(() => {
    readHelper();
    const timer = setInterval(readHelper, 3000);
    return () => clearInterval(timer);
  }, [apiUrl]);

  useEffect(() => {
    findPools();
  }, [apiUrl]);

  const findPools = async () => {
    setBusy('scan');
    setHelperProblem('');
    try {
      const req = await fetch(`${apiUrl}/api/engine/pools/scan`);
      const res = await req.json();
      setFound(res.found || []);
    } catch(e) {
      setHelperProblem(e.message);
    } finally {
      setBusy('');
    }
  };

  const joinPool = async (host) => {
    setBusy('join');
    setHelperProblem('');
    try {
      const req = await fetch(`${apiUrl}/api/engine/worker/join`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({host})
      });
      const res = await req.json();
      if(!req.ok) throw new Error(res.message);
      setHelper(res);
    } catch(e) {
      setHelperProblem(e.message);
    } finally {
      setBusy('');
    }
  };

  const leaveHelper = async () => {
    setBusy('leave');
    try {
      await fetch(`${apiUrl}/api/engine/worker/leave`, {method: 'POST'});
      await readHelper();
    } finally {
      setBusy('');
    }
  };

  const openFirewall = async () => {
    setBusy('firewall');
    try {
      await fetch(`${apiUrl}/api/engine/firewall`, {method: 'POST'});
      for(let i = 0; i < 40; i++){
        await new Promise(done => setTimeout(done, 3000));
        if(await readFirewall()) break;
      }
    } finally {
      setBusy('');
    }
  };

  useEffect(() => {
    let alive = true;
    let timer = null;

    const tick = async () => {
      try {
        const req = await fetch(`${apiUrl}/api/engine/pool`);
        const res = await req.json();

        if(!req.ok){
          throw new Error(res.message || res.error || tr('Pool answered {0}', req.status));
        }

        const hostReq = await fetch(`${apiUrl}/api/engine/pool/host`).catch(() => null);
        const hostRes = hostReq && hostReq.ok ? await hostReq.json().catch(() => ({})) : {};
        const shown = hostRes.joined && hostRes.pool ? hostRes.pool : res;

        if(!alive){
          return;
        }

        setRemote(hostRes.joined ? {host: hostRes.host, name: hostRes.pool?.self || hostRes.host, viewer: res.self, error: hostRes.error || ''} : null);
        setPool(shown);
        setError('');

        if(!dirtyRef.current || hostRes.joined){
          setSlots(readSlots(shown.slots));
          setContext(shown.context || 131072);
          setFrontLayers(Number(shown.frontLayers || 0));
        }

        if(!networkDirtyRef.current && res.network){
          setNetwork(prev => ({
            ...prev,
            acceptRequests: res.network.acceptRequests,
            doorPort: String(res.network.doorPort),
            apiListen: res.network.apiListen
          }));
        }

      } catch(e) {
        if(alive){
          setError(e instanceof SyntaxError ? tr('The SNEO server has no /api/engine route yet. Restart it so engine.js gets loaded.') : e.message);
        }
      }

      if(alive){
        timer = setTimeout(tick, 2000);
      }
    };

    tick();

    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [apiUrl]);

  const runAction = async (action, body = {}, key = action) => {
    setBusy(key);
    setNotice('');

    try {
      const req = await fetch(`${apiUrl}/api/engine/pool/${action}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || res.error || tr('Pool answered {0}', req.status));
      }

      setPool(res);

      if(!dirtyRef.current){
        setSlots(readSlots(res.slots));
      }

    } catch(e) {
      setNotice(e.message);
    }

    setBusy('');
  };

  const switchModel = async (action) => {
    setBusy(action);
    setNotice('');

    try {
      const req = await fetch(`${apiUrl}/api/models/${action}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: '{}'
      });
      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || res.error || tr('Engine answered {0}', req.status));
      }

      if(res.slots) setPool(res);
    } catch(e) {
      setNotice(e.message);
    }

    setBusy('');
  };

  const savePool = async () => {
    setBusy('save');
    setNotice('');

    try {
      const req = await fetch(`${apiUrl}/api/engine/pool`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          slots: slots.map(row => ({
            node: row.node,
            kind: row.kind,
            gb: row.unit === 'gb' ? Number(row.value) || 0 : 0,
            percent: row.unit === 'percent' ? Number(row.value) || 0 : 0,
            keep: row.unit === 'keep' ? Number(row.value) || 0 : 0
          })),
          context: Number(context),
          frontLayers: Math.max(0, Math.round(Number(frontLayers) || 0))
        })
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || res.error || tr('Pool answered {0}', req.status));
      }

      setPool(res);
      setSlots(readSlots(res.slots));
      setContext(res.context || Number(context));
      setFrontLayers(Number(res.frontLayers || 0));
      setDirty(false);

    } catch(e) {
      setNotice(e.message);
    }

    setBusy('');
  };

  const changeNetwork = (patch) => {
    setNetwork(prev => ({...prev, ...patch}));
    setNetworkDirty(true);
  };

  const revertNetwork = () => {
    setNetwork({
      acceptRequests: pool?.network?.acceptRequests ?? true,
      doorPort: String(pool?.network?.doorPort ?? 9097),
      apiListen: pool?.network?.apiListen || 'local',
      allowNetwork: sneo?.allowNetwork === true
    });
    setNetworkDirty(false);
  };

  const saveNetwork = async () => {
    const doorPort = Number(network.doorPort);

    if(!Number.isInteger(doorPort) || doorPort < 1 || doorPort > 65535){
      setNotice(tr('The door port must be a number from 1 to 65535'));
      return;
    }

    setBusy('network');
    setNotice('');

    try {
      const body = {acceptRequests: network.acceptRequests, doorPort, apiListen: network.apiListen};

      if(sneo){
        body.allowNetwork = network.allowNetwork;
      }

      const req = await fetch(`${apiUrl}/api/engine/network`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });

      const res = await req.json();

      if(!req.ok){
        throw new Error(res.message || res.error || tr('Pool answered {0}', req.status));
      }

      if(res.network){
        setPool(res);
      }

      setNetworkDirty(false);
      await readSneo();

    } catch(e) {
      setNotice(e instanceof SyntaxError ? tr('Restart the SNEO server so it can save the network settings.') : e.message);
    }

    setBusy('');
  };

  const copyText = async (key, value) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(key);
      setTimeout(() => setCopied(''), 1400);
    } catch(e) {
      setNotice(tr('Clipboard is not available here'));
    }
  };

  const nodes = pool?.nodes || [];
  const machines = nodes.filter(item => item.self || item.accepted || !item.connected);
  const requests = pool?.requests || [];
  const members = nodes.filter(item => item.self || item.accepted);
  const net = pool?.network || null;
  const addresses = net?.addresses || [];
  const answering = busy.startsWith('accept:') || busy.startsWith('decline:');
  const plan = pool?.plan || {};
  const model = pool?.model || {};
  const helperNode = nodes.find(item => !item.self && item.connected && item.accepted && item.stats?.gpu?.present);
  const layerCount = Number(model.layers || 0);
  const local = pool?.local || {};
  const liveSlots = pool?.slots || [];
  const demand = Number(plan.demand || 0);
  const placed = liveSlots.reduce((sum, item) => sum + Number(item.used || 0), 0);
  const fillTotal = Math.max(demand, placed, 1);
  const unplaced = Math.max(0, demand - placed);
  const addNode = newSlot.node || members[0]?.name || '';
  const addStats = nodes.find(item => item.name === addNode)?.stats;
  const modelFiles = models.filter(item => !/nomic|embed|llava|clip|bge|minilm/i.test(item.name));
  const pickedModel = pick || (modelFiles.some(item => item.name === currentModel) ? currentModel : modelFiles[0]?.name || '');

  const statsOf = (name) => nodes.find(item => item.name === name)?.stats || null;

  const totalOf = (row) => {
    const stats = statsOf(row.node);

    if(!stats){
      return 0;
    }

    return row.kind === 'gpu'
      ? (stats.gpu?.present ? stats.gpu.vramTotal : 0)
      : row.kind === 'ram'
        ? stats.ramTotal
        : stats.diskFree;
  };

  const freeOf = (row) => {
    const stats = statsOf(row.node);

    if(!stats){
      return 0;
    }

    return row.kind === 'gpu'
      ? (stats.gpu?.present ? stats.gpu.vramFree : 0)
      : row.kind === 'ram'
        ? stats.ramFree
        : stats.diskFree;
  };

  const changeRow = (index, patch) => {
    setSlots(prev => prev.map((row, i) => i === index ? {...row, ...patch} : row));
    setDirty(true);
  };

  const reachableOf = (row) => {
    const stats = statsOf(row.node);
    const total = totalOf(row);

    return row.kind === 'ram' && stats
      ? Math.min(total, Number(stats.ramFree || 0) + Number(stats.heldRam || 0))
      : total;
  };

  const limitOf = (row) => {
    const total = totalOf(row);
    const value = Number(row.value || 0);

    const limit = row.unit === 'gb'
      ? value * gib
      : row.unit === 'percent'
        ? value / 100 * total
        : Math.max(0, reachableOf(row) - value * gib);

    return total ? Math.min(total, limit) : limit;
  };

  const changeUnit = (index, unit) => {
    const row = slots[index];
    const total = totalOf(row);

    if(row.unit === unit){
      return;
    }

    if(!total){
      changeRow(index, {unit});
      return;
    }

    const limit = limitOf(row);

    const value = unit === 'gb'
      ? Math.round(limit / gib * 10) / 10
      : unit === 'percent'
        ? Math.min(100, Math.round(limit / total * 100))
        : Math.round(Math.max(0, reachableOf(row) - limit) / gib * 10) / 10;

    changeRow(index, {unit, value});
  };

  const moveRow = (to) => {
    const from = dragRef.current;

    if(from < 0 || from === to){
      return;
    }

    dragRef.current = to;

    setSlots(prev => {
      const copy = [...prev];
      const [moved] = copy.splice(from, 1);
      copy.splice(to, 0, moved);
      return copy;
    });
    setDragIndex(to);
    setDirty(true);
  };

  const addRow = () => {
    if(!addNode){
      return;
    }

    const total = totalOf({node: addNode, kind: newSlot.kind});
    const share = newSlot.kind === 'gpu'
      ? total * 0.9
      : newSlot.kind === 'ram'
        ? total * 0.5
        : Math.min(64 * gib, total * 0.25);

    setSlots(prev => [...prev, {
      id: `new|${Date.now()}`,
      node: addNode,
      kind: newSlot.kind,
      unit: 'gb',
      value: total ? Math.round(share / gib * 2) / 2 : (newSlot.kind === 'ssd' ? 32 : 8)
    }]);
    setDirty(true);
  };

  const revert = () => {
    setSlots(readSlots(pool?.slots));
    setContext(pool?.context || 131072);
    setFrontLayers(Number(pool?.frontLayers || 0));
    setDirty(false);
  };

  if(!pool){
    return (
      <div className="view-page">
        <div className="view-head">
          <div>
            <h1>{tr('Pool')}</h1>
            <span>{tr('Machines that share the model')}</span>
          </div>
        </div>
        <div className="view-empty">
          <i className={error ? 'fa-solid fa-plug-circle-xmark' : 'fa-solid fa-circle-notch fa-spin'}></i>
          <p>{error || tr('Reading the pool...')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="view-page">
      <div className="view-head">
        <div>
          <h1>{tr('Pool')}</h1>
          <span>{pool.self} · {members.length} {members.length === 1 ? tr('machine') : tr('machines')} {tr('· context {0}K', Math.round((pool.context || 0) / 1024))}</span>
        </div>

        <div className="view-head-actions">
          <span className={plan.ok ? 'state-pill pill-ok' : 'state-pill pill-warn'}>
            <i className={plan.ok ? 'fa-solid fa-check' : 'fa-solid fa-triangle-exclamation'}></i>
            {plan.ok ? tr('Plan ready') : tr('Plan not ready')}
          </span>
          <button type="button" className="btn btn-primary" disabled={!!remote || busy === 'apply' || !plan.ok || dirty || pool.generation === 'loading'} onClick={() => runAction('apply')} title={dirty ? tr('Save the priority first') : pool.generation === 'loading' ? tr('The model is still loading') : tr('Load the model with this split')}>
            <i className="fa-solid fa-play"></i>
            {tr('Apply')}
          </button>
          <button type="button" className="btn" disabled={!!remote || busy === 'unload'} onClick={() => runAction('unload')} title={tr('Release the memory on every machine')}>
            <i className="fa-solid fa-eject"></i>
            {tr('Empty')}
          </button>
        </div>
      </div>

      {remote && (
        <div className="view-alert alert-warn">
          <i className="fa-solid fa-circle-info"></i>
          {remote.error
            ? tr('This PC is in the pool of {0}, but that pool could not be read: {1}.', remote.host, remote.error)
            : tr('This PC is helping the pool of {0}. You see that pool here; change it on {1}.', remote.name, remote.name)}
        </div>
      )}
      {error && <div className="view-alert alert-bad"><i className="fa-solid fa-plug-circle-xmark"></i>{error}</div>}
      {notice && <div className="view-alert alert-bad"><i className="fa-solid fa-circle-exclamation"></i>{notice}</div>}
      {pool.notice && <div className="view-alert alert-warn"><i className="fa-solid fa-circle-info"></i>{pool.notice}</div>}
      {firewall && !firewall.ready && network.acceptRequests && (
        <div className="view-alert alert-warn">
          <i className="fa-solid fa-shield-halved"></i>
          <span>
            {firewall.publicNetwork
              ? tr('Other PCs cannot join yet: the network {0} is marked public, so Windows keeps the pool door closed.', firewall.publicNetwork)
              : tr('Other PCs cannot join yet: the Windows firewall keeps the pool door closed.')}
          </span>
          <button type="button" className="btn btn-ghost" disabled={busy === 'firewall'} onClick={openFirewall}>
            {busy === 'firewall' ? tr('Waiting for Windows...') : tr('Fix it')}
          </button>
        </div>
      )}

      <div className="load-strip">
        <div className="load-strip-state">
          <span className="load-label">{tr('Load')}</span>
          <span className={`state-pill pill-${local.state === 'ready' ? 'ok' : local.state === 'failed' ? 'bad' : local.state === 'loading' ? 'busy' : 'idle'}`}>{local.state || tr('empty')}</span>
        </div>
        <div className="load-strip-bar">
          <div className="meter">
            <div className="meter-fill" style={{width: `${local.total ? Math.min(100, local.done / local.total * 100) : local.state === 'ready' ? 100 : 0}%`}}></div>
          </div>
          <span>{gb(local.done)} / {gb(local.total)}</span>
        </div>
        <div className="load-strip-mem">
          <span><i className={kindIcons.gpu} style={{color: kindColors.gpu}}></i>{gb(local.gpu)}</span>
          <span><i className={kindIcons.ram} style={{color: kindColors.ram}}></i>{gb(local.ram)}</span>
        </div>
        {local.problem && <div className="load-strip-problem">{local.problem}</div>}
      </div>

      <div className="card join-card">
        <div className="card-head">
          <h3>{tr('Join another PC\'s pool')}</h3>
          <span className="card-count">{helper.running ? helper.status : tr('off')}</span>
        </div>
        <p className="card-note">{tr('Lend this PC\'s GPU to the pool of another PC on your network. Its own model is unloaded while it helps.')}</p>
        {helper.running && helper.status !== 'idle' ? (
          <div className="join-live">
            <span>
              {helper.status === 'joined'
                ? tr('Helping the pool at {0}{1}', helper.host, helper.linked ? tr(', computing its first layers') : tr(', waiting for it to connect'))
                : helper.status === 'waiting'
                  ? tr('Waiting for {0} to accept', helper.host)
                  : helper.problem || `${helper.status} ${helper.host || ''}`}
            </span>
            <button type="button" className="btn" disabled={busy === 'leave'} onClick={leaveHelper}>
              {busy === 'leave' ? tr('Disconnecting...') : tr('Disconnect')}
            </button>
          </div>
        ) : (
          <>
            <div className="join-row">
              <button type="button" className="btn btn-primary" disabled={busy === 'scan'} onClick={findPools}>
                <i className="fa-solid fa-magnifying-glass"></i>
                {busy === 'scan' ? tr('Looking on your network...') : tr('Find pools on my network')}
              </button>
              <input type="text" placeholder={tr('or type its IPv4')} value={joinHost} onChange={e => setJoinHost(e.target.value)} />
              <button type="button" className="btn" disabled={busy === 'join' || !joinHost.trim()} onClick={() => joinPool(joinHost.trim())}>{tr('Join')}</button>
            </div>
            {found && !found.length && <p className="card-note">{tr('No Sneo pool answered on your network. The other PC needs Sneo open with Accept join requests on.')}</p>}
            {found && found.map(item => {
              const routes = item.routes?.length ? item.routes : [{ip: item.ip, kind: 'other', mbps: 0, ms: null}];
              return (
                <div key={item.name || item.ip} className="join-found">
                  <div>
                    <strong>{item.name}</strong>
                    <span>{item.gpu || tr('no GPU')}{item.model ? ` · ${item.model}` : ''}</span>
                  </div>
                  <div className="join-routes">
                    {routes.map((route, i) => (
                      <button key={route.ip} type="button" className={i === 0 && !route.blocked ? 'btn btn-primary' : 'btn'} disabled={busy === 'join' || !item.accepting || route.blocked}
                        onClick={() => joinPool(route.ip)} title={route.blocked ? tr('{0} blocks this connection in its firewall', item.name) : item.accepting ? tr('Join through {0}', route.ip) : tr('That PC is not accepting join requests')}>
                        <i className={route.kind === 'wifi' ? 'fa-solid fa-wifi' : route.kind === 'ethernet' ? 'fa-solid fa-ethernet' : route.kind === 'tailscale' ? 'fa-solid fa-shield-halved' : 'fa-solid fa-network-wired'}></i>
                        {busy === 'join'
                          ? tr('Joining...')
                          : `${route.kind === 'wifi' ? tr('Wi-Fi') : route.kind === 'ethernet' ? tr('Ethernet') : route.kind === 'tailscale' ? tr('Tailscale') : tr('Network')}${route.mbps > 0 ? ` · ${route.mbps >= 1000 ? tr('{0} Gbps', route.mbps / 1000) : tr('{0} Mbps', route.mbps)}` : ''}${route.blocked ? tr(' · blocked') : route.ms !== null ? tr(' · {0} ms', route.ms) : ''}`}
                      </button>
                    ))}
                  </div>
                  {!item.accepting && <p className="card-note">{tr('{0} is not accepting join requests right now.', item.name)}</p>}
                  {routes.some(route => route.blocked) && <p className="card-note bad">{tr('The cable to {0} is connected but its firewall blocks it. On {1}, press Fix it in the Pool view.', item.name, item.name)}</p>}
                </div>
              );
            })}
          </>
        )}
        {helperProblem && <p className="card-note bad">{helperProblem}</p>}
      </div>

      <fieldset className="plain-fieldset" disabled={!!remote}>
      <div className="pool-grid">
        <div className="pool-col">
          <section className="card">
            <div className="card-head">
              <h3>{tr('Model')}</h3>
              <span className={pool.generation === 'ready'
                ? 'state-pill pill-ok'
                : pool.generation === 'loading'
                  ? 'state-pill pill-busy'
                  : pool.generation === 'failed'
                    ? 'state-pill pill-bad'
                    : 'state-pill pill-idle'}>
                {pool.generation === 'ready'
                  ? tr('running on {0}', pool.running)
                  : pool.generation === 'loading'
                    ? tr('loading')
                    : pool.generation === 'failed'
                      ? tr('failed')
                      : pool.loaded
                        ? tr('unloaded')
                        : tr('none')}
              </span>
            </div>
            {pool.loaded ? (
              <div className="model-box">
                <strong className="model-name">{model.name || model.path}</strong>
                <span className="model-path">{model.path}</span>
                <div className="stat-row">
                  <div><span>{tr('Size')}</span><strong>{gb(model.bytes)}</strong></div>
                  <div><span>{tr('Layers')}</span><strong>{model.layers}</strong></div>
                  <div><span>{tr('Experts')}</span><strong>{model.experts ? `${model.expertsUsed} / ${model.experts}` : tr('dense')}</strong></div>
                  <div><span>{tr('Context')}</span><strong>{tr('{0}K of {1}K', Math.round((pool.context || 0) / 1024), Math.round((model.contextLength || 0) / 1024))}</strong></div>
                </div>
                <span className="model-arch">{model.arch}</span>
                <div className="model-actions">
                  <button type="button" className="btn" disabled={busy !== '' || (pool.generation !== 'ready' && pool.generation !== 'loading' && pool.generation !== 'failed')} onClick={() => switchModel('unload')} title={pool.generation === 'loading' ? tr('Stop the load and free the memory') : tr('Free the memory')}>
                    <i className={busy === 'unload' ? 'fa-solid fa-circle-notch fa-spin' : 'fa-solid fa-eject'}></i>
                    {busy === 'unload' ? tr('Freeing memory...') : tr('Unload')}
                  </button>
                  <button type="button" className="btn btn-primary" disabled={busy !== '' || pool.generation === 'ready' || pool.generation === 'loading'} onClick={() => switchModel('load')}>
                    <i className={busy === 'load' ? 'fa-solid fa-circle-notch fa-spin' : 'fa-solid fa-play'}></i>
                    {tr('Load')}
                  </button>
                </div>
              </div>
            ) : (
              <p className="card-note">{tr('No model loaded. Pick one below.')}</p>
            )}
            {model.error && <div className="view-alert alert-bad">{model.error}</div>}

            <div className="model-pick">
              <div className="field">
                <span className="field-label">{tr('Models folder')}</span>
                <div className="field-inline">
                  <input
                    value={modelFolder}
                    placeholder={tr('modelos or F:\\path\\modelos')}
                    onChange={(e) => onFolderChange && onFolderChange(e.target.value)}
                  />
                  <button type="button" className="btn" disabled={scanning || !modelFolder.trim()} onClick={() => onScan && onScan()}>
                    <i className={scanning ? 'fa-solid fa-circle-notch fa-spin' : 'fa-solid fa-magnifying-glass'}></i>
                    {tr('Scan')}
                  </button>
                </div>
              </div>

              <div className="field">
                <span className="field-label">{tr('Model file')}</span>
                <div className="field-inline">
                  <select value={pickedModel} onChange={(e) => setPick(e.target.value)}>
                    {modelFiles.length === 0 && <option value="">{tr('No models found')}</option>}
                    {modelFiles.map(item => (
                      <option key={item.name} value={item.name}>
                        {item.name} {item.size ? `(${gb(item.size)})` : ''}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={loadingModel || !pickedModel || busy !== '' || pool.generation === 'ready' || pool.generation === 'loading'}
                    title={pool.generation === 'ready' || pool.generation === 'loading' ? tr('Unload the current model first') : ''}
                    onClick={async () => {
                      const found = modelFiles.find(item => item.name === pickedModel);

                      if(!onLoad){
                        return;
                      }

                      setNotice('');
                      const res = await onLoad(found?.path || '', pickedModel);

                      if(!res?.cargado || res?.success === false){
                        setNotice(res?.error || res?.message || tr('The model did not load'));
                      }
                    }}
                  >
                    <i className={loadingModel ? 'fa-solid fa-circle-notch fa-spin' : 'fa-solid fa-play'}></i>
                    {loadingModel ? tr('Loading') : tr('Load')}
                  </button>
                </div>
                {(pool.generation === 'ready' || pool.generation === 'loading') && (
                  <em className="field-help">{tr('Unload the current model to load another one, the memory is freed first.')}</em>
                )}
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <h3>{tr('Machines')}</h3>
              <span className="card-count">{machines.length}</span>
            </div>
            <div className="machine-list">
              {machines.map(node => {
                const stats = node.stats || {};
                const gpu = stats.gpu || {};
                const transferring = !node.self && node.transferTotal > 0 && node.transfer !== 'idle';
                const here = remote ? node.name === remote.viewer : node.self;

                return (
                  <div key={node.name} className={node.connected ? 'machine' : 'machine machine-off'}>
                    <div className="machine-top">
                      <i className={here ? 'fa-solid fa-desktop machine-icon' : 'fa-solid fa-server machine-icon'}></i>
                      <div className="machine-title">
                        <strong>{node.name}</strong>
                        <span>{here ? tr('this machine') : node.self ? tr('pool host') : `${node.ip}:${node.port}`} · {node.self ? tr('master') : tr('{0} ms', node.pingMs)}</span>
                      </div>
                      <span className={`state-pill ${!node.connected ? 'pill-bad' : node.self ? 'pill-busy' : 'pill-ok'}`}>
                        {!node.connected ? tr('offline') : node.self ? tr('master') : tr('connected')}
                      </span>
                      {!node.self && (
                        <button type="button" className="btn btn-ghost" title={tr('Disconnect this PC from the pool; it frees its GPU and RAM')} disabled={busy === 'remove'} onClick={() => runAction('remove', {name: node.name})}>
                          <i className="fa-solid fa-plug-circle-xmark"></i>
                          {tr('Disconnect')}
                        </button>
                      )}
                    </div>

                    <div className="machine-gpu">
                      <i className={kindIcons.gpu} style={{color: kindColors.gpu}}></i>
                      {gpu.present ? gpu.name : tr('No GPU')}
                    </div>

                    {gpu.present && (
                      <div className="mini-meter">
                        <span>{tr('VRAM')}</span>
                        <div className="meter"><div className="meter-fill" style={{width: `${gpu.vramTotal ? (1 - gpu.vramFree / gpu.vramTotal) * 100 : 0}%`, background: kindColors.gpu}}></div></div>
                        <em>{tr('{0} free / {1}', gb(gpu.vramFree), gb(gpu.vramTotal))}</em>
                      </div>
                    )}

                    <div className="mini-meter">
                      <span>{tr('RAM')}</span>
                      <div className="meter"><div className="meter-fill" style={{width: `${stats.ramTotal ? (1 - stats.ramFree / stats.ramTotal) * 100 : 0}%`, background: kindColors.ram}}></div></div>
                      <em>{tr('{0} free / {1}', gb(stats.ramFree), gb(stats.ramTotal))}</em>
                    </div>

                    <div className="mini-meter" title={tr(slowDrives[stats.drive] || 'Disk holding the model files')}>
                      <span className={slowDrives[stats.drive] ? 'drive-slow' : ''}>{driveLabels[stats.drive] || tr('Disk')}</span>
                      <div className="meter"><div className="meter-fill" style={{width: `${stats.diskTotal ? (1 - stats.diskFree / stats.diskTotal) * 100 : 0}%`, background: slowDrives[stats.drive] ? 'var(--danger)' : kindColors.ssd}}></div></div>
                      <em>{tr('{0} free / {1}', gb(stats.diskFree), gb(stats.diskTotal))}</em>
                    </div>

                    <div className="machine-cpu">
                      <span>{stats.cpu || tr('CPU')}</span>
                      <em>{stats.cores}C / {stats.threads}T · {Number(stats.cpuLoad || 0).toFixed(0)}%</em>
                    </div>

                    {transferring && (
                      <div className="mini-meter">
                        <span>{node.transfer}</span>
                        <div className="meter"><div className="meter-fill" style={{width: `${Math.min(100, node.transferDone / node.transferTotal * 100)}%`}}></div></div>
                        <em>{gb(node.transferDone)} / {gb(node.transferTotal)}</em>
                      </div>
                    )}

                    {!node.self && !transferring && node.transfer && node.transfer !== 'idle' && (
                      <div className="machine-transfer">{tr('weights {0}', node.transfer)}</div>
                    )}

                    {node.problem && <div className="machine-problem">{node.problem}</div>}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <h3>{tr('Join requests')}</h3>
              <span className="card-count">{requests.length}</span>
            </div>
            {requests.length === 0 && (
              <p className="card-note">
                {!net
                  ? tr('This Sneo build does not take join requests. Restart it after updating.')
                  : net.acceptRequests
                    ? tr('Nobody is waiting. On the other PC, start worker.exe and send a request to {0} port {1}.', addresses[0] || tr('this PC'), net.doorPort)
                    : tr('Join requests are off. Turn them on in Network below.')}
              </p>
            )}
            {requests.map(item => (
              <div key={item.id} className="join-row">
                <div className="join-info">
                  <div className="join-ask"><strong>{item.name}</strong> {tr('({0}) wants to join this pool', specsOf(item))}</div>
                  <span>{tr('from {0} · {1}{2}', item.ip, ago(item.age), item.cpu ? ` · ${item.cpu.trim()}` : '')}</span>
                  {item.problem && <div className="machine-problem">{item.problem}</div>}
                </div>
                <button type="button" className="btn btn-ok" disabled={answering} onClick={() => runAction('request/accept', {id: item.id}, `accept:${item.id}`)}>
                  {busy === `accept:${item.id}` ? tr('Joining...') : tr('Accept')}
                </button>
                <button type="button" className="btn" disabled={answering} onClick={() => runAction('request/decline', {id: item.id}, `decline:${item.id}`)}>{tr('Decline')}</button>
              </div>
            ))}
            {(pool.refused || []).map(item => (
              <div key={item.address} className="refused-row">
                <i className="fa-solid fa-ban"></i>
                <div>
                  <strong>{item.name || item.address}</strong>
                  <span>{item.address} · {item.why}</span>
                </div>
              </div>
            ))}
          </section>

          <section className="card">
            <div className="card-head">
              <h3>{tr('Network')}</h3>
              <div className="card-head-right">
                {networkDirty && <span className="state-pill pill-warn">{tr('unsaved')}</span>}
                {networkDirty && <button type="button" className="btn btn-ghost" onClick={revertNetwork}>{tr('Revert')}</button>}
                <button type="button" className="btn btn-primary" disabled={!networkDirty || busy === 'network' || !net} onClick={saveNetwork}>
                  {busy === 'network' ? tr('Saving...') : tr('Save')}
                </button>
              </div>
            </div>

            <div className="network-form">
              <div className="field">
                <span className="field-label">{tr('Accept join requests')}</span>
                <div className="segmented">
                  <button type="button" className={network.acceptRequests ? 'active' : ''} onClick={() => changeNetwork({acceptRequests: true})}>{tr('On')}</button>
                  <button type="button" className={network.acceptRequests ? '' : 'active'} onClick={() => changeNetwork({acceptRequests: false})}>{tr('Off')}</button>
                </div>
                <span className="field-help">
                  {!net
                    ? tr('Restart Sneo to use join requests.')
                    : net.doorOpen
                      ? tr('Other PCs can ask to join on port {0}. Changes apply as soon as you save.', net.doorPort)
                      : net.doorProblem || (net.acceptRequests ? tr('Opening the door...') : tr('Closed, nobody can ask to join.'))}
                </span>
              </div>

              <div className="field">
                <span className="field-label">{tr('Door port')}</span>
                <input
                  className="port-input"
                  value={network.doorPort}
                  disabled={!network.acceptRequests}
                  placeholder="9097"
                  onChange={(e) => changeNetwork({doorPort: e.target.value.replace(/\D/g, '')})}
                />
                <span className="field-help">{tr('The worker page asks for this port next to the address.')}</span>
              </div>

              <div className="field">
                <span className="field-label">{tr('Engine API')}{net ? tr(' (port {0})', net.apiPort) : ''}</span>
                <div className="segmented">
                  <button type="button" className={network.apiListen === 'local' ? 'active' : ''} onClick={() => changeNetwork({apiListen: 'local'})}>{tr('This PC only')}</button>
                  <button type="button" className={network.apiListen === 'network' ? 'active' : ''} onClick={() => changeNetwork({apiListen: 'network'})}>{tr('Local network')}</button>
                </div>
                <span className="field-help">
                  {net && network.apiListen !== net.apiBound
                    ? tr('Takes effect when Sneo restarts.')
                    : network.apiListen === 'network'
                      ? tr('Any device on your network can use the engine and change this pool.')
                      : tr('Only programs on this PC can reach the engine.')}
                </span>
              </div>

              <div className="field">
                <span className="field-label">{tr('Allow other devices to open SNEO')}</span>
                <div className="segmented">
                  <button type="button" disabled={!sneo} className={network.allowNetwork ? '' : 'active'} onClick={() => changeNetwork({allowNetwork: false})}>{tr('Off')}</button>
                  <button type="button" disabled={!sneo} className={network.allowNetwork ? 'active' : ''} onClick={() => changeNetwork({allowNetwork: true})}>{tr('On')}</button>
                </div>
                <span className="field-help">
                  {!sneo
                    ? tr('Restart the SNEO server to change this.')
                    : network.allowNetwork !== sneo.sneoNetwork
                      ? tr('Takes effect when the SNEO server restarts.')
                      : sneo.sneoNetwork
                        ? tr('Other devices open SNEO at http://{0}:{1} once they are paired in Phone', addresses[0] || tr('this-pc'), sneo.sneoPort)
                        : tr('Only this PC can open SNEO.')}
                </span>
              </div>

              <div className="field">
                <span className="field-label">{tr('Addresses of this PC')}</span>
                {addresses.length === 0 && <span className="field-help">{tr('No network address found.')}</span>}
                {addresses.map(ip => (
                  <div key={ip} className="copy-line">
                    <code>{ip}</code>
                    <button type="button" className="icon-btn" title={tr('Copy address')} onClick={() => copyText(ip, ip)}>
                      <i className={copied === ip ? 'fa-solid fa-check' : 'fa-regular fa-copy'}></i>
                    </button>
                  </div>
                ))}
                {addresses.length > 0 && <span className="field-help">{tr('Type one of these in the worker page on the other PC.')}</span>}
              </div>
            </div>
          </section>
        </div>

        <div className="pool-col pool-col-wide">
          <section className="card">
            <div className="card-head">
              <h3>{tr('Priority')}</h3>
              <div className="card-head-right">
                {dirty && <span className="state-pill pill-warn">{tr('unsaved')}</span>}
                {dirty && <button type="button" className="btn btn-ghost" onClick={revert}>{tr('Revert')}</button>}
                <button type="button" className="btn btn-primary" disabled={!dirty || busy === 'save'} onClick={savePool}>
                  {busy === 'save' ? tr('Saving...') : tr('Save')}
                </button>
              </div>
            </div>

            <p className="card-note">{tr('The model fills these slots from top to bottom. Drag to change the order.')}</p>

            <div className="context-pick">
              <span>{tr('Context')}</span>
              <div className="segmented">
                {contextSizes.map(item => (
                  <button
                    key={item.value}
                    type="button"
                    className={Number(context) === item.value ? 'active' : ''}
                    disabled={model.contextLength > 0 && item.value > model.contextLength}
                    onClick={() => {
                      setContext(item.value);
                      setDirty(true);
                    }}
                  >
                    {tr(item.label)}
                  </button>
                ))}
              </div>
            </div>

            {helperNode && layerCount > 4 && (
              <div className="context-pick front-pick">
                <span>{tr('Layers on {0}', helperNode.name)}</span>
                <div className="segmented">
                  <button
                    type="button"
                    className={!Number(frontLayers) ? 'active' : ''}
                    onClick={() => {
                      setFrontLayers(0);
                      setDirty(true);
                    }}
                  >
                    {tr('Auto')}
                  </button>
                  <button
                    type="button"
                    className={Number(frontLayers) > 0 ? 'active' : ''}
                    onClick={() => {
                      setFrontLayers(Number(pool.frontEnd) || Math.round(layerCount / 3));
                      setDirty(true);
                    }}
                  >
                    {tr('Pick')}
                  </button>
                </div>
                {Number(frontLayers) > 0 && (
                  <input
                    type="number"
                    min="2"
                    max={layerCount - 2}
                    value={frontLayers}
                    onChange={(e) => {
                      setFrontLayers(Math.min(layerCount - 2, Math.max(2, Math.round(Number(e.target.value) || 2))));
                      setDirty(true);
                    }}
                  />
                )}
                <em>
                  {pool.frontEnd
                    ? tr('now {0} of {1}', pool.frontEnd, layerCount)
                    : tr('used when the model loads')}
                </em>
              </div>
            )}

            <div className="slot-list">
              {slots.map((row, index) => {
                const live = liveSlots.find(item => item.node === row.node && item.kind === row.kind);
                const total = totalOf(row);
                const used = Number(live?.used || 0);
                const limit = Number(live?.limit || 0);
                const drive = row.kind === 'ssd' ? statsOf(row.node)?.drive : '';
                const hardware = statsOf(row.node);
                const device = row.kind === 'gpu'
                  ? hardware?.gpu?.present ? hardware.gpu.name : tr('no GPU')
                  : row.kind === 'ram'
                    ? hardware?.cpu ? tr('{0} computes these experts', hardware.cpu.trim()) : tr('system RAM')
                    : tr('the drive that holds the models');

                return (
                  <div
                    key={row.id}
                    className={`slot-row ${dragIndex === index ? 'slot-dragging' : ''}`}
                    draggable={grabbed === row.id}
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = 'move';
                      e.dataTransfer.setData('text/plain', row.id);
                      dragRef.current = index;
                      setDragIndex(index);
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      moveRow(index);
                    }}
                    onDragEnd={() => {
                      dragRef.current = -1;
                      setDragIndex(-1);
                      setGrabbed('');
                    }}
                  >
                    <span
                      className="slot-grip"
                      title={tr('Drag to reorder')}
                      onMouseDown={() => setGrabbed(row.id)}
                      onMouseUp={() => setGrabbed('')}
                    >
                      <i className="fa-solid fa-grip-vertical"></i>
                    </span>

                    <span className="slot-order">{index + 1}</span>

                    <div className="slot-who">
                      <span className="slot-kind" style={{color: kindColors[row.kind], borderColor: kindColors[row.kind]}}>
                        <i className={kindIcons[row.kind]}></i>
                        {driveLabels[drive] || row.kind}
                      </span>
                      <div className="slot-name">
                        <strong>{row.node}{row.node === (remote ? remote.viewer : pool.self) ? tr(' (this PC)') : ''}</strong>
                        <em>{device}</em>
                      </div>
                      {slowDrives[drive] && (
                        <span className="drive-slow" title={tr(slowDrives[drive])}>
                          <i className="fa-solid fa-triangle-exclamation"></i>
                          {tr('slow disk')}
                        </span>
                      )}
                    </div>

                    <div className="slot-amount">
                      <div className="slot-amount-row">
                        <input
                          type="number"
                          min="0"
                          step={row.unit === 'percent' ? '1' : '0.5'}
                          max={row.unit === 'percent' ? '100' : undefined}
                          value={row.value}
                          placeholder="0"
                          onChange={(e) => changeRow(index, {value: e.target.value})}
                        />
                        <div className="unit-pick">
                          {amountModes.map(mode => (
                            <button
                              key={mode.value}
                              type="button"
                              title={tr(mode.title)}
                              className={row.unit === mode.value ? 'active' : ''}
                              onClick={() => changeUnit(index, mode.value)}
                            >
                              {tr(mode.label)}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="slot-amount-hint">
                        {row.unit === 'keep'
                          ? tr('keep {0} GB free · model gets {1}', Number(row.value || 0), gb(dirty ? limitOf(row) : limit))
                          : row.unit === 'percent'
                            ? tr('{0}% of {1} · {2}', Number(row.value || 0), gb(total), gb(dirty ? limitOf(row) : limit))
                            : tr('up to {0} GB{1}', Number(row.value || 0), total && limitOf(row) < Number(row.value || 0) * gib ? tr(' · capped at {0}', gb(total)) : '')}
                      </div>
                    </div>

                    <div className="slot-live">
                      <div className="slot-free">
                        <span>{tr('{0} free', gb(freeOf(row)))}</span>
                        <em>{tr('of {0}', gb(total))}</em>
                      </div>
                      <div className="meter" title={tr('{0} used of {1} limit', gb(used), gb(limit))}>
                        <div className="meter-fill" style={{width: `${limit ? Math.min(100, used / limit * 100) : 0}%`, background: kindColors[row.kind]}}></div>
                      </div>
                      <div className="slot-used">{gb(used)} / {gb(limit)}</div>
                    </div>

                    <button type="button" className="icon-btn icon-danger" title={tr('Remove slot')} onClick={() => {
                      setSlots(prev => prev.filter((item, i) => i !== index));
                      setDirty(true);
                    }}>
                      <i className="fa-solid fa-trash"></i>
                    </button>
                  </div>
                );
              })}

              {slots.length === 0 && <p className="card-note">{tr('No slots. Add one below.')}</p>}
            </div>

            <div className="slot-add">
              <select value={addNode} onChange={(e) => setNewSlot(prev => ({...prev, node: e.target.value}))}>
                {members.map(item => (
                  <option key={item.name} value={item.name}>{item.name}</option>
                ))}
              </select>
              <select value={newSlot.kind} onChange={(e) => setNewSlot(prev => ({...prev, kind: e.target.value}))}>
                <option value="gpu" disabled={addStats && !addStats.gpu?.present}>{tr('GPU')}</option>
                <option value="ram">{tr('RAM')}</option>
                <option value="ssd">{tr('SSD')}</option>
              </select>
              <button type="button" className="btn" onClick={addRow} disabled={!addNode}>
                <i className="fa-solid fa-plus"></i>
                {tr('Add slot')}
              </button>
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <h3>{tr('Fill')}</h3>
              <span className="card-sub">{tr('{0} placed of {1} needed', gb(placed), gb(demand))}</span>
            </div>

            <div className="fill-bar">
              {liveSlots.filter(item => item.used > 0).map((item, i) => (
                <div
                  key={`${item.node}-${item.kind}-${i}`}
                  className="fill-seg"
                  style={{width: `${item.used / fillTotal * 100}%`, background: kindColors[item.kind]}}
                  title={`${item.node} · ${item.kind.toUpperCase()} · ${gb(item.used)}`}
                ></div>
              ))}
              {unplaced > 0 && (
                <div className="fill-seg fill-missing" style={{width: `${unplaced / fillTotal * 100}%`}} title={tr('Not placed · {0}', gb(unplaced))}></div>
              )}
            </div>

            <div className="fill-legend">
              {liveSlots.map((item, i) => (
                <div key={`${item.node}-${item.kind}-legend-${i}`} className={item.used > 0 ? 'legend-item' : 'legend-item legend-empty'}>
                  <span className="legend-swatch" style={{background: kindColors[item.kind]}}></span>
                  <span className="legend-name">{item.node} · {item.kind.toUpperCase()}</span>
                  <strong>{gb(item.used)}</strong>
                  <em>{demand ? `${(item.used / demand * 100).toFixed(0)}%` : ''}</em>
                </div>
              ))}
              {unplaced > 0 && (
                <div className="legend-item">
                  <span className="legend-swatch fill-missing"></span>
                  <span className="legend-name">{tr('Not placed')}</span>
                  <strong>{gb(unplaced)}</strong>
                  <em>{`${(unplaced / demand * 100).toFixed(0)}%`}</em>
                </div>
              )}
            </div>
            {dirty && <p className="card-note">{tr('Save the priority to see the new fill.')}</p>}
          </section>

          <section className="card">
            <div className="card-head">
              <h3>{tr('Split')}</h3>
              <span className="card-sub">{tr('{0} layers', (plan.layers || []).length)}</span>
            </div>

            {plan.problem && <div className="view-alert alert-bad"><i className="fa-solid fa-triangle-exclamation"></i>{plan.problem}</div>}

            <div className="split-list">
              {(plan.nodes || []).map(item => {
                const sum = Number(item.gpu || 0) + Number(item.ram || 0) + Number(item.ssd || 0);

                return (
                  <div key={item.node} className="split-row">
                    <div className="split-head">
                      <strong>{item.node}</strong>
                      <span>{tr('layers {0} – {1} ({2})', item.firstLayer, item.lastLayer, item.lastLayer - item.firstLayer + 1)}</span>
                      {item.holdsEdges && <span className="state-pill pill-idle">{tr('embeddings + head')}</span>}
                    </div>
                    <div className="fill-bar fill-bar-thin">
                      {['gpu', 'ram', 'ssd'].filter(kind => item[kind] > 0).map(kind => (
                        <div key={kind} className="fill-seg" style={{width: `${item[kind] / Math.max(sum, 1) * 100}%`, background: kindColors[kind]}} title={`${kind.toUpperCase()} · ${gb(item[kind])}`}></div>
                      ))}
                    </div>
                    <div className="split-bytes">
                      {['gpu', 'ram', 'ssd'].map(kind => (
                        <span key={kind}>
                          <i className={kindIcons[kind]} style={{color: kindColors[kind]}}></i>
                          {kind.toUpperCase()} <strong>{gb(item[kind])}</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            {(plan.layers || []).length > 0 && (
              <div className="layer-map">
                <div className="layer-map-cols">
                  {plan.layers.map(layer => {
                    const sum = layer.gpuExperts + layer.ramExperts + layer.ssdExperts;

                    return (
                      <div
                        key={layer.layer}
                        className="layer-col"
                        title={tr('Layer {0} · {1} · dense on {2}{3}', layer.layer, layer.node, layer.dense.toUpperCase(), sum ? tr(' · experts GPU {0} / RAM {1} / SSD {2}', layer.gpuExperts, layer.ramExperts, layer.ssdExperts) : '')}
                      >
                        {sum > 0 ? (
                          <>
                            <span style={{flexGrow: layer.ssdExperts, background: kindColors.ssd}}></span>
                            <span style={{flexGrow: layer.ramExperts, background: kindColors.ram}}></span>
                            <span style={{flexGrow: layer.gpuExperts, background: kindColors.gpu}}></span>
                          </>
                        ) : (
                          <span style={{flexGrow: 1, background: kindColors[layer.dense] || kindColors.ram}}></span>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="layer-map-axis">
                  <span>{tr('layer 0')}</span>
                  <span>{tr('expert placement per layer')}</span>
                  <span>{tr('layer {0}', plan.layers.length - 1)}</span>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
      </fieldset>
    </div>
  );
}

export default PoolView;
