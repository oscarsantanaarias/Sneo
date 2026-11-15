import React from 'react';
import { useState, useEffect } from 'react';
import { tr } from './i18n.js';

const gib = 1073741824;
const keep = 90;
const history = {};
const reads = {at: 0, bytes: 0, rate: 0};

const gb = (bytes) => `${(Number(bytes || 0) / gib).toFixed(1)} GB`;

const rate = (bytes) => {
  const value = Number(bytes || 0);

  return value >= gib
    ? `${(value / gib).toFixed(1)} GB/s`
    : `${(value / 1048576).toFixed(1)} MB/s`;
};

function Spark({values = [], max = 0, color = 'var(--accent)', format = (value) => value}) {
  const [hover, setHover] = useState(-1);
  const width = 100;

  if(values.length < 2){
    return <div className="spark spark-empty"><span>{tr('collecting...')}</span></div>;
  }

  const top = max || Math.max(...values, 1);
  const step = width / (values.length - 1);
  const yOf = (value) => 27 - Math.min(1, Math.max(0, value / top)) * 25;
  const points = values.map((value, i) => `${(i * step).toFixed(2)},${yOf(value).toFixed(2)}`).join(' ');
  const shown = hover >= 0 && hover < values.length ? hover : -1;
  const hoverX = shown * step;

  return (
    <div
      className="spark"
      onMouseMove={(e) => {
        const box = e.currentTarget.getBoundingClientRect();
        const index = Math.round((e.clientX - box.left) / box.width * (values.length - 1));
        setHover(Math.max(0, Math.min(values.length - 1, index)));
      }}
      onMouseLeave={() => setHover(-1)}
    >
      <svg viewBox={`0 0 ${width} 28`} preserveAspectRatio="none">
        <polygon points={`0,28 ${points} ${width},28`} fill={color} opacity="0.13" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        {shown >= 0 && <line x1={hoverX} x2={hoverX} y1="0" y2="28" stroke="var(--text-faint)" strokeWidth="1" vectorEffect="non-scaling-stroke" />}
      </svg>
      {shown >= 0 && (
        <>
          <span className="spark-dot" style={{left: `${hoverX}%`, top: `${yOf(values[shown]) / 28 * 100}%`, background: color}}></span>
          <span className={hoverX > 60 ? 'spark-tip spark-tip-left' : 'spark-tip'} style={{left: `${hoverX}%`}}>
            {format(values[shown])}
          </span>
        </>
      )}
    </div>
  );
}

function Tile({label, icon, value, sub, children, tone = ''}) {
  return (
    <div className={`tile ${tone}`}>
      <div className="tile-label">
        <i className={icon}></i>
        {label}
      </div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
      {children}
    </div>
  );
}

function EngineView({apiUrl}) {
  const [monitor, setMonitor] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    let timer = null;

    const tick = async () => {
      try {
        const req = await fetch(`${apiUrl}/api/engine/monitor`);
        const res = await req.json();

        if(!req.ok){
          throw new Error(res.message || res.error || tr('Monitor answered {0}', req.status));
        }

        if(!alive){
          return;
        }

        for(const node of res.nodes || []){
          const stats = node.stats || {};
          const gpu = stats.gpu || {};
          const list = history[node.name] || [];

          list.push({
            gpu: Number(gpu.load || 0),
            vram: Number(gpu.vramTotal || 0) - Number(gpu.vramFree || 0),
            heat: Number(gpu.temperature || 0),
            power: Number(gpu.power || 0),
            bus: Number(gpu.pcieTx || 0) + Number(gpu.pcieRx || 0),
            cpu: Number(stats.cpuLoad || 0),
            ram: Number(stats.ramTotal || 0) - Number(stats.ramFree || 0),
            disk: Number(stats.diskRead || 0) + Number(stats.diskWrite || 0),
            more: (node.moreGpus || []).map(extra => {
              const card = extra.stats || {};
              return {
                gpu: Number(card.load || 0),
                vram: Number(card.vramTotal || 0) - Number(card.vramFree || 0),
                heat: Number(card.temperature || 0),
                power: Number(card.power || 0),
                bus: Number(card.pcieTx || 0) + Number(card.pcieRx || 0)
              };
            })
          });

          history[node.name] = list.slice(-keep);
        }

        const now = Date.now();
        const engineRead = Number(res.disk?.engineRead || 0);
        reads.rate = reads.at && engineRead >= reads.bytes
          ? (engineRead - reads.bytes) / Math.max(0.2, (now - reads.at) / 1000)
          : 0;
        reads.at = now;
        reads.bytes = engineRead;
        const self = (history[(res.nodes || []).find(node => node.self)?.name] || []);
        if(self.length) self[self.length - 1].engineDisk = reads.rate;

        setMonitor(res);
        setError('');

      } catch(e) {
        if(alive){
          setError(e instanceof SyntaxError ? tr('The SNEO server has no /api/engine route yet. Restart it so engine.js gets loaded.') : e.message);
        }
      }

      if(alive){
        timer = setTimeout(tick, 1000);
      }
    };

    tick();

    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [apiUrl]);

  if(!monitor){
    return (
      <div className="view-page">
        <div className="view-head">
          <div>
            <h1>{tr('Engine')}</h1>
            <span>{tr('Live hardware and request numbers')}</span>
          </div>
        </div>
        <div className="view-empty">
          <i className={error ? 'fa-solid fa-plug-circle-xmark' : 'fa-solid fa-circle-notch fa-spin'}></i>
          <p>{error || tr('Reading the engine...')}</p>
        </div>
      </div>
    );
  }

  const requests = monitor.requests || [];
  const last = requests[0] || null;
  const speeds = [...requests].reverse().map(item => Number(item.outputSpeed || 0));

  return (
    <div className="view-page">
      <div className="view-head">
        <div>
          <h1>{tr('Engine')}</h1>
          <span>{monitor.model || tr('no model')} {tr('· context {0}K · {1}% of experts in VRAM', Math.round((monitor.context || 0) / 1024), (Number(monitor.expertsInVram || 0) * 100).toFixed(0))}</span>
        </div>

        <div className="view-head-actions">
          <span className={`state-pill pill-${monitor.state === 'busy' ? 'busy' : monitor.state === 'idle' ? 'ok' : 'idle'}`}>
            <span className={monitor.state === 'busy' ? 'state-dot dot-live' : 'state-dot'}></span>
            {monitor.state}
            {monitor.active > 0 ? ` · ${monitor.active}` : ''}
          </span>
        </div>
      </div>

      {error && <div className="view-alert alert-bad"><i className="fa-solid fa-plug-circle-xmark"></i>{error}</div>}

      {(monitor.nodes || []).map(node => {
        const stats = node.stats || {};
        const gpu = stats.gpu || {};
        const samples = history[node.name] || [];
        const series = (key) => samples.map(item => item[key]);
        const shared = Number(monitor.disk?.firstLayer || 0) > 0;
        const part = shared && Number(monitor.disk?.share || 0) > 0 ? Number(monitor.disk.share) : Number(monitor.disk?.current || 0);
        const vramUsed = Number(gpu.vramTotal || 0) - Number(gpu.vramFree || 0);
        const heldGpu = Number(stats.heldGpu || 0);
        const heldRam = Number(stats.heldRam || 0);
        const ramUsed = Number(stats.ramTotal || 0) - Number(stats.ramFree || 0);

        return (
          <section key={node.name} className={node.connected ? 'card machine-card' : 'card machine-card machine-off'}>
            <div className="card-head">
              <div className="machine-card-title">
                <i className={node.self ? 'fa-solid fa-desktop' : 'fa-solid fa-server'}></i>
                <h3>{node.name}</h3>
                <span>{node.self ? tr('master') : tr('{0} ms', node.pingMs)}</span>
              </div>
              <div className="card-head-right">
                {stats.loadTotal > 0 && stats.loadDone < stats.loadTotal && (
                  <span className="card-sub">{tr('loading {0} / {1}', gb(stats.loadDone), gb(stats.loadTotal))}</span>
                )}
                <span className={node.connected ? 'state-pill pill-ok' : 'state-pill pill-bad'}>{node.connected ? tr('online') : tr('offline')}</span>
              </div>
            </div>

            <div className="tile-grid">
              {node.self && (
                <Tile
                  label={tr('Throughput')}
                  icon="fa-solid fa-bolt"
                  value={last ? `${Number(last.outputSpeed || 0).toFixed(1)} tok/s` : '--'}
                  sub={last ? tr('prompt {0} tok/s', Number(last.promptSpeed || 0).toFixed(0)) : tr('no requests yet')}
                >
                  <Spark values={speeds.length > 1 ? speeds : []} format={(value) => `${value.toFixed(1)} tok/s`} />
                </Tile>
              )}

              {gpu.present && (
                <>
                  <Tile label={tr('GPU')} icon="fa-solid fa-microchip" value={`${gpu.load}%`} sub={`${gpu.name} · ${gpu.clock} MHz`}>
                    <Spark values={series('gpu')} max={100} format={(value) => `${value}%`} />
                  </Tile>

                  <Tile label={tr('VRAM')} icon="fa-solid fa-layer-group" value={`${gb(vramUsed)} / ${gb(gpu.vramTotal)}`} sub={tr('Sneo holds {0}', gb(heldGpu))}>
                    <div className="held-bar" title={tr('Sneo {0} · other {1}', gb(heldGpu), gb(Math.max(0, vramUsed - heldGpu)))}>
                      <span className="held-own" style={{width: `${gpu.vramTotal ? Math.min(100, heldGpu / gpu.vramTotal * 100) : 0}%`}}></span>
                      <span className="held-other" style={{width: `${gpu.vramTotal ? Math.max(0, (vramUsed - heldGpu) / gpu.vramTotal * 100) : 0}%`}}></span>
                    </div>
                    <Spark values={series('vram')} max={gpu.vramTotal} format={gb} />
                  </Tile>

                  <Tile label={tr('Heat')} icon="fa-solid fa-temperature-half" value={`${gpu.temperature} °C`} sub={`fan ${gpu.fan}%`} tone={gpu.temperature >= 83 ? 'tile-hot' : ''}>
                    <Spark values={series('heat')} max={100} format={(value) => `${value} °C`} />
                  </Tile>

                  <Tile label={tr('Power')} icon="fa-solid fa-plug" value={`${Number(gpu.power || 0).toFixed(0)} W`} sub={tr('of {0} W limit · {1} Wh used', Number(gpu.powerLimit || 0).toFixed(0), Number(gpu.energy || 0).toFixed(1))}>
                    <Spark values={series('power')} max={gpu.powerLimit} format={(value) => `${value.toFixed(0)} W`} />
                  </Tile>

                  <Tile label={tr('Bus')} icon="fa-solid fa-right-left" value={`PCIe ${gpu.pcieGen}.0 x${gpu.pcieWidth}`} sub={`tx ${rate(gpu.pcieTx)} · rx ${rate(gpu.pcieRx)}`}>
                    <Spark values={series('bus')} format={rate} />
                  </Tile>
                </>
              )}

              <Tile label={tr('CPU')} icon="fa-solid fa-gears" value={`${Number(stats.cpuLoad || 0).toFixed(0)}%`} sub={`${stats.cpu} · ${stats.cores}C / ${stats.threads}T`}>
                <Spark values={series('cpu')} max={100} format={(value) => `${value.toFixed(0)}%`} />
              </Tile>

              <Tile label={tr('RAM')} icon="fa-solid fa-memory" value={tr('{0} free', gb(stats.ramFree))} sub={tr('of {0} · Sneo holds {1}', gb(stats.ramTotal), gb(heldRam))}>
                <div className="held-bar" title={tr('Sneo {0} · other {1}', gb(heldRam), gb(Math.max(0, ramUsed - heldRam)))}>
                  <span className="held-own" style={{width: `${stats.ramTotal ? Math.min(100, heldRam / stats.ramTotal * 100) : 0}%`}}></span>
                  <span className="held-other" style={{width: `${stats.ramTotal ? Math.max(0, (ramUsed - heldRam) / stats.ramTotal * 100) : 0}%`}}></span>
                </div>
                <Spark values={series('ram')} max={stats.ramTotal} format={gb} />
              </Tile>

              {node.self && monitor.disk ? (
                <Tile
                  label={tr('Model disk')}
                  icon="fa-solid fa-hard-drive"
                  value={(
                    <span className="disk-value">
                      {rate(stats.diskRead)}
                      <em className={Number(stats.diskRead || 0) > 5242880 ? 'disk-flag reading' : 'disk-flag'}>
                        {Number(stats.diskRead || 0) > 5242880 ? tr('reading') : tr('quiet')}
                      </em>
                    </span>
                  )}
                  sub={Number(monitor.disk.current || 0) > 0
                    ? tr('{0} model on {1} {2}', gb(monitor.disk.current), String(monitor.disk.kind || '').toUpperCase(), monitor.disk.drive)
                    : tr('No model loaded · {0} {1}', String(monitor.disk.kind || '').toUpperCase(), monitor.disk.drive)}
                >
                  <div className="held-bar" title={tr('in VRAM and RAM {0} · read from the disk {1}', gb(Math.max(0, part - monitor.disk.fromDisk)), gb(monitor.disk.fromDisk))}>
                    <span className="held-own" style={{width: `${part ? Math.min(100, Math.max(0, part - monitor.disk.fromDisk) / part * 100) : 0}%`}}></span>
                    <span className="held-other" style={{width: `${part ? Math.min(100, monitor.disk.fromDisk / part * 100) : 0}%`}}></span>
                  </div>
                  <div className="disk-lines">
                    <span>
                      {monitor.state === 'empty' || monitor.generation === 'off'
                        ? tr('No model in memory, the disk is not used.')
                        : !monitor.disk.ready
                          ? tr('The model is loading into VRAM and RAM.')
                          : Number(monitor.disk.fromDisk || 0) > 0
                          ? tr('{0}{1} in VRAM and RAM · {2} read from the disk when needed', shared ? tr('This PC runs layers {0} and up ({1}): ', monitor.disk.firstLayer, gb(part)) : '', gb(Math.max(0, part - monitor.disk.fromDisk)), gb(monitor.disk.fromDisk))
                          : shared
                            ? tr('This PC runs layers {0} and up ({1}), all of it in VRAM and RAM.', monitor.disk.firstLayer, gb(part))
                            : tr('The whole model is in VRAM and RAM, the disk is not used.')}
                    </span>
                  </div>
                  <Spark values={series('disk')} format={rate} />
                </Tile>
              ) : (
                <Tile label={tr('Disk')} icon="fa-solid fa-hard-drive" value={rate(Number(stats.diskRead || 0) + Number(stats.diskWrite || 0))} sub={`read ${rate(stats.diskRead)} · write ${rate(stats.diskWrite)} · ${gb(stats.diskFree)} free`}>
                  <Spark values={series('disk')} format={rate} />
                </Tile>
              )}
            </div>

            {(node.moreGpus || []).map((extra, index) => {
              const card = extra.stats || {};
              const used = Number(card.vramTotal || 0) - Number(card.vramFree || 0);
              const held = Number(extra.held || 0);
              const trail = (key) => samples.map(item => (item.more && item.more[index] ? item.more[index][key] : 0));
              return (
                <div key={`${card.name}-${index}`} className="extra-gpu">
                  <div className="extra-gpu-head">
                    <i className="fa-solid fa-microchip"></i>
                    <span>{card.name}</span>
                    <em>{extra.role}{Number(extra.experts || 0) > 0 ? tr(' · {0} experts per layer', extra.experts) : ''}</em>
                  </div>
                  <div className="tile-grid">
                    <Tile label={tr('GPU')} icon="fa-solid fa-microchip" value={`${card.load}%`} sub={`${card.name} · ${card.clock} MHz`}>
                      <Spark values={trail('gpu')} max={100} format={(value) => `${value}%`} />
                    </Tile>
                    <Tile label={tr('VRAM')} icon="fa-solid fa-layer-group" value={`${gb(used)} / ${gb(card.vramTotal)}`} sub={tr('Sneo holds {0}', gb(held))}>
                      <div className="held-bar" title={tr('Sneo {0} · other {1}', gb(held), gb(Math.max(0, used - held)))}>
                        <span className="held-own" style={{width: `${card.vramTotal ? Math.min(100, held / card.vramTotal * 100) : 0}%`}}></span>
                        <span className="held-other" style={{width: `${card.vramTotal ? Math.max(0, (used - held) / card.vramTotal * 100) : 0}%`}}></span>
                      </div>
                      <Spark values={trail('vram')} max={card.vramTotal} format={gb} />
                    </Tile>
                    <Tile label={tr('Heat')} icon="fa-solid fa-temperature-half" value={`${card.temperature} °C`} sub={`fan ${card.fan}%`} tone={card.temperature >= 83 ? 'tile-hot' : ''}>
                      <Spark values={trail('heat')} max={100} format={(value) => `${value} °C`} />
                    </Tile>
                    <Tile label={tr('Power')} icon="fa-solid fa-plug" value={`${Number(card.power || 0).toFixed(0)} W`} sub={tr('of {0} W limit · {1} Wh used', Number(card.powerLimit || 0).toFixed(0), Number(card.energy || 0).toFixed(1))}>
                      <Spark values={trail('power')} max={card.powerLimit} format={(value) => `${value.toFixed(0)} W`} />
                    </Tile>
                    <Tile label={tr('Bus')} icon="fa-solid fa-right-left" value={`PCIe ${card.pcieGen}.0 x${card.pcieWidth}`} sub={`tx ${rate(card.pcieTx)} · rx ${rate(card.pcieRx)}`}>
                      <Spark values={trail('bus')} format={rate} />
                    </Tile>
                  </div>
                </div>
              );
            })}

            {!gpu.present && <p className="card-note">{tr('No GPU on this machine.')}</p>}
          </section>
        );
      })}

      <section className="card">
        <div className="card-head">
          <h3>{tr('Last requests')}</h3>
          <span className="card-count">{requests.length}</span>
        </div>

        {requests.length === 0 ? (
          <p className="card-note">{tr('Nothing asked yet.')}</p>
        ) : (
          <div className="table-wrap">
            <table className="req-table">
              <thead>
                <tr>
                  <th>{tr('Time')}</th>
                  <th>{tr('Status')}</th>
                  <th>{tr('Prompt')}</th>
                  <th>{tr('Cached')}</th>
                  <th>{tr('Output')}</th>
                  <th>{tr('tok/s')}</th>
                  <th>{tr('Prompt tok/s')}</th>
                  <th>{tr('Duration')}</th>
                </tr>
              </thead>
              <tbody>
                {requests.map((item, i) => (
                  <tr key={`${item.when}-${i}`}>
                    <td>{new Date(item.when).toLocaleTimeString()}</td>
                    <td><span className={item.status >= 200 && item.status < 300 ? 'code-ok' : 'code-bad'}>{item.status}</span></td>
                    <td>{Number(item.prompt || 0).toLocaleString()}</td>
                    <td>{Number(item.cached || 0).toLocaleString()}</td>
                    <td>{Number(item.output || 0).toLocaleString()}</td>
                    <td>{Number(item.outputSpeed || 0).toFixed(1)}</td>
                    <td>{Number(item.promptSpeed || 0).toFixed(0)}</td>
                    <td>{(Number(item.durationMs || 0) / 1000).toFixed(1)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default EngineView;
