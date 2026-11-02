import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const PAGES = [
  { id: 'overview', label: 'Overview', mark: '◫' },
  { id: 'anomalies', label: 'Anomalies', mark: '⌁' },
  { id: 'cameras', label: 'Cameras', mark: '◎' },
];

function formatDate(value) {
  return value ? new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' }) : '—';
}

function PageIcon({ id }) {
  const paths = {
    overview: <><rect x="3" y="3" width="5" height="5" rx="1"/><rect x="12" y="3" width="5" height="5" rx="1"/><rect x="3" y="12" width="5" height="5" rx="1"/><rect x="12" y="12" width="5" height="5" rx="1"/></>,
    anomalies: <><path d="M10 2.7 18 17H2L10 2.7Z"/><path d="M10 7v4.5M10 14.3v.1"/></>,
    cameras: <><rect x="2.5" y="5" width="15" height="11" rx="2"/><path d="m7 5 1-2h4l1 2"/><circle cx="10" cy="10.5" r="3"/></>,
  };
  return <svg viewBox="0 0 20 20" aria-hidden="true">{paths[id]}</svg>;
}

function App() {
  const [page, setPage] = useState(() => location.hash.slice(1) || 'overview');
  const [theme, setTheme] = useState(() => localStorage.getItem('edgewatch-theme') || 'light');
  const [cameras, setCameras] = useState([]);
  const [events, setEvents] = useState([]);
  const [form, setForm] = useState({ id: '', name: '', device_id: '' });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);
  const [cameraFilter, setCameraFilter] = useState('all');
  const [anomalyFilter, setAnomalyFilter] = useState('all');

  async function refresh() {
    try {
      const [cameraResponse, eventResponse] = await Promise.all([
        fetch(`${API}/api/cameras`), fetch(`${API}/api/events`),
      ]);
      if (!cameraResponse.ok || !eventResponse.ok) throw new Error('The API returned an error.');
      setCameras(await cameraResponse.json());
      setEvents(await eventResponse.json());
      setError('');
    } catch {
      setError('Cannot reach FastAPI. Check that the backend is running and the API address is correct.');
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);

  function navigate(id) {
    setPage(id);
    location.hash = id;
  }

  function toggleTheme() {
    setTheme(current => {
      const next = current === 'light' ? 'dark' : 'light';
      localStorage.setItem('edgewatch-theme', next);
      return next;
    });
  }

  async function register(event) {
    event.preventDefault();
    setMessage('');
    setError('');
    try {
      const response = await fetch(`${API}/api/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Camera registration failed.');
      setMessage(`Camera registered. Set CAMERA_ID=${body.id} in that Pi's edge-ai-camera/.env.`);
      setForm({ id: '', name: '', device_id: '' });
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const taggedEvents = useMemo(() => events.filter(event => event.anomaly?.label), [events]);
  const anomalies = useMemo(() => taggedEvents.filter(event => event.anomaly.label === 'ANOMALY'), [taggedEvents]);
  const visibleTagged = useMemo(() => taggedEvents.filter(event =>
    (anomalyFilter === 'all' || event.anomaly.label === anomalyFilter)
      && (cameraFilter === 'all' || event.camera_id === cameraFilter)), [taggedEvents, anomalyFilter, cameraFilter]);
  return <div className={`app-shell theme-${theme}`}>
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={() => navigate('overview')}>
        <span className="brand-icon">E</span><span>Edgewatch<small>CAMERAS &amp; EVENTS</small></span>
      </a>
      <div className="nav-label">WORKSPACE</div>
      <nav>{PAGES.map(item => <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`}
        onClick={() => navigate(item.id)}><span className="nav-mark"><PageIcon id={item.id}/></span>{item.label}
        {item.id === 'anomalies' && anomalies.length > 0 && <span className="nav-count">{anomalies.length}</span>}
      </button>)}</nav>
      <div className="sidebar-bottom"><span className="live-dot"/> {error ? 'Backend not connected' : 'Backend connected'} <small>Checks for new events every 5 sec</small>
        <button className="theme-toggle" onClick={toggleTheme}><span>{theme === 'light' ? '◐' : '☼'}</span>{theme === 'light' ? 'Dark theme' : 'Light theme'}</button>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <div className="breadcrumb">Edgewatch <span>/</span> {PAGES.find(item => item.id === page)?.label || 'Overview'}</div>
        <div className="top-actions"><span className="connection"><i className={error ? 'offline-dot' : ''}/>{error ? 'Not connected' : 'Connected'}</span>
          <button className="refresh-button" onClick={refresh} aria-label="Refresh data" title="Refresh">
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16.4 8A6.5 6.5 0 0 0 5.2 5.4L3.5 7.2"/><path d="M3.5 3.8v3.4h3.4M3.6 12a6.5 6.5 0 0 0 11.2 2.6l1.7-1.8"/><path d="M16.5 16.2v-3.4h-3.4"/></svg>
          </button></div>
      </header>
      {error && <div className="alert error-alert">{error}</div>}

      {page === 'overview' && <>
        <section className="page-heading"><div><div className="eyebrow">CAMERA NETWORK</div><h1>Overview</h1><p>A quick look at your cameras and the latest events they sent.</p></div>
          <button className="button primary" onClick={() => navigate('cameras')}>Add a camera <span>＋</span></button></section>
        <section className="stat-grid">
          <StatCard label="Cameras" value={cameras.length} note="Registered on this system" />
          <StatCard label="Online" value={cameras.filter(camera => camera.status === 'online').length} note="Seen in the last 45 seconds" />
          <StatCard label="Flagged by model" value={anomalies.length} note="Events tagged as unusual" alert={anomalies.length > 0} />
          <StatCard label="Recent events" value={events.length} note="Most recent 100 received" />
        </section>
        <section className="section-card">
          <div className="section-heading"><div><h2>Recent events</h2><p>Events sent by your cameras</p></div>
            <button className="text-button" onClick={() => navigate('anomalies')}>See model tags <span>→</span></button></div>
          <EventTable events={events.slice(0, 8)} onSelect={setSelected} empty="No events received yet. Events appear after a motion event closes on a camera." />
        </section>
        <section className="section-card camera-summary">
          <div className="section-heading"><div><h2>Cameras</h2><p>Last contact from each device</p></div>
            <button className="text-button" onClick={() => navigate('cameras')}>View all cameras <span>→</span></button></div>
          {cameras.length ? <div className="camera-grid">{cameras.slice(0, 4).map(camera => <CameraCard key={camera.id} camera={camera}/>)}</div> : <Empty text="No cameras registered yet." action="Register your first camera" onClick={() => navigate('cameras')} />}
        </section>
      </>}

      {page === 'anomalies' && <>
        <section className="page-heading"><div><div className="eyebrow">EVENT REVIEW</div><h1>Model flagged</h1><p>Events the camera model marked as unusual. Open a recording to review it.</p></div>
          <div className="anomaly-total"><strong>{anomalies.length}</strong><span>flagged events</span></div></section>
        <section className="filter-bar">
          <label>Model tag<select value={anomalyFilter} onChange={event => setAnomalyFilter(event.target.value)}><option value="all">All model tags</option><option value="ANOMALY">Anomaly</option><option value="NORMAL">Normal</option></select></label>
          <label>Camera<select value={cameraFilter} onChange={event => setCameraFilter(event.target.value)}><option value="all">All cameras</option>{cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.name}</option>)}</select></label>
          <span className="filter-result">{visibleTagged.length} tagged event{visibleTagged.length === 1 ? '' : 's'}</span>
        </section>
        <section className="anomaly-list">
          {visibleTagged.length ? visibleTagged.map(event => <AnomalyCard key={event.event_id} event={event} onSelect={setSelected}/>)
            : <Empty text={taggedEvents.length ? 'No tagged events match these filters.' : 'No model-tagged events yet. Enable INFERENCE_ENABLED on a Pi to attach model results to uploaded events.'} action={taggedEvents.length ? 'Clear filters' : null} onClick={() => { setCameraFilter('all'); setAnomalyFilter('all'); }} />}
        </section>
        <div className="quiet-note">A model tag is the on-device model output. It is a review signal, not a confirmed incident.</div>
      </>}

      {page === 'cameras' && <>
        <section className="page-heading"><div><div className="eyebrow">EDGE DEVICES</div><h1>Cameras</h1><p>Register Raspberry Pis and see which devices are sending events.</p></div></section>
        {message && <div className="alert success-alert">{message}</div>}
        <section className="camera-layout">
          <article className="section-card register-card"><div className="card-kicker">ADD DEVICE</div><h2>Register a camera</h2><p>Add each camera once. Copy its ID into the device's configuration.</p>
            <form onSubmit={register}>
              <label>Camera ID <span>Use letters, numbers, hyphens or underscores</span><input required pattern="[A-Za-z0-9_-]{1,80}" placeholder="front-door" value={form.id} onChange={e=>setForm({...form,id:e.target.value})}/></label>
              <label>Name <span>A place people will recognize</span><input required maxLength="160" placeholder="Front door" value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></label>
              <label>Device name <span>Optional Pi hostname or asset tag</span><input maxLength="160" placeholder="pi-front-door" value={form.device_id} onChange={e=>setForm({...form,device_id:e.target.value})}/></label>
              <button className="button primary submit-button">Save camera <span>→</span></button>
            </form>
          </article>
          <article className="section-card setup-card"><div className="card-kicker">ON THE PI</div><h2>Point it to this laptop</h2><p>Put these settings in the Pi's <code>edge-ai-camera/.env</code> file:</p>
            <div className="code-block"><span>CAMERA_ID</span><b>{form.id || 'pi-entrance-01'}</b><span>BACKEND_URL</span><b>http://&lt;laptop-lan-ip&gt;:8000</b><span>SHOW_PREVIEW</span><b>true</b></div>
            <div className="setup-foot"><span className="step-number">1</span><span>Start the dashboard backend on your laptop.</span></div>
            <div className="setup-foot"><span className="step-number">2</span><span>On the Pi, run <code>python -m edge.main</code>.</span></div>
          </article>
        </section>
        <section className="section-card registered-section"><div className="section-heading"><div><h2>Registered cameras</h2><p>{cameras.length} device{cameras.length === 1 ? '' : 's'} configured</p></div></div>
          {cameras.length ? <div className="camera-grid">{cameras.map(camera => <CameraCard key={camera.id} camera={camera}/>)}</div> : <Empty text="Your registered devices will appear here."/>}
        </section>
      </>}
    </main>

    {selected && <EventModal event={selected} onClose={() => setSelected(null)}/>}
  </div>;
}

function StatCard({ label, value, note, icon, good, alert }) {
  return <article className="stat-card"><div className={`stat-icon ${good ? 'good' : ''} ${alert ? 'alert' : ''}`}>{icon}</div><span className="stat-label">{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

function CameraCard({ camera }) {
  return <article className="camera-card"><div className="camera-card-top"><span className="camera-icon">◎</span><span className={`status-pill ${camera.status === 'online' ? 'online' : ''}`}><i/>{camera.status || 'offline'}</span></div>
    <h3>{camera.name}</h3><div className="camera-id">{camera.id}</div><div className="camera-meta"><span>DEVICE</span><b>{camera.device_id || 'Not provided'}</b></div>
    <div className="camera-meta"><span>LAST SEEN</span><b>{camera.last_seen ? formatDate(camera.last_seen) : 'No heartbeat yet'}</b></div></article>;
}

function EventTable({ events, onSelect, empty }) {
  if (!events.length) return <Empty text={empty}/>;
  return <div className="table-wrap"><table><thead><tr><th>CAMERA</th><th>TIME</th><th>DURATION</th><th>MODEL TAG</th><th>OBJECTS</th><th/></tr></thead>
    <tbody>{events.map(event => <tr key={event.event_id}><td><b>{event.camera_name || event.camera_id}</b><small className="table-sub">{event.camera_id}</small></td>
      <td>{formatDate(event.started_at)}</td><td>{Number(event.duration || 0).toFixed(1)} sec</td><td><ModelTag anomaly={event.anomaly}/></td>
      <td>{[...new Set((event.detections || []).map(d => d.class_name))].slice(0, 3).join(', ') || '—'}</td>
      <td><button className="row-action" onClick={() => onSelect(event)}>Details</button></td></tr>)}</tbody></table></div>;
}

function ModelTag({ anomaly, large = false }) {
  if (!anomaly?.label) return <span className="no-tag">No model tag</span>;
  const flagged = anomaly.label === 'ANOMALY';
  return <span className={`model-tag ${flagged ? 'flagged' : 'normal'} ${large ? 'large' : ''}`}><i/>{anomaly.label}<b>{Number(anomaly.anomaly_probability).toFixed(2)}</b></span>;
}

function AnomalyCard({ event, onSelect }) {
  const score = Number(event.anomaly.anomaly_probability);
  const detections = [...new Set((event.detections || []).map(item => item.class_name))];
  return <article className="anomaly-card"><div className="anomaly-accent"/><div className="anomaly-card-main"><div className="anomaly-card-head"><div><ModelTag anomaly={event.anomaly} large/><span className="camera-name">{event.camera_name || event.camera_id}</span></div>
      <div className="anomaly-time">{formatDate(event.started_at)}</div></div>
    <div className="score-line"><span>Anomaly score</span><b>{score.toFixed(3)}</b><div className="score-track"><i style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }}/></div><small>threshold {Number(event.anomaly.threshold ?? 0.5).toFixed(2)}</small></div>
    <div className="anomaly-card-foot"><span>{Number(event.duration || 0).toFixed(1)} sec · Camera {event.camera_id}</span><span>{detections.length ? `Objects: ${detections.join(', ')}` : 'No objects detected'}</span>
      <button className="button" onClick={() => onSelect(event)}>Review event <span>→</span></button></div></div>
    {event.video_available && <a className="video-link" href={`${API}/api/events/${event.event_id}/video`} target="_blank" rel="noreferrer" aria-label="Open event video">▶</a>}
  </article>;
}

function Empty({ text, action, onClick }) {
  return <div className="empty-state"><span className="empty-mark">⌁</span><p>{text}</p>{action && <button className="button" onClick={onClick}>{action}</button>}</div>;
}

function EventModal({ event, onClose }) {
  return <div className="modal" onClick={onClose}><article className="event-modal" onClick={e=>e.stopPropagation()}><button className="modal-close" onClick={onClose}>×</button>
    <div className="eyebrow">EVENT REVIEW</div><h2>{event.camera_name || event.camera_id}</h2><p className="modal-subtitle">{formatDate(event.started_at)} · {Number(event.duration || 0).toFixed(1)} seconds</p>
    <div className="modal-tag-row"><span>ON-DEVICE MODEL TAG</span><ModelTag anomaly={event.anomaly} large/></div>
    {event.anomaly && <div className="modal-score">Anomaly score <b>{Number(event.anomaly.anomaly_probability).toFixed(3)}</b><span>Threshold {Number(event.anomaly.threshold ?? 0.5).toFixed(2)}</span></div>}
    {event.video_available ? <video className="event-video" controls preload="metadata" src={`${API}/api/events/${event.event_id}/video`}/> : <div className="video-unavailable">No video uploaded with this event.</div>}
    <div className="modal-section"><h3>Detected objects</h3>{event.detections?.length ? event.detections.map((detection, index) => <div className="detection-row" key={`${detection.class_name}-${index}`}><span>{detection.class_name}</span><b>{(Number(detection.confidence) * 100).toFixed(0)}%</b><small>{formatDate(detection.timestamp)}</small></div>) : <p>No object detections attached.</p>}</div>
    <div className="modal-event-id">Event ID <code>{event.event_id}</code></div>
  </article></div>;
}

createRoot(document.getElementById('root')).render(<App/>);
