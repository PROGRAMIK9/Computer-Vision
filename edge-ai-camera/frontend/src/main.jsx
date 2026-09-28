import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [cameras, setCameras] = useState([]);
  const [events, setEvents] = useState([]);
  const [form, setForm] = useState({ id: '', name: '', device_id: '' });
  const [message, setMessage] = useState('');
  const [selected, setSelected] = useState(null);

  async function refresh() {
    try {
      const [c, e] = await Promise.all([fetch(`${API}/api/cameras`), fetch(`${API}/api/events`)]);
      if (c.ok) setCameras(await c.json());
      if (e.ok) setEvents(await e.json());
    } catch { setMessage('Cannot reach FastAPI. Is the backend running?'); }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 4000); return () => clearInterval(timer); }, []);

  async function register(event) {
    event.preventDefault(); setMessage('Registering camera…');
    try {
      const response = await fetch(`${API}/api/cameras`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Registration failed');
      setMessage(`Registered ${body.name} (${body.id}). Set CAMERA_ID=${body.id} in edge-ai-camera/.env.`);
      setForm({ id: '', name: '', device_id: '' }); refresh();
    } catch (error) { setMessage(error.message); }
  }

  return <main>
    <header><div><span className="eyebrow">LOCAL EDGE MONITORING</span><h1>Camera dashboard</h1></div><button onClick={refresh}>Refresh</button></header>
    <section className="grid">
      <article><h2>Register a phone camera</h2>
        <p>Registration creates the camera record. The phone sends its local stream directly to the laptop edge processor.</p>
        <form onSubmit={register}>
          <label>Camera ID<input required pattern="[A-Za-z0-9_-]{1,80}" placeholder="phone-01" value={form.id} onChange={e=>setForm({...form,id:e.target.value})}/></label>
          <label>Display name<input required placeholder="Front door phone" value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></label>
          <label>Device ID (optional)<input placeholder="Android device name" value={form.device_id} onChange={e=>setForm({...form,device_id:e.target.value})}/></label>
          <button className="primary">Register camera</button>
        </form>{message && <p className="message">{message}</p>}
      </article>
      <article><h2>Registered cameras <small>({cameras.length})</small></h2>
        {cameras.length ? cameras.map(c=><div className="row" key={c.id}><div><b>{c.name}</b><small>{c.id} · {c.device_id || 'device id unavailable'}</small></div><span className={`pill ${c.status}`}>{c.status}</span></div>) : <p>No cameras registered yet.</p>}
        <p className="hint">After registration, configure the matching CAMERA_ID and phone MJPEG URL in the edge processor’s .env.</p>
      </article>
    </section>
    <section className="panel"><h2>Events <small>updates every 4 seconds</small></h2>
      {events.length ? <div className="table-wrap"><table><thead><tr><th>Camera</th><th>Started</th><th>Duration</th><th>Model</th><th>Objects</th><th></th></tr></thead><tbody>
        {events.map(e=><tr key={e.event_id}><td>{e.camera_name || e.camera_id}</td><td>{new Date(e.started_at).toLocaleString()}</td><td>{Number(e.duration).toFixed(1)} s</td><td>{e.anomaly?.label || '—'}{e.anomaly ? ` (${Number(e.anomaly.anomaly_probability).toFixed(2)})` : ''}</td><td>{(e.detections||[]).map(d=>d.class_name).join(', ') || '—'}</td><td><button onClick={()=>setSelected(e)}>Details</button></td></tr>)}
      </tbody></table></div> : <p>No completed events yet. Events appear after motion ends and the no-motion timeout passes.</p>}
    </section>
    {selected && <div className="modal" onClick={()=>setSelected(null)}><article onClick={e=>e.stopPropagation()}><button className="close" onClick={()=>setSelected(null)}>Close</button><h2>Event details</h2><p><b>ID:</b> {selected.event_id}</p><p><b>Started:</b> {new Date(selected.started_at).toLocaleString()}</p><p><b>Ended:</b> {new Date(selected.ended_at).toLocaleString()}</p><p><b>Anomaly:</b> {selected.anomaly?.label || 'No inference data'} {selected.anomaly ? `(${selected.anomaly.anomaly_probability})` : ''}</p>{selected.video_available && <video controls src={`${API}/api/events/${selected.event_id}/video`}/>}<pre>{JSON.stringify(selected.detections || [], null, 2)}</pre></article></div>}
  </main>;
}

createRoot(document.getElementById('root')).render(<App/>);
