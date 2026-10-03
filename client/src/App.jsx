import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Camera, Upload, Activity, Sliders, Shield, RefreshCw, 
  Trash2, CheckCircle, Layers, Server, Cpu, Link, Settings
} from 'lucide-react';

// Cloudflare Public HTTPS Tunnel for port 5000 (Express + AI microservice proxy)
const CLOUDFLARE_URL = 'https://lifestyle-role-collection-artists.trycloudflare.com/api';

const getDefaultApi = () => {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem('object_vision_api_url');
    // Discard old dead localtunnel URLs
    if (saved && !saved.includes('loca.lt')) {
      return saved;
    }
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:5000/api';
    }
  }
  return CLOUDFLARE_URL;
};

export default function App() {
  const [apiUrl, setApiUrl] = useState(getDefaultApi);
  const [showApiSettings, setShowApiSettings] = useState(false);
  const [tempApiUrl, setTempApiUrl] = useState(apiUrl);

  const [activeTab, setActiveTab] = useState('image'); // 'webcam', 'image', 'events'
  const [systemStatus, setSystemStatus] = useState({ backend: 'checking', ai: 'checking', model: 'Loading...' });
  const [availableClasses, setAvailableClasses] = useState([]);
  const [selectedClasses, setSelectedClasses] = useState([]);
  const [confidence, setConfidence] = useState(0.35);

  // Image detection state
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);

  // Events & Analytics state
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({ total_inferences: 0, total_objects: 0, avg_latency: 0 });

  // Webcam stream state
  const [webcamActive, setWebcamActive] = useState(false);
  const fileInputRef = useRef(null);

  // 1. Fetch system health and classes on mount or API URL change
  useEffect(() => {
    checkHealth();
    fetchEvents();
    fetchStats();
    const interval = setInterval(() => {
      checkHealth();
      fetchEvents();
      fetchStats();
    }, 4000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const checkHealth = async (target = apiUrl) => {
    try {
      const res = await axios.get(`${target}/health`, { 
        timeout: 4000
      });
      setSystemStatus({
        backend: 'connected',
        ai: res.data.ai_service,
        model: res.data.model_details?.model_path || 'models/best.pt',
        classesCount: res.data.model_details?.classes_count || 20
      });
      if (res.data.model_details?.classes && availableClasses.length === 0) {
        setAvailableClasses(res.data.model_details.classes);
      }
      if (target !== apiUrl) {
        setApiUrl(target);
        localStorage.setItem('object_vision_api_url', target);
      }
    } catch (err) {
      // If primary target fails and we are not already on Cloudflare tunnel, try auto-connecting to tunnel
      if (target !== CLOUDFLARE_URL) {
        try {
          const cfRes = await axios.get(`${CLOUDFLARE_URL}/health`, { timeout: 4000 });
          if (cfRes.data && cfRes.data.server) {
            setSystemStatus({
              backend: 'connected',
              ai: cfRes.data.ai_service,
              model: cfRes.data.model_details?.model_path || 'models/best.pt',
              classesCount: cfRes.data.model_details?.classes_count || 20
            });
            if (cfRes.data.model_details?.classes && availableClasses.length === 0) {
              setAvailableClasses(cfRes.data.model_details.classes);
            }
            setApiUrl(CLOUDFLARE_URL);
            localStorage.setItem('object_vision_api_url', CLOUDFLARE_URL);
            return;
          }
        } catch (e2) {}
      }
      setSystemStatus({ backend: 'disconnected', ai: 'disconnected', model: 'Offline' });
    }
  };

  const fetchEvents = async () => {
    try {
      const res = await axios.get(`${apiUrl}/events`, {
        timeout: 4000
      });
      setEvents(res.data.events || []);
    } catch (err) {}
  };

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${apiUrl}/stats`, {
        headers: { 'Bypass-Tunnel-Reminder': 'true' }
      });
      setStats({
        total_inferences: res.data.total_inferences || 0,
        total_objects: res.data.total_objects_detected || 0,
        avg_latency: res.data.avg_latency_ms || 0
      });
    } catch (err) {}
  };

  // 2. Handle Image Upload
  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setDetectionResult(null);
    }
  };

  // 3. Run Inference on selected image
  const handleRunDetection = async () => {
    if (!selectedImage) return;

    setIsDetecting(true);
    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('conf', confidence);
    if (selectedClasses.length > 0) {
      formData.append('classes', selectedClasses.join(','));
    }

    try {
      const res = await axios.post(`${apiUrl}/detect`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Bypass-Tunnel-Reminder': 'true'
        }
      });
      setDetectionResult(res.data);
      fetchEvents();
      fetchStats();
    } catch (err) {
      alert(`Detection failed. Please verify API is running at: ${apiUrl}`);
    } finally {
      setIsDetecting(false);
    }
  };

  // 4. Save API URL
  const handleSaveApiUrl = () => {
    let clean = tempApiUrl.trim().replace(/\/+$/, '');
    if (!clean.endsWith('/api') && !clean.includes('/api/')) {
      clean += '/api';
    }
    setApiUrl(clean);
    localStorage.setItem('object_vision_api_url', clean);
    setShowApiSettings(false);
  };

  // 5. Toggle class filter
  const toggleClass = (cls) => {
    if (selectedClasses.includes(cls)) {
      setSelectedClasses(selectedClasses.filter(c => c !== cls));
    } else {
      setSelectedClasses([...selectedClasses, cls]);
    }
  };

  // 6. Clear events
  const handleClearEvents = async () => {
    await axios.delete(`${apiUrl}/events`, {
      headers: { 'Bypass-Tunnel-Reminder': 'true' }
    });
    fetchEvents();
    fetchStats();
  };

  const aiFeedUrl = `${apiUrl}/video_feed`;

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px' }}>
      {/* Top Navbar */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px', borderBottom: '1px solid var(--border)', paddingBottom: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', width: '38px', height: '38px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Shield size={22} color="#fff" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.4rem', fontWeight: '700', letterSpacing: '-0.02em' }}>ObjectVision <span style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem', fontWeight: '500' }}>MERN AI Suite</span></h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>React + Express + Node.js + YOLOv11 Deep Learning Pipeline</p>
            </div>
          </div>
        </div>

        {/* System Badges & API Settings */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div className={`badge ${systemStatus.backend === 'connected' ? 'badge-success' : 'badge-warning'}`}>
            <Server size={14} /> Express API: {systemStatus.backend}
          </div>
          <div className={`badge ${systemStatus.ai === 'online' || systemStatus.ai === 'connected' ? 'badge-success' : 'badge-warning'}`}>
            <Cpu size={14} /> AI Engine: {systemStatus.ai}
          </div>
          <div className="badge badge-primary">
            <Layers size={14} /> Model: {systemStatus.model}
          </div>
          <button 
            onClick={() => setShowApiSettings(!showApiSettings)}
            title="Configure Backend API URL"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', borderRadius: '8px', padding: '6px 10px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem' }}
          >
            <Settings size={14} /> API Settings
          </button>
        </div>
      </header>

      {/* API Endpoint Configuration Modal/Banner */}
      {showApiSettings && (
        <div className="card" style={{ marginBottom: '20px', background: 'var(--bg-secondary)', borderColor: 'var(--accent-emerald)' }}>
          <h4 style={{ fontSize: '0.9rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-emerald)' }}>
            <Link size={16} /> Backend API Endpoint Configuration
          </h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
            Current API URL: <code>{apiUrl}</code>
          </p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input 
              type="text" 
              value={tempApiUrl} 
              onChange={(e) => setTempApiUrl(e.target.value)}
              placeholder="e.g. https://...trycloudflare.com/api or http://localhost:5000/api"
              style={{ flex: 1, padding: '8px 12px', borderRadius: '6px', background: 'var(--bg-primary)', border: '1px solid var(--border)', color: '#fff', fontSize: '0.85rem' }}
            />
            <button onClick={handleSaveApiUrl} className="btn btn-primary" style={{ padding: '8px 16px' }}>Save & Connect</button>
            <button onClick={() => { setTempApiUrl('http://localhost:5000/api'); }} className="btn btn-secondary" style={{ padding: '8px 12px' }}>Use Localhost</button>
            <button onClick={() => { setTempApiUrl(CLOUDFLARE_URL); }} className="btn btn-secondary" style={{ padding: '8px 12px' }}>Use Cloud Tunnel</button>
          </div>
        </div>
      )}

      {/* Main Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px' }}>
        
        {/* LEFT COLUMN: Controls & Settings */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Quick Metrics */}
          <div className="card">
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={16} color="var(--accent-emerald)" /> Real-Time Telemetry
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Inferences</span>
                <p style={{ fontSize: '1.3rem', fontWeight: '700', color: 'var(--text-primary)' }}>{stats.total_inferences}</p>
              </div>
              <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Avg Latency</span>
                <p style={{ fontSize: '1.3rem', fontWeight: '700', color: 'var(--accent-cyan)' }}>{stats.avg_latency} ms</p>
              </div>
            </div>
          </div>

          {/* Model Inference Controls */}
          <div className="card">
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={16} color="var(--accent-cyan)" /> Detection Parameters
            </h3>

            {/* Confidence Slider */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Confidence Threshold</label>
                <span style={{ fontWeight: '600', color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>{(confidence * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" 
                min="0.10" 
                max="0.95" 
                step="0.05" 
                value={confidence} 
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-emerald)', cursor: 'pointer' }}
              />
            </div>

            {/* Target Class Filter */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Filter Classes ({selectedClasses.length === 0 ? 'All 20' : selectedClasses.length})</label>
                {selectedClasses.length > 0 && (
                  <button onClick={() => setSelectedClasses([])} style={{ background: 'none', border: 'none', color: 'var(--accent-red)', fontSize: '0.75rem', cursor: 'pointer' }}>Reset</button>
                )}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', maxHeight: '180px', overflowY: 'auto', paddingRight: '4px' }}>
                {availableClasses.map((cls) => {
                  const active = selectedClasses.includes(cls);
                  return (
                    <button
                      key={cls}
                      onClick={() => toggleClass(cls)}
                      style={{
                        padding: '4px 8px',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                        border: '1px solid',
                        borderColor: active ? 'var(--accent-emerald)' : 'var(--border)',
                        background: active ? 'rgba(16, 185, 129, 0.2)' : 'var(--bg-secondary)',
                        color: active ? '#34d399' : 'var(--text-secondary)',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      {cls}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Desktop App Notice */}
          <div className="card" style={{ background: 'rgba(59, 130, 246, 0.05)', borderColor: 'rgba(59, 130, 246, 0.2)' }}>
            <h4 style={{ fontSize: '0.85rem', color: '#60a5fa', marginBottom: '8px' }}>💡 Desktop Live Stream</h4>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
              For high-FPS video streaming with line crossing and dwell-time alerts:
            </p>
            <code style={{ display: 'block', background: 'var(--bg-primary)', padding: '8px', borderRadius: '6px', fontSize: '0.75rem', marginTop: '8px', color: '#38bdf8' }}>
              python app/main.py --model models/best.pt
            </code>
          </div>

        </aside>

        {/* RIGHT COLUMN: Interactive Workspaces */}
        <main style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Navigation Tabs */}
          <div style={{ display: 'flex', gap: '10px', borderBottom: '1px solid var(--border)', paddingBottom: '12px' }}>
            <button
              onClick={() => setActiveTab('image')}
              className={`btn ${activeTab === 'image' ? 'btn-primary' : 'btn-secondary'}`}
            >
              <Upload size={16} /> Image Detection
            </button>
            <button
              onClick={() => setActiveTab('webcam')}
              className={`btn ${activeTab === 'webcam' ? 'btn-primary' : 'btn-secondary'}`}
            >
              <Camera size={16} /> Live AI Webcam Stream
            </button>
            <button
              onClick={() => setActiveTab('events')}
              className={`btn ${activeTab === 'events' ? 'btn-primary' : 'btn-secondary'}`}
            >
              <Activity size={16} /> Event Log & History ({events.length})
            </button>
          </div>

          {/* TAB 1: IMAGE INFERENCE */}
          {activeTab === 'image' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              
              {/* Upload Controls Bar */}
              <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageSelect}
                    accept="image/*"
                    style={{ display: 'none' }}
                  />
                  <button onClick={() => fileInputRef.current?.click()} className="btn btn-secondary">
                    <Upload size={16} /> Choose Image
                  </button>
                  {selectedImage && (
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Selected: <b>{selectedImage.name}</b>
                    </span>
                  )}
                </div>

                <button 
                  onClick={handleRunDetection} 
                  disabled={!selectedImage || isDetecting}
                  className="btn btn-primary"
                  style={{ opacity: !selectedImage || isDetecting ? 0.6 : 1 }}
                >
                  {isDetecting ? <RefreshCw className="animate-spin" size={16} /> : <CheckCircle size={16} />}
                  {isDetecting ? 'Running YOLO...' : 'Detect Objects'}
                </button>
              </div>

              {/* Visual Display Comparison */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                
                {/* Source Image */}
                <div className="card" style={{ minHeight: '340px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ alignSelf: 'flex-start', fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '12px' }}>Input Image</span>
                  {previewUrl ? (
                    <img src={previewUrl} alt="Input" style={{ maxWidth: '100%', maxHeight: '380px', borderRadius: '8px', objectFit: 'contain' }} />
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                      <Upload size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                      <p>Select or drag an image to begin</p>
                    </div>
                  )}
                </div>

                {/* Annotated Output */}
                <div className="card" style={{ minHeight: '340px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>YOLO Annotated Result</span>
                    {detectionResult && (
                      <span className="badge badge-success">
                        {detectionResult.count} Objects • {detectionResult.latency_ms} ms
                      </span>
                    )}
                  </div>

                  {detectionResult?.annotated_image ? (
                    <img src={detectionResult.annotated_image} alt="Annotated" style={{ maxWidth: '100%', maxHeight: '380px', borderRadius: '8px', objectFit: 'contain' }} />
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                      <Shield size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                      <p>Click "Detect Objects" to view detections</p>
                    </div>
                  )}
                </div>

              </div>

              {/* Detected Objects Table */}
              {detectionResult?.detections && detectionResult.detections.length > 0 && (
                <div className="card">
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '14px', color: 'var(--text-secondary)' }}>
                    Detected Entities Breakdown ({detectionResult.detections.length})
                  </h4>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '8px' }}>#</th>
                        <th style={{ padding: '8px' }}>Class</th>
                        <th style={{ padding: '8px' }}>Confidence</th>
                        <th style={{ padding: '8px' }}>Bounding Box [x1, y1, x2, y2]</th>
                        <th style={{ padding: '8px' }}>Center [cx, cy]</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detectionResult.detections.map((det, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '10px 8px' }}>{idx + 1}</td>
                          <td style={{ padding: '10px 8px', fontWeight: '600', color: 'var(--accent-emerald)', textTransform: 'capitalize' }}>
                            {det.class_name}
                          </td>
                          <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)' }}>
                            {(det.confidence * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            [{det.bbox.join(', ')}]
                          </td>
                          <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            [{det.center.join(', ')}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

            </div>
          )}

          {/* TAB 2: LIVE WEBCAM STREAM */}
          {activeTab === 'webcam' && (
            <div className="card" style={{ textAlign: 'center', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Live AI Camera Stream</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Real-time MJPEG feed with 20-class YOLO inference directly inside React</p>
                </div>
                <button 
                  onClick={() => setWebcamActive(!webcamActive)} 
                  className={`btn ${webcamActive ? 'btn-secondary' : 'btn-primary'}`}
                >
                  <Camera size={16} /> {webcamActive ? 'Stop Stream' : 'Start Live Stream'}
                </button>
              </div>

              {webcamActive ? (
                <div style={{ background: '#000', borderRadius: '12px', overflow: 'hidden', display: 'inline-block', boxShadow: '0 8px 30px rgba(0,0,0,0.5)' }}>
                  <img 
                    src={aiFeedUrl} 
                    alt="Live YOLO Stream" 
                    style={{ width: '100%', maxWidth: '960px', height: 'auto', display: 'block' }} 
                    onError={() => alert('Could not connect to webcam stream. Ensure camera is plugged in.')}
                  />
                </div>
              ) : (
                <div style={{ padding: '60px 20px', background: 'var(--bg-secondary)', borderRadius: '12px', border: '1px dashed var(--border)' }}>
                  <Camera size={48} style={{ opacity: 0.3, margin: '0 auto 16px' }} />
                  <h4>Stream is Currently Inactive</h4>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '6px' }}>
                    Click "Start Live Stream" to activate your connected webcam with real-time detection boxes.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: EVENTS LOG */}
          {activeTab === 'events' && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h3 style={{ fontSize: '1rem', fontWeight: '600' }}>Event & Alert Log</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Stored in Express database for security auditing & tracking</p>
                </div>
                <button onClick={handleClearEvents} className="btn btn-secondary" style={{ color: 'var(--accent-red)' }}>
                  <Trash2 size={14} /> Clear History
                </button>
              </div>

              {events.length === 0 ? (
                <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px 0' }}>No detection events recorded yet.</p>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '8px' }}>Event ID</th>
                      <th style={{ padding: '8px' }}>Timestamp</th>
                      <th style={{ padding: '8px' }}>Detected Classes</th>
                      <th style={{ padding: '8px' }}>Count</th>
                      <th style={{ padding: '8px' }}>Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((ev) => (
                      <tr key={ev.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{ev.id.toString().slice(-6)}</td>
                        <td style={{ padding: '10px 8px' }}>{new Date(ev.timestamp).toLocaleTimeString()}</td>
                        <td style={{ padding: '10px 8px' }}>
                          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                            {ev.classes?.map((c, i) => (
                              <span key={i} className="badge badge-success" style={{ fontSize: '0.7rem' }}>{c}</span>
                            ))}
                          </div>
                        </td>
                        <td style={{ padding: '10px 8px', fontWeight: '600' }}>{ev.count}</td>
                        <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{ev.latency_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
