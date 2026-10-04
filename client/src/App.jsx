import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Camera, Upload, Activity, Sliders, Shield, RefreshCw, 
  Trash2, CheckCircle, Layers, Cpu
} from 'lucide-react';

// Production Cloud AI Backend on Render
const CLOUD_API_URL = 'https://live-object-detection-ai.onrender.com/api';

// Automatically downscales large camera/phone images (e.g. 4000x3000) to max 1200px before uploading
const compressImageForInference = (file) => {
  return new Promise((resolve) => {
    if (file.size < 400 * 1024) {
      resolve(file);
      return;
    }
    const img = new Image();
    const reader = new FileReader();
    reader.onload = (e) => {
      img.onload = () => {
        const maxDim = 1200;
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          if (width > height) {
            height = Math.round((height * maxDim) / width);
            width = maxDim;
          } else {
            width = Math.round((width * maxDim) / height);
            height = maxDim;
          }
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const compressedFile = new File([blob], file.name.replace(/\.[^/.]+$/, "") + ".jpg", { type: 'image/jpeg' });
              resolve(compressedFile);
            } else {
              resolve(file);
            }
          },
          'image/jpeg',
          0.85
        );
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
};

export default function App() {
  const [activeTab, setActiveTab] = useState('image'); // 'image', 'webcam', 'events'
  const [systemStatus, setSystemStatus] = useState({ state: 'checking', message: 'Connecting to Cloud AI Engine...' });
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

  // 1. Polling System Health (gentle interval to avoid CPU thrashing)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('object_vision_api_url');
    }
    checkHealth();
    fetchEvents();
    fetchStats();
    const interval = setInterval(() => {
      checkHealth();
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${CLOUD_API_URL}/health`, { timeout: 20000 });
      if (res.data) {
        setSystemStatus({
          state: 'online',
          message: 'Cloud AI Online',
          model: res.data.model_details?.model_path || 'models/best.pt',
          classesCount: res.data.model_details?.classes_count || 20
        });
        if (res.data.model_details?.classes && availableClasses.length === 0) {
          setAvailableClasses(res.data.model_details.classes);
        }
      }
    } catch (err) {
      setSystemStatus({ state: 'waking', message: 'Waking up Cloud AI Engine (Render free tier cold-boot ~30s)...' });
    }
  };

  const fetchEvents = async () => {
    try {
      const res = await axios.get(`${CLOUD_API_URL}/events`, { timeout: 8000 });
      setEvents(res.data.events || []);
    } catch (err) {}
  };

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${CLOUD_API_URL}/stats`, { timeout: 8000 });
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
    try {
      const optimizedImage = await compressImageForInference(selectedImage);
      const formData = new FormData();
      formData.append('file', optimizedImage);
      formData.append('image', optimizedImage);
      formData.append('conf', confidence);
      if (selectedClasses.length > 0) {
        formData.append('classes', selectedClasses.join(','));
      }

      const res = await axios.post(`${CLOUD_API_URL}/detect`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000
      });
      setDetectionResult(res.data);
      fetchEvents();
      fetchStats();
    } catch (err) {
      alert('Detection failed. The cloud server may be waking up from cold boot. Please retry in a few seconds.');
    } finally {
      setIsDetecting(false);
    }
  };

  // 4. Toggle class filter
  const toggleClass = (cls) => {
    if (selectedClasses.includes(cls)) {
      setSelectedClasses(selectedClasses.filter(c => c !== cls));
    } else {
      setSelectedClasses([...selectedClasses, cls]);
    }
  };

  // 5. Clear events
  const handleClearEvents = async () => {
    try {
      await axios.delete(`${CLOUD_API_URL}/events`);
      fetchEvents();
      fetchStats();
    } catch (err) {}
  };

  const aiFeedUrl = `${CLOUD_API_URL}/video_feed`;

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px' }}>
      {/* Top Navbar */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid var(--border)', paddingBottom: '20px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', width: '40px', height: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Shield size={22} color="#fff" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.45rem', fontWeight: '700', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                ObjectVision <span style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem', fontWeight: '600' }}>AI</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>Production Real-Time Object Detection & Visual Analytics</p>
            </div>
          </div>
        </div>

        {/* Clean System Badges */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div className={`badge ${systemStatus.state === 'online' ? 'badge-success' : 'badge-warning'}`} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ 
              width: '8px', 
              height: '8px', 
              borderRadius: '50%', 
              background: systemStatus.state === 'online' ? '#10b981' : '#f59e0b',
              boxShadow: systemStatus.state === 'online' ? '0 0 8px #10b981' : 'none'
            }} />
            {systemStatus.message}
          </div>
          <div className="badge badge-primary">
            <Layers size={14} /> Fine-Tuned YOLOv11 (20 Classes)
          </div>
        </div>
      </header>

      {/* Cloud Wake-Up Banner if Cold Booting */}
      {systemStatus.state === 'waking' && (
        <div style={{ 
          background: 'rgba(245, 158, 11, 0.1)', 
          border: '1px solid rgba(245, 158, 11, 0.3)', 
          borderRadius: '8px', 
          padding: '12px 16px', 
          marginBottom: '20px', 
          display: 'flex', 
          alignItems: 'center', 
          gap: '12px',
          color: '#fbbf24',
          fontSize: '0.85rem'
        }}>
          <RefreshCw className="animate-spin" size={16} />
          <span>Cloud instance is spinning up on Render. Requests will complete automatically once ready.</span>
        </div>
      )}

      {/* Main Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px' }}>
        
        {/* LEFT COLUMN: Controls & Settings */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Quick Metrics */}
          <div className="card">
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={16} color="var(--accent-emerald)" /> Live Telemetry
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
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', maxHeight: '200px', overflowY: 'auto', paddingRight: '4px' }}>
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
              <Activity size={16} /> Event History ({events.length})
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
                      <p>Select an image to analyze</p>
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
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Real-time camera feed with YOLO deep learning inference</p>
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
                    onError={() => alert('Could not connect to webcam stream. Ensure camera permissions are granted.')}
                  />
                </div>
              ) : (
                <div style={{ padding: '60px 20px', background: 'var(--bg-secondary)', borderRadius: '12px', border: '1px dashed var(--border)' }}>
                  <Camera size={48} style={{ opacity: 0.3, margin: '0 auto 16px' }} />
                  <h4>Stream is Currently Inactive</h4>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '6px' }}>
                    Click "Start Live Stream" to activate your connected camera with real-time detection boxes.
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
                  <h3 style={{ fontSize: '1rem', fontWeight: '600' }}>Event & Alert History</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Audit log of detected objects and inference metrics</p>
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
