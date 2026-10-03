const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000';

// Middleware
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', '*');
    res.setHeader('Access-Control-Allow-Headers', '*');
    res.setHeader('Access-Control-Allow-Private-Network', 'true');
    if (req.method === 'OPTIONS') {
        return res.sendStatus(200);
    }
    next();
});
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// In-memory events database (with persistent JSON storage)
const EVENTS_FILE = path.join(__dirname, 'events_store.json');
let eventsDb = [];

if (fs.existsSync(EVENTS_FILE)) {
    try {
        eventsDb = JSON.parse(fs.readFileSync(EVENTS_FILE, 'utf8'));
    } catch (e) {
        eventsDb = [];
    }
}

function saveEvents() {
    try {
        fs.writeFileSync(EVENTS_FILE, JSON.stringify(eventsDb.slice(-500), null, 2));
    } catch (e) {
        console.error('Failed to save events to disk:', e);
    }
}

// Multer memory storage for image uploads
const upload = multer({ storage: multer.memoryStorage() });

// --- ROUTES ---

// 1. Health & Status
app.get('/api/health', async (req, res) => {
    let aiStatus = 'disconnected';
    let aiDetails = null;

    try {
        const response = await axios.get(`${AI_SERVICE_URL}/health`, { timeout: 3000 });
        if (response.data && response.data.status === 'online') {
            aiStatus = 'connected';
            aiDetails = response.data;
        }
    } catch (err) {
        aiStatus = 'disconnected';
    }

    res.json({
        server: 'Express backend online',
        port: PORT,
        ai_service: aiStatus,
        model_details: aiDetails,
        stored_events_count: eventsDb.length
    });
});

// 2. Get available classes
app.get('/api/classes', async (req, res) => {
    try {
        const response = await axios.get(`${AI_SERVICE_URL}/classes`, { timeout: 3000 });
        res.json(response.data);
    } catch (err) {
        res.status(502).json({ error: 'AI microservice unavailable' });
    }
});

// 3. Detect objects in uploaded image
app.post('/api/detect', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname || 'upload.jpg',
            contentType: req.file.mimetype || 'image/jpeg',
        });

        if (req.body.conf) {
            formData.append('conf', req.body.conf);
        }
        if (req.body.classes) {
            formData.append('classes', req.body.classes);
        }

        const aiResponse = await axios.post(`${AI_SERVICE_URL}/detect`, formData, {
            headers: formData.getHeaders(),
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            timeout: 10000
        });

        const data = aiResponse.data;

        // Log event into database
        const event = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            classes: data.detections.map(d => d.class_name),
            count: data.count,
            latency_ms: data.latency_ms,
            filename: req.file.originalname || 'image.jpg'
        };
        eventsDb.unshift(event);
        saveEvents();

        res.json({
            ...data,
            logged_event_id: event.id
        });
    } catch (err) {
        console.error('Detection error:', err.message);
        res.status(500).json({ error: 'Detection failed', details: err.message });
    }
});

// 4. Get events history
app.get('/api/events', (req, res) => {
    res.json({
        total: eventsDb.length,
        events: eventsDb.slice(0, 50)
    });
});

// 5. Clear events
app.delete('/api/events', (req, res) => {
    eventsDb = [];
    saveEvents();
    res.json({ success: true, message: 'Event logs cleared' });
});

// 6. Analytics & stats
app.get('/api/stats', (req, res) => {
    const classFrequencies = {};
    let totalObjects = 0;
    let totalLatency = 0;

    eventsDb.forEach(e => {
        totalObjects += e.count || 0;
        totalLatency += e.latency_ms || 0;
        (e.classes || []).forEach(c => {
            classFrequencies[c] = (classFrequencies[c] || 0) + 1;
        });
    });

    res.json({
        total_inferences: eventsDb.length,
        total_objects_detected: totalObjects,
        avg_latency_ms: eventsDb.length > 0 ? (totalLatency / eventsDb.length).toFixed(1) : 0,
        class_frequencies: classFrequencies
    });
});

// 7. Video Feed Stream Proxy
app.get('/api/video_feed', async (req, res) => {
    try {
        const response = await axios({
            method: 'get',
            url: `${AI_SERVICE_URL}/video_feed`,
            responseType: 'stream',
            timeout: 10000
        });
        res.setHeader('Content-Type', response.headers['content-type'] || 'multipart/x-mixed-replace; boundary=frame');
        response.data.pipe(res);
        req.on('close', () => {
            if (response.data && response.data.destroy) response.data.destroy();
        });
    } catch (err) {
        res.status(502).send('Camera stream unavailable');
    }
});

app.listen(PORT, () => {
    console.log(`[Express] Backend server running at http://localhost:${PORT}`);
    console.log(`[Express] Connected to AI Engine at ${AI_SERVICE_URL}`);
});
