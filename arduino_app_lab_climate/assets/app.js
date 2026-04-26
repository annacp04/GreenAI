// Initialize socket.io connection using the exact pattern from test_camera
const socket = io(`http://${window.location.host}`); 

let canvas, ctx, noSignal, connStatus, errorOverlay;

let currentImageBitmap = null;
const sparklines = {};

console.log("GreenAI Dashboard Initializing with host:", window.location.host);

const metricConfigs = {
    'temperature': { id: 'temp', color: '#ff8c42', min: 10, max: 40 },
    'humidity': { id: 'hum', color: '#42a5f5', min: 0, max: 100 },
    'light': { id: 'light', color: '#ffeb3b', min: 0, max: 1024 },
    'dew_point': { id: 'dew', color: '#9c27b0', min: 0, max: 30 },
    'heat_index': { id: 'heat', color: '#f44336', min: 10, max: 50 },
    'absolute_humidity': { id: 'abs', color: '#00ff88', min: 0, max: 30 }
};

function initSocketIO() {
    socket.on('connect', () => {
        console.log("Socket connected to board!");
        connStatus.textContent = 'Connected';
        connStatus.className = 'status-badge connected';
        errorOverlay.classList.add('hidden');
    });

    socket.on('disconnect', (reason) => {
        console.warn("Socket disconnected:", reason);
        connStatus.textContent = 'Disconnected';
        connStatus.className = 'status-badge disconnected';
        errorOverlay.classList.remove('hidden');
    });

    socket.on('connect_error', (err) => {
        console.error("Socket connection error:", err);
    });

    // Camera Frame Handler
    socket.on('camera_frame', async (message) => {
        if (noSignal && !noSignal.classList.contains('hidden')) {
            console.log("Stream active: Frame received");
            noSignal.classList.add('hidden');
        }
        await renderFrame(message.image, message.image_type);
    });

    // Metrics Handlers
    Object.keys(metricConfigs).forEach(key => {
        socket.on(key, (data) => {
            updateMetricUI(key, data.value);
        });
    });
}

async function renderFrame(base64Image, type) {
    try {
        const bytes = base64ToUint8Array(base64Image);
        const blob = new Blob([bytes], { type: type });

        if (currentImageBitmap) currentImageBitmap.close();
        currentImageBitmap = await createImageBitmap(blob);

        if (canvas.width !== currentImageBitmap.width) canvas.width = currentImageBitmap.width;
        if (canvas.height !== currentImageBitmap.height) canvas.height = currentImageBitmap.height;

        ctx.drawImage(currentImageBitmap, 0, 0);
    } catch (e) {
        // console.error("Frame render error:", e);
    }
}

function base64ToUint8Array(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

function updateMetricUI(key, value) {
    const config = metricConfigs[key];
    const el = document.getElementById(`val-${config.id}`);
    if (el) el.textContent = value.toFixed(1);
    updateSparkline(key, value);
}

function updateSparkline(key, value) {
    const config = metricConfigs[key];
    if (!sparklines[key]) {
        const c = document.getElementById(`chart-${config.id}`);
        if (!c) return;
        
        sparklines[key] = new Chart(c, {
            type: 'line',
            data: {
                labels: Array(20).fill(''),
                datasets: [{
                    data: Array(20).fill(null),
                    borderColor: config.color,
                    borderWidth: 2,
                    fill: true,
                    backgroundColor: hexToRgba(config.color, 0.1),
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { 
                    x: { display: false }, 
                    y: { 
                        display: true, 
                        position: 'right',
                        suggestedMin: config.min, 
                        suggestedMax: config.max,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 } }
                    } 
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    const chart = sparklines[key];
    chart.data.datasets[0].data.push(value);
    chart.data.datasets[0].data.shift();
    chart.update('none');
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('videoCanvas');
    ctx = canvas.getContext('2d');
    noSignal = document.getElementById('no-signal');
    connStatus = document.getElementById('connection-status');
    errorOverlay = document.getElementById('error-overlay');
    
    initSocketIO();
});
