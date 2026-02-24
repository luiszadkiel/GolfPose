function switchMode(mode) {
    const uploadMain = document.getElementById('upload-mode');
    const liveMain = document.getElementById('live-mode');
    const buttons = document.querySelectorAll('nav button');

    if (mode === 'upload') {
        uploadMain.classList.remove('hidden');
        liveMain.classList.add('hidden');
        buttons[0].classList.add('active');
        buttons[1].classList.remove('active');
        stopLive();
    } else {
        uploadMain.classList.add('hidden');
        liveMain.classList.remove('hidden');
        buttons[0].classList.remove('active');
        buttons[1].classList.add('active');
        startLive();
    }
}

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const processingView = document.getElementById('processing-view');
const dropContent = document.querySelector('.drop-content');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

async function handleFileUpload(file) {
    dropContent.classList.add('hidden');
    processingView.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        let progressValue = 0;
        const interval = setInterval(() => {
            progressValue += 1;
            document.getElementById('progress').style.width = progressValue + '%';
            if (progressValue >= 95) clearInterval(interval);
        }, 100);

        const response = await fetch('/analyze_video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        clearInterval(interval);

        if (data.success) {
            showResults(data);
        } else {
            alert('Error en el análisis: ' + data.error);
            resetUpload();
        }
    } catch (err) {
        console.error(err);
        alert('Error al conectar con el servidor.');
        resetUpload();
    }
}

function showResults(data) {
    document.getElementById('upload-mode').classList.remove('hidden');
    document.getElementById('drop-zone').classList.add('hidden');
    document.getElementById('results-view').classList.remove('hidden');

    document.getElementById('output-video').src = data.output_url;
    document.getElementById('shoulder-angle').innerText = data.stats.shoulder_angle + '°';
    document.getElementById('knee-angle').innerText = data.stats.knee_angle + '°';
    document.getElementById('club-speed').innerText = data.stats.club_speed + ' mph';
}

function resetUpload() {
    dropContent.classList.remove('hidden');
    processingView.classList.add('hidden');
    document.getElementById('progress').style.width = '0%';
}

// ============ LIVE MODE ============

let liveStream = null;
let liveInterval = null;
let isProcessing = false;

// GolfPose 22-keypoint skeleton (índices de conexión por hueso)
const SKELETON_BONES = [
    // Pierna izquierda (verde)
    { from: 0, to: 4, color: '#22c55e' },
    { from: 4, to: 5, color: '#22c55e' },
    { from: 5, to: 6, color: '#22c55e' },
    // Pierna derecha (naranja)
    { from: 0, to: 1, color: '#f97316' },
    { from: 1, to: 2, color: '#f97316' },
    { from: 2, to: 3, color: '#f97316' },
    // Columna (azul)
    { from: 0, to: 7, color: '#3b82f6' },
    { from: 7, to: 8, color: '#3b82f6' },
    { from: 8, to: 9, color: '#3b82f6' },
    { from: 9, to: 10, color: '#3b82f6' },
    // Brazo izquierdo (verde)
    { from: 8, to: 11, color: '#22c55e' },
    { from: 11, to: 12, color: '#22c55e' },
    { from: 12, to: 13, color: '#22c55e' },
    // Brazo derecho (naranja)
    { from: 8, to: 14, color: '#f97316' },
    { from: 14, to: 15, color: '#f97316' },
    { from: 15, to: 16, color: '#f97316' },
    // Palo de golf (blanco)
    { from: 17, to: 18, color: '#ffffff' },
    { from: 18, to: 19, color: '#ffffff' },
    { from: 19, to: 20, color: '#ffffff' },
    { from: 20, to: 21, color: '#ffffff' },
];

// Color por keypoint (cuerpo vs palo)
const KP_COLORS = [
    '#3b82f6', // 0  root
    '#f97316', // 1  right_hip
    '#f97316', // 2  right_knee
    '#f97316', // 3  right_foot
    '#22c55e', // 4  left_hip
    '#22c55e', // 5  left_knee
    '#22c55e', // 6  left_foot
    '#3b82f6', // 7  spine
    '#3b82f6', // 8  thorax
    '#3b82f6', // 9  neck_base
    '#3b82f6', // 10 head
    '#22c55e', // 11 left_shoulder
    '#22c55e', // 12 left_elbow
    '#22c55e', // 13 left_wrist
    '#f97316', // 14 right_shoulder
    '#f97316', // 15 right_elbow
    '#f97316', // 16 right_wrist
    '#ffffff', // 17 shaft
    '#ffffff', // 18 hosel
    '#ffffff', // 19 heel
    '#ffffff', // 20 toe_down
    '#ffffff', // 21 toe_up
];

async function startLive() {
    const webcam = document.getElementById('webcam');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(
            'La cámara no está disponible.\n\n' +
            'El navegador requiere HTTPS para acceder a la cámara.\n' +
            'Accede usando https:// o desde localhost.'
        );
        return;
    }

    try {
        liveStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        webcam.srcObject = liveStream;
        liveInterval = setInterval(processLiveFrame, 250);
    } catch (err) {
        alert('No se pudo acceder a la cámara: ' + err.message);
    }
}

function stopLive() {
    if (liveStream) {
        liveStream.getTracks().forEach(track => track.stop());
        liveStream = null;
    }
    if (liveInterval) {
        clearInterval(liveInterval);
        liveInterval = null;
    }
}

const MIN_KP_SCORE = 0.3;

async function processLiveFrame() {
    if (isProcessing) return;
    isProcessing = true;

    const webcam = document.getElementById('webcam');
    if (!webcam.videoWidth) { isProcessing = false; return; }

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(webcam, 0, 0);

    const blob = await new Promise(resolve =>
        captureCanvas.toBlob(resolve, 'image/jpeg', 0.8)
    );
    const formData = new FormData();
    formData.append('file', blob);

    try {
        const response = await fetch('/analyze_frame', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success && data.keypoints) {
            drawSkeleton(data.keypoints, data.scores || []);
            updateLiveStats(data.stats);
        } else {
            drawNoDetection();
        }
    } catch (err) {
        console.error('Error en frame live:', err);
    } finally {
        isProcessing = false;
    }
}

function drawSkeleton(kps, scores) {
    const canvas = document.getElementById('canvas-overlay');
    const webcam = document.getElementById('webcam');

    if (!webcam.videoWidth || !webcam.videoHeight) return;

    const ctx = canvas.getContext('2d');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!kps || kps.length === 0) return;

    const isValid = (idx) => {
        if (!kps[idx]) return false;
        if (scores.length > 0 && scores[idx] < MIN_KP_SCORE) return false;
        return true;
    };

    // Dibujar huesos con grosor y glow
    SKELETON_BONES.forEach(bone => {
        if (!isValid(bone.from) || !isValid(bone.to)) return;
        const a = kps[bone.from];
        const b = kps[bone.to];

        ctx.save();
        ctx.shadowColor = bone.color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.strokeStyle = bone.color;
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        ctx.restore();
    });

    // Dibujar keypoints solo los confiables
    kps.forEach((kp, i) => {
        if (!isValid(i)) return;
        const radius = (i === 10) ? 9 : 6;
        ctx.beginPath();
        ctx.fillStyle = KP_COLORS[i] || '#4ade80';
        ctx.arc(kp[0], kp[1], radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawNoDetection() {
    const canvas = document.getElementById('canvas-overlay');
    const webcam = document.getElementById('webcam');
    const ctx = canvas.getContext('2d');

    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'rgba(239, 68, 68, 0.7)';
    ctx.font = 'bold 24px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No se detectó golfista', canvas.width / 2, 50);
}

function updateLiveStats(stats) {
    if (!stats) return;

    const badge = document.getElementById('mode-badge');
    if (stats.mode === '3D') {
        badge.textContent = '3D';
        badge.classList.add('active-3d');
    } else {
        badge.textContent = '2D';
        badge.classList.remove('active-3d');
    }

    const bufferPct = Math.round((stats.buffer / 27) * 100);
    const bufferFill = document.getElementById('buffer-fill');
    const bufferCount = document.getElementById('buffer-count');
    if (bufferFill) bufferFill.style.width = bufferPct + '%';
    if (bufferCount) bufferCount.textContent = stats.buffer;

    const map = {
        's-rknee': stats.right_knee_angle,
        's-lknee': stats.left_knee_angle,
        's-rshoulder': stats.right_shoulder_angle,
        's-lshoulder': stats.left_shoulder_angle,
        's-relbow': stats.right_elbow_angle,
        's-lelbow': stats.left_elbow_angle,
        's-spine': stats.spine_angle,
    };
    for (const [id, val] of Object.entries(map)) {
        const el = document.getElementById(id);
        if (el && val !== undefined) el.textContent = val + '°';
    }

    const clubEl = document.getElementById('s-clubspeed');
    if (clubEl && stats.club_speed != null) clubEl.textContent = stats.club_speed + ' mph';

    updateSwingState(stats);
}

function updateSwingState(stats) {
    const dot = document.getElementById('swing-dot');
    const text = document.getElementById('swing-text');
    const result = document.getElementById('swing-result');
    if (!dot || !text) return;

    const st = stats.swing_state;
    if (st === 'idle') {
        dot.className = 'swing-dot idle';
        text.textContent = 'Colócate en posición de address';
    } else if (st === 'ready') {
        dot.className = 'swing-dot ready';
        text.textContent = 'Listo — Haz tu swing';
        if (result) result.classList.add('hidden');
    } else if (st === 'active') {
        dot.className = 'swing-dot active';
        text.textContent = 'Analizando swing...';
    } else if (st === 'done') {
        dot.className = 'swing-dot done';
        text.textContent = 'Swing completado';
        if (stats.swing_summary) showSwingSummary(stats.swing_summary);
    }
}

function showSwingSummary(s) {
    const el = document.getElementById('swing-result');
    if (!el) return;
    el.classList.remove('hidden');

    const spd = s.peak_club_speed;
    document.getElementById('sr-speed').textContent = spd ? spd + ' mph' : '-- mph';

    const phases = { a: 'address', i: 'impact', f: 'finish' };
    const metrics = {
        rk: 'right_knee_angle', lk: 'left_knee_angle',
        rs: 'right_shoulder_angle', ls: 'left_shoulder_angle',
        re: 'right_elbow_angle', le: 'left_elbow_angle',
        sp: 'spine_angle',
    };

    for (const [pk, pn] of Object.entries(phases)) {
        const phase = s[pn];
        if (!phase) continue;
        for (const [mk, mn] of Object.entries(metrics)) {
            const cell = document.getElementById('sr-' + pk + '-' + mk);
            if (cell) {
                const v = phase[mn];
                cell.textContent = v != null ? v + '°' : '--';
            }
        }
    }
}
