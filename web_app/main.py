from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
import subprocess
import cv2
import numpy as np
import uuid
from inference_engine import GolfInferenceEngine

app = FastAPI()

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Iniciar motor (tardará un poco)
engine = GolfInferenceEngine()

# --- Dibujo de esqueleto (mismo esqueleto que el frontend, colores BGR) ---
SKELETON_BONES = [
    (0, 4, (94, 197, 34)),   (4, 5, (94, 197, 34)),   (5, 6, (94, 197, 34)),
    (0, 1, (22, 115, 249)),  (1, 2, (22, 115, 249)),  (2, 3, (22, 115, 249)),
    (0, 7, (246, 130, 59)),  (7, 8, (246, 130, 59)),  (8, 9, (246, 130, 59)),  (9, 10, (246, 130, 59)),
    (8, 11, (94, 197, 34)),  (11, 12, (94, 197, 34)), (12, 13, (94, 197, 34)),
    (8, 14, (22, 115, 249)), (14, 15, (22, 115, 249)),(15, 16, (22, 115, 249)),
    (17, 18, (255, 255, 255)),(18, 19, (255, 255, 255)),(19, 20, (255, 255, 255)),(20, 21, (255, 255, 255)),
]
KP_COLORS_BGR = [
    (246,130,59),(22,115,249),(22,115,249),(22,115,249),
    (94,197,34),(94,197,34),(94,197,34),
    (246,130,59),(246,130,59),(246,130,59),(246,130,59),
    (94,197,34),(94,197,34),(94,197,34),
    (22,115,249),(22,115,249),(22,115,249),
    (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
]
MIN_KP_SCORE = 0.3

def draw_skeleton_cv(frame, kps, kp_scores):
    for fr, to, color in SKELETON_BONES:
        if kp_scores[fr] < MIN_KP_SCORE or kp_scores[to] < MIN_KP_SCORE:
            continue
        pt1 = (int(kps[fr][0]), int(kps[fr][1]))
        pt2 = (int(kps[to][0]), int(kps[to][1]))
        cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

    for i, (kp, score) in enumerate(zip(kps, kp_scores)):
        if score < MIN_KP_SCORE:
            continue
        center = (int(kp[0]), int(kp[1]))
        radius = 7 if i == 10 else 5
        cv2.circle(frame, center, radius, KP_COLORS_BGR[i], -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, (0, 0, 0), 2, cv2.LINE_AA)

def reencode_h264(src):
    """Re-encodes to H.264 with ffmpeg for browser compatibility."""
    tmp = src + ".tmp.mp4"
    os.rename(src, tmp)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", src],
            check=True, capture_output=True,
        )
        os.remove(tmp)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.rename(tmp, src)

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    output_path = os.path.join(RESULT_DIR, f"{file_id}_out.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    engine.kps_buffer = []
    engine.prev_kps_2d = None
    engine.prev_kps_3d = None
    engine.prev_time = None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return JSONResponse({"success": False, "error": "No se pudo abrir el video"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    all_stats = []
    peak_club_speed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kps_2d, kp_scores, stats = engine.process_frame(frame, fps=fps)
        if kps_2d is not None:
            draw_skeleton_cv(frame, kps_2d, kp_scores)
            all_stats.append(stats)
            if stats.get("club_speed") is not None:
                peak_club_speed = max(peak_club_speed, stats["club_speed"])

        writer.write(frame)

    cap.release()
    writer.release()
    reencode_h264(output_path)

    final_stats = {}
    if all_stats:
        last = all_stats[-1]
        final_stats = {
            "shoulder_angle": last.get("right_shoulder_angle", 0),
            "knee_angle": last.get("right_knee_angle", 0),
            "club_speed": peak_club_speed if peak_club_speed > 0 else "--",
            "right_knee_angle": last.get("right_knee_angle", 0),
            "left_knee_angle": last.get("left_knee_angle", 0),
            "right_shoulder_angle": last.get("right_shoulder_angle", 0),
            "left_shoulder_angle": last.get("left_shoulder_angle", 0),
            "right_elbow_angle": last.get("right_elbow_angle", 0),
            "left_elbow_angle": last.get("left_elbow_angle", 0),
            "spine_angle": last.get("spine_angle", 0),
        }

    return {
        "success": True,
        "output_url": f"/results/{file_id}_out.mp4",
        "stats": final_stats,
    }

@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    kps_2d, kp_scores, stats = engine.process_frame(img)
    
    if kps_2d is None:
        return {"success": False, "error": stats.get("error", "Sin detección")}
    
    return {
        "success": True,
        "keypoints": kps_2d,
        "scores": kp_scores,
        "stats": stats
    }

# Servir estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

@app.get("/test_response")
async def test_response():
    """Endpoint de prueba para verificar formato de respuesta."""
    return {
        "success": True,
        "keypoints": [[100, 200]] * 22,
        "stats": {
            "mode": "2D",
            "buffer": 5,
            "buffer_ready": False,
            "right_knee_angle": 158.2,
            "left_knee_angle": 155.0,
            "right_shoulder_angle": 42.5,
            "left_shoulder_angle": 40.1,
            "right_elbow_angle": 120.3,
            "left_elbow_angle": 118.7,
            "spine_angle": 170.5,
        }
    }

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
