import torch
import numpy as np
import cv2
import os
import sys
import pickle
import time

# Asegurar que el directorio raíz esté en el path para importar 'common'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    from mmengine.registry import DefaultScope
    ERROR_ML = None
except ImportError as e:
    ERROR_ML = str(e)

from common.model_cross import MixSTE2

class GolfInferenceEngine:
    NUM_FRAMES = 27
    NUM_JOINTS = 22

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ready = False
        self.error = ERROR_ML
        self.kps_buffer = []
        self.last_kps_3d = None
        self.prev_kps_2d = None
        self.prev_kps_3d = None
        self.prev_time = None
        self.last_3d_time = 0
        self.MIN_3D_INTERVAL = 0.5 if self.device == 'cuda' else 2.0

        self.swing_state = "idle"
        self.swing_frames = []
        self.swing_summary = None
        self.still_count = 0
        self.cool_count = 0
        self._still_start = None
        self._done_time = 0

        if self.error:
            print(f"⚠️ Error cargando librerías ML: {self.error}")
            return

        print(f"🖥️ Dispositivo seleccionado: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        det_dir = 'entrenamiento_kaggle/work_dirs/detector_yolox_2cls'
        self.det_config = os.path.join(det_dir, 'golfpose_detector_2cls_yolox_s.py')
        
        # Intentar determinar el checkpoint dinámicamente desde 'last_checkpoint'
        last_cp_path = os.path.join(det_dir, 'last_checkpoint')
        if os.path.exists(last_cp_path):
            try:
                with open(last_cp_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        checkpoint_name = os.path.basename(content)
                        potential_path = os.path.join(det_dir, checkpoint_name)
                        
                        if os.path.exists(potential_path):
                            self.det_checkpoint = potential_path
                        else:
                            # Búsqueda recursiva por si está dentro de una carpeta de fecha
                            print(f"⚠️ {checkpoint_name} no está en la raíz, buscando en subdirectorios...")
                            found = False
                            for root, dirs, files in os.walk(det_dir):
                                if checkpoint_name in files:
                                    self.det_checkpoint = os.path.join(root, checkpoint_name)
                                    found = True
                                    break
                            
                            if found:
                                print(f"✨ Encontrado en: {self.det_checkpoint}")
                            else:
                                print(f"❌ No se encontró {checkpoint_name} en ninguna parte de {det_dir}")
                                self.det_checkpoint = os.path.join(det_dir, 'best_coco_bbox_mAP_epoch_22.pth')
                        
                        print(f"🔍 Checkpoint final seleccionado: {self.det_checkpoint}")
                    else:
                        raise ValueError("last_checkpoint está vacío")
            except Exception as e:
                print(f"⚠️ No se pudo leer last_checkpoint, usando valor por defecto: {e}")
                self.det_checkpoint = os.path.join(det_dir, 'best_coco_bbox_mAP_epoch_22.pth')
        else:
            self.det_checkpoint = os.path.join(det_dir, 'best_coco_bbox_mAP_epoch_22.pth')

        self.pose2d_config = 'configs/mmpose/golfpose_golfer_hrnetw48.py'
        self.pose2d_checkpoint = 'entrenamiento_kaggle/work_dirs/pose2d_hrnet/pose2d_best.pth'
        self.pose3d_checkpoint = 'entrenamiento_kaggle/work_dirs/pose3d_golfpose/pose3d_epoch80.bin'

        try:
            print("📦 Cargando Detector YOLOX...")
            self.detector = init_detector(self.det_config, self.det_checkpoint, device=self.device)

            print("📦 Cargando Pose 2D HRNet...")
            self.pose_model = init_model(self.pose2d_config, self.pose2d_checkpoint, device=self.device)

            print("📦 Cargando Pose 3D MixSTE2...")
            self.model_pos = MixSTE2(
                num_frame=self.NUM_FRAMES, num_joints=self.NUM_JOINTS, in_chans=2,
                embed_dim_ratio=512, depth=8, num_heads=8, mlp_ratio=2.,
                qkv_bias=True, qk_scale=None, drop_path_rate=0,
            )

            import pickle as _std_pickle
            from types import ModuleType as _ModuleType

            class _RandomDummy:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): return _RandomDummy()
                def __setstate__(self, s): pass

            class _SafeUnpickler(_std_pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('numpy.random'):
                        return _RandomDummy
                    return super().find_class(module, name)

            _safe_pkl = _ModuleType('_safe_pkl')
            _safe_pkl.Unpickler = _SafeUnpickler
            _safe_pkl.UnpicklingError = _std_pickle.UnpicklingError

            try:
                checkpoint = torch.load(
                    self.pose3d_checkpoint, map_location=self.device,
                    weights_only=False, pickle_module=_safe_pkl,
                )
                state_dict = checkpoint['model_pos'] if isinstance(checkpoint, dict) and 'model_pos' in checkpoint else checkpoint
                new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
                self.model_pos.load_state_dict(new_state_dict, strict=False)
                print("✅ Pose 3D cargada (Parche de compatibilidad activado).")
            except Exception as e:
                print(f"❌ Error crítico en carga 3D: {e}")

            self.model_pos.to(self.device)
            self.model_pos.eval()
            self.ready = True
            print("✅ Todos los modelos cargados correctamente.")
        except Exception as e:
            self.error = f"Error cargando checkpoints: {str(e)}"
            print(f"❌ {self.error}")

    def process_frame(self, img, fps=None):
        if not self.ready:
            return None, None, {"error": self.error or "Motores no listos"}

        with DefaultScope.overwrite_default_scope('mmdet'):
            det_results = inference_detector(self.detector, img)

        bboxes = det_results.pred_instances.bboxes.cpu().numpy()
        scores = det_results.pred_instances.scores.cpu().numpy()
        labels = det_results.pred_instances.labels.cpu().numpy()

        # Filtrar detecciones con score mínimo
        valid_mask = scores > 0.3
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        # Buscar la mejor detección de persona (clase 0)
        person_mask = labels == 0
        if not np.any(person_mask):
            return None, None, {"error": "No se detectó golfista"}

        person_bboxes = bboxes[person_mask]
        person_scores = scores[person_mask]
        best_idx = int(np.argmax(person_scores))
        bbox = person_bboxes[best_idx]

        # Expandir bbox un 20% para capturar extremidades y palo
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        expand = 1.2
        h_img, w_img = img.shape[:2]
        bbox_expanded = np.array([
            max(0, cx - w * expand / 2),
            max(0, cy - h * expand / 2),
            min(w_img, cx + w * expand / 2),
            min(h_img, cy + h * expand / 2),
        ])

        pose_results = inference_topdown(self.pose_model, img, bboxes=[bbox_expanded])
        data_samples = merge_data_samples(pose_results)
        keypoints = data_samples.pred_instances.keypoints[0]  # [22, 2]
        kp_scores = data_samples.pred_instances.keypoint_scores[0]  # [22]

        self.kps_buffer.append(keypoints.copy())
        if len(self.kps_buffer) > self.NUM_FRAMES:
            self.kps_buffer.pop(0)

        kps_3d = None
        buffer_count = len(self.kps_buffer)

        now = time.time()
        if buffer_count >= self.NUM_FRAMES:
            if fps is not None:
                kps_3d = self.lift_to_3d()
            elif now - self.last_3d_time >= self.MIN_3D_INTERVAL:
                kps_3d = self.lift_to_3d()
                self.last_3d_time = now
            else:
                kps_3d = self.last_kps_3d
        if fps is not None:
            effective_fps = fps
        elif self.prev_time is not None:
            effective_fps = 1.0 / max(now - self.prev_time, 0.001)
        else:
            effective_fps = 30.0

        stats = self._compute_stats(keypoints, kps_3d, effective_fps)
        stats["buffer"] = buffer_count
        stats["buffer_ready"] = buffer_count >= self.NUM_FRAMES

        club_prop = self._get_club_proportional_speed(keypoints, kps_3d)
        self._update_swing_state(club_prop, stats)
        stats["swing_state"] = self.swing_state
        print(f"   🎯 mov={club_prop:.3f} estado={self.swing_state} quieto={self.still_count}/{self.STILL_FRAMES_NEEDED}")
        if self.swing_state == "done" and self.swing_summary is not None:
            stats["swing_summary"] = self.swing_summary

        self.prev_kps_2d = keypoints.copy()
        self.prev_kps_3d = kps_3d.copy() if kps_3d is not None else None
        self.prev_time = now

        return keypoints.tolist(), kp_scores.tolist(), stats

    def lift_to_3d(self):
        frames = np.array(self.kps_buffer[-self.NUM_FRAMES:])  # (27, 22, 2)

        root = frames[:, 0:1, :]  # (27, 1, 2) — joint 0 = root
        centered = frames - root

        scale = np.abs(centered).max() + 1e-6
        normalized = centered / scale

        inp = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)  # (1, 27, 22, 2)

        with torch.no_grad():
            out = self.model_pos(inp)  # (1, 27, 22, 3)

        kps_3d = out[0, -1].cpu().numpy()  # último frame: (22, 3)
        self.last_kps_3d = kps_3d
        return kps_3d

    REAL_TORSO_M = 0.50

    def _compute_stats(self, kps_2d, kps_3d, fps=30.0):
        stats = {}

        kps = kps_3d if kps_3d is not None else kps_2d
        use_3d = kps_3d is not None

        if use_3d:
            stats["mode"] = "3D"
            stats["kps_3d"] = kps_3d.tolist()
        else:
            stats["mode"] = "2D"

        def angle_3pts(a, b, c):
            ba = a - b
            bc = c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        stats["right_knee_angle"] = round(angle_3pts(kps[1], kps[2], kps[3]), 1)
        stats["left_knee_angle"] = round(angle_3pts(kps[4], kps[5], kps[6]), 1)
        stats["right_shoulder_angle"] = round(angle_3pts(kps[9], kps[14], kps[15]), 1)
        stats["left_shoulder_angle"] = round(angle_3pts(kps[9], kps[11], kps[12]), 1)
        stats["right_elbow_angle"] = round(angle_3pts(kps[14], kps[15], kps[16]), 1)
        stats["left_elbow_angle"] = round(angle_3pts(kps[11], kps[12], kps[13]), 1)
        stats["spine_angle"] = round(angle_3pts(kps[0], kps[7], kps[8]), 1)

        stats["club_speed"] = self._compute_club_speed(kps_2d, kps_3d, fps)

        club_str = f"{stats['club_speed']} mph" if stats['club_speed'] is not None else "--"
        print(
            f"📊 [{stats['mode']}] "
            f"Rod.D={stats['right_knee_angle']}° Rod.I={stats['left_knee_angle']}° | "
            f"Hom.D={stats['right_shoulder_angle']}° Hom.I={stats['left_shoulder_angle']}° | "
            f"Cod.D={stats['right_elbow_angle']}° Cod.I={stats['left_elbow_angle']}° | "
            f"Col={stats['spine_angle']}° | Vel.Palo={club_str}"
        )

        return stats

    def _compute_club_speed(self, kps_2d, kps_3d, fps):
        HOSEL_IDX = 18

        if kps_3d is not None and self.prev_kps_3d is not None:
            kps_now = np.array(kps_3d)
            kps_prev = np.array(self.prev_kps_3d)
        elif self.prev_kps_2d is not None:
            kps_now = np.array(kps_2d)
            kps_prev = np.array(self.prev_kps_2d)
        else:
            return None

        club_disp = np.linalg.norm(kps_now[HOSEL_IDX] - kps_prev[HOSEL_IDX])

        torso_len = np.linalg.norm(kps_now[9] - kps_now[0])
        if torso_len < 1e-6:
            return None

        disp_m = club_disp * (self.REAL_TORSO_M / torso_len)
        speed_mph = disp_m * fps * 2.23694

        return round(speed_mph, 1)

    # --- Swing detection ---

    STILL_THRESH = 0.15
    SWING_THRESH = 0.40
    STILL_FRAMES_NEEDED = 3
    COOL_FRAMES_NEEDED = 3
    MIN_SWING_FRAMES = 5
    STILL_TIME_S = 3.0
    DONE_COOLDOWN_S = 15.0

    def _get_club_proportional_speed(self, kps_2d, kps_3d):
        HOSEL_IDX = 18
        if self.prev_kps_2d is None:
            return 0.0
        kps_now = np.array(kps_2d)
        kps_prev = np.array(self.prev_kps_2d)
        club_disp = np.linalg.norm(kps_now[HOSEL_IDX] - kps_prev[HOSEL_IDX])
        torso_len = np.linalg.norm(kps_now[9] - kps_now[0])
        if torso_len < 1e-6:
            return 0.0
        return club_disp / torso_len

    def _update_swing_state(self, club_prop, stats):
        now = time.time()

        if self.swing_state == "idle":
            if club_prop < self.STILL_THRESH:
                if self._still_start is None:
                    self._still_start = now
                elapsed = now - self._still_start
                stats["timer"] = round(max(self.STILL_TIME_S - elapsed, 0), 1)
                if elapsed >= self.STILL_TIME_S:
                    self.swing_state = "ready"
                    self.swing_summary = None
                    self._still_start = None
                    print("🏌️ Posición inicial detectada — Listo para swing")
            else:
                self._still_start = None
                stats["timer"] = round(self.STILL_TIME_S, 1)

        elif self.swing_state == "ready":
            if club_prop >= self.SWING_THRESH:
                self.swing_state = "active"
                self.swing_frames = [stats.copy()]
                self.cool_count = 0
                print("🔴 Swing iniciado — Grabando...")

        elif self.swing_state == "active":
            self.swing_frames.append(stats.copy())
            if club_prop < self.STILL_THRESH:
                self.cool_count += 1
                if self.cool_count >= self.COOL_FRAMES_NEEDED:
                    if len(self.swing_frames) >= self.MIN_SWING_FRAMES:
                        self._finalize_swing()
                        self.swing_state = "done"
                        self._done_time = now
                    else:
                        print(f"⏭️ Movimiento descartado — solo {len(self.swing_frames)} frames (mínimo {self.MIN_SWING_FRAMES})")
                        self.swing_state = "ready"
                        self.swing_frames = []
                        self.cool_count = 0
            else:
                self.cool_count = 0

        elif self.swing_state == "done":
            remaining = self.DONE_COOLDOWN_S - (now - self._done_time)
            stats["timer"] = round(max(remaining, 0), 1)
            if remaining <= 0:
                self.swing_state = "idle"
                self._still_start = None
                print("🔄 Cooldown terminado — Listo para otro swing")

    _SWING_KEYS = [
        "right_knee_angle", "left_knee_angle",
        "right_shoulder_angle", "left_shoulder_angle",
        "right_elbow_angle", "left_elbow_angle",
        "spine_angle",
    ]

    IDEAL_RANGES = {
        "right_knee_angle":     {"address": (150, 165), "impact": (140, 155), "finish": (160, 175)},
        "left_knee_angle":      {"address": (150, 165), "impact": (155, 170), "finish": (170, 180)},
        "right_shoulder_angle": {"address": (0, 10),    "impact": (30, 45),   "finish": (80, 100)},
        "left_shoulder_angle":  {"address": (0, 10),    "impact": (30, 45),   "finish": (80, 100)},
        "right_elbow_angle":    {"address": (150, 170), "impact": (130, 150), "finish": (160, 180)},
        "left_elbow_angle":     {"address": (150, 170), "impact": (140, 160), "finish": (160, 180)},
        "spine_angle":          {"address": (30, 40),   "impact": (25, 35),   "finish": (20, 30)},
    }

    _METRIC_NAMES = {
        "right_knee_angle": "Rodilla Der.",
        "left_knee_angle": "Rodilla Izq.",
        "right_shoulder_angle": "Hombro Der.",
        "left_shoulder_angle": "Hombro Izq.",
        "right_elbow_angle": "Codo Der.",
        "left_elbow_angle": "Codo Izq.",
        "spine_angle": "Columna",
    }

    _PHASE_NAMES = {"address": "Inicio", "impact": "Impacto", "finish": "Final"}

    def _finalize_swing(self):
        if not self.swing_frames:
            return

        address = self.swing_frames[0]
        finish = self.swing_frames[-1]

        peak_speed = 0
        impact = address
        for f in self.swing_frames:
            spd = f.get("club_speed")
            if spd is not None and spd > peak_speed:
                peak_speed = spd
                impact = f

        extract = lambda fr: {k: fr.get(k) for k in self._SWING_KEYS}

        self.swing_summary = {
            "peak_club_speed": peak_speed if peak_speed > 0 else None,
            "total_frames": len(self.swing_frames),
            "address": extract(address),
            "impact": extract(impact),
            "finish": extract(finish),
        }

        self.swing_summary["evaluation"] = self._evaluate_swing(self.swing_summary)

        spd_str = f"{peak_speed} mph" if peak_speed > 0 else "--"
        ev = self.swing_summary["evaluation"]
        v_icon = {"bueno": "🟢", "aceptable": "🟡", "malo": "🔴"}[ev["verdict"]]

        print("=" * 60)
        print(f"⛳ SWING COMPLETO — {len(self.swing_frames)} frames")
        print(f"   💨 Vel. Pico Palo: {spd_str}")
        print(f"   📍 Inicio  → Rod.D={address.get('right_knee_angle')}° Hom.D={address.get('right_shoulder_angle')}° Col={address.get('spine_angle')}°")
        print(f"   💥 Impacto → Rod.D={impact.get('right_knee_angle')}° Hom.D={impact.get('right_shoulder_angle')}° Col={impact.get('spine_angle')}°")
        print(f"   🏁 Final   → Rod.D={finish.get('right_knee_angle')}° Hom.D={finish.get('right_shoulder_angle')}° Col={finish.get('spine_angle')}°")
        print(f"   📈 Score: {ev['score']}% ({ev['in_range']}/{ev['total']} en rango)")
        print(f"   {v_icon} Veredicto: SWING {ev['verdict'].upper()}")
        for p in ev["problems"]:
            print(f"   ⚠️ {p}")
        print("=" * 60)

    _FEEDBACK = {
        ("right_knee_angle", "address", "low"):  "Flexionaste demasiado la rodilla derecha al inicio — postura inestable",
        ("right_knee_angle", "address", "high"): "Pierna derecha muy recta al inicio — necesitas más flexión para generar potencia",
        ("right_knee_angle", "impact", "low"):   "Tu rodilla derecha colapsó en el impacto — pierdes potencia y equilibrio",
        ("right_knee_angle", "impact", "high"):  "Pierna derecha rígida en el impacto — no transferiste el peso hacia adelante",
        ("right_knee_angle", "finish", "low"):   "Rodilla derecha muy doblada al final — falta estabilidad en el seguimiento",
        ("right_knee_angle", "finish", "high"):  "Pierna derecha bloqueada al final — falta fluidez en el finish",

        ("left_knee_angle", "address", "low"):   "Rodilla izquierda muy flexionada al inicio — desbalance en la postura",
        ("left_knee_angle", "address", "high"):  "Pierna izquierda muy recta al inicio — necesitas más flex de rodilla",
        ("left_knee_angle", "impact", "low"):    "Tu rodilla izquierda colapsó en el impacto — base inestable, pierdes dirección",
        ("left_knee_angle", "impact", "high"):   "Pierna izquierda rígida en el impacto — no rotaste la cadera correctamente",
        ("left_knee_angle", "finish", "low"):    "Rodilla izquierda muy doblada al final — pérdida de balance",
        ("left_knee_angle", "finish", "high"):   "Pierna izquierda se extendió correctamente pero demasiado recta",

        ("right_shoulder_angle", "address", "high"): "Hombro derecho ya rotado antes de empezar — corrige tu postura inicial",
        ("right_shoulder_angle", "impact", "low"):   "Poca rotación de hombro derecho en el impacto — swing débil, falta potencia",
        ("right_shoulder_angle", "impact", "high"):  "Rotaste demasiado el hombro derecho — pierdes control del palo",
        ("right_shoulder_angle", "finish", "low"):   "El hombro derecho no completó la rotación — swing incompleto",
        ("right_shoulder_angle", "finish", "high"):  "Sobre-rotación del hombro derecho al final — riesgo de lesión",

        ("left_shoulder_angle", "address", "high"):  "Hombro izquierdo ya rotado al inicio — alinea los hombros con la bola",
        ("left_shoulder_angle", "impact", "low"):    "Poca rotación de hombro izquierdo en el impacto — falta potencia",
        ("left_shoulder_angle", "impact", "high"):   "Rotaste demasiado el hombro izquierdo — pierdes precisión",
        ("left_shoulder_angle", "finish", "low"):    "El hombro izquierdo no completó el giro — finish incompleto",
        ("left_shoulder_angle", "finish", "high"):   "Sobre-rotación del hombro izquierdo al final — pérdida de control",

        ("right_elbow_angle", "address", "low"):  "Codo derecho muy doblado al inicio — brazos deben estar más extendidos",
        ("right_elbow_angle", "address", "high"): "Codo derecho bloqueado al inicio — necesitas un poco de flex natural",
        ("right_elbow_angle", "impact", "low"):   "Codo derecho muy flexionado en el impacto — swing 'quebrado', pierdes alcance",
        ("right_elbow_angle", "impact", "high"):  "Codo derecho muy rígido en el impacto — falta fluidez, riesgo de slice",
        ("right_elbow_angle", "finish", "low"):   "Codo derecho muy doblado al final — no completaste la extensión",
        ("right_elbow_angle", "finish", "high"):  "Codo derecho hiperextendido al final — cuidado con lesiones",

        ("left_elbow_angle", "address", "low"):   "Codo izquierdo muy doblado al inicio — el brazo guía debe estar más recto",
        ("left_elbow_angle", "address", "high"):  "Codo izquierdo bloqueado al inicio — relaja un poco el brazo",
        ("left_elbow_angle", "impact", "low"):    "Codo izquierdo se dobló en el impacto — 'chicken wing', pierdes distancia",
        ("left_elbow_angle", "impact", "high"):   "Codo izquierdo muy rígido en el impacto — falta naturalidad",
        ("left_elbow_angle", "finish", "low"):    "Codo izquierdo muy doblado al final — el brazo no se extendió bien",
        ("left_elbow_angle", "finish", "high"):   "Codo izquierdo hiperextendido al final — cuidado con la articulación",

        ("spine_angle", "address", "low"):  "Espalda demasiado inclinada al inicio — riesgo de topping y slices",
        ("spine_angle", "address", "high"): "Postura muy erguida al inicio — inclínate más desde la cadera",
        ("spine_angle", "impact", "low"):   "Te inclinaste demasiado en el impacto — pierdes balance y dirección",
        ("spine_angle", "impact", "high"):  "Te levantaste en el impacto — 'early extension', causa tops y chunks",
        ("spine_angle", "finish", "low"):   "Espalda muy inclinada al final — posible pérdida de balance",
        ("spine_angle", "finish", "high"):  "Torso muy erguido al final — falta seguimiento natural del cuerpo",
    }

    def _evaluate_swing(self, summary):
        phases = {
            "address": summary["address"],
            "impact": summary["impact"],
            "finish": summary["finish"],
        }

        total = 0
        in_range = 0
        problems = []

        for metric, ranges in self.IDEAL_RANGES.items():
            for phase, (lo, hi) in ranges.items():
                val = phases[phase].get(metric)
                if val is None:
                    continue
                total += 1
                if lo <= val <= hi:
                    in_range += 1
                else:
                    direction = "low" if val < lo else "high"
                    fb = self._FEEDBACK.get((metric, phase, direction))
                    if fb:
                        problems.append(f"{fb} ({val}°, ideal {lo}°–{hi}°)")
                    else:
                        name = self._METRIC_NAMES[metric]
                        pname = self._PHASE_NAMES[phase]
                        problems.append(f"{name} en {pname}: {val}° (ideal {lo}°–{hi}°)")

        score = round((in_range / total) * 100) if total > 0 else 0

        for metric in self.IDEAL_RANGES:
            a_val = phases["address"].get(metric)
            i_val = phases["impact"].get(metric)
            if a_val is not None and i_val is not None:
                delta = abs(i_val - a_val)
                if delta > 40:
                    name = self._METRIC_NAMES[metric]
                    problems.append(f"{name}: cambio brusco de {delta:.0f}° entre inicio e impacto — movimiento descontrolado")

        sym_pairs = [
            ("right_knee_angle", "left_knee_angle", "Rodillas"),
            ("right_shoulder_angle", "left_shoulder_angle", "Hombros"),
            ("right_elbow_angle", "left_elbow_angle", "Codos"),
        ]
        for r, l, name in sym_pairs:
            for phase in ["address", "impact", "finish"]:
                r_val = phases[phase].get(r)
                l_val = phases[phase].get(l)
                if r_val is not None and l_val is not None:
                    diff = abs(r_val - l_val)
                    if diff > 20:
                        pname = self._PHASE_NAMES[phase]
                        problems.append(f"{name} desbalanceadas en {pname}: {diff:.0f}° de diferencia — distribuye mejor el peso")

        has_critical = False
        for metric in ["spine_angle", "right_knee_angle", "left_knee_angle"]:
            val = phases["impact"].get(metric)
            if val is None:
                continue
            lo, hi = self.IDEAL_RANGES[metric]["impact"]
            if val < lo - 15 or val > hi + 15:
                has_critical = True

        has_collapse = any("descontrolado" in p for p in problems)

        if score >= 80 and not has_critical and not has_collapse:
            verdict = "bueno"
        elif score >= 65 and not has_critical:
            verdict = "aceptable"
        else:
            verdict = "malo"

        return {
            "score": score,
            "in_range": in_range,
            "total": total,
            "verdict": verdict,
            "problems": problems[:8],
        }
