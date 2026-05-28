import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
import requests
import urllib.parse
import mediapipe as mp

# --- Configuración WhatsApp ---
WHATSAPP_PHONE = "+51902697385"
CALLMEBOT_API_KEY = "2408114"

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- Clasificadores OpenCV ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Modelo LBPH global ---
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)
label_map = {}
model_trained = False
pending_batches = {}

# ─────────────────────────────────────────────
#  MEDIAPIPE — detector de landmarks faciales
# ─────────────────────────────────────────────
# FaceMesh detecta 468 puntos del rostro incluyendo contornos
# detallados de ambos ojos (16 puntos por ojo).
# Se inicializa una sola vez al arrancar para no pagar el costo
# de inicialización en cada petición.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,    # True para imágenes individuales (no video)
    max_num_faces=1,           # Solo el rostro más prominente
    refine_landmarks=False,    # Sin landmarks extra de iris (ahorra RAM y CPU)
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ── Índices de los puntos del contorno de cada ojo en FaceMesh ──────────
# Estos son los 6 puntos estándar usados para calcular EAR (Eye Aspect Ratio)
# extraídos del mapa de 468 landmarks de MediaPipe.
# Ojo izquierdo (desde la perspectiva del observador = ojo derecho de la persona)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Ojo derecho (desde la perspectiva del observador = ojo izquierdo de la persona)
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Umbral EAR: por debajo de este valor el ojo se considera cerrado.
# Ojo abierto normal: EAR ≈ 0.25–0.35
# Ojo medio cerrado:  EAR ≈ 0.18–0.25
# Ojo cerrado (guiño):EAR < 0.18
EAR_THRESHOLD = 0.20


# ─────────────────────────────────────────────
#  PREPROCESAMIENTO
# ─────────────────────────────────────────────

def preprocess_image(img):
    """
    Mejora la calidad de la imagen usando CLAHE y filtro bilateral.
    Normaliza diferencias entre cámaras (ESP32-CAM vs cámaras de smartphone).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
#  DETECCIÓN DE ROSTRO (para LBPH)
# ─────────────────────────────────────────────

def detect_face(img):
    """
    Detecta el rostro principal usando Haar Cascade.
    Retorna (face_roi_gray, face_rect) o (None, None).
    Parámetros relajados para imágenes del ESP32-CAM.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        logging.warning(
            f"No se detectó rostro. Tamaño: {img.shape[1]}×{img.shape[0]} | "
            f"Brillo medio: {np.mean(gray):.1f}"
        )
        return None, None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    logging.info(f"Rostro detectado: x={x} y={y} w={w} h={h}")

    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_gray = cv2.resize(face_roi_gray, (100, 100))
    return face_roi_gray, (x, y, w, h)


# ─────────────────────────────────────────────
#  CARGA Y ENTRENAMIENTO CON LBPH
# ─────────────────────────────────────────────

def load_known_faces():
    """
    Lee todas las fotos de base_rostros/ y entrena el modelo LBPH.
    Convención: nombre_1.jpg, nombre_2.jpg → persona 'nombre'
    """
    global label_map, model_trained

    faces_list    = []
    labels_list   = []
    name_to_label = {}
    label_map     = {}
    current_label = 0

    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning("Carpeta base_rostros/ no encontrada.")
        return

    archivos = [
        f for f in os.listdir(BASE_ROSTROS_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not archivos:
        logging.warning("No hay fotos en base_rostros/.")
        return

    for filename in archivos:
        img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"No se pudo leer: {filename}")
            continue

        img_proc = preprocess_image(img)
        face_roi, _ = detect_face(img_proc)

        if face_roi is None:
            logging.warning(f"No se detectó rostro en: {filename}")
            continue

        base_name = filename.rsplit('.', 1)[0]
        nombre    = base_name.split('_')[0]

        if nombre not in name_to_label:
            name_to_label[nombre] = current_label
            label_map[current_label] = nombre
            current_label += 1

        faces_list.append(face_roi)
        labels_list.append(name_to_label[nombre])
        logging.info(f"Foto cargada: {filename} → persona '{nombre}'")

    if len(faces_list) == 0:
        logging.error("Ninguna foto válida para entrenar.")
        model_trained = False
        return

    recognizer.train(faces_list, np.array(labels_list))
    model_trained = True
    logging.info(
        f"Modelo LBPH entrenado con {len(faces_list)} fotos "
        f"de {len(label_map)} personas: {list(label_map.values())}"
    )


# ─────────────────────────────────────────────
#  RECONOCIMIENTO FACIAL (LBPH)
# ─────────────────────────────────────────────

def recognize_face(img, confidence_threshold=150):
    """
    Reconoce el rostro usando LBPH.
    Distancia LBPH: menor = mejor coincidencia.
    Umbral 150 para tolerar diferencia ESP32-CAM vs smartphone.
    """
    if not model_trained:
        logging.warning("Modelo LBPH no entrenado.")
        return None, 0, None

    img_proc = preprocess_image(img)
    face_roi, face_rect = detect_face(img_proc)

    if face_roi is None:
        return None, 0, None

    label, distance = recognizer.predict(face_roi)

    logging.info(
        f"LBPH predict → label={label} ({label_map.get(label,'?')}) "
        f"distancia={distance:.2f} umbral={confidence_threshold}"
    )

    if distance > confidence_threshold:
        logging.warning(
            f"Distancia {distance:.2f} > umbral {confidence_threshold} → NO reconocido"
        )
        return None, 0, face_rect

    nombre = label_map.get(label, "Desconocido")
    confianza_pct = max(0, round((1 - distance / confidence_threshold) * 100, 2))
    logging.info(f"Reconocido: {nombre} | Distancia: {distance:.2f} | Confianza: {confianza_pct}%")
    return nombre, confianza_pct, face_rect


# ─────────────────────────────────────────────
#  CÁLCULO EAR (Eye Aspect Ratio)
# ─────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Calcula el EAR (Eye Aspect Ratio) para un ojo dado.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Donde p1-p6 son los 6 puntos del contorno del ojo en orden:
    p1=esquina izquierda, p2=superior izquierdo, p3=superior derecho,
    p4=esquina derecha,   p5=inferior derecho,   p6=inferior izquierdo.

    Un ojo abierto tiene EAR ≈ 0.25–0.35.
    Un ojo cerrado (guiño) tiene EAR < 0.20.
    """
    def point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    p1 = point(eye_indices[0])
    p2 = point(eye_indices[1])
    p3 = point(eye_indices[2])
    p4 = point(eye_indices[3])
    p5 = point(eye_indices[4])
    p6 = point(eye_indices[5])

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-5:
        return 0.30  # valor neutro si el denominador es 0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# ─────────────────────────────────────────────
#  DETECCIÓN DE GUIÑO (MediaPipe EAR)
# ─────────────────────────────────────────────

def detect_wink(img, face_rect):
    """
    Detecta guiño usando MediaPipe FaceMesh + EAR.

    Lógica:
    - Se calculan EAR del ojo izquierdo y ojo derecho.
    - Si UN solo ojo tiene EAR < EAR_THRESHOLD y el otro está abierto
      (EAR > EAR_THRESHOLD + margen) → es guiño.
    - Si AMBOS ojos tienen EAR bajo → es parpadeo normal, no guiño.

    Ventaja sobre el método anterior de asimetría de brillo:
    - Simétrico: detecta igual ojo izquierdo y derecho.
    - No afectado por iluminación lateral.
    - No genera falsos positivos por movimiento de cabeza.
    """
    if face_rect is None:
        return False

    h_img, w_img = img.shape[:2]

    # Convertir a RGB que es lo que espera MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Reducir resolución antes de MediaPipe ───────────────────────────
    # El ESP32 ya envía QVGA (320×240) pero si viene de smartphone puede
    # ser más grande. MediaPipe procesa igual de bien con 320×240 y
    # tarda mucho menos en CPU limitada.
    h_proc, w_proc = img_rgb.shape[:2]
    if w_proc > 320 or h_proc > 240:
        img_rgb = cv2.resize(img_rgb, (320, 240))
        h_proc, w_proc = 240, 320

    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        # MediaPipe no encontró landmarks — fallback al método de asimetría
        logging.warning("MediaPipe no detectó landmarks, usando fallback de asimetría")
        return detect_wink_fallback(img, face_rect)

    landmarks = results.multi_face_landmarks[0].landmark

    ear_left  = eye_aspect_ratio(landmarks, LEFT_EYE,  w_proc, h_proc)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w_proc, h_proc)

    # Margen de separación: para considerar guiño, el ojo abierto debe
    # tener EAR claramente mayor que el umbral (no justo en el límite)
    OPEN_MARGIN = 0.05

    left_closed  = ear_left  < EAR_THRESHOLD
    right_closed = ear_right < EAR_THRESHOLD
    left_open    = ear_left  > (EAR_THRESHOLD + OPEN_MARGIN)
    right_open   = ear_right > (EAR_THRESHOLD + OPEN_MARGIN)

    # Guiño = exactamente un ojo cerrado y el otro claramente abierto
    wink = (left_closed and right_open) or (right_closed and left_open)

    logging.info(
        f"EAR → izquierdo={ear_left:.3f} derecho={ear_right:.3f} "
        f"umbral={EAR_THRESHOLD} → guiño={wink}"
    )
    return wink


def detect_wink_fallback(img, face_rect):
    """
    Método de respaldo basado en asimetría de brillo.
    Se usa solo cuando MediaPipe no detecta landmarks
    (imagen muy oscura, rostro muy pequeño, etc).
    """
    if face_rect is None:
        return False

    x, y, w, h = face_rect
    face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return False

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_face = clahe.apply(gray_face)

    eye_band = gray_face[int(h * 0.20):int(h * 0.50), :]
    if eye_band.size == 0:
        return False

    half      = eye_band.shape[1] // 2
    left_eye  = eye_band[:, :half]
    right_eye = eye_band[:, half:]

    mean_l    = np.mean(left_eye)
    mean_r    = np.mean(right_eye)
    asymmetry = abs(mean_l - mean_r) / max(mean_l, mean_r, 1e-5)

    var_l     = np.var(left_eye)
    var_r     = np.var(right_eye)
    var_ratio = abs(var_l - var_r) / max(var_l, var_r, 1e-5)

    max_l    = float(np.max(left_eye))
    max_r    = float(np.max(right_eye))
    max_diff = abs(max_l - max_r) / max(max_l, max_r, 1e-5)

    wink = (
        (asymmetry > 0.06 and var_ratio > 0.18 and max_diff > 0.09) or
        (asymmetry > 0.08 and var_ratio > 0.28) or
        (asymmetry > 0.08 and max_diff > 0.09) or
        (var_ratio > 0.35)
    )

    logging.info(
        f"Fallback → Asym={asymmetry:.3f} VarRatio={var_ratio:.3f} "
        f"MaxDiff={max_diff:.3f} → {wink}"
    )
    return wink


# ─────────────────────────────────────────────
#  WHATSAPP
# ─────────────────────────────────────────────

def send_whatsapp_message(message):
    try:
        encoded = urllib.parse.quote(message)
        url = (
            f"https://api.callmebot.com/whatsapp.php"
            f"?phone={WHATSAPP_PHONE}&text={encoded}&apikey={CALLMEBOT_API_KEY}"
        )
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logging.info("WhatsApp enviado correctamente.")
        else:
            logging.error(f"Error WhatsApp HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Excepción WhatsApp: {e}")


# ─────────────────────────────────────────────
#  RUTAS FLASK
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "modelo_reconocimiento": "LBPH",
        "modelo_guino": "MediaPipe FaceMesh EAR",
        "personas_registradas": len(label_map),
        "nombres": list(label_map.values()),
        "modelo_entrenado": model_trained
    })


@app.route("/recargar_rostros", methods=["POST"])
def recargar_rostros():
    """Recarga el modelo LBPH sin reiniciar el servidor."""
    try:
        load_known_faces()
        return jsonify({
            "ok": True,
            "personas": list(label_map.values()),
            "modelo_entrenado": model_trained
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/recibir", methods=["POST"])
def recibir():
    try:
        data = request.get_json()
        if not data or 'foto' not in data or 'batch_id' not in data:
            return jsonify({
                "error": "Faltan campos: foto y batch_id",
                "activar_rele": False
            }), 400

        batch_id = data['batch_id']
        foto_b64 = data['foto']

        img_data = base64.b64decode(foto_b64)
        nparr    = np.frombuffer(img_data, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Error decodificando imagen",
                "activar_rele": False
            }), 400

        logging.info(f"Batch {batch_id} — foto recibida ({len(img_data)} bytes)")

        nombre, confianza, face_rect = recognize_face(img)

        tiene_guino = False
        if nombre:
            tiene_guino = detect_wink(img, face_rect)

        individual_response = {
            "activar_rele": bool(nombre is not None and not tiene_guino),
            "nombre":       nombre if nombre else None,
            "confianza":    float(confianza),
            "guino":        bool(tiene_guino)
        }

        if batch_id not in pending_batches:
            pending_batches[batch_id] = {
                "count": 0,
                "results": [],
                "final_notified": False
            }

        batch = pending_batches[batch_id]
        batch["count"]   += 1
        batch["results"].append(individual_response)

        if batch["count"] >= 3 and not batch["final_notified"]:
            batch["final_notified"] = True
            results = batch["results"]

            any_wink          = any(r.get("guino", False)        for r in results)
            any_recognized_ok = any(r.get("activar_rele", False) for r in results)

            if any_wink:
                names    = [r["nombre"] for r in results if r.get("nombre")]
                name_str = names[0] if names else "Desconocido"
                msg = (
                    f"⚠️ Timbre activado. Rostro reconocido: {name_str}. "
                    f"Se detectó un GUIÑO (posible emergencia)."
                )
                logging.info(f"Batch {batch_id} — resultado final: GUIÑO DETECTADO")

            elif any_recognized_ok:
                name_str = next(
                    (r["nombre"] for r in results if r.get("activar_rele") and r.get("nombre")),
                    "Reconocido"
                )
                msg = f"✅ Timbre activado. Rostro reconocido: {name_str}. Sin guiño."
                logging.info(f"Batch {batch_id} — resultado final: ACCESO PERMITIDO")

            else:
                msg = "❗ Timbre activado. Rostro NO reconocido."
                logging.info(f"Batch {batch_id} — resultado final: ACCESO DENEGADO")

            send_whatsapp_message(msg)
            del pending_batches[batch_id]

        return jsonify(individual_response), 200

    except Exception as e:
        logging.error(f"Error en /recibir: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500


# ─────────────────────────────────────────────
#  ARRANQUE
# ─────────────────────────────────────────────

load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
