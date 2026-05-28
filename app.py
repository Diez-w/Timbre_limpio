import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
import requests
import urllib.parse

# --- Configuración WhatsApp ---
WHATSAPP_PHONE = "+51902697385"
CALLMEBOT_API_KEY = "2408114"

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- Clasificadores OpenCV ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# --- Modelo LBPH global ---
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)
label_map     = {}
model_trained = False
pending_batches = {}


# ─────────────────────────────────────────────
#  PREPROCESAMIENTO
# ─────────────────────────────────────────────

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
#  DETECCIÓN DE ROSTRO
# ─────────────────────────────────────────────

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
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
#  CARGA Y ENTRENAMIENTO LBPH
# ─────────────────────────────────────────────

def load_known_faces():
    global label_map, model_trained
    faces_list, labels_list = [], []
    name_to_label, label_map = {}, {}
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
        img = cv2.imread(os.path.join(BASE_ROSTROS_FOLDER, filename))
        if img is None:
            continue
        img_proc = preprocess_image(img)
        face_roi, _ = detect_face(img_proc)
        if face_roi is None:
            logging.warning(f"Sin rostro en: {filename}")
            continue
        nombre = filename.rsplit('.', 1)[0].split('_')[0]
        if nombre not in name_to_label:
            name_to_label[nombre] = current_label
            label_map[current_label] = nombre
            current_label += 1
        faces_list.append(face_roi)
        labels_list.append(name_to_label[nombre])
        logging.info(f"Foto cargada: {filename} → persona '{nombre}'")

    if not faces_list:
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
#  RECONOCIMIENTO FACIAL
# ─────────────────────────────────────────────

def recognize_face(img, confidence_threshold=150):
    if not model_trained:
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
        logging.warning(f"Distancia {distance:.2f} > umbral → NO reconocido")
        return None, 0, face_rect
    nombre = label_map.get(label, "Desconocido")
    confianza_pct = max(0, round((1 - distance / confidence_threshold) * 100, 2))
    logging.info(f"Reconocido: {nombre} | Distancia: {distance:.2f} | Confianza: {confianza_pct}%")
    return nombre, confianza_pct, face_rect


# ─────────────────────────────────────────────
#  DETECCIÓN DE GUIÑO — MÉTODO COMBINADO
#
#  Estrategia de dos capas:
#
#  CAPA 1 — Haar Cascade de ojos (primario)
#  Detecta cuántos ojos hay en la banda ocular.
#  - 2 ojos detectados → ambos abiertos, sin guiño
#  - 1 ojo detectado  → posible guiño, confirmar con capa 2
#  - 0 ojos           → imagen oscura/borrosa, usar solo capa 2
#
#  CAPA 2 — Asimetría de brillo (confirmación)
#  Si el cascade detectó 1 ojo, confirma con métricas de
#  asimetría para reducir falsos positivos.
#  Si el cascade detectó 0 ojos, usa umbrales más altos
#  para evitar falsos positivos por imagen de mala calidad.
# ─────────────────────────────────────────────

def detect_wink(img, face_rect):
    if face_rect is None:
        return False

    x, y, w, h = face_rect
    if w <= 0 or h <= 0:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── CAPA 1: Haar Cascade de ojos ────────────────────────────────────
    # Recortar solo la franja de ojos (20%–50% vertical del rostro)
    eye_top    = int(y + h * 0.20)
    eye_bottom = int(y + h * 0.50)
    eye_band_gray = gray[eye_top:eye_bottom, x:x+w]

    if eye_band_gray.size == 0:
        return False

    # CLAHE en la banda para mejorar contraste de ojos pequeños
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    eye_band_gray = clahe.apply(eye_band_gray)

    eyes_detected = eye_cascade.detectMultiScale(
        eye_band_gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(15, 15)
    )
    num_eyes = len(eyes_detected)
    logging.info(f"Ojos detectados por cascade: {num_eyes}")

    # 2 ojos → ambos abiertos, no hay guiño
    if num_eyes >= 2:
        logging.info("Guiño → False (2 ojos detectados)")
        return False

    # ── CAPA 2: Asimetría de brillo ──────────────────────────────────────
    half      = eye_band_gray.shape[1] // 2
    left_eye  = eye_band_gray[:, :half]
    right_eye = eye_band_gray[:, half:]

    mean_l    = np.mean(left_eye)
    mean_r    = np.mean(right_eye)
    asymmetry = abs(mean_l - mean_r) / max(mean_l, mean_r, 1e-5)

    var_l     = np.var(left_eye)
    var_r     = np.var(right_eye)
    var_ratio = abs(var_l - var_r) / max(var_l, var_r, 1e-5)

    max_l    = float(np.max(left_eye))
    max_r    = float(np.max(right_eye))
    max_diff = abs(max_l - max_r) / max(max_l, max_r, 1e-5)

    logging.info(
        f"Asimetría → Asym={asymmetry:.3f} VarRatio={var_ratio:.3f} "
        f"MaxDiff={max_diff:.3f} | Ojos cascade={num_eyes}"
    )

    if num_eyes == 1:
        # Cascade detectó 1 ojo → probablemente hay guiño.
        # Umbrales BAJOS porque el cascade ya da alta confianza.
        wink = (
            (asymmetry > 0.05 and var_ratio > 0.15) or
            (asymmetry > 0.05 and max_diff > 0.08)  or
            (var_ratio > 0.25)
        )
    else:
        # Cascade detectó 0 ojos → imagen problemática.
        # Umbrales ALTOS para evitar falsos positivos.
        wink = (
            (asymmetry > 0.08 and var_ratio > 0.28 and max_diff > 0.09) or
            (var_ratio > 0.40)
        )

    logging.info(f"Guiño → {wink}")
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
        "modelo_guino": "Haar Cascade ojos + asimetría",
        "personas_registradas": len(label_map),
        "nombres": list(label_map.values()),
        "modelo_entrenado": model_trained
    })


@app.route("/recargar_rostros", methods=["POST"])
def recargar_rostros():
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
            return jsonify({"error": "Error decodificando imagen", "activar_rele": False}), 400

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
                "count": 0, "results": [], "final_notified": False
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
