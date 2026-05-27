import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
from datetime import datetime
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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Ya no usaremos eye_cascade para detectar ojos cerrados, solo para referencia si se desea
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

pending_batches = {}
known_face_histograms = {}

def preprocess_image(img):
    """Mejora la calidad de la imagen usando CLAHE y filtro bilateral."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def send_whatsapp_message(message):
    try:
        encoded_message = urllib.parse.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={encoded_message}&apikey={CALLMEBOT_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logging.info("WhatsApp enviado correctamente")
        else:
            logging.error(f"Error WhatsApp: {response.status_code}")
    except Exception as e:
        logging.error(f"Excepción WhatsApp: {e}")

def extract_face_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_small = cv2.resize(face_roi, (32, 32))
    hist = cv2.calcHist([face_small], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist, (x, y, w, h)

def load_known_faces():
    global known_face_histograms
    known_face_histograms = {}
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        return
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_proc = preprocess_image(img)
                hist, _ = extract_face_histogram(img_proc)
                if hist is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    known_face_histograms[nombre] = hist
                    logging.info(f"Rostro registrado: {nombre}")
    logging.info(f"Total rostros: {len(known_face_histograms)}")

def recognize_face_histogram(img, threshold=0.35):  # Umbral más tolerante (0.35 en lugar de 0.30)
    hist, face_rect = extract_face_histogram(img)
    if hist is None or len(known_face_histograms) == 0:
        return None, 0, None
    best_name = None
    best_score = float('inf')
    for name, known_hist in known_face_histograms.items():
        score = cv2.compareHist(hist, known_hist, cv2.HISTCMP_BHATTACHARYYA)
        if score < best_score:
            best_score = score
            best_name = name
    if best_score < threshold:
        confidence = max(0, min(100, (1 - best_score / threshold) * 100))
        return best_name, round(float(confidence), 2), face_rect
    return None, 0, None

def detect_wink_lightweight(img, face_rect):
    """
    Detecta guiño usando exclusivamente asimetría de brillo y diferencia de varianza.
    No depende de la detección de ojos (falla con ojos cerrados).
    """
    if face_rect is None:
        return False
    x, y, w, h = face_rect
    if w <= 0 or h <= 0:
        return False

    face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return False

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # CLAHE para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_face = clahe.apply(gray_face)

    # Dividir el rostro en dos mitades (izquierda y derecha)
    half = w // 2
    left_half = gray_face[:, :half]
    right_half = gray_face[:, half:]

    # 1. Diferencia de brillo medio (asimetría)
    mean_left = np.mean(left_half)
    mean_right = np.mean(right_half)
    if max(mean_left, mean_right) == 0:
        asymmetry = 0.0
    else:
        asymmetry = abs(mean_left - mean_right) / max(mean_left, mean_right)

    # 2. Diferencia de varianza (un ojo cerrado reduce la textura en ese lado)
    var_left = np.var(left_half)
    var_right = np.var(right_half)
    if max(var_left, var_right) == 0:
        var_ratio = 0.0
    else:
        var_ratio = abs(var_left - var_right) / max(var_left, var_right)

    # 3. Diferencia en el valor máximo (pico de brillo)
    max_left = np.max(left_half)
    max_right = np.max(right_half)
    max_diff = abs(max_left - max_right) / max(max_left, max_right) if max(max_left, max_right) > 0 else 0

    # Umbrales ajustados para máxima sensibilidad (valores bajos)
    ASYMMETRY_THRESHOLD = 0.08      # Muy sensible
    VAR_RATIO_THRESHOLD = 0.15      # Diferencia de textura
    MAX_DIFF_THRESHOLD = 0.10       # Diferencia de picos

    # Condiciones combinadas
    wink_condition = (
        (asymmetry > ASYMMETRY_THRESHOLD and var_ratio > VAR_RATIO_THRESHOLD) or
        (asymmetry > ASYMMETRY_THRESHOLD and max_diff > MAX_DIFF_THRESHOLD) or
        (var_ratio > 0.25)  # Si la varianza es muy diferente, casi seguro guiño
    )
    
    # Log para depuración (opcional)
    # logging.debug(f"Asym={asymmetry:.3f}, VarRatio={var_ratio:.3f}, MaxDiff={max_diff:.3f} -> Wink={wink_condition}")
    
    return wink_condition

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "rostros_cargados": len(known_face_histograms)
    })

@app.route("/recibir", methods=["POST"])
def recibir():
    try:
        data = request.get_json()
        if not data or 'foto' not in data or 'batch_id' not in data:
            return jsonify({"error": "Faltan campos: foto y batch_id", "activar_rele": False}), 400

        batch_id = data['batch_id']
        foto_b64 = data['foto']
        img_data = base64.b64decode(foto_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Error decodificando imagen", "activar_rele": False}), 400

        img = preprocess_image(img)
        # img = cv2.resize(img, (320, 240))  # opcional

        logging.info(f"Batch {batch_id} - Foto recibida ({len(img_data)} bytes)")

        nombre, confianza, face_rect = recognize_face_histogram(img)
        tiene_guino = False
        if nombre:
            tiene_guino = detect_wink_lightweight(img, face_rect)

        individual_response = {
            "activar_rele": bool(nombre is not None and not tiene_guino),
            "nombre": nombre if nombre else None,
            "confianza": float(confianza) if nombre else 0,
            "guino": bool(tiene_guino)
        }

        if batch_id not in pending_batches:
            pending_batches[batch_id] = {
                "count": 0,
                "results": [],
                "final_notified": False
            }
        batch = pending_batches[batch_id]
        batch["count"] += 1
        batch["results"].append(individual_response)

        if batch["count"] >= 3 and not batch["final_notified"]:
            batch["final_notified"] = True
            any_wink = any(r.get("guino", False) for r in batch["results"])
            any_recognized_no_wink = any(r.get("activar_rele", False) for r in batch["results"])

            if any_wink:
                names = [r.get("nombre") for r in batch["results"] if r.get("nombre")]
                name_str = names[0] if names else "Desconocido"
                final_message = f"⚠️ Timbre activado. Rostro reconocido: {name_str}. Se detectó un GUIÑO (posible emergencia)."
                logging.info(f"Batch {batch_id} - Resultado final: GUIÑO")
                send_whatsapp_message(final_message)
            elif any_recognized_no_wink:
                for r in batch["results"]:
                    if r.get("activar_rele") and r.get("nombre"):
                        final_message = f"✅ Timbre activado. Rostro reconocido: {r['nombre']}. Sin guiño."
                        break
                else:
                    final_message = "✅ Timbre activado. Rostro reconocido. Sin guiño."
                logging.info(f"Batch {batch_id} - Resultado final: PERMITIDO")
                send_whatsapp_message(final_message)
            else:
                final_message = "❗ Timbre activado. Rostro NO reconocido."
                logging.info(f"Batch {batch_id} - Resultado final: DENEGADO")
                send_whatsapp_message(final_message)

            del pending_batches[batch_id]

        return jsonify(individual_response), 200

    except Exception as e:
        logging.error(f"Error en /recibir: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
