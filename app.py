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
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

known_face_histograms = {}

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 3)
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

def recognize_face_histogram(img, threshold=0.25):
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
        return best_name, round(confidence, 2), face_rect
    return None, 0, None

def detect_wink_lightweight(img, face_rect):
    """
    Detección de guiño sin MediaPipe, usando solo OpenCV.
    Tiempo estimado: 5-10ms.
    """
    if face_rect is None:
        return False
    
    x, y, w, h = face_rect
    face_roi = img[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Dividir el rostro en mitades izquierda y derecha
    half = w // 2
    left_half = gray_face[:, :half]
    right_half = gray_face[:, half:]
    
    # Detectar ojos con Haar (rápido)
    eyes_left = eye_cascade.detectMultiScale(left_half, scaleFactor=1.05, minNeighbors=3, minSize=(10, 10))
    eyes_right = eye_cascade.detectMultiScale(right_half, scaleFactor=1.05, minNeighbors=3, minSize=(10, 10))
    total_eyes = len(eyes_left) + len(eyes_right)
    
    # Asimetría de brillo entre ambas mitades
    mean_left = np.mean(left_half)
    mean_right = np.mean(right_half)
    if max(mean_left, mean_right) == 0:
        asymmetry = 0
    else:
        asymmetry = abs(mean_left - mean_right) / max(mean_left, mean_right)
    
    # Condición: pocos ojos detectados o asimetría alta
    return (total_eyes < 2) or (asymmetry > 0.15)

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "rostros_cargados": len(known_face_histograms)
    })

@app.route("/recibir", methods=["POST"])
def recibir():
    start_time = datetime.now()
    try:
        data = request.get_json()
        if not data or 'foto' not in data:
            return jsonify({"error": "No se recibió foto", "activar_rele": False}), 400
        
        foto_b64 = data['foto']
        img_data = base64.b64decode(foto_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Error decodificando imagen", "activar_rele": False}), 400
        
        img = preprocess_image(img)
        # (Opcional) img = cv2.resize(img, (320, 240))
        
        logging.info(f"Foto recibida ({len(img_data)} bytes)")
        
        nombre, confianza, face_rect = recognize_face_histogram(img)
        tiempo = (datetime.now() - start_time).total_seconds()
        
        if nombre:
            tiene_guino = detect_wink_lightweight(img, face_rect)
            if tiene_guino:
                mensaje = f"⚠️ Timbre activado. Rostro reconocido: {nombre}. Se detectó un GUIÑO (posible emergencia)."
                logging.info(f"GUIÑO detectado - {tiempo:.2f}s")
                send_whatsapp_message(mensaje)
                return jsonify({
                    "activar_rele": False,
                    "motivo": "guiño",
                    "nombre": nombre,
                    "mensaje": "GUIÑO detectado. Acceso DENEGADO.",
                    "tiempo": round(tiempo, 2)
                }), 200
            else:
                mensaje = f"✅ Timbre activado. Rostro reconocido: {nombre}. Sin guiño."
                logging.info(f"Reconocido: {nombre} ({confianza}%) - {tiempo:.2f}s")
                send_whatsapp_message(mensaje)
                return jsonify({
                    "activar_rele": True,
                    "nombre": nombre,
                    "confianza": confianza,
                    "mensaje": f"Rostro reconocido: {nombre}. Acceso PERMITIDO.",
                    "tiempo": round(tiempo, 2)
                }), 200
        else:
            mensaje = "❗ Timbre activado. Rostro NO reconocido."
            logging.info(f"No reconocido - {tiempo:.2f}s")
            send_whatsapp_message(mensaje)
            return jsonify({
                "activar_rele": False,
                "mensaje": "Rostro no reconocido. Acceso DENEGADO.",
                "motivo": "no_reconocido",
                "tiempo": round(tiempo, 2)
            }), 200
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
