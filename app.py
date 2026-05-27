import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
import base64
import logging
from datetime import datetime

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ==================================================
# CONFIGURACIÓN CRÍTICA PARA 0.1 CPU / 512 MB RAM
# ==================================================
mp_face_detection = mp.solutions.face_detection

# Usar model_selection=0 (distancias cortas) que es MÁS RÁPIDO
# Reducir min_detection_confidence para más detecciones (0.3 en lugar de 0.5)
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,           # 0 = corta distancia (<2m), MÁS RÁPIDO
    min_detection_confidence=0.3 # Más bajo = más detecciones, menos precisión
)
# ==================================================

# Estructura para almacenar rostros conocidos
known_face_data = {}  # {nombre: "huella_digital"}

def extract_face_signature(img):
    """
    Extrae una "huella digital" del rostro usando solo los keypoints de Face Detection.
    Esto es ULTRA-RÁPIDO porque NO usa Face Mesh.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    
    if not results.detections:
        return None
    
    # Tomamos el primer rostro detectado
    detection = results.detections[0]
    keypoints = detection.location_data.relative_keypoints
    
    # Face Detection tiene 6 keypoints:
    # 0: ojo izquierdo, 1: ojo derecho, 2: nariz, 3: boca, 4: oreja izq, 5: oreja der
    signature = []
    for kp in keypoints:
        signature.append(kp.x)
        signature.append(kp.y)
    
    return np.array(signature)

def load_known_faces():
    """Carga rostros conocidos (solo sus signatures)"""
    global known_face_data
    known_face_data = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                signature = extract_face_signature(img)
                if signature is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    known_face_data[nombre] = signature
                    logging.info(f"✅ Rostro registrado: {nombre}")
                else:
                    logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_face_data)}")

def recognize_face_fast(img, threshold=0.25):
    """
    Reconocimiento facial ULTRA-RÁPIDO usando distancia euclidiana entre signatures.
    Tiempo estimado: 50-100ms por foto.
    """
    signature = extract_face_signature(img)
    if signature is None or len(known_face_data) == 0:
        return None, 0
    
    best_name = None
    best_distance = float('inf')
    
    for name, known_sig in known_face_data.items():
        distance = np.linalg.norm(signature - known_sig)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    
    # Convertir distancia a porcentaje (ajustado para ser más tolerante)
    confidence = max(0, min(100, 100 - (best_distance / 0.35 * 100)))
    
    if confidence >= threshold * 100:
        return best_name, round(confidence, 2)
    return None, 0

def detect_wink_fast(img):
    """
    Detección de GUIÑO usando Face Detection (NO Face Mesh).
    Un guiño ocurre cuando un ojo está significativamente más bajo que el otro.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    
    if not results.detections:
        return False
    
    detection = results.detections[0]
    keypoints = detection.location_data.relative_keypoints
    
    # keypoints[0] = ojo izquierdo, keypoints[1] = ojo derecho
    left_eye_y = keypoints[0].y
    right_eye_y = keypoints[1].y
    
    # Calcular diferencia de altura entre ojos
    eye_diff = abs(left_eye_y - right_eye_y)
    
    # Si la diferencia es grande (>0.03), puede ser un guiño
    # Ajusta este valor según tus pruebas
    return eye_diff > 0.025

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "servicio": "Reconocimiento Facial Ultra-Rápido",
        "rostros_cargados": len(known_face_data)
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
        
        # REDUCIR RESOLUCIÓN para acelerar (opcional, pero recomendado)
        # 320x240 es suficiente para detección y reduce el tiempo a la mitad
        # img = cv2.resize(img, (320, 240))
        
        logging.info(f"📥 Foto recibida ({len(img_data)} bytes)")
        
        # Reconocimiento facial (50-100ms)
        nombre, confianza = recognize_face_fast(img)
        tiempo = (datetime.now() - start_time).total_seconds()
        
        if nombre:
            # Detectar guiño (solo si hay rostro, 20-30ms adicionales)
            tiene_guino = detect_wink_fast(img)
            
            if tiene_guino:
                logging.info(f"👁️ {nombre} - GUIÑO detectado (bloquear) - {tiempo:.2f}s")
                return jsonify({
                    "activar_rele": False,
                    "motivo": "guiño",
                    "nombre": nombre,
                    "mensaje": f"GUIÑO detectado. Acceso DENEGADO.",
                    "tiempo": round(tiempo, 2)
                }), 200
            else:
                logging.info(f"✅ {nombre} - reconocido ({confianza}%) - {tiempo:.2f}s")
                return jsonify({
                    "activar_rele": True,
                    "nombre": nombre,
                    "confianza": confianza,
                    "mensaje": f"Rostro reconocido: {nombre}. Acceso PERMITIDO.",
                    "tiempo": round(tiempo, 2)
                }), 200
        else:
            logging.info(f"❌ Rostro NO reconocido - {tiempo:.2f}s")
            return jsonify({
                "activar_rele": False,
                "mensaje": "Rostro no reconocido. Acceso DENEGADO.",
                "motivo": "no_reconocido",
                "tiempo": round(tiempo, 2)
            }), 200
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

# Cargar rostros al iniciar
load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
