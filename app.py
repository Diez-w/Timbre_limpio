import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
from datetime import datetime

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ==================================================
# CONFIGURACIÓN PARA 0.1 CPU / 512 MB RAM
# Solo OpenCV - Sin MediaPipe, TensorFlow, etc.
# ==================================================

# Clasificadores Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Estructura para almacenar rostros conocidos
known_face_histograms = {}  # {nombre: histograma}

def preprocess_image(img):
    """Mejora la calidad de la imagen para que se parezca más a las fotos de referencia"""
    # Convertir a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ecualizar histograma (mejora contraste)
    gray = cv2.equalizeHist(gray)
    # Reducir ruido
    gray = cv2.medianBlur(gray, 3)
    # Volver a BGR para mantener compatibilidad (aunque luego se volverá a gris)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def extract_face_histogram(img):
    """
    Extrae histograma del rostro usando OpenCV.
    Tiempo estimado: 20-30ms.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None, None
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Reducir tamaño para acelerar
    face_small = cv2.resize(face_roi, (32, 32))
    
    # Histograma de 32 bins
    hist = cv2.calcHist([face_small], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist, (x, y, w, h)

def load_known_faces():
    """Carga rostros conocidos (fotos)"""
    global known_face_histograms
    known_face_histograms = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning(f"⚠️ Carpeta {BASE_ROSTROS_FOLDER} no existe")
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Preprocesar también las imágenes de referencia (opcional)
                img_proc = preprocess_image(img)
                hist, face_rect = extract_face_histogram(img_proc)
                if hist is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    known_face_histograms[nombre] = hist
                    logging.info(f"✅ Rostro registrado: {nombre}")
                else:
                    logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_face_histograms)}")

def recognize_face_histogram(img, threshold=0.25):
    """
    Reconoce rostro comparando histogramas usando Bhattacharyya (más tolerante).
    Tiempo estimado: 30-50ms.
    """
    hist, face_rect = extract_face_histogram(img)
    if hist is None or len(known_face_histograms) == 0:
        return None, 0, None
    
    best_name = None
    best_score = float('inf')  # Para Bhattacharyya, menor = mejor coincidencia
    
    for name, known_hist in known_face_histograms.items():
        # Usar Bhattacharyya en lugar de correlación (más flexible)
        score = cv2.compareHist(hist, known_hist, cv2.HISTCMP_BHATTACHARYYA)
        if score < best_score:
            best_score = score
            best_name = name
    
    # Convertir score (0 = perfecto, 1 = muy diferente) a porcentaje de confianza
    # Si best_score < threshold, consideramos coincidencia
    if best_score < threshold:
        confidence = max(0, min(100, (1 - best_score / threshold) * 100))
        return best_name, round(confidence, 2), face_rect
    return None, 0, None

def detect_wink_haar(img, face_rect):
    """
    Detecta guiño usando clasificadores de ojos Haar.
    Tiempo estimado: 10-20ms.
    """
    if face_rect is None:
        return False
    
    x, y, w, h = face_rect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = gray[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))
    
    # Si se detectan menos de 2 ojos, puede ser un guiño
    return len(eyes) < 2

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "servicio": "Reconocimiento Facial con OpenCV (mejorado)",
        "rostros_cargados": len(known_face_histograms),
        "mensaje": "Optimizado para fotos ESP32-CAM vs POCO F5 Pro"
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
        
        # Mejorar calidad de la imagen del ESP32-CAM
        img = preprocess_image(img)
        # Opcional: reducir resolución para acelerar (comentar si no quieres)
        img = cv2.resize(img, (320, 240))
        
        logging.info(f"📥 Foto recibida ({len(img_data)} bytes)")
        
        # Reconocimiento facial
        nombre, confianza, face_rect = recognize_face_histogram(img)
        tiempo = (datetime.now() - start_time).total_seconds()
        
        if nombre:
            # Detectar guiño
            tiene_guino = detect_wink_haar(img, face_rect)
            
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
    logging.info(f"🚀 Servidor iniciando en puerto {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
