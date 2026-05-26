import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from fer import FER
import base64
import logging
from datetime import datetime

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Inicializar FER para detección de emociones (guiños)
emotion_detector = FER(mtcnn=False)

# Estructura para rostros conocidos (usando histogramas de color)
known_faces_data = {}

def extract_face(img):
    """Extrae rostro usando Haar Cascade de OpenCV"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

def preprocess_face(face_img):
    """Preprocesa rostro para comparación (50x50, ecualizado)"""
    if face_img is None:
        return None
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, (50, 50))

def load_known_faces():
    """Carga rostros conocidos desde base_rostros"""
    global known_faces_data
    known_faces_data = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning(f"⚠️ Carpeta {BASE_ROSTROS_FOLDER} no existe")
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                face = extract_face(img)
                if face is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    processed = preprocess_face(face)
                    if processed is not None:
                        known_faces_data[nombre] = processed.flatten()
                        logging.info(f"✅ Rostro registrado: {nombre}")
                    else:
                        logging.warning(f"⚠️ Error preprocesando: {filename}")
                else:
                    logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_faces_data)}")

def recognize_face(img, threshold=0.3):
    """Reconoce rostro por comparación de histogramas"""
    face = extract_face(img)
    if face is None:
        return None, 0
    
    query = preprocess_face(face)
    if query is None:
        return None, 0
    
    query_vector = query.flatten()
    
    if len(known_faces_data) == 0:
        return None, 0
    
    mejor_nombre = None
    mejor_distancia = float('inf')
    
    for nombre, known_vector in known_faces_data.items():
        # Distancia euclidiana entre vectores de 2500 elementos (50x50)
        distancia = np.linalg.norm(query_vector - known_vector)
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
    
    # Convertir distancia a confianza (heurística)
    confianza = max(0, min(100, 100 - (mejor_distancia / 30)))
    
    if confianza >= threshold * 100:
        return mejor_nombre, round(confianza, 2)
    return None, 0

def detectar_guino(img):
    """Detecta guiño usando FER (happy/surprise)"""
    try:
        # FER espera RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emociones = emotion_detector.detect_emotions(rgb_img)
        
        if emociones:
            emocion_data = emociones[0]['emotions']
            happy = emocion_data.get('happy', 0)
            surprise = emocion_data.get('surprise', 0)
            
            # Si felicidad o sorpresa superan 0.6, puede ser un guiño
            if happy > 0.6 or surprise > 0.6:
                return True
    except Exception as e:
        logging.warning(f"Error en detección FER: {e}")
    return False

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "servicio": "Reconocimiento Facial + Guiño",
        "rostros_cargados": len(known_faces_data),
        "mensaje": "Servidor listo para recibir fotos"
    })

@app.route("/recibir", methods=["POST"])
def recibir():
    start_time = datetime.now()
    
    try:
        # Recibir JSON con foto en base64
        data = request.get_json()
        if not data or 'foto' not in data:
            return jsonify({"error": "No se recibió foto", "activar_rele": False}), 400
        
        foto_b64 = data['foto']
        img_data = base64.b64decode(foto_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Error decodificando imagen", "activar_rele": False}), 400
        
        logging.info(f"📥 Foto recibida ({len(img_data)} bytes)")
        
        # Reconocimiento facial
        nombre, confianza = recognize_face(img)
        
        if nombre:
            # Detectar guiño
            tiene_guino = detectar_guino(img)
            tiempo = (datetime.now() - start_time).total_seconds()
            
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
                logging.info(f"✅ {nombre} - reconocido (confianza {confianza}%) - {tiempo:.2f}s")
                return jsonify({
                    "activar_rele": True,
                    "motivo": "reconocido",
                    "nombre": nombre,
                    "confianza": confianza,
                    "mensaje": f"Rostro reconocido: {nombre}. Acceso PERMITIDO.",
                    "tiempo": round(tiempo, 2)
                }), 200
        else:
            tiempo = (datetime.now() - start_time).total_seconds()
            logging.info(f"❌ Rostro NO reconocido - {tiempo:.2f}s")
            return jsonify({
                "activar_rele": False,
                "motivo": "no_reconocido",
                "mensaje": "Rostro no reconocido. Acceso DENEGADO.",
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
