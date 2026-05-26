import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from datetime import datetime
import logging
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuración ---
UPLOAD_FOLDER = "static/uploads"
BASE_ROSTROS_FOLDER = "base_rostros"
EMBEDDINGS_FILE = "face_embeddings.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

app = Flask(__name__)

# --- Inicializar MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# --- Cargar o crear base de embeddings faciales ---
known_faces = {}  # {nombre: embedding}

def extract_face_embedding(img):
    """Extrae embedding facial usando MediaPipe Face Mesh"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        # Extraer coordenadas de los 468 landmarks como embedding
        landmarks = results.multi_face_landmarks[0].landmark
        embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return embedding

def load_known_faces():
    """Carga rostros conocidos y calcula sus embeddings"""
    global known_faces
    known_faces = {}
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                embedding = extract_face_embedding(img)
                if embedding is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    known_faces[nombre] = embedding
                    logging.info(f"Cargado: {nombre}")
    
    logging.info(f"✅ Cargados {len(known_faces)} rostros conocidos")

# Cargar rostros al iniciar
load_known_faces()

def detectar_guiño(imagen):
    """Detecta guiño usando MediaPipe Face Mesh"""
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return False
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Puntos para ojo izquierdo y derecho
            left_eye_top = landmarks[159].y
            left_eye_bottom = landmarks[145].y
            right_eye_top = landmarks[386].y
            right_eye_bottom = landmarks[374].y
            
            left_ear = abs(left_eye_top - left_eye_bottom)
            right_ear = abs(right_eye_top - right_eye_bottom)
            
            umbral = 0.015  # Umbral ajustado para detección rápida
            
            # Guiño: un ojo cerrado y el otro abierto
            if (left_ear < umbral and right_ear > umbral * 2) or \
               (right_ear < umbral and left_ear > umbral * 2):
                return True
    except Exception as e:
        logging.warning(f"Error detectar guiño: {e}")
    return False

def reconocer_rostro(embedding, threshold=0.4):
    """Compara embedding con rostros conocidos usando similitud de coseno"""
    if embedding is None or len(known_faces) == 0:
        return None, 0
    
    mejor_nombre = None
    mejor_similitud = 0
    
    for nombre, known_embedding in known_faces.items():
        similitud = cosine_similarity([embedding], [known_embedding])[0][0]
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_nombre = nombre
    
    if mejor_similitud >= threshold:
        return mejor_nombre, mejor_similitud
    return None, 0

@app.route("/")
def index():
    return jsonify({"status": "online"})

@app.route("/recibir", methods=["POST"])
def recibir():
    # Recibir imagen
    if request.headers.get('Content-Type') == 'image/jpeg':
        raw_data = request.get_data()
        if not raw_data:
            return jsonify({"error": "No se recibió imagen"}), 400
        
        nparr = np.frombuffer(raw_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Error al decodificar"}), 400
    else:
        return jsonify({"error": "Formato no válido"}), 400
    
    logging.info("📥 Imagen recibida")
    
    # Detectar rostro
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector.process(rgb)
    
    if not detections.detections:
        logging.info("❌ No se detectó rostro")
        return jsonify({
            "reconocido": False,
            "nombre": None,
            "guino": False,
            "activar_rele": False,
            "mensaje": "No se detectó rostro"
        }), 200
    
    # Extraer embedding del rostro detectado
    embedding = extract_face_embedding(img)
    if embedding is None:
        return jsonify({
            "reconocido": False,
            "nombre": None,
            "guino": False,
            "activar_rele": False,
            "mensaje": "No se pudieron extraer landmarks"
        }), 200
    
    # Reconocer rostro
    nombre, similitud = reconocer_rostro(embedding)
    
    if nombre:
        # Detectar guiño
        tiene_guino = detectar_guiño(img)
        
        if tiene_guino:
            logging.info(f"⚠️ {nombre} reconocido (similitud: {similitud:.2f}) con GUIÑO - BLOQUEADO")
            return jsonify({
                "reconocido": True,
                "nombre": nombre,
                "guino": True,
                "activar_rele": False,
                "mensaje": f"Rostro reconocido: {nombre}. GUIÑO detectado. Acceso BLOQUEADO"
            }), 200
        else:
            logging.info(f"✅ {nombre} reconocido (similitud: {similitud:.2f}) SIN guiño - ACCESO PERMITIDO")
            return jsonify({
                "reconocido": True,
                "nombre": nombre,
                "guino": False,
                "activar_rele": True,
                "mensaje": f"Rostro reconocido: {nombre}. Sin guiño. Acceso PERMITIDO"
            }), 200
    else:
        logging.info("❌ Rostro NO reconocido")
        return jsonify({
            "reconocido": False,
            "nombre": None,
            "guino": False,
            "activar_rele": False,
            "mensaje": "Rostro no reconocido"
        }), 200

if __name__ == "__main__":
    import os
    # Toma el puerto de la variable de entorno PORT, o usa el 10000 por defecto.
    port = int(os.environ.get("PORT", 10000))
    # Es MUY importante que el host sea '0.0.0.0'
    app.run(host="0.0.0.0", port=port)
