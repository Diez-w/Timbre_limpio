import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
from PIL import Image

# --- Configuración ---
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB para 3 fotos

# --- Inicializar MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

logging.basicConfig(level=logging.INFO)

# --- Base de rostros conocidos ---
known_faces = {}  # {nombre: embedding}

def extract_face_embedding(img):
    """Extrae embedding facial usando MediaPipe Face Mesh"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return embedding

def load_known_faces():
    """Carga rostros conocidos desde la carpeta base_rostros"""
    global known_faces
    known_faces = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        return
    
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

def reconocer_rostro(embedding, threshold=0.4):
    """Compara embedding con rostros conocidos"""
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

def detectar_guiño(imagen):
    """Detecta guiño usando MediaPipe"""
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return False
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye_top = landmarks[159].y
            left_eye_bottom = landmarks[145].y
            right_eye_top = landmarks[386].y
            right_eye_bottom = landmarks[374].y
            
            left_ear = abs(left_eye_top - left_eye_bottom)
            right_ear = abs(right_eye_top - right_eye_bottom)
            
            umbral = 0.015
            
            if (left_ear < umbral and right_ear > umbral * 2) or \
               (right_ear < umbral and left_ear > umbral * 2):
                return True
    except Exception as e:
        logging.warning(f"Error detectar guiño: {e}")
    return False

def procesar_una_foto(img):
    """Procesa una foto individual y devuelve resultado"""
    # Detectar rostro
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector.process(rgb)
    
    if not detections.detections:
        return {"rostro_detectado": False, "reconocido": False, "nombre": None, "guino": False}
    
    # Extraer embedding
    embedding = extract_face_embedding(img)
    if embedding is None:
        return {"rostro_detectado": True, "reconocido": False, "nombre": None, "guino": False}
    
    # Reconocer rostro
    nombre, similitud = reconocer_rostro(embedding)
    
    if nombre:
        tiene_guino = detectar_guiño(img)
        return {
            "rostro_detectado": True,
            "reconocido": True,
            "nombre": nombre,
            "similitud": float(similitud),
            "guino": tiene_guino
        }
    else:
        return {"rostro_detectado": True, "reconocido": False, "nombre": None, "guino": False}

@app.route("/")
def index():
    return jsonify({"status": "online", "rostros_cargados": len(known_faces)})

@app.route("/recibir", methods=["POST"])
def recibir():
    try:
        # Recibir JSON con las 3 fotos en base64
        data = request.get_json()
        
        if not data or 'fotos' not in data:
            return jsonify({"error": "No se recibieron fotos"}), 400
        
        fotos_b64 = data['fotos']
        if len(fotos_b64) != 3:
            return jsonify({"error": "Se esperaban 3 fotos"}), 400
        
        logging.info(f"📥 Recibidas {len(fotos_b64)} fotos")
        
        resultados_fotos = []
        
        for i, foto_b64 in enumerate(fotos_b64):
            # Decodificar imagen base64
            img_data = base64.b64decode(foto_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                resultado = procesar_una_foto(img)
                resultados_fotos.append(resultado)
                logging.info(f"📸 Foto {i+1}: {resultado}")
            else:
                resultados_fotos.append({"error": "No se pudo decodificar"})
        
        # Determinar respuesta final basada en las 3 fotos
        # Lógica: Si ALGUNA foto tiene guiño -> BLOQUEAR
        #         Si ALGUNA foto tiene rostro reconocido SIN guiño -> ACTIVAR
        #         Si ninguna foto reconoce rostro -> NO ACTIVAR
        
        hay_guiño = any(r.get('guino', False) for r in resultados_fotos)
        hay_reconocido_sin_guino = any(r.get('reconocido', False) and not r.get('guino', False) for r in resultados_fotos)
        
        if hay_guiño:
            respuesta_final = {
                "activar_rele": False,
                "motivo": "guiño_detectado",
                "mensaje": "Se detectó un guiño en al menos una foto. Acceso BLOQUEADO.",
                "detalles": resultados_fotos
            }
        elif hay_reconocido_sin_guino:
            # Obtener el nombre del primer rostro reconocido sin guiño
            nombre_reconocido = next((r['nombre'] for r in resultados_fotos if r.get('reconocido') and not r.get('guino')), "Desconocido")
            respuesta_final = {
                "activar_rele": True,
                "motivo": "rostro_reconocido_sin_guiño",
                "mensaje": f"Rostro reconocido: {nombre_reconocido}. Acceso PERMITIDO.",
                "detalles": resultados_fotos
            }
        else:
            respuesta_final = {
                "activar_rele": False,
                "motivo": "no_reconocido",
                "mensaje": "No se reconoció ningún rostro válido. Acceso DENEGADO.",
                "detalles": resultados_fotos
            }
        
        logging.info(f"🏁 Respuesta final: {respuesta_final['mensaje']}")
        return jsonify(respuesta_final), 200
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

# Cargar rostros al iniciar
load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
