import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import base64

# --- Configuración ---
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# --- Inicializar MediaPipe (solo Face Detection) ---
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

logging.basicConfig(level=logging.INFO)

# --- Base de rostros conocidos ---
known_faces = {}

def extract_simple_embedding(img):
    """Extrae embedding simple de los 6 puntos clave de Face Detection"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = face_detector.process(rgb)
    
    if not resultados.detections:
        return None
    
    deteccion = resultados.detections[0]
    keypoints = deteccion.location_data.relative_keypoints
    
    embedding = []
    for kp in keypoints:
        embedding.append(kp.x)
        embedding.append(kp.y)
    
    return np.array(embedding)

def load_known_faces():
    global known_faces
    known_faces = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning(f"⚠️ Carpeta {BASE_ROSTROS_FOLDER} no existe")
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                embedding = extract_simple_embedding(img)
                if embedding is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    known_faces[nombre] = embedding
                    logging.info(f"✅ Cargado: {nombre}")
                else:
                    logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_faces)}")

def reconocer_rostro_simple(embedding, threshold=0.15):
    if embedding is None or len(known_faces) == 0:
        return None, 0
    
    mejor_nombre = None
    mejor_distancia = float('inf')
    
    for nombre, known_embedding in known_faces.items():
        distancia = np.linalg.norm(embedding - known_embedding)
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
    
    if mejor_distancia <= threshold:
        confianza = max(0, min(100, (1 - mejor_distancia / threshold) * 100))
        return mejor_nombre, confianza
    return None, 0

def detectar_guino_rapido(imagen):
    """Detecta guiño usando puntos clave de Face Detection"""
    try:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = face_detector.process(rgb)
        
        if not resultados.detections:
            return False
        
        deteccion = resultados.detections[0]
        keypoints = deteccion.location_data.relative_keypoints
        
        altura_ojo_izq = keypoints[0].y
        altura_ojo_der = keypoints[1].y
        altura_nariz = keypoints[2].y
        
        diff_izq = abs(altura_ojo_izq - altura_nariz)
        diff_der = abs(altura_ojo_der - altura_nariz)
        
        umbral = 0.045
        
        if diff_izq > umbral and diff_der < umbral * 0.6:
            return True
        if diff_der > umbral and diff_izq < umbral * 0.6:
            return True
            
        return False
    except Exception as e:
        logging.warning(f"Error en detección de guiño: {e}")
        return False

def procesar_una_foto(img):
    """Procesa una sola foto y devuelve resultado"""
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = face_detector.process(rgb)
        
        if not detections.detections:
            return {
                "rostro_detectado": False,
                "reconocido": False,
                "nombre": None,
                "guino": False,
                "confianza": 0
            }
        
        embedding = extract_simple_embedding(img)
        if embedding is None:
            return {
                "rostro_detectado": True,
                "reconocido": False,
                "nombre": None,
                "guino": False,
                "confianza": 0
            }
        
        nombre, confianza = reconocer_rostro_simple(embedding)
        tiene_guino = detectar_guino_rapido(img)
        
        return {
            "rostro_detectado": True,
            "reconocido": nombre is not None,
            "nombre": nombre,
            "confianza": round(confianza, 2),
            "guino": tiene_guino
        }
    except Exception as e:
        logging.error(f"Error procesando foto: {e}")
        return {
            "rostro_detectado": False,
            "reconocido": False,
            "nombre": None,
            "guino": False,
            "confianza": 0,
            "error": str(e)
        }

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "servicio": "Reconocimiento Facial + Detección de Guiño",
        "rostros_cargados": len(known_faces),
        "versión": "Optimizada para Render"
    })

@app.route("/recibir", methods=["POST"])
def recibir():
    start_time = datetime.now()
    
    try:
        # Obtener datos según Content-Type
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            if not data or 'fotos' not in data:
                return jsonify({"error": "No se recibieron fotos"}), 400
            
            fotos_b64 = data['fotos']
            if not isinstance(fotos_b64, list):
                fotos_b64 = [fotos_b64]  # Si es una sola foto, convertir a lista
            
            logging.info(f"📥 Recibidas {len(fotos_b64)} fotos")
            
            resultados = []
            for i, foto_b64 in enumerate(fotos_b64):
                try:
                    img_data = base64.b64decode(foto_b64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        resultado = procesar_una_foto(img)
                        resultados.append(resultado)
                        logging.info(f"📸 Foto {i+1}: reconocido={resultado.get('reconocido')}, guiño={resultado.get('guino')}")
                    else:
                        resultados.append({"error": "Decodificación fallida"})
                except Exception as e:
                    logging.error(f"Error decodificando foto {i+1}: {e}")
                    resultados.append({"error": str(e)})
        
        else:
            # Modo binario directo (una sola foto)
            raw_data = request.get_data()
            if not raw_data:
                return jsonify({"error": "No se recibió imagen"}), 400
            
            nparr = np.frombuffer(raw_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({"error": "Error al decodificar imagen"}), 400
            
            resultado = procesar_una_foto(img)
            resultados = [resultado]
        
        # Determinar respuesta final
        fotos_con_guiño = [r for r in resultados if r.get('guino', False)]
        fotos_reconocidas_sin_guino = [
            r for r in resultados 
            if r.get('reconocido', False) and not r.get('guino', False)
        ]
        
        tiempo_procesamiento = (datetime.now() - start_time).total_seconds()
        
        if fotos_con_guiño:
            respuesta = {
                "activar_rele": False,
                "motivo": "guiño_detectado",
                "mensaje": "Se detectó un guiño. Acceso DENEGADO.",
                "detalles": resultados,
                "tiempo_procesamiento": round(tiempo_procesamiento, 2)
            }
            logging.info(f"🚫 RESULTADO: Guiño detectado - Acceso DENEGADO")
            
        elif fotos_reconocidas_sin_guino:
            nombre = fotos_reconocidas_sin_guino[0].get('nombre', 'Desconocido')
            confianza = fotos_reconocidas_sin_guino[0].get('confianza', 0)
            
            respuesta = {
                "activar_rele": True,
                "motivo": "rostro_reconocido",
                "nombre": nombre,
                "confianza": confianza,
                "mensaje": f"Rostro reconocido: {nombre}. Acceso PERMITIDO.",
                "detalles": resultados,
                "tiempo_procesamiento": round(tiempo_procesamiento, 2)
            }
            logging.info(f"✅ RESULTADO: {nombre} reconocido - Acceso PERMITIDO")
            
        else:
            respuesta = {
                "activar_rele": False,
                "motivo": "no_reconocido",
                "mensaje": "No se reconoció ningún rostro. Acceso DENEGADO.",
                "detalles": resultados,
                "tiempo_procesamiento": round(tiempo_procesamiento, 2)
            }
            logging.info(f"❌ RESULTADO: Rostro no reconocido - Acceso DENEGADO")
        
        return jsonify(respuesta), 200
        
    except Exception as e:
        logging.error(f"Error en /recibir: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

# Cargar rostros conocidos al iniciar
load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"🚀 Servidor iniciando en puerto {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
