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

# --- Inicializar MediaPipe (solo Face Detection, NO Face Mesh) ---
mp_face_detection = mp.solutions.face_detection
# model_selection=0 para distancias cortas (menos de 2 metros)
# model_selection=1 para distancias lejanas (mayor rendimiento)
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

logging.basicConfig(level=logging.INFO)

# --- Base de rostros conocidos (embeddings simples) ---
known_faces = {}  # {nombre: embedding}

def extract_simple_embedding(img):
    """
    Extrae un embedding simple del rostro detectado
    Usa solo los puntos clave de Face Detection (6 puntos)
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = face_detector.process(rgb)
    
    if not resultados.detections:
        return None
    
    deteccion = resultados.detections[0]
    keypoints = deteccion.location_data.relative_keypoints
    
    # Extraer coordenadas de los 6 puntos clave
    embedding = []
    for kp in keypoints:
        embedding.append(kp.x)
        embedding.append(kp.y)
    
    return np.array(embedding)

def load_known_faces():
    """Carga rostros conocidos y calcula sus embeddings simples"""
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
    """
    Compara embedding con rostros conocidos usando distancia euclidiana
    Threshold más bajo = más preciso (0.10-0.20 recomendado)
    """
    if embedding is None or len(known_faces) == 0:
        return None, 0
    
    mejor_nombre = None
    mejor_distancia = float('inf')
    
    for nombre, known_embedding in known_faces.items():
        # Distancia euclidiana (más rápida que cosine_similarity)
        distancia = np.linalg.norm(embedding - known_embedding)
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
    
    if mejor_distancia <= threshold:
        # Convertir distancia a porcentaje de confianza
        confianza = max(0, min(100, (1 - mejor_distancia / threshold) * 100))
        return mejor_nombre, confianza
    return None, 0

def detectar_guino_rapido(imagen):
    """
    Detecta guiño usando los puntos clave de Face Detection
    Muy rápido (~10-20ms por imagen)
    """
    try:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = face_detector.process(rgb)
        
        if not resultados.detections:
            return False
        
        deteccion = resultados.detections[0]
        keypoints = deteccion.location_data.relative_keypoints
        
        # keypoints[0] = ojo izquierdo
        # keypoints[1] = ojo derecho
        # keypoints[2] = nariz
        # keypoints[3] = boca
        # keypoints[4] = oreja izquierda
        # keypoints[5] = oreja derecha
        
        # Calcular altura relativa de cada ojo
        # Usamos una aproximación: la posición Y del ojo indica apertura
        # Un ojo cerrado tiene un valor Y más alto (porque el párpado baja)
        
        altura_ojo_izq = keypoints[0].y
        altura_ojo_der = keypoints[1].y
        altura_nariz = keypoints[2].y
        
        # Normalizar respecto a la nariz
        diff_izq = abs(altura_ojo_izq - altura_nariz)
        diff_der = abs(altura_ojo_der - altura_nariz)
        
        # Un ojo cerrado tiene mayor diferencia con la nariz
        umbral = 0.045  # Ajustar según pruebas
        
        # Guiño: un ojo significativamente más cerrado que el otro
        if diff_izq > umbral and diff_der < umbral * 0.6:
            return True
        if diff_der > umbral and diff_izq < umbral * 0.6:
            return True
            
        return False
        
    except Exception as e:
        logging.warning(f"Error en detección de guiño: {e}")
        return False

def procesar_una_foto(img):
    """
    Procesa una foto individual: detección de rostro, reconocimiento y guiño
    Tiempo estimado: 50-150ms por foto
    """
    try:
        # Detectar rostro
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
        
        # Extraer embedding para reconocimiento
        embedding = extract_simple_embedding(img)
        
        if embedding is None:
            return {
                "rostro_detectado": True,
                "reconocido": False,
                "nombre": None,
                "guino": False,
                "confianza": 0
            }
        
        # Reconocer rostro
        nombre, confianza = reconocer_rostro_simple(embedding)
        
        # Detectar guiño (solo si hay rostro)
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
    """Endpoint que recibe las 3 fotos y devuelve la decisión final"""
    start_time = datetime.now()
    
    try:
        # Verificar si es JSON con base64 o binario directo
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            if not data or 'fotos' not in data:
                return jsonify({"error": "No se recibieron fotos"}), 400
            
            fotos_b64 = data['fotos']
            if len(fotos_b64) != 3:
                return jsonify({"error": f"Se esperaban 3 fotos, se recibieron {len(fotos_b64)}"}), 400
            
            # Procesar cada foto
            resultados = []
            for i, foto_b64 in enumerate(fotos_b64):
                img_data = base64.b64decode(foto_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    resultado = procesar_una_foto(img)
                    resultados.append(resultado)
                    logging.info(f"📸 Foto {i+1}: {resultado}")
                else:
                    resultados.append({"error": "Decodificación fallida"})
            
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
        
        # Determinar respuesta final basada en las 3 fotos
        # Reglas:
        # 1. Si ALGUNA foto tiene guiño -> BLOQUEAR
        # 2. Si ALGUNA foto tiene rostro reconocido SIN guiño -> ACTIVAR
        # 3. Si ninguna foto reconoce rostro -> NO ACTIVAR
        
        fotos_con_guiño = [r for r in resultados if r.get('guino', False)]
        fotos_reconocidas_sin_guino = [
            r for r in resultados 
            if r.get('reconocido', False) and not r.get('guino', False)
        ]
        
        tiempo_procesamiento = (datetime.now() - start_time).total_seconds()
        
        if fotos_con_guiño:
            # Hay guiño en al menos una foto
            respuesta = {
                "activar_rele": False,
                "motivo": "guiño_detectado",
                "mensaje": "Se detectó un guiño. Acceso DENEGADO.",
                "detalles": resultados,
                "tiempo_procesamiento": round(tiempo_procesamiento, 2)
            }
            logging.info(f"🚫 RESULTADO: Guiño detectado - Acceso DENEGADO")
            
        elif fotos_reconocidas_sin_guino:
            # Obtener el nombre del primer rostro reconocido sin guiño
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
