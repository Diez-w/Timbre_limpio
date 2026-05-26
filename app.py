import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
from datetime import datetime
from fer import FER
import pickle

# Configuración
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Inicializar FER para detección de emociones (guiños)
emotion_detector = FER(mtcnn=False)  # MTCNN más preciso pero más lento, desactivado para velocidad

# Estructura para almacenar rostros conocidos (para Edge Face)
# Edge face usa un enfoque simple: guardamos las imágenes de referencia
known_faces_data = {}  # {nombre: lista_de_imagenes_preprocesadas}

def preprocess_face_for_edge(face_img):
    """Preprocesa la imagen para Edge Face (50x50 color BGR)"""
    if face_img is None:
        return None
    # Redimensionar a 50x50 como espera edgeface-knn
    resized = cv2.resize(face_img, (50, 50))
    return resized

def extract_face(img):
    """Extrae el rostro de la imagen usando Haar Cascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    return face_img

def load_known_faces():
    """Carga rostros conocidos desde la carpeta base_rostros"""
    global known_faces_data
    known_faces_data = {}
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = cv2.imread(img_path)
            if img is not None:
                face = extract_face(img)
                if face is not None:
                    nombre = filename.rsplit('.', 1)[0]
                    processed_face = preprocess_face_for_edge(face)
                    known_faces_data[nombre] = processed_face
                    logging.info(f"✅ Rostro registrado: {nombre}")
                else:
                    logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_faces_data)}")

def recognize_face_edge(img, threshold=0.4):
    """
    Reconoce rostro usando distancia euclidiana entre imágenes 50x50
    Basado en el enfoque de edgeface-knn (sin dependencias externas)
    """
    face = extract_face(img)
    if face is None:
        return None, 0
    
    processed_face = preprocess_face_for_edge(face)
    if processed_face is None:
        return None, 0
    
    # Vectorizar imagen
    query_vector = processed_face.flatten()
    
    if len(known_faces_data) == 0:
        return None, 0
    
    mejor_nombre = None
    mejor_distancia = float('inf')
    
    # Comparar con todos los rostros conocidos
    for nombre, known_face in known_faces_data.items():
        known_vector = known_face.flatten()
        distancia = np.linalg.norm(query_vector - known_vector)
        
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
    
    # Calcular confianza (heurística edgeface)
    confianza = 100.0 * np.exp(-mejor_distancia / 4500.0)
    
    if confianza >= threshold * 100:
        return mejor_nombre, round(confianza, 2)
    return None, 0

def detectar_guino_fer(img):
    """
    Detecta guiño usando FER (Facial Expression Recognition)
    Un guiño suele estar asociado con 'happy' o 'surprise'
    """
    try:
        # FER espera RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emociones = emotion_detector.detect_emotions(rgb_img)
        
        if emociones:
            emocion_data = emociones[0]['emotions']
            # Verificar si la emoción dominante es happy o surprise
            happy_score = emocion_data.get('happy', 0)
            surprise_score = emocion_data.get('surprise', 0)
            
            # Si felicidad o sorpresa superan 0.6, puede ser un guiño
            if happy_score > 0.6 or surprise_score > 0.6:
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
        "tiempo_estimado": "200ms por lote"
    })

@app.route("/recibir", methods=["POST"])
def recibir():
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        if not data or 'fotos' not in data:
            return jsonify({"error": "No se recibieron fotos", "activar_rele": False}), 400
        
        fotos_b64 = data['fotos']
        if isinstance(fotos_b64, str):
            fotos_b64 = [fotos_b64]
        
        logging.info(f"📥 Recibidas {len(fotos_b64)} fotos")
        
        resultados = []
        for i, foto_b64 in enumerate(fotos_b64):
            img_data = base64.b64decode(foto_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                resultados.append({"foto": i+1, "error": "Decodificación fallida"})
                continue
            
            # Paso 1: Reconocimiento facial (rápido)
            nombre, confianza = recognize_face_edge(img)
            
            # Paso 2: Detección de guiño con FER (solo si hay rostro)
            tiene_guino = False
            if nombre is not None:
                tiene_guino = detectar_guino_fer(img)
                logging.info(f"📸 Foto {i+1}: {nombre} (confianza {confianza}%) - Guiño: {tiene_guino}")
            else:
                logging.info(f"📸 Foto {i+1}: No reconocido")
            
            resultados.append({
                "foto": i+1,
                "reconocido": nombre is not None,
                "nombre": nombre,
                "confianza": confianza if nombre else 0,
                "guino": tiene_guino
            })
        
        # Determinar respuesta final
        hay_guiño = any(r.get('guino', False) for r in resultados)
        hay_reconocido = any(r.get('reconocido', False) and not r.get('guino', False) for r in resultados)
        nombre_reconocido = next((r.get('nombre') for r in resultados if r.get('reconocido') and not r.get('guino')), None)
        
        tiempo = (datetime.now() - start_time).total_seconds()
        
        if hay_guiño:
            respuesta = {
                "activar_rele": False,
                "motivo": "guiño_detectado",
                "mensaje": "Se detectó un guiño. Acceso DENEGADO.",
                "tiempo_procesamiento": round(tiempo, 2)
            }
            logging.info(f"🚫 RESULTADO: Guiño detectado - Acceso DENEGADO ({tiempo:.2f}s)")
        elif hay_reconocido:
            respuesta = {
                "activar_rele": True,
                "motivo": "rostro_reconocido",
                "nombre": nombre_reconocido,
                "mensaje": f"Rostro reconocido: {nombre_reconocido}. Acceso PERMITIDO.",
                "tiempo_procesamiento": round(tiempo, 2)
            }
            logging.info(f"✅ RESULTADO: {nombre_reconocido} - Acceso PERMITIDO ({tiempo:.2f}s)")
        else:
            respuesta = {
                "activar_rele": False,
                "motivo": "no_reconocido",
                "mensaje": "No se reconoció ningún rostro. Acceso DENEGADO.",
                "tiempo_procesamiento": round(tiempo, 2)
            }
            logging.info(f"❌ RESULTADO: Rostro no reconocido - Acceso DENEGADO ({tiempo:.2f}s)")
        
        return jsonify(respuesta), 200
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500

# Cargar rostros al iniciar
load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
