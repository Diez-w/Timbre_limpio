import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import face_recognition
import base64
import logging
from datetime import datetime

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Estructura para almacenar codificaciones faciales
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Carga rostros conocidos usando face_recognition"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning(f"⚠️ Carpeta {BASE_ROSTROS_FOLDER} no existe")
        return
    
    for filename in os.listdir(BASE_ROSTROS_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            
            if encodings:
                nombre = filename.rsplit('.', 1)[0]
                known_face_encodings.append(encodings[0])
                known_face_names.append(nombre)
                logging.info(f"✅ Rostro registrado: {nombre}")
            else:
                logging.warning(f"⚠️ No se detectó rostro en: {filename}")
    
    logging.info(f"📊 Total rostros cargados: {len(known_face_names)}")

def recognize_face(img):
    """Reconoce rostro usando face_recognition"""
    # Convertir imagen de BGR (OpenCV) a RGB (face_recognition)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detectar rostros y obtener codificaciones
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    if not face_encodings:
        return None, 0
    
    # Comparar con rostros conocidos
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if True in matches:
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                confianza = (1 - distances[best_match_index]) * 100
                return known_face_names[best_match_index], round(confianza, 2)
    
    return None, 0

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "servicio": "Reconocimiento Facial",
        "rostros_cargados": len(known_face_names)
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
        
        logging.info(f"📥 Foto recibida ({len(img_data)} bytes)")
        
        nombre, confianza = recognize_face(img)
        tiempo = (datetime.now() - start_time).total_seconds()
        
        if nombre:
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
