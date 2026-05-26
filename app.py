import os
import json
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import mediapipe as mp
from datetime import datetime
import requests
import logging
import threading

# --- Configuración ---
WHATSAPP_PHONE = "+51902697385"
CALLMEBOT_API_KEY = "2408114"

UPLOAD_FOLDER = "static/uploads"
BASE_ROSTROS_FOLDER = "base_rostros"
ALERTA_GUIÑO_FOLDER = "alertas_guiño"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)
os.makedirs(ALERTA_GUIÑO_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB máximo

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Función para detección de guiño ---
def detectar_guiño(ruta_imagen):
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                return False
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(img_rgb)
            if not resultados.multi_face_landmarks:
                return False
            for rostro in resultados.multi_face_landmarks:
                landmarks = rostro.landmark
                # Obtener puntos de referencia para ojos
                ojo_izq_superior = landmarks[159].y
                ojo_izq_inferior = landmarks[145].y
                ojo_der_superior = landmarks[386].y
                ojo_der_inferior = landmarks[374].y
                
                apertura_izq = abs(ojo_izq_superior - ojo_izq_inferior)
                apertura_der = abs(ojo_der_superior - ojo_der_inferior)
                
                umbral = 0.02  # Umbral para considerar ojo cerrado
                
                # Detectar si un ojo está cerrado y el otro abierto
                if (apertura_izq < umbral and apertura_der >= umbral) or \
                   (apertura_der < umbral and apertura_izq >= umbral):
                    return True
    except Exception as e:
        logging.warning(f"Fallo en detectar_guiño: {e}")
    return False

# --- Enviar mensaje por WhatsApp ---
def enviar_mensaje_whatsapp(texto):
    try:
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={requests.utils.quote(texto)}&apikey={CALLMEBOT_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code not in [200, 201]:
            logging.error(f"❌ Error al enviar WhatsApp: {response.status_code}")
        else:
            logging.info("✅ WhatsApp enviado")
    except Exception as e:
        logging.error(f"❌ Excepción al enviar WhatsApp: {e}")

# --- Función de procesamiento (devuelve resultado) ---
def procesar_imagen(ruta_imagen):
    resultado = {
        "reconocido": False,
        "nombre": None,
        "guino": False,
        "activar_rele": False,
        "mensaje": ""
    }
    
    try:
        umbral = 0.30
        mejor_match = None
        mejor_distancia = float("inf")

        if not os.path.exists(ruta_imagen):
            resultado["mensaje"] = "Error: archivo no existe"
            return resultado

        rostros = os.listdir(BASE_ROSTROS_FOLDER)
        if not rostros:
            resultado["mensaje"] = "No hay rostros en la base de datos"
            return resultado

        for rostro in rostros:
            ruta_rostro = os.path.join(BASE_ROSTROS_FOLDER, rostro)
            try:
                resultado_verify = DeepFace.verify(
                    img1_path=ruta_imagen,
                    img2_path=ruta_rostro,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=False
                )
                distancia = resultado_verify["distance"]
                if distancia <= umbral and distancia < mejor_distancia:
                    mejor_match = rostro
                    mejor_distancia = distancia
            except Exception as e:
                logging.warning(f"Saltando rostro ({rostro}): {e}")
                continue

        if mejor_match:
            resultado["reconocido"] = True
            resultado["nombre"] = mejor_match.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            precision = 90 + ((umbral - mejor_distancia) / umbral) * 10
            resultado["mensaje"] = f"Rostro reconocido: {resultado['nombre']} con {precision:.2f}% precisión"
            logging.info(resultado["mensaje"])
            
            # Enviar WhatsApp de reconocimiento
            enviar_mensaje_whatsapp(f"🔔 {resultado['nombre']} reconocido con {precision:.2f}% precisión.")
            
            # Detectar guiño
            try:
                if detectar_guiño(ruta_imagen):
                    resultado["guino"] = True
                    resultado["mensaje"] = f"Rostro reconocido: {resultado['nombre']}. GUIÑO detectado - EMERGENCIA"
                    resultado["activar_rele"] = False
                    logging.info(resultado["mensaje"])
                    enviar_mensaje_whatsapp(f"🚨 ¡EMERGENCIA! {resultado['nombre']} ha realizado un GUIÑO.")
                    # Guardar alerta
                    alerta = os.path.join(ALERTA_GUIÑO_FOLDER, f"alerta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    imagen_cv = cv2.imread(ruta_imagen)
                    if imagen_cv is not None:
                        cv2.imwrite(alerta, imagen_cv)
                else:
                    resultado["guino"] = False
                    resultado["mensaje"] = f"Rostro reconocido: {resultado['nombre']}. Sin guiño"
                    resultado["activar_rele"] = True
                    logging.info(resultado["mensaje"])
                    enviar_mensaje_whatsapp(f"✅ Acceso permitido: {resultado['nombre']} (sin guiño).")
            except Exception as e:
                logging.error(f"Error en detección de guiño: {e}")
                resultado["guino"] = False
                resultado["activar_rele"] = True
        else:
            resultado["reconocido"] = False
            resultado["nombre"] = None
            resultado["guino"] = False
            resultado["activar_rele"] = False
            resultado["mensaje"] = "Rostro NO reconocido"
            logging.info("❌ Rostro no reconocido")
            enviar_mensaje_whatsapp("❌ Acceso DENEGADO: Rostro no reconocido.")

    except Exception as e:
        logging.error(f"Error crítico en procesamiento: {e}")
        resultado["mensaje"] = f"Error: {str(e)}"
    
    return resultado

@app.route("/")
def index():
    return jsonify({"status": "online", "service": "Reconocimiento facial"})

@app.route("/recibir", methods=["POST"])
def recibir():
    # Verificar si la imagen viene como binario directo (ESP32)
    if request.headers.get('Content-Type') == 'image/jpeg':
        raw_data = request.get_data()
        if not raw_data:
            return jsonify({"error": "No se recibió imagen"}), 400
        
        nombre_archivo = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
        
        with open(ruta_imagen, 'wb') as f:
            f.write(raw_data)
        logging.info(f"📥 Imagen recibida (binario): {nombre_archivo}")
    
    elif 'imagen' in request.files:
        archivo = request.files['imagen']
        if not archivo.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Formato no soportado"}), 400
        
        nombre_archivo = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
        archivo.save(ruta_imagen)
        logging.info(f"📥 Imagen recibida (multipart): {nombre_archivo}")
    
    else:
        return jsonify({"error": "Formato no válido"}), 400

    # Procesar la imagen y obtener resultado
    resultado = procesar_imagen(ruta_imagen)
    
    # Eliminar archivo temporal
    try:
        if os.path.exists(ruta_imagen):
            os.remove(ruta_imagen)
    except Exception as e:
        logging.warning(f"No se pudo eliminar: {e}")

    # Devolver respuesta JSON para el ESP32
    return jsonify({
        "reconocido": resultado["reconocido"],
        "nombre": resultado["nombre"],
        "guino": resultado["guino"],
        "activar_rele": resultado["activar_rele"],
        "mensaje": resultado["mensaje"]
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
