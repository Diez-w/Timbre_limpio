import os
from flask import Flask, request
from deepface import DeepFace
import cv2
import mediapipe as mp
from datetime import datetime
import requests
import logging

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
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Máximo 2MB por imagen

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def detectar_guiño(ruta_imagen):
    try:
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            return False
        img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = face_mesh.process(img_rgb)
        if not resultados.multi_face_landmarks:
            return False
        for rostro in resultados.multi_face_landmarks:
            landmarks = rostro.landmark
            apertura_izq = abs(landmarks[159].y - landmarks[145].y)
            apertura_der = abs(landmarks[386].y - landmarks[374].y)
            umbral = 0.02
            if (apertura_izq < umbral and apertura_der >= umbral) or (apertura_der < umbral and apertura_izq >= umbral):
                return True
    except Exception as e:
        logging.warning(f"Fallo en detectar_guiño: {e}")
    return False

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

@app.route("/")
def index():
    return "Servidor de reconocimiento activo"

@app.route("/recibir", methods=["POST"])
def recibir():
    if 'imagen' not in request.files:
        return "❌ No se envió imagen", 400

    archivo = request.files['imagen']
    nombre_archivo = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
    archivo.save(ruta_imagen)
    logging.info(f"📥 Imagen recibida: {nombre_archivo}")

    try:
        umbral = 0.30
        mejor_match = None
        mejor_distancia = float("inf")

        rostros = os.listdir(BASE_ROSTROS_FOLDER)
        if not rostros:
            return "⚠️ No hay imágenes en la base de rostros", 500

        for rostro in rostros:
            ruta_rostro = os.path.join(BASE_ROSTROS_FOLDER, rostro)
            try:
                resultado = DeepFace.verify(
                    img1_path=ruta_imagen,
                    img2_path=ruta_rostro,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=False
                )
                distancia = resultado["distance"]
                if distancia <= umbral and distancia < mejor_distancia:
                    mejor_match = rostro
                    mejor_distancia = distancia
            except Exception as e:
                logging.warning(f"No se pudo comparar con {rostro}: {e}")
                continue

        if mejor_match:
            precision = 90 + ((umbral - mejor_distancia) / umbral) * 10
            mensaje = f"🔔 {mejor_match} reconocido con {precision:.2f}% precisión."
            logging.info(mensaje)
            enviar_mensaje_whatsapp(mensaje)

            if detectar_guiño(ruta_imagen):
                alerta = os.path.join(ALERTA_GUIÑO_FOLDER, f"alerta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(alerta, cv2.imread(ruta_imagen))
                enviar_mensaje_whatsapp("🚨 ¡Emergencia! Se detectó un guiño.")
            return mensaje, 200

        return "❌ Rostro no reconocido con precisión mínima requerida (≥90%)", 404

    except Exception as e:
        logging.error("❌ ERROR DETECTADO:", exc_info=True)
        return f"❌ Error al procesar imagen: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=False)

