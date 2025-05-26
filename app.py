import os
from flask import Flask, request
from datetime import datetime
import cv2
import mediapipe as mp
from deepface import DeepFace
import requests

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
BASE_ROSTROS_FOLDER = "base_rostros"
ALERTA_GUIÑO_FOLDER = "alertas_guiño"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)
os.makedirs(ALERTA_GUIÑO_FOLDER, exist_ok=True)

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

WHATSAPP_PHONE = "+51902697385"
CALLMEBOT_API_KEY = "2408114"

def enviar_mensaje_whatsapp(texto):
    try:
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={requests.utils.quote(texto)}&apikey={CALLMEBOT_API_KEY}"
        response = requests.get(url)
        if response.status_code in [200, 201]:
            print("✅ Mensaje enviado por WhatsApp")
        else:
            print(f"❌ Error al enviar mensaje: {response.status_code}")
    except Exception as e:
        print("❌ Excepción al enviar WhatsApp:", e)

def detectar_guiño(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return False
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(img_rgb)
    if not resultados.multi_face_landmarks:
        return False
    for rostro in resultados.multi_face_landmarks:
        try:
            landmarks = rostro.landmark
            apertura_izq = abs(landmarks[159].y - landmarks[145].y)
            apertura_der = abs(landmarks[386].y - landmarks[374].y)
            umbral = 0.02
            if (apertura_izq < umbral and apertura_der >= umbral) or (apertura_der < umbral and apertura_izq >= umbral):
                return True
        except:
            continue
    return False

@app.route("/")
def index():
    return "Servidor de reconocimiento activo."

@app.route("/recibir", methods=["POST"])
def recibir_imagen():
    if 'file' not in request.files:
        return "No se envió imagen", 400

    archivo = request.files['file']
    ruta_imagen = os.path.join(UPLOAD_FOLDER, "foto_recibida.jpg")
    archivo.save(ruta_imagen)

    try:
        mejor_match = None
        mejor_distancia = float("inf")
        umbral = 0.30

        for nombre_archivo in os.listdir(BASE_ROSTROS_FOLDER):
            ruta_rostro = os.path.join(BASE_ROSTROS_FOLDER, nombre_archivo)
            resultado = DeepFace.verify(
                img1_path=ruta_imagen,
                img2_path=ruta_rostro,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=True
            )
            distancia = resultado["distance"]
            if distancia <= umbral and distancia < mejor_distancia:
                mejor_match = nombre_archivo
                mejor_distancia = distancia

        if mejor_match:
            precision = 90 + ((umbral - mejor_distancia) / umbral) * 10
            mensaje = f"🔔 {mejor_match} reconocido con {precision:.2f}% de precisión."
            enviar_mensaje_whatsapp(mensaje)

            if detectar_guiño(ruta_imagen):
                alerta = f"🚨 ¡Guiño detectado en {mejor_match}!"
                enviar_mensaje_whatsapp(alerta)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ruta_alerta = os.path.join(ALERTA_GUIÑO_FOLDER, f"alerta_{timestamp}.jpg")
                cv2.imwrite(ruta_alerta, cv2.imread(ruta_imagen))

            return f"{mejor_match} reconocido con {precision:.2f}% de precisión", 200
        else:
            return "❌ Rostro no reconocido con suficiente precisión", 404

    except Exception as e:
        print("❌ ERROR DETECTADO:", e)
        return f"❌ Error al procesar imagen: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
