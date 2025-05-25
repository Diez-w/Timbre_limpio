import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, request
from deepface import DeepFace
import cv2
import mediapipe as mp
from datetime import datetime
import requests

# Configuración de WhatsApp (CallMeBot)
WHATSAPP_PHONE = "+51902697385"  # <-- tu número real
CALLMEBOT_API_KEY = "2408114"  # <-- tu API key de CallMeBot

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
BASE_ROSTROS_FOLDER = "base_rostros"
ALERTA_GUIÑO_FOLDER = "alertas_guiño"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)
os.makedirs(ALERTA_GUIÑO_FOLDER, exist_ok=True)

# Inicializar MediaPipe Face Mesh en modo imagen
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

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

def enviar_mensaje_whatsapp(texto):
    try:
        url = f"https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_PHONE}&text={requests.utils.quote(texto)}&apikey={CALLMEBOT_API_KEY}"
        response = requests.get(url)
        if response.status_code not in [200, 201]:
            print(f"❌ Error al enviar mensaje de WhatsApp: {response.status_code}")
        else:
            print("✅ Mensaje enviado por WhatsApp")
    except Exception as e:
        print(f"❌ Excepción al enviar mensaje de WhatsApp: {e}")

@app.route("/")
def index():
    return "Servidor de reconocimiento facial activo"

@app.route("/recibir", methods=["POST"])
def recibir_imagen():
    if 'imagen' not in request.files:
        return "No se envió ningún archivo", 400

    archivo = request.files['imagen']
    ruta_imagen = os.path.join(UPLOAD_FOLDER, "foto_prueba.jpeg")
    archivo.save(ruta_imagen)

    try:
        umbral = 0.30
        mejor_match = None
        mejor_distancia = float("inf")

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
            respuesta = (
                f"✅ Rostro reconocido: {mejor_match}\n"
                f"Distancia: {mejor_distancia:.4f}\n"
                f"Precisión estimada: {precision:.2f}%"
            )
            enviar_mensaje_whatsapp(f"🔔 Alerta: {mejor_match} ha sido reconocido con {precision:.2f}% de precisión.")

            if detectar_guiño(ruta_imagen):
                respuesta += "\n⚠️ Alerta: se detectó guiño (posible situación de emergencia)"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ruta_alerta = os.path.join(ALERTA_GUIÑO_FOLDER, f"alerta_{timestamp}.jpeg")
                cv2.imwrite(ruta_alerta, cv2.imread(ruta_imagen))
                enviar_mensaje_whatsapp("🚨 ¡Emergencia! Se detectó guiño durante el reconocimiento facial.")

            return respuesta

        else:
            return "❌ Rostro no reconocido con la precisión mínima requerida (≥90%)", 404

    except Exception as e:
        print("❌ ERROR DETECTADO:", e)
        return f"❌ Error al procesar la imagen: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
