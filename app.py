import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
from datetime import datetime
import requests
import urllib.parse

# --- Configuración WhatsApp ---
WHATSAPP_PHONE = "+51902697385"
CALLMEBOT_API_KEY = "2408114"

app = Flask(__name__)
BASE_ROSTROS_FOLDER = "base_rostros"
os.makedirs(BASE_ROSTROS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- Clasificadores OpenCV ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Modelo LBPH global ---
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,       # Radio del patrón LBP (1 es estándar)
    neighbors=8,    # Puntos vecinos a comparar
    grid_x=8,       # Grilla horizontal de celdas
    grid_y=8        # Grilla vertical de celdas
)
label_map = {}       # {indice_int: "nombre_persona"}
model_trained = False
pending_batches = {}


# ─────────────────────────────────────────────
#  PREPROCESAMIENTO
# ─────────────────────────────────────────────

def preprocess_image(img):
    """
    Mejora la calidad de la imagen usando CLAHE y filtro bilateral.
    Normaliza diferencias entre cámaras (ESP32-CAM vs cámaras de smartphone).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
#  DETECCIÓN DE ROSTRO
# ─────────────────────────────────────────────

def detect_face(img):
    """
    Detecta el rostro principal en la imagen.
    Retorna (face_roi_gray, face_rect) o (None, None) si no hay rostro.

    Parámetros relajados para imágenes del ESP32-CAM (ruidosas, baja resolución):
    - scaleFactor=1.05 → busca rostros en más escalas (más lento pero más sensible)
    - minNeighbors=3   → menos estricto que 5, acepta detecciones con menos confirmaciones
    - minSize=(30,30)  → acepta rostros más pequeños en el frame (antes era 50×50)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Intento 1: parámetros relajados para ESP32-CAM
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        logging.warning(
            f"No se detectó rostro. Tamaño imagen: {img.shape[1]}×{img.shape[0]} | "
            f"Brillo medio: {np.mean(gray):.1f}"
        )
        return None, None

    # Tomar el rostro más grande detectado (más confiable)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    logging.info(f"Rostro detectado: x={x} y={y} w={w} h={h}")

    face_roi_gray = gray[y:y+h, x:x+w]
    # Normalizar tamaño para que LBPH siempre trabaje con el mismo input
    face_roi_gray = cv2.resize(face_roi_gray, (100, 100))
    return face_roi_gray, (x, y, w, h)


# ─────────────────────────────────────────────
#  CARGA Y ENTRENAMIENTO CON LBPH
# ─────────────────────────────────────────────

def load_known_faces():
    """
    Lee todas las fotos de base_rostros/, detecta el rostro en cada una
    y entrena el modelo LBPH.

    Convención de nombres de archivo:
        nombre_1.jpg, nombre_2.jpg, nombre_3.jpg ...
    El texto antes del primer '_' (o antes del '.') se usa como nombre.
    Ejemplo: juan_1.jpg → "juan" | maria.jpg → "maria"
    """
    global label_map, model_trained

    faces_list = []
    labels_list = []
    name_to_label = {}   # {"juan": 0, "maria": 1, ...}
    label_map = {}       # {0: "juan", 1: "maria", ...}
    current_label = 0

    if not os.path.exists(BASE_ROSTROS_FOLDER):
        logging.warning("Carpeta base_rostros/ no encontrada.")
        return

    archivos = [
        f for f in os.listdir(BASE_ROSTROS_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not archivos:
        logging.warning("No hay fotos en base_rostros/.")
        return

    for filename in archivos:
        img_path = os.path.join(BASE_ROSTROS_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"No se pudo leer: {filename}")
            continue

        img_proc = preprocess_image(img)
        face_roi, _ = detect_face(img_proc)

        if face_roi is None:
            logging.warning(f"No se detectó rostro en: {filename}")
            continue

        # Extraer nombre: "juan_2.jpg" → "juan"
        base_name = filename.rsplit('.', 1)[0]          # "juan_2"
        nombre = base_name.split('_')[0]                # "juan"

        if nombre not in name_to_label:
            name_to_label[nombre] = current_label
            label_map[current_label] = nombre
            current_label += 1

        faces_list.append(face_roi)
        labels_list.append(name_to_label[nombre])
        logging.info(f"Foto cargada: {filename} → persona '{nombre}'")

    if len(faces_list) == 0:
        logging.error("Ninguna foto válida para entrenar. Revisar base_rostros/.")
        model_trained = False
        return

    recognizer.train(faces_list, np.array(labels_list))
    model_trained = True
    logging.info(
        f"Modelo LBPH entrenado con {len(faces_list)} fotos "
        f"de {len(label_map)} personas: {list(label_map.values())}"
    )


# ─────────────────────────────────────────────
#  RECONOCIMIENTO FACIAL (LBPH)
# ─────────────────────────────────────────────

def recognize_face(img, confidence_threshold=150):
    """
    Reconoce el rostro en la imagen usando LBPH.

    En LBPH, la confianza es una DISTANCIA: menor valor = mejor coincidencia.
    - < 80   → reconocimiento muy seguro (misma cámara)
    - 80–120 → aceptable
    - 120–150→ tolerable (diferencia de cámara ESP32 vs smartphone)
    - > 150  → demasiado diferente, se considera desconocido

    Umbral en 150 para tolerar la diferencia entre ESP32-CAM y fotos de referencia
    tomadas con smartphone. Bajar a 120 si aparecen falsos positivos.
    """
    if not model_trained:
        logging.warning("Modelo LBPH no entrenado aún.")
        return None, 0, None

    img_proc = preprocess_image(img)
    face_roi, face_rect = detect_face(img_proc)

    if face_roi is None:
        return None, 0, None

    label, distance = recognizer.predict(face_roi)

    # Log siempre visible para calibrar el umbral
    logging.info(
        f"LBPH predict → label={label} ({label_map.get(label,'?')}) "
        f"distancia={distance:.2f} umbral={confidence_threshold}"
    )

    if distance > confidence_threshold:
        # Distancia alta = no reconocido
        logging.warning(
            f"Rostro detectado pero distancia {distance:.2f} > umbral {confidence_threshold} → NO reconocido"
        )
        return None, 0, face_rect

    nombre = label_map.get(label, "Desconocido")
    # Convertir distancia a porcentaje legible (inverso normalizado)
    confianza_pct = max(0, round((1 - distance / confidence_threshold) * 100, 2))

    logging.info(f"Reconocido: {nombre} | Distancia LBPH: {distance:.2f} | Confianza: {confianza_pct}%")
    return nombre, confianza_pct, face_rect


# ─────────────────────────────────────────────
#  DETECCIÓN DE GUIÑO (MEJORADA)
# ─────────────────────────────────────────────

def detect_wink(img, face_rect):
    """
    Detecta guiño analizando SOLO la franja de ojos del rostro.
    Usa asimetría de brillo + diferencia de varianza (textura).

    Mejora clave vs versión anterior:
    - Recorta únicamente la banda ocular (20%–50% vertical del rostro)
    - Elimina ruido de mejillas, frente e iluminación lateral
    - Umbrales más precisos porque la señal es más limpia
    """
    if face_rect is None:
        return False

    x, y, w, h = face_rect
    if w <= 0 or h <= 0:
        return False

    face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return False

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # CLAHE para normalizar contraste dentro del rostro
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_face = clahe.apply(gray_face)

    # ── Recorte de la banda ocular ──────────────────────────────────────
    # Entre el 20% y el 50% vertical del rostro es donde están los ojos.
    # Ignorar frente (0–20%) y parte inferior (50–100%) elimina el ruido
    # de iluminación que causaba falsos positivos/negativos en la versión anterior.
    eye_top    = int(h * 0.20)
    eye_bottom = int(h * 0.50)
    eye_band   = gray_face[eye_top:eye_bottom, :]

    if eye_band.size == 0:
        return False

    half = eye_band.shape[1] // 2
    left_eye  = eye_band[:, :half]
    right_eye = eye_band[:, half:]

    # ── Métrica 1: Asimetría de brillo medio ───────────────────────────
    mean_l = np.mean(left_eye)
    mean_r = np.mean(right_eye)
    denom_mean = max(mean_l, mean_r, 1e-5)
    asymmetry = abs(mean_l - mean_r) / denom_mean

    # ── Métrica 2: Diferencia de varianza (textura) ─────────────────────
    # Un ojo cerrado tiene menos textura (varianza más baja) que uno abierto.
    var_l = np.var(left_eye)
    var_r = np.var(right_eye)
    denom_var = max(var_l, var_r, 1e-5)
    var_ratio = abs(var_l - var_r) / denom_var

    # ── Métrica 3: Diferencia de pico de brillo ─────────────────────────
    max_l = float(np.max(left_eye))
    max_r = float(np.max(right_eye))
    denom_max = max(max_l, max_r, 1e-5)
    max_diff = abs(max_l - max_r) / denom_max

    # ── Umbrales calibrados para la banda ocular ────────────────────────
    ASYM_THR    = 0.06   # Asimetría de brillo
    VAR_THR     = 0.18   # Diferencia de textura
    MAX_THR     = 0.09   # Diferencia de pico

    wink = (
        (asymmetry > ASYM_THR and var_ratio > VAR_THR and max_diff > MAX_THR) or  # Las 3 métricas juntas
        (asymmetry > 0.08 and var_ratio > 0.28) or   # Asimetría alta + mucha diferencia de textura
        (asymmetry > 0.08 and max_diff > MAX_THR) or  # Asimetría alta + pico diferente
        (var_ratio > 0.35)                             # Varianza extremadamente distinta = guiño claro
    )

    logging.info(
        f"Guiño → Asym={asymmetry:.3f} VarRatio={var_ratio:.3f} "
        f"MaxDiff={max_diff:.3f} → {wink}"
    )
    return wink


# ─────────────────────────────────────────────
#  WHATSAPP
# ─────────────────────────────────────────────

def send_whatsapp_message(message):
    try:
        encoded = urllib.parse.quote(message)
        url = (
            f"https://api.callmebot.com/whatsapp.php"
            f"?phone={WHATSAPP_PHONE}&text={encoded}&apikey={CALLMEBOT_API_KEY}"
        )
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logging.info("WhatsApp enviado correctamente.")
        else:
            logging.error(f"Error WhatsApp HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Excepción WhatsApp: {e}")


# ─────────────────────────────────────────────
#  RUTAS FLASK
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "modelo": "LBPH",
        "personas_registradas": len(label_map),
        "nombres": list(label_map.values()),
        "modelo_entrenado": model_trained
    })


@app.route("/recargar_rostros", methods=["POST"])
def recargar_rostros():
    """
    Endpoint opcional: recarga el modelo LBPH sin reiniciar el servidor.
    Útil cuando se agregan nuevas fotos a base_rostros/ en caliente.
    """
    try:
        load_known_faces()
        return jsonify({
            "ok": True,
            "personas": list(label_map.values()),
            "modelo_entrenado": model_trained
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/recibir", methods=["POST"])
def recibir():
    try:
        data = request.get_json()
        if not data or 'foto' not in data or 'batch_id' not in data:
            return jsonify({
                "error": "Faltan campos: foto y batch_id",
                "activar_rele": False
            }), 400

        batch_id  = data['batch_id']
        foto_b64  = data['foto']

        # Decodificar imagen
        img_data = base64.b64decode(foto_b64)
        nparr    = np.frombuffer(img_data, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Error decodificando imagen",
                "activar_rele": False
            }), 400

        logging.info(f"Batch {batch_id} — foto recibida ({len(img_data)} bytes)")

        # Reconocimiento facial con LBPH
        nombre, confianza, face_rect = recognize_face(img)

        # Detección de guiño solo si el rostro fue reconocido
        tiene_guino = False
        if nombre:
            tiene_guino = detect_wink(img, face_rect)

        individual_response = {
            "activar_rele": bool(nombre is not None and not tiene_guino),
            "nombre":       nombre if nombre else None,
            "confianza":    float(confianza),
            "guino":        bool(tiene_guino)
        }

        # ── Acumulación del batch ────────────────────────────────────────
        if batch_id not in pending_batches:
            pending_batches[batch_id] = {
                "count": 0,
                "results": [],
                "final_notified": False
            }

        batch = pending_batches[batch_id]
        batch["count"]   += 1
        batch["results"].append(individual_response)

        # Al recibir la 3ª foto → evaluar resultado global y notificar
        if batch["count"] >= 3 and not batch["final_notified"]:
            batch["final_notified"] = True
            results = batch["results"]

            any_wink             = any(r.get("guino", False)       for r in results)
            any_recognized_ok    = any(r.get("activar_rele", False) for r in results)

            if any_wink:
                names    = [r["nombre"] for r in results if r.get("nombre")]
                name_str = names[0] if names else "Desconocido"
                msg = (
                    f"⚠️ Timbre activado. Rostro reconocido: {name_str}. "
                    f"Se detectó un GUIÑO (posible emergencia)."
                )
                logging.info(f"Batch {batch_id} — resultado final: GUIÑO DETECTADO")

            elif any_recognized_ok:
                name_str = next(
                    (r["nombre"] for r in results if r.get("activar_rele") and r.get("nombre")),
                    "Reconocido"
                )
                msg = f"✅ Timbre activado. Rostro reconocido: {name_str}. Sin guiño."
                logging.info(f"Batch {batch_id} — resultado final: ACCESO PERMITIDO")

            else:
                msg = "❗ Timbre activado. Rostro NO reconocido."
                logging.info(f"Batch {batch_id} — resultado final: ACCESO DENEGADO")

            send_whatsapp_message(msg)
            del pending_batches[batch_id]

        return jsonify(individual_response), 200

    except Exception as e:
        logging.error(f"Error en /recibir: {e}")
        return jsonify({"error": str(e), "activar_rele": False}), 500


# ─────────────────────────────────────────────
#  ARRANQUE
# ─────────────────────────────────────────────

load_known_faces()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
