from deepface import DeepFace
import cv2
import os
import numpy as np

ORIGEN = "base_original"
DESTINO = "base_rostros"

def alinear_base():
    if not os.path.exists(DESTINO):
        os.makedirs(DESTINO)

    for nombre_archivo in os.listdir(ORIGEN):
        ruta_entrada = os.path.join(ORIGEN, nombre_archivo)
        try:
            rostros = DeepFace.extract_faces(
                img_path=ruta_entrada,
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )
            if rostros:
                rostro_alineado = rostros[0]["face"]
                rostro_alineado = (np.array(rostro_alineado) * 255).astype("uint8")
                rostro_alineado = cv2.cvtColor(rostro_alineado, cv2.COLOR_RGB2BGR)
                ruta_salida = os.path.join(DESTINO, nombre_archivo)
                cv2.imwrite(ruta_salida, rostro_alineado)
                print(f"Guardado: {ruta_salida}")
        except Exception as e:
            print(f"Error con {nombre_archivo}: {e}")

if __name__ == "__main__":
    alinear_base()
