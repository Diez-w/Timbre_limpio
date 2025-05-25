import requests

# Dirección local del servidor Flask
url = "http://127.0.0.1:5000/recibir"

# Imagen simulada tomada por la ESP32-CAM
imagen = {'imagen': open('foto_prueba.jpeg', 'rb')}  # asegúrate de tener esta imagen

response = requests.post(url, files=imagen)

print("Status:", response.status_code)

if response.status_code == 200:
    texto = response.text
    if "⚠️" in texto:
        print("\033[93m" + texto + "\033[0m")  # Amarillo (alerta guiño)
    else:
        print("\033[92m" + texto + "\033[0m")  # Verde (reconocimiento exitoso)
elif response.status_code == 404:
    print("\033[91m❌ Rostro no reconocido\033[0m")  # Rojo
else:
    print("\033[91m" + response.text + "\033[0m")  # Rojo para errores 500
