# Configuración de Gunicorn para Render gratuito
# El timeout largo es necesario porque MediaPipe con TFLite
# puede tardar más en la primera ejecución (cold start de CPU).

timeout = 120        # segundos antes de matar un worker (default: 30)
workers = 1          # un solo worker para no multiplicar el uso de RAM
threads = 1          # un solo hilo por worker
worker_class = "sync"
bind = "0.0.0.0:10000"
