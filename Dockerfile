# 1. IMAGEN BASE
# Usamos una imagen de Python base compatible con Azure
FROM mcr.microsoft.com/azure-app-service/python:3.11-slim

# 2. DIRECTORIO DE TRABAJO
# Establecemos el directorio donde se ejecutará la aplicación
WORKDIR /home/site/wwwroot

# 3. INSTALACIÓN DE DEPENDENCIAS (EL PASO CLAVE)
# Copiamos solo el archivo requirements.txt para aprovechar el caching de Docker
COPY requirements.txt .

# Instalamos todas las librerías, incluyendo las grandes (pandas, torch, etc.)
# Esto garantiza que pandas esté disponible en la ruta correcta.
RUN pip install --no-cache-dir -r requirements.txt

# 4. COPIA DEL CÓDIGO RESTANTE
# Copiamos el resto de los archivos (app.py, plantillas, archivos .csv, archivos .npy)
COPY . .

# 5. COMANDO DE INICIO
# Configuramos el comando de inicio que ejecutará Azure.
# Utilizamos la sintaxis gunicorn archivo_principal:instancia_app.
# app.py tiene la instancia Flask llamada 'app', por lo que es app:app
CMD gunicorn --bind 0.0.0.0 --timeout 600 --workers 4 app:app