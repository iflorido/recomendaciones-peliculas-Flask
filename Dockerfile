# Imagen base Python 3.11 ligera
FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar caché
COPY requirements.txt .

# Instalar dependencias de Python
# El flag --no-cache-dir reduce el peso de la imagen final
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto interno de la aplicación
EXPOSE 5000

# Ejecutar con Gunicorn (1 worker para ahorrar RAM en VPS, timeout alto para cargar modelos)
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]