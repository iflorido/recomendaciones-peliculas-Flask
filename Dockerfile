FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiamos requirements
COPY requirements.txt .

# 2. EL TRUCO DE LA DIETA:
# Instalamos explícitamente torch para CPU antes de nada.
# Esto evita que se baje la versión monstruosa de 3GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Instalamos el resto.
# Al detectar que torch ya está instalado, no lo volverá a bajar.
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código y los modelos (.npy, .csv)
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]