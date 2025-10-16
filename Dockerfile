# Usa Python 3.11 como base
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del repositorio al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expone el puerto que usa Flask
EXPOSE 5000

# Comando para ejecutar la app
CMD ["python", "app.py"]
