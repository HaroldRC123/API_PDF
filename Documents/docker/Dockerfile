# Usamos una imagen oficial de Python ligera
FROM python:3.10-slim

# Evita que Python genere archivos .pyc y fuerza salida en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalamos Tesseract OCR y el paquete de idioma español (sin paquetes gráficos obsoletos)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los requerimientos y los instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos (tu main.py)
COPY . .

# Exponemos el puerto
EXPOSE 8000

# Comando para arrancar la API con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
