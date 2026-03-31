FROM python:3.10-slim

# Logs en tiempo real (muy útil para ver entrenamiento)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Permite pasar argumento (Golden, Eugene, Cocoa)
ENTRYPOINT ["python", "client.py"]