FROM python:3.12-slim

WORKDIR /

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verifica que los archivos se copian correctamente
RUN ls -la /

# Verifica que ultralytics está instalado
RUN pip list | grep ultralytics

EXPOSE 8080

CMD ["python", "app.py"]
