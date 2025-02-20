# Utilizar una imagen base de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Copiar el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que Flask estará escuchando
EXPOSE 5000

# Definir el comando por defecto para ejecutar la aplicación
CMD ["python", "main.py"]