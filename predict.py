from flask import request, jsonify
from PIL import Image
import numpy as np
import io
from model import load_model, preprocess_input
import cv2

model = load_model()

def predict():
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img)

        # Asegurarse de que la imagen esté en formato uint8
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        img_array = preprocess_input(img_array)

        # Convertir la imagen a escala de grises para el modelo de Harris
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Aplicar el modelo para detectar esquinas
        corners = model(gray, 2, 3, 0.04)

        # Obtener las coordenadas de las esquinas detectadas
        coordinates = np.argwhere(corners > corners.mean())

        results = [
            {"coordinate": (int(coord[1]), int(coord[0])), "response": float(corners[coord[0], coord[1]])}
            for coord in coordinates
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
