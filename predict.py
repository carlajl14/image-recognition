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

        # Verificar tipo de dato antes de la conversión
        print(f"Tipo de dato antes de preprocess_input: {img_array.dtype}")

        # Aplicar preprocesamiento del modelo
        img_array = preprocess_input(img_array)

        # Verificar tipo de dato después del preprocesamiento
        print(f"Tipo de dato después de preprocess_input: {img_array.dtype}")

        # Convertir a uint8 si es necesario
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        print(f"Tipo de dato después de conversión a uint8: {img_array.dtype}")

        # Convertir a escala de grises para Harris
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
