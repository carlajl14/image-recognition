from flask import request, jsonify
from PIL import Image
import numpy as np
import io
from model import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import setuptools

model = load_model()

def predict():
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        results = [
            {"label": pred[1], "confidence": float(pred[2])}
            for pred in decoded_predictions
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
