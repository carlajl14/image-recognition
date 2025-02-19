from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Cargar el modelo pre-entrenado InceptionV3
model = InceptionV3(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar el encabezado de RapidAPI
    rapidapi_secret = request.headers.get('X-RapidAPI-Proxy-Secret')
    if rapidapi_secret != '5ec60b00-eddc-11ef-9c15-59f608f93374':
        return jsonify({'error': 'Acceso no autorizado'}), 403

    # Obtener la imagen de la solicitud
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Preprocesar la imagen
    img = img.resize((299, 299))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizar la predicción
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Obtener solo la predicción más probable

    # Formatear la respuesta
    top_prediction = decoded_predictions[0]
    result = {
        "label": top_prediction[0],
        "description": top_prediction[1],
        "score": float(top_prediction[2])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)
