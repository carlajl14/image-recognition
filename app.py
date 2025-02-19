from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Cargar el modelo pre-entrenado MobileNetV2
def load_model():
    return MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar el encabezado de RapidAPI
    #rapidapi_secret = request.headers.get('X-RapidAPI-Proxy-Secret')
    #if rapidapi_secret != '5ec60b00-eddc-11ef-9c15-59f608f93374':
        #return jsonify({'error': 'Acceso no autorizado'}), 403

    # Obtener la imagen de la solicitud
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Preprocesar la imagen
    img = img.resize((224, 224))  # Cambiar el tamaño para MobileNetV2
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Cargar el modelo y realizar la predicción
    model = load_model()
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

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
