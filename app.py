from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

def load_model():
    return MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = load_model()
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    results = [
        {"label": pred[1], "confidence": float(pred[2])}
        for pred in decoded_predictions
    ]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False)
