from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import cv2

app = Flask(__name__)

# Cargar las clases de objetos de COCO (personas, autos, perros, etc.)
LABELS_FILE = "coco.names"  # Asegúrate de tener este archivo con las clases de COCO
CONFIG_FILE = "yolov3.cfg"  # Archivo de configuración de YOLO
WEIGHTS_FILE = "yolov3.weights"  # Pesos preentrenados de YOLOv3

# Cargar las clases de COCO
with open(LABELS_FILE, "r") as f:
    CLASSES = f.read().strip().split("\n")

# Cargar el modelo YOLO
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_objects(image):
    """
    Detecta objetos en una imagen usando YOLOv3.
    """
    H, W = image.shape[:2]

    # Crear un blob desde la imagen
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtener los nombres de las capas de salida de YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Hacer la predicción
    layer_outputs = net.forward(output_layers)

    results = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Filtrar objetos con baja confianza
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                results.append({
                    "object": CLASSES[class_id],
                    "confidence": float(confidence),
                    "bounding_box": {"x": x, "y": y, "width": int(width), "height": int(height)}
                })

    return results

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Leer la imagen
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert("RGB")  # Asegurar que está en RGB
        img_array = np.array(img)

        # Convertir a formato OpenCV
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detectar objetos en la imagen
        detections = detect_objects(img_cv2)

        return jsonify({"objects_detected": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)