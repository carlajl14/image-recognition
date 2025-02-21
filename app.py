from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolo11n.pt")

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

        # Realizar detección con YOLOv8
        results = model(img_array)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "object": model.names[int(box.cls)],  # Nombre del objeto detectado
                    "confidence": float(box.conf),  # Confianza de la detección
                    "bounding_box": {
                        "x_min": float(box.xyxy[0][0]),
                        "y_min": float(box.xyxy[0][1]),
                        "x_max": float(box.xyxy[0][2]),
                        "y_max": float(box.xyxy[0][3]),
                    }
                })

        return jsonify({"objects_detected": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
