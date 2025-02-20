from skimage import data, io, filters, feature, color
from skimage.transform import resize
import numpy as np

def load_model():
    # Cargar el modelo preentrenado, en este caso el modelo de detección de características de Harris
    return feature.corner_harris

def preprocess_input(image_array):
    # Normalizar la imagen
    return image_array / 255.0
