import cv2
import numpy as np

def load_model():
    # Aquí deberías cargar el modelo de Harris o cualquier otro modelo más ligero.
    return cv2.cornerHarris

def preprocess_input(image_array):
    # Normalizar la imagen
    return image_array / 255.0
