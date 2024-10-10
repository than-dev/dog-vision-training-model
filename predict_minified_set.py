import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from utils import select_random_image, load_labels_from_csv, preprocess_image

def load_model_and_predict(model_path, image_path, target_size=(224, 224)):
    # Carrega o modelo salvo, incluindo a camada customizada 'KerasLayer'
    model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Carrega e processa a imagem
    img_array = preprocess_image(image_path)

    # Faz a previsão
    predictions = model.predict(img_array)
    
    # Encontra a classe com a maior probabilidade
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    class_labels = load_labels_from_csv()
    predicted_label = class_labels[predicted_class]

    confidence = predictions[0][predicted_class]  # Confiança da classe prevista
    
    # Exibir a imagem junto com o resultado
    plt.imshow(image.load_img(image_path))
    plt.title(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    # Retorna a previsão e a confiança
    return predicted_label, confidence


load_model_and_predict(
    'models/20240205-18121707156779-1000-images-mobilenetv2-Adam.h5',
    select_random_image('dog-breed-identification/test')
);