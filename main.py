import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tf_keras

from PIL import Image
import numpy as np
import tensorflow_hub as hub
from utils import predict_breed, display_prediction  

# Suprimindo mensagens de log do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Caminho do modelo salvo
MODEL_PATH = 'models/meu_modelo.h5'

def preprocess_image(image_path, target_size=(224, 224)):
    """Carrega e preprocessa a imagem para entrada no modelo."""
    try:
        # Abrindo a imagem com PIL (em vez de cv2)
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0  # Normaliza os pixels
        return np.expand_dims(image_array, axis=0)  # Adiciona uma dimensão extra para batch
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        sys.exit(1)

def main():
    # Carregar o modelo e o codificador de labels
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo não encontrado em {MODEL_PATH}. Certifique-se de treinar o modelo primeiro.")
        sys.exit(1)
    
    print("Carregando modelo...")
    model = tf_keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

    print("Carregando encoder de labels...")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('models/label_classes.npy', allow_pickle=True)
    
    # Receber o caminho da imagem
    image_path = input("Insira o caminho da imagem de um cachorro: ")

    if not os.path.exists(image_path):
        print(f"Imagem não encontrada no caminho: {image_path}")
        sys.exit(1)

    print("Preprocessando a imagem...")
    preprocessed_image = preprocess_image(image_path)
    
    print("Realizando previsão...")
    predicted_breed = predict_breed(preprocessed_image, model, label_encoder)
    
    print(f"Raça prevista: {predicted_breed}")
    
    # Mostrar imagem e previsão
    print("Exibindo a imagem com a previsão...")
    display_prediction(image_path, predicted_breed)

if __name__ == "__main__":
    main()


