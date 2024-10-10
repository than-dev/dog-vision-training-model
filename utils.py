import pandas as pd
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def display_prediction(image_path, predicted_breed):
    # Carregar a imagem
    img = load_img(image_path, target_size=(224, 224))
    
    # Criar uma figura
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted Breed: {predicted_breed}")
    plt.axis('off')  # Esconder os eixos
    plt.show()

def select_random_image(directory):
    # Lista todos os arquivos no diretório
    images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Seleciona uma imagem aleatória
    if images:
        random_image = random.choice(images)
        return os.path.join(directory, random_image)
    else:
        return None

    if __name__ == "__main__":
        # Defina o caminho do diretório de imagens
        test_directory = 'dog-breed-identification/test'  # Ajuste o caminho conforme necessário
        
        random_image_path = select_random_image(test_directory)
        
        if random_image_path:
            print(f'Randomly selected image: {random_image_path}')
        else:
            print('No images found in the directory.')

def load_images_from_csv(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_id = row['id'] 
        img_extension = '.jpg'  
        img_path = os.path.join(img_folder, img_id + img_extension)  
        try:
            img = load_img(img_path, target_size=(224, 224))  
            img = img_to_array(img) / 255.0  
            images.append(img)
            labels.append(row['breed'])
        except FileNotFoundError:
            print(f"Imagem não encontrada: {img_path}")  
    return np.array(images), np.array(labels)

def predict_breed(image_path, model, label_encoder):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_breed = label_encoder.inverse_transform([np.argmax(prediction)])

    display_prediction(image_path, predict_breed)

    return predicted_breed

def load_model(model_path):
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def load_labels_from_csv():
    # Carrega o arquivo CSV contendo os nomes das classes
    labels_df = pd.read_csv('dog-breed-identification/labels.csv')
    # Supondo que o CSV tenha uma coluna chamada 'breed' com os nomes das classes
    class_labels = labels_df['breed'].values
    
    return class_labels

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  
    return img_array