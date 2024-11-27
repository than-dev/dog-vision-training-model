import pandas as pd
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def display_prediction(image, predicted_breed):
    # Se a imagem já é um numpy array com o formato (1, 224, 224, 3), remova a primeira dimensão
    if isinstance(image, np.ndarray):
        img = np.squeeze(image)  # Remove a dimensão do batch (1, 224, 224, 3) -> (224, 224, 3)
    else:
        img = load_img(image, target_size=(224, 224))  # Se for caminho, carregamos com load_img
    
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

def load_images_from_csv(csv_path, img_folder, limit=None):
    print(f"Iniciando carregamento de imagens a partir de: {csv_path}")
    
    # Carregar o CSV com limite opcional
    try:
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)
        print(f"CSV carregado. Total de entradas a serem processadas: {len(df)}")
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        return None, None

    images = []
    labels = []

    for index, row in df.iterrows():
        img_id = row['id']
        img_extension = '.jpg'
        img_path = os.path.join(img_folder, img_id + img_extension)
        
        if index % 100 == 0:
            print(f"Processando imagem {index + 1}/{len(df)}: {img_path}")

        try:
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(row['breed'])
        except FileNotFoundError:
            print(f"Imagem não encontrada: {img_path}")
        except Exception as e:
            print(f"Erro ao processar a imagem {img_path}: {e}")

    print(f"Carregamento concluído. Total de imagens processadas: {len(images)}")
    
    # Verificar consistência entre images e labels
    if len(images) != len(labels):
        print("Erro: O número de imagens não corresponde ao número de labels.")
        return None, None

    return np.array(images), np.array(labels)


def predict_breed(image, model, label_encoder):
    # Verifica se a imagem precisa ser carregada (no caso do path)
    if isinstance(image, np.ndarray):
        img = image  # Se já for numpy array, usamos diretamente
    else:
        # Se for um caminho de imagem, carregamos e preprocessamos
        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

    # Fazendo a previsão
    prediction = model.predict(img)
    predicted_breed = label_encoder.inverse_transform([np.argmax(prediction)])

    # Exibe a imagem com a previsão (se for uma imagem preprocessada, não passamos o caminho)
    display_prediction(img, predicted_breed)

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