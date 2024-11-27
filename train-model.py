import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from create_model import create_model  
from utils import load_images_from_csv, predict_breed, select_random_image, display_prediction, preprocess_image  

IMAGES_TO_TRAIN = 1000

csv_path = 'dog-breed-identification/labels.csv'
img_folder = 'dog-breed-identification/train'

images, labels = load_images_from_csv(csv_path, img_folder, limit=IMAGES_TO_TRAIN)

print('labels', labels)
print('labels', images)

print('IMAGENS E LABELS CARREGADAS')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

print('DADOS NORMALIZADOS')

X_train, X_test, y_train, y_test = train_test_split(images[:IMAGES_TO_TRAIN], labels_encoded[:IMAGES_TO_TRAIN], test_size=0.2, random_state=42)

print('SEPARADO TREINO E TESTE')

print(label_encoder.classes_)

model = create_model(label_encoder.classes_)  
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

print('MODELO CRIADO E TREINADO')

model.save('models/meu_modelo.h5')


np.save('models/label_classes.npy', label_encoder.classes_)
print("Classes salvas em 'models/label_classes.npy'")


test_image_path = select_random_image('dog-breed-identification/test')
preprocessed_image = preprocess_image(test_image_path)
predicted_breed = predict_breed(preprocessed_image, model, label_encoder)

# # Avaliar o modelo
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Visualizar a acertividade
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

