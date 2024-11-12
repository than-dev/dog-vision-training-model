# tests/test_predict_breed.py
import unittest
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from utils import predict_breed, select_random_image, load_labels_from_csv

class TestPredictBreed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configura o modelo e o encoder uma vez antes dos testes
        cls.model = load_model('models/meu_modelo.h5')
        cls.label_encoder = LabelEncoder()
        cls.label_encoder.fit(load_labels_from_csv())

    def test_predict_breed(self):
        image_path = select_random_image('dog-breed-identification/test')
        
        # Verifica se a imagem foi carregada corretamente
        self.assertIsNotNone(image_path, "No image found in test directory")
        self.assertTrue(os.path.exists(image_path), "Image path does not exist")
        
        # Faz uma previsão
        predicted_breed = predict_breed(image_path, self.model, self.label_encoder)
        
        # Verifica se a previsão é uma string e não está vazia
        self.assertIsInstance(predicted_breed[0], str, "Predicted breed is not a string")
        self.assertGreater(len(predicted_breed[0]), 0, "Predicted breed is empty")

if __name__ == '__main__':
    unittest.main()
