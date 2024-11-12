import unittest
import os
from tensorflow.keras.models import load_model
import numpy as np

class TestPredictBreed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Carrega o modelo de um dos três arquivos disponíveis na pasta models
        cls.model_path = 'models/20240205-16521707151979-1000-images-mobilenetv2-Adam.h5'
        cls.model = load_model(cls.model_path)

    def test_predict_breed(self):
        # Cria uma imagem de teste aleatória para simular a entrada do modelo
        test_image = np.random.rand(1, 224, 224, 3)  # Ajuste para o tamanho esperado pelo modelo
        predictions = self.model.predict(test_image)
        
        # Verifica se o output está no formato esperado
        self.assertEqual(predictions.shape[1], 120)  # Exemplo: 120 classes de raças

    def test_model_loaded(self):
        # Teste para garantir que o modelo foi carregado corretamente
        self.assertTrue(self.model is not None, "Falha ao carregar o modelo.")

if __name__ == '__main__':
    unittest.main()
