import unittest
import os
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
import numpy as np

class TestPredictBreed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configura o caminho do modelo
        cls.model_path = 'models/20240205-16521707151979-1000-images-mobilenetv2-Adam.h5'
        
        # Carrega o modelo com a camada customizada 'KerasLayer'
        cls.model = tf_keras.models.load_model(cls.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    def test_predict_breed(self):
        # Simula uma entrada processada para o modelo (imagem pré-processada)
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Array normalizado no intervalo [0, 1]
        predictions = self.model.predict(test_image)
        
        # Verifica se o output está no formato esperado
        self.assertEqual(predictions.shape[0], 1, "A saída deve ter uma única amostra.")
        self.assertGreaterEqual(predictions.shape[1], 1, "A saída deve ter pelo menos uma classe.")
        self.assertAlmostEqual(np.sum(predictions[0]), 1, delta=0.01, msg="As probabilidades devem somar aproximadamente 1.")

    def test_model_loaded(self):
        # Testa se o modelo foi carregado corretamente
        self.assertIsNotNone(self.model, "Falha ao carregar o modelo.")

    def test_layer_in_model(self):
        # Verifica se a camada KerasLayer está no modelo
        keras_layer_found = any(isinstance(layer, hub.KerasLayer) for layer in self.model.layers)
        self.assertTrue(keras_layer_found, "O modelo não contém a camada KerasLayer.")

if __name__ == '__main__':
    unittest.main()
