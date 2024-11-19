# tests/test_load_images.py
import unittest
import os
from utils import load_images_from_csv

class TestLoadImages(unittest.TestCase):
    def test_load_images_from_csv(self):
        csv_path = '../dog-breed-identification/labels.csv'
        img_folder = '../dog-breed-identification/train'
        
        # Certifique-se de que os caminhos existam antes de rodar o teste
        self.assertTrue(os.path.exists(csv_path), "CSV path does not exist")
        self.assertTrue(os.path.exists(img_folder), "Image folder does not exist")
        
        # Carrega as imagens e rótulos
        images, labels = load_images_from_csv(csv_path, img_folder)
        
        # Verifica se as imagens e labels foram carregados corretamente
        self.assertGreater(len(images), 0, "No images were loaded")
        self.assertGreater(len(labels), 0, "No labels were loaded")
        self.assertEqual(len(images), len(labels), "Mismatch between images and labels count")

if __name__ == '__main__':
    unittest.main()
