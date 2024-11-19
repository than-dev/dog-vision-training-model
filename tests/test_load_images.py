# tests/test_load_images.py
import unittest
import os
from utils import load_images_from_csv

class TestLoadImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Caminhos para o CSV e a pasta de imagens, calculados de forma absoluta
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo de teste
        cls.csv_path = os.path.join(base_dir, "../dog-breed-identification/labels.csv")
        cls.img_folder = os.path.join(base_dir, "../dog-breed-identification/train")

    def test_paths_exist(self):
        # Testa se os caminhos existem
        self.assertTrue(os.path.exists(self.csv_path), f"CSV path does not exist: {self.csv_path}")
        self.assertTrue(os.path.exists(self.img_folder), f"Image folder does not exist: {self.img_folder}")

    def test_load_images_from_csv(self):
        # Garante que o teste só será executado se os arquivos existirem
        if not os.path.exists(self.csv_path) or not os.path.exists(self.img_folder):
            self.skipTest("Required files are missing, skipping test_load_images_from_csv")

        # Carrega as imagens e rótulos
        images, labels = load_images_from_csv(self.csv_path, self.img_folder)

        # Verifica se as imagens e rótulos foram carregados corretamente
        self.assertGreater(len(images), 0, "No images were loaded")
        self.assertGreater(len(labels), 0, "No labels were loaded")
        self.assertEqual(len(images), len(labels), "Mismatch between images and labels count")

if __name__ == '__main__':
    unittest.main()
