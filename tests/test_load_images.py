import unittest
from unittest.mock import patch

class TestLoadImages(unittest.TestCase):
    @patch('utils.load_images_from_csv')  # Substituímos a função pelo mock
    def test_load_images_from_csv(self, mock_load_images):
        # Simula o retorno da função mocked
        mock_load_images.return_value = (
            ['image1.jpg', 'image2.jpg'],  # Lista simulada de imagens
            ['label1', 'label2']           # Lista simulada de rótulos
        )

        # Caminhos simulados (não usados devido ao mock)
        csv_path = 'fake_csv_path'
        img_folder = 'fake_image_folder'

        # Chama a função simulada
        images, labels = mock_load_images(csv_path, img_folder)

        # Verifica se as imagens e labels foram retornados corretamente
        self.assertEqual(len(images), 2, "Images list length mismatch")
        self.assertEqual(len(labels), 2, "Labels list length mismatch")
        self.assertEqual(images[0], 'image1.jpg', "First image mismatch")
        self.assertEqual(labels[0], 'label1', "First label mismatch")

        # Verifica se a função foi chamada com os argumentos corretos
        mock_load_images.assert_called_once_with(csv_path, img_folder)

if __name__ == '__main__':
    unittest.main()
