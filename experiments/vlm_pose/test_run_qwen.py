import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from run_qwen import build_request


class RequestTests(unittest.TestCase):
    def test_image_resize_updates_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new('RGB', (1000, 500)).save(folder / 'view.jpg')
            data = {'cad_local_bounds': [[-1, -1, -1], [1, 1, 1]], 'views': [
                {'image': 'view.jpg', 'K': [[800, 0, 500], [0, 800, 250], [0, 0, 1]],
                 'camera_to_world': np.eye(4).tolist()}]}
            request = build_request(data, folder, Image.new('RGB', (20, 20)),
                                    'qwen3.5:9b', 500, 42, 16384)
            content = request['messages'][0]['content']
            self.assertIn('400.0', content)
            self.assertIn('250.0', content)
            self.assertIn('"width": 500', content)
            self.assertIn('"height": 250', content)
            self.assertEqual(len(request['messages'][0]['images']), 2)
            self.assertEqual(request['model'], 'qwen3.5:9b')


if __name__ == '__main__':
    unittest.main()
