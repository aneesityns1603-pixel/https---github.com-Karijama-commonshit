import unittest
import numpy as np
from evaluate_pose import metrics, pose_matrix


class EvaluationTests(unittest.TestCase):
    def test_known_translation_and_rotation(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        gt = {'T_cad_to_scannet': np.eye(4).tolist(), 'sym': '__SYM_NONE'}
        pred = {'translation': [0, 0, 0], 'rotation_wxyz': [1, 0, 0, 0], 'scale_xyz': [1, 1, 1]}
        self.assertEqual(metrics(vertices, pred, gt)['mean_corresponding_vertex_error_m'], 0)
        pred['translation'] = [0, 0, 1]
        result = metrics(vertices, pred, gt)
        self.assertAlmostEqual(result['translation_error_m'], 1)
        self.assertAlmostEqual(result['mean_corresponding_vertex_error_m'], 1)
        pred['rotation_wxyz'] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        self.assertAlmostEqual(metrics(vertices, pred, gt)['rotation_error_degrees_raw'], 90)

    def test_invalid_prediction_rejected(self):
        with self.assertRaises(ValueError):
            pose_matrix({'translation': [0, 0, 0], 'rotation_wxyz': [0, 0, 0, 0],
                         'scale_xyz': [1, 1, 1]})


if __name__ == '__main__':
    unittest.main()
