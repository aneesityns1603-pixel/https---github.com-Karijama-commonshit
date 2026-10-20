import unittest
import numpy as np
from select_views import project, transform, zbuffer, score_vertices, choose


class GeometryTests(unittest.TestCase):
    def test_camera_inverse_and_frustum(self):
        camera = np.eye(4)
        camera[0, 3] = 10
        K = np.array([[10, 0, 20], [0, 10, 20], [0, 0, 1]])
        points = np.array([[10, 0, 2], [10, 0, -2], [30, 0, 2]])
        uv, z, valid = project(points, K, camera, 40, 40)
        np.testing.assert_allclose(uv[0], [20, 20])
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(transform(points, np.linalg.inv(camera))[0], [0, 0, 2])

    def test_self_and_scan_occlusion_and_unknown(self):
        uv = np.array([[1, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
        z = np.array([2., 3., 4., 5.])
        valid = np.ones(4, dtype=bool)
        scan = np.full((5, 5), np.inf)
        scan[1, 2] = 2
        scores, keep = score_vertices(uv, z, valid, scan, 5, 5, 1, .1)
        np.testing.assert_array_equal(keep, [True, False, False, True])
        self.assertEqual(scores['unknown_scan_depth_vertices'], 2)
        self.assertEqual(scores['scan_blocked_front_vertices'], 1)
        depth = zbuffer(uv, z, valid, 5, 5, 1)
        self.assertEqual(depth[1, 1], 2)

    def test_top_count_and_optional_diversity(self):
        rows = [dict(frame_id=str(i), score=s, view_direction=d) for i, s, d in
                [(0, 10, [1, 0, 0]), (1, 9, [1, 0, 0]), (2, 8, [0, 1, 0])]]
        self.assertEqual([r['frame_id'] for r in choose(rows, 2, 'score', 0)], ['0', '1'])
        self.assertEqual([r['frame_id'] for r in choose(rows, 2, 'score', 15)], ['0', '2'])


if __name__ == '__main__':
    unittest.main()
