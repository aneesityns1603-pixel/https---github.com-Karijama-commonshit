"""Evaluate an external model prediction; never optimize or generate predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from select_views import load_vertices, transform, write_json


def pose_matrix(prediction):
    t = np.asarray(prediction['translation'], dtype=float)
    q = np.asarray(prediction['rotation_wxyz'], dtype=float)
    s = np.asarray(prediction['scale_xyz'], dtype=float)
    if t.shape != (3,) or q.shape != (4,) or s.shape != (3,):
        raise ValueError('Expected translation[3], rotation_wxyz[4], scale_xyz[3]')
    if not all(np.isfinite(x).all() for x in (t, q, s)) or np.any(s <= 0):
        raise ValueError('Values must be finite, scales positive')
    if abs(np.linalg.norm(q) - 1) > .01:
        raise ValueError('Quaternion must be unit length (tolerance .01)')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ np.diag(s)
    matrix[:3, 3] = t
    return matrix


def metrics(vertices, prediction, gt):
    predicted = pose_matrix(prediction)
    target = np.asarray(gt['T_cad_to_scannet'])
    ps = np.linalg.norm(predicted[:3, :3], axis=0)
    ts = np.linalg.norm(target[:3, :3], axis=0)
    pr = predicted[:3, :3] / ps
    tr = target[:3, :3] / ts
    angle = np.degrees(np.arccos(np.clip((np.trace(pr.T @ tr) - 1) / 2, -1, 1)))
    # Deterministic cap; report vertex distances, not surface Chamfer.
    if len(vertices) > 10000:
        vertices = vertices[np.linspace(0, len(vertices) - 1, 10000, dtype=int)]
    p, t = transform(vertices, predicted), transform(vertices, target)
    return {'translation_error_m': float(np.linalg.norm(predicted[:3, 3] - target[:3, 3])),
            'rotation_error_degrees_raw': float(angle),
            'symmetry': gt.get('sym'),
            'rotation_warning': 'Raw rotation is not symmetry invariant',
            'scale_relative_error_xyz': (np.abs(ps - ts) / ts).tolist(),
            'mean_corresponding_vertex_error_m': float(np.linalg.norm(p - t, axis=1).mean()),
            'mean_predicted_to_gt_nearest_vertex_error_m': float(cKDTree(t).query(p)[0].mean()),
            'evaluated_vertex_count': len(vertices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prediction = json.loads(args.prediction.read_text(encoding='utf-8'))
    gt = json.loads(args.ground_truth.read_text(encoding='utf-8'))
    result = metrics(load_vertices(args.root / gt['asset_path']), prediction, gt)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
