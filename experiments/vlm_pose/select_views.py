"""GT-assisted view selection. Does not estimate poses or call any model.

Visibility is a point z-buffer approximation, not triangle ray tracing.
Unknown scan depth is recorded separately; it is not proof of visibility.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter


def load_vertices(path):
    vertices = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"Invalid OBJ vertices: {path}")
    return result


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, K, c2w, width, height):
    camera = transform(points, np.linalg.inv(c2w))
    z = camera[:, 2]
    pixels = camera @ K.T
    uv = np.full((len(points), 2), np.nan)
    front = np.isfinite(camera).all(axis=1) & (z > 1e-6)
    uv[front] = pixels[front, :2] / pixels[front, 2:3]
    valid = front & (uv[:, 0] >= 0) & (uv[:, 0] < width)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, z, valid


def zbuffer(uv, z, valid, width, height, cell):
    w, h = int(np.ceil(width / cell)), int(np.ceil(height / cell))
    depth = np.full(h * w, np.inf)
    xy = (uv[valid] / cell).astype(int)
    np.minimum.at(depth, xy[:, 1] * w + xy[:, 0], z[valid])
    return depth.reshape(h, w)


def score_vertices(uv, z, valid, scan_depth, width, height, cell, tolerance):
    ids = np.flatnonzero(valid)
    xy = (uv[ids] / cell).astype(int)
    own = zbuffer(uv, z, valid, width, height, cell)
    self_front = z[ids] <= own[xy[:, 1], xy[:, 0]] + tolerance
    observed = scan_depth[xy[:, 1], xy[:, 0]]
    known = np.isfinite(observed)
    blocked = known & (z[ids] > observed + tolerance)
    accepted = self_front & ~blocked
    keep = np.zeros(len(z), dtype=bool)
    keep[ids[accepted]] = True
    return {
        "in_frustum_vertices": int(valid.sum()),
        "approx_unoccluded_vertices": int(accepted.sum()),
        "unknown_scan_depth_vertices": int((self_front & ~known).sum()),
        "scan_blocked_front_vertices": int((self_front & blocked).sum()),
    }, keep


def choose(rows, count, score, min_angle):
    ranked = sorted(rows, key=lambda r: (-r[score], r["frame_id"]))
    selected = []
    for row in ranked:
        if row[score] <= 0:
            continue
        direction = np.asarray(row["view_direction"])
        if any(np.degrees(np.arccos(np.clip(direction @ np.asarray(s["view_direction"]), -1, 1)))
               < min_angle for s in selected):
            continue
        selected.append(row)
        if len(selected) == count:
            break
    return selected


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_scene(root, scene_id, output, args):
    scene = root / "scenes" / scene_id
    destination = output / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    annotation = json.loads((scene / "annotations/instances_scannet.json").read_text(encoding="utf-8"))
    instances = annotation["instances"]
    K = np.loadtxt(scene / "posed_images/intrinsic.txt")[:3, :3]
    with (scene / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = [r for r in csv.DictReader(stream) if r["pose_valid"] == "1"]
    with Image.open(scene / frames[0]["rgb"]) as image:
        width, height = image.size
    geometry = []
    for instance in instances:
        local = load_vertices(root / instance["asset_path"])
        world = transform(local, np.asarray(instance["T_cad_to_scannet"]))
        geometry.append((local, world))
    with np.load(scene / "points/pointcloud.npz") as cloud:
        scan = cloud["points"].astype(np.float64)
    scores = [[] for _ in instances]
    for fi, frame in enumerate(frames):
        c2w = np.loadtxt(scene / frame["pose"])
        uv_scan, z_scan, valid_scan = project(scan, K, c2w, width, height)
        scan_depth = zbuffer(uv_scan, z_scan, valid_scan, width, height, args.cell)
        # Fill small sampling holes. This is explicitly an approximate blocker.
        if args.scan_radius:
            scan_depth = minimum_filter(scan_depth, size=2 * args.scan_radius + 1)
        for ii, (_, world) in enumerate(geometry):
            uv, z, valid = project(world, K, c2w, width, height)
            metrics, _ = score_vertices(uv, z, valid, scan_depth, width, height,
                                        args.cell, args.tolerance)
            direction = c2w[:3, 3] - world.mean(axis=0)
            direction /= max(np.linalg.norm(direction), 1e-12)
            scores[ii].append(dict(frame_id=frame["frame_id"], rgb=frame["rgb"],
                                   view_direction=direction.tolist(), **metrics))
        if fi % 100 == 0:
            print(f"{scene_id}: {fi + 1}/{len(frames)} frames", flush=True)
    gallery = []
    summary = []
    for instance, (local, world), rows in zip(instances, geometry, scores):
        iid = instance["instance_id"]
        folder = destination / f"instance_{iid:02d}"
        folder.mkdir(exist_ok=True)
        selected = choose(rows, args.views, args.score, args.min_angle)
        # Never export target transform, GT overlays, or GT bbox to model_input.
        model_folder = folder / "model_input"
        model_folder.mkdir(exist_ok=True)
        model_views = []
        cards = []
        for rank, row in enumerate(selected):
            frame = next(f for f in frames if f["frame_id"] == row["frame_id"])
            c2w = np.loadtxt(scene / frame["pose"])
            with Image.open(scene / frame["rgb"]) as source:
                image = source.convert("RGB")
            image.save(model_folder / f"view_{rank}.jpg", quality=95)
            model_views.append({"image": f"view_{rank}.jpg", "frame_id": row["frame_id"],
                                "K": K.tolist(), "camera_to_world": c2w.tolist()})
            uv, _, valid = project(world, K, c2w, width, height)
            image.thumbnail((648, 484))
            draw = ImageDraw.Draw(image)
            factor = image.width / width
            # GT overlay is diagnostic only, not a model input or segmentation mask.
            for x, y in uv[valid][::max(1, int(valid.sum()) // 2500)]:
                x, y = x * factor, y * factor
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 50, 20))
            preview = f"gt_projection_{rank}.jpg"
            image.save(folder / preview)
            relative = f"instance_{iid:02d}/{preview}"
            cards.append(f'<figure><img width="420" src="{relative}"><figcaption>'
                         f'{row["frame_id"]}: frustum={row["in_frustum_vertices"]}, '
                         f'approx={row["approx_unoccluded_vertices"]}</figcaption></figure>')
        write_json(model_folder / "input.json", {
            "scene_id": scene_id, "instance_id": iid, "asset_path": instance["asset_path"],
            "cad_local_bounds": [local.min(axis=0).tolist(), local.max(axis=0).tolist()],
            "coordinate_convention": "ScanNet world in metres; camera x right, y down, z forward; column vectors",
            "protocol": "known GT cameras; GT-assisted view selection; object pose withheld",
            "views": model_views,
            "required_prediction": {"translation": "3 numbers in world metres",
                                    "rotation_wxyz": "unit quaternion, CAD local to world",
                                    "scale_xyz": "3 positive local-axis scale factors"},
        })
        write_json(folder / "evaluation_only.json", instance)
        write_json(folder / "selection.json", {"score": args.score,
                   "min_angle_degrees": args.min_angle, "selected": selected,
                   "all_frames": rows})
        gallery.append(f'<h2>Instance {iid}: {html.escape(instance["category_id"])}</h2>'
                       + '<div class="row">' + ''.join(cards) + '</div>')
        summary.append({"instance_id": iid, "vertices": len(local),
                        "frames": [r["frame_id"] for r in selected]})
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>GT-assisted view selection</title>'
        '<style>body{font:16px sans-serif;margin:24px}.row{display:flex;flex-wrap:wrap}figure{margin:8px}</style>'
        '<h1>' + scene_id + '</h1><p>Diagnostic GT projections. Red points are NOT masks. '
        'Approximate visibility uses vertex and sparse scan z-buffers. Unknown depth is not verified visibility.</p>'
        + ''.join(gallery), encoding="utf-8")
    write_json(destination / "summary.json", summary)
    print(json.dumps({"scene": scene_id, "instances": summary}), flush=True)
    return {"scene": scene_id, "instances": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--scenes", nargs="+", default=["scene0084_01"])
    parser.add_argument("--views", type=int, choices=[2, 3], default=3)
    parser.add_argument("--score", choices=["in_frustum_vertices", "approx_unoccluded_vertices"],
                        default="in_frustum_vertices")
    parser.add_argument("--cell", type=int, default=4)
    parser.add_argument("--scan-radius", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--min-angle", type=float, default=0,
                        help="0 = strict top count; e.g. 15 avoids near-duplicate viewpoints")
    args = parser.parse_args()
    if args.cell < 1 or args.scan_radius < 0 or args.tolerance < 0 or not 0 <= args.min_angle <= 180:
        parser.error("Invalid visibility or angle parameter")
    scenes = args.scenes
    if scenes == ["all"]:
        scenes = [s["scene_id"] for s in json.loads((args.root / "manifest.json").read_text())["scenes"]]
    args.output.mkdir(parents=True, exist_ok=True)
    result = [run_scene(args.root, scene, args.output, args) for scene in scenes]
    write_json(args.output / "run.json", {"configuration": {k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()}, "results": result})


if __name__ == "__main__":
    main()
