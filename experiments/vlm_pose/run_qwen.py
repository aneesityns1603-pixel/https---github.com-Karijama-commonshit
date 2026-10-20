"""Local Ollama Qwen3.5-9B experiment. No training, no GT object pose access."""
import argparse
import base64
import io
import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from PIL import Image, ImageDraw
import requests
from select_views import load_vertices, write_json
from evaluate_pose import pose_matrix


def cad_preview(path):
    """Orthographic canonical mesh views; painter ordering, no materials."""
    vertices = load_vertices(path)
    faces = []
    seen = 0
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'v':
                seen += 1
            if fields[0] == 'f':
                ids = [int(x.split('/')[0]) for x in fields[1:]]
                ids = [x - 1 if x > 0 else seen + x for x in ids]
                for j in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[j], ids[j + 1]])
    faces = np.asarray(faces, dtype=int)
    if faces.size == 0:
        raise ValueError('CAD has no faces')
    center = (vertices.min(0) + vertices.max(0)) / 2
    extent = max(np.ptp(vertices, axis=0).max(), 1e-8)
    sheet = Image.new('RGB', (800, 800), 'white')
    for i, direction in enumerate([[1, 1, 1], [-1, 1, 1], [1, 1, -1], [-1, 1, -1]]):
        toward = np.asarray(direction, dtype=float)
        toward /= np.linalg.norm(toward)
        right = np.cross([0, 1, 0], toward)
        right /= np.linalg.norm(right)
        up = np.cross(toward, right)
        basis = np.stack([right, up, toward], axis=1)
        camera = (vertices - center) @ basis
        screen = camera[:, :2] * [230 / extent, -230 / extent] + [200, 205]
        tile = Image.new('RGB', (400, 400), (245, 245, 245))
        draw = ImageDraw.Draw(tile)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shade = 100 + 125 * np.abs(normals @ toward)
        order = np.argsort(camera[faces, 2].mean(axis=1))
        for f in order:
            color = int(shade[f])
            draw.polygon([tuple(x) for x in screen[faces[f]]], fill=(color, color, min(255, color + 15)))
        draw.text((10, 10), f'CAD local; camera direction {direction}', fill='black')
        draw.text((10, 30), 'X red, Y green, Z blue; arrows positive', fill='black')
        origin = np.array([65., 330.])
        for axis, color, label in zip(np.eye(3), ['red', 'green', 'blue'], ['+X', '+Y', '+Z']):
            d = axis @ basis
            end = origin + d[:2] * [40, -40]
            draw.line([tuple(origin), tuple(end)], fill=color, width=3)
            draw.ellipse((end[0]-2, end[1]-2, end[0]+2, end[1]+2), fill=color)
            draw.text(tuple(end), label, fill=color)
        sheet.paste(tile, ((i % 2) * 400, (i // 2) * 400))
    return sheet


def encode(image, max_side):
    image = image.convert('RGB')
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def build_request(data, input_dir, preview, model, max_side, seed, context):
    images = [encode(preview, max_side)]
    view_info = []
    for view in data['views']:
        with Image.open(input_dir / view['image']) as image:
            old_width, old_height = image.size
            image.thumbnail((max_side, max_side))
            new_width, new_height = image.size
            K = np.asarray(view['K'], dtype=float).copy()
            K[0] *= new_width / old_width
            K[1] *= new_height / old_height
            images.append(encode(image, max_side))
        view_info.append({'image_index': len(images) - 1, 'width': new_width,
                          'height': new_height, 'K': K.tolist(),
                          'camera_to_world': view['camera_to_world']})
    prompt = '''Estimate the pose of the object matching the canonical CAD shown in image 0.
The other images are observations of the same static scene. Image 0 shows four canonical
CAD views with signed local axes. Identify the same object across the scene views.
If several indistinguishable instances prevent identification, return status ambiguous.
Do not invent an exact answer if the object is invisible or the evidence is insufficient.
Cameras are known. Camera axes: X right, Y down, Z forward. All world coordinates are
ScanNet metres, column vectors. Camera-to-world maps camera coordinates to world.
Return the CAD LOCAL TO WORLD transform, not camera-to-object, not its inverse.
p_world = R(q_wxyz) @ diag(scale_xyz) @ p_cad + translation.
Scale has three positive factors along the CAD local axes. Quaternion order is w,x,y,z.
Return only JSON with status (ok/ambiguous/not_visible/insufficient), translation[3],
rotation_wxyz[4], scale_xyz[3], confidence between 0 and 1, explanation.
For a non-ok status the three pose fields must be null. No ground-truth object pose is given.
CAD local bounds and observed camera information follow:\n'''
    prompt += json.dumps({'cad_local_bounds': data['cad_local_bounds'], 'views': view_info})
    return {'model': model, 'stream': False, 'think': False, 'format': 'json',
            'messages': [{'role': 'user', 'content': prompt, 'images': images}],
            'options': {'temperature': 0, 'seed': seed, 'num_ctx': context, 'num_predict': 2048},
            'keep_alive': '0s'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='scannet5 dataset root')
    parser.add_argument('--input', type=Path, required=True, help='model_input/input.json only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--server', default='http://127.0.0.1:11434')
    parser.add_argument('--max-side', type=int, default=896)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if urlparse(args.server).hostname not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('Local inference only: run this script on the model computer using loopback')
    if args.max_side < 64 or args.context < 1024:
        parser.error('Invalid image/context size')
    # Refuse reuse to avoid accidentally retaining stale predictions after failures.
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if len(data['views']) < 2:
        raise ValueError('Need at least two selected views')
    asset = (args.root / data['asset_path']).resolve()
    if not asset.is_relative_to((args.root / 'assets').resolve()):
        raise ValueError('CAD must be inside dataset assets/')
    preview = cad_preview(asset)
    preview.save(args.output / 'cad_preview.jpg')
    request = build_request(data, args.input.parent, preview, args.model,
                            args.max_side, args.seed, args.context)
    write_json(args.output / 'request.json', request)
    if args.prepare_only:
        print('Prepared request; inference NOT run.', flush=True)
        return
    session = requests.Session()
    session.trust_env = False  # A loopback server must not use an external proxy.
    server = args.server.rstrip('/')
    tags = session.get(server + '/api/tags', timeout=10)
    tags.raise_for_status()
    write_json(args.output / 'ollama_models.json', tags.json())
    if not any(m.get('name') == args.model for m in tags.json().get('models', [])):
        raise ValueError(f'Model absent. Run: ollama pull {args.model}')
    show = session.post(server + '/api/show', json={'model': args.model}, timeout=30)
    show.raise_for_status()
    write_json(args.output / 'model_info.json', show.json())
    print('Running local model; waiting for response...', flush=True)
    response = session.post(server + '/api/chat', json=request, timeout=args.timeout)
    response.raise_for_status()
    raw = response.json()
    write_json(args.output / 'response.json', raw)
    prediction = json.loads(raw['message']['content'])
    status = prediction.get('status')
    if status not in ('ok', 'ambiguous', 'not_visible', 'insufficient'):
        raise ValueError('Invalid prediction status; raw response preserved')
    if status == 'ok':
        pose_matrix(prediction)  # Reject malformed transforms before evaluation.
    elif any(prediction.get(k) is not None for k in ['translation', 'rotation_wxyz', 'scale_xyz']):
        raise ValueError('Non-ok prediction must have null pose fields')
    write_json(args.output / 'prediction.json', prediction)
    print(f'Model finished; status={status}. No geometric refinement applied.', flush=True)


if __name__ == '__main__':
    main()
