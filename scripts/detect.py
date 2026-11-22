"""Detecta rostos em cache/img e grava embeddings ArcFace em cache/faces.npz."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "cache" / "img"
ARC_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
MIN_FACE = 36  # px no lado menor do rosto, na imagem de 1600px
SCORE = 0.8

files = json.loads((ROOT / "data" / "files.json").read_text())
seen, photos = set(), []
for f in files:
    if f["id"] not in seen and (IMG_DIR / f"{f['id']}.jpg").exists():
        seen.add(f["id"])
        photos.append(f)

sess = ort.InferenceSession(
    str(ROOT / "models" / "buffalo" / "w600k_r50.onnx"),
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name


def detect(photo):
    img = cv2.imread(str(IMG_DIR / f"{photo['id']}.jpg"))
    if img is None:
        return photo, None, []
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(
        str(ROOT / "models" / "yunet.onnx"), "", (w, h), SCORE, 0.3, 5000)
    _, faces = det.detect(img)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = f[:4]
        if min(bw, bh) < MIN_FACE:
            continue
        lm = f[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm, ARC_TEMPLATE, method=cv2.LMEDS)
        crop = cv2.warpAffine(img, M, (112, 112), borderValue=0)
        out.append({"box": [float(x) / w, float(y) / h, float(bw) / w, float(bh) / h],
                    "score": float(f[14]), "crop": crop, "px": float(min(bw, bh))})
    return photo, (w, h), out


records, crops = [], []
with ThreadPoolExecutor(8) as ex:
    for i, (photo, size, faces) in enumerate(ex.map(detect, photos)):
        for fc in faces:
            records.append({"photo": photo["id"], "box": fc["box"],
                            "score": fc["score"], "px": fc["px"]})
            crops.append(fc["crop"])
        if i % 100 == 0:
            print(f"{i}/{len(photos)} fotos, {len(records)} rostos", flush=True)

embs = []
for i in range(0, len(crops), 64):
    batch = np.stack(crops[i:i + 64])[:, :, :, ::-1].astype(np.float32)
    batch = ((batch - 127.5) / 127.5).transpose(0, 3, 1, 2)
    embs.append(sess.run(None, {in_name: batch})[0])
emb = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

np.save(ROOT / "cache" / "emb.npy", emb)
(ROOT / "cache" / "faces.json").write_text(json.dumps(records))
(ROOT / "cache" / "photos.json").write_text(json.dumps(photos))
print(f"pronto: {len(photos)} fotos, {len(records)} rostos", file=sys.stderr)
