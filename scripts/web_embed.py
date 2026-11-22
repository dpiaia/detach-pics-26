"""Embeddings com o modelo leve (MobileFaceNet) para a busca por selfie no navegador.

Refaz a mesma detecção do detect.py (mesmos parâmetros, mesma ordem), confere que os
rostos batem com cache/faces.json e grava cache/emb_mbf.npy.
"""
import json
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
MIN_FACE, SCORE = 36, 0.8

faces = json.loads((ROOT / "cache" / "faces.json").read_text())
photos = json.loads((ROOT / "cache" / "photos.json").read_text())
sess = ort.InferenceSession(str(ROOT / "models" / "buffalo_s" / "w600k_mbf.onnx"),
                            providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name


def detect(photo):
    img = cv2.imread(str(IMG_DIR / f"{photo['id']}.jpg"))
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "yunet.onnx"), "", (w, h), SCORE, 0.3, 5000)
    _, found = det.detect(img)
    out = []
    for f in found if found is not None else []:
        if min(f[2], f[3]) < MIN_FACE:
            continue
        lm = f[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm, ARC_TEMPLATE, method=cv2.LMEDS)
        out.append((photo["id"], [float(f[0]) / w, float(f[1]) / h], cv2.warpAffine(img, M, (112, 112))))
    return out


crops, ids = [], []
with ThreadPoolExecutor(8) as ex:
    for res in ex.map(detect, photos):
        for pid, xy, crop in res:
            ids.append((pid, xy))
            crops.append(crop)

assert len(crops) == len(faces), (len(crops), len(faces))
for (pid, xy), f in zip(ids, faces):
    assert pid == f["photo"] and abs(xy[0] - f["box"][0]) < 1e-4 and abs(xy[1] - f["box"][1]) < 1e-4

embs = []
for c in crops:  # o modelo leve tem lote fixo de 1
    x = ((c[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    embs.append(sess.run(None, {in_name: x})[0][0])
emb = np.array(embs, np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
np.save(ROOT / "cache" / "emb_mbf.npy", emb)
print("ok", emb.shape)
