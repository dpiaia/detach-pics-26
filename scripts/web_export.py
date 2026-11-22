"""Exporta para o site o que a busca por selfie precisa: média de cada pessoa + modelos.

Lê as pessoas de site/data.js (gerado pelo build.py) e os embeddings leves de
cache/emb_mbf.npy (gerados pelo web_embed.py). Rodar sempre depois desses dois.
"""
import base64
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

emb = np.load(ROOT / "cache" / "emb_mbf.npy")
faces = json.loads((ROOT / "cache" / "faces.json").read_text())
data = json.loads((SITE / "data.js").read_text()[len("window.DATA="):-1])

# rótulo de cada rosto, na mesma ordem de faces.json
person_of = {(p["id"], k): f[4] for p in data["photos"] for k, f in enumerate(p["faces"])}
seen, labels = {}, []
for f in faces:
    k = seen.get(f["photo"], 0)
    seen[f["photo"]] = k + 1
    labels.append(person_of[(f["photo"], k)])
labels = np.array(labels)

n_people = len(data["people"])
cent = np.zeros((n_people, emb.shape[1]), np.float32)
for pid in range(n_people):
    cent[pid] = emb[labels == pid].mean(0)
cent /= np.linalg.norm(cent, axis=1, keepdims=True)

q = np.clip(np.round(cent * 127), -127, 127).astype(np.int8)
payload = {"dim": int(emb.shape[1]), "n": n_people, "q": base64.b64encode(q.tobytes()).decode()}
(SITE / "selfie-data.js").write_text("window.SELFIE=" + json.dumps(payload) + ";")

(SITE / "models").mkdir(exist_ok=True)
shutil.copy(ROOT / "models" / "yunet.onnx", SITE / "models" / "yunet.onnx")
shutil.copy(ROOT / "models" / "buffalo_s" / "w600k_mbf.onnx", SITE / "models" / "mbf.onnx")
print(f"{n_people} pessoas exportadas")
