"""Agrupa os rostos por pessoa e gera site/data.js, site/faces/ e site/thumbs/."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
IMG = ROOT / "cache" / "img"
THRESHOLD = 0.6     # distância cosseno máxima (média) dentro de uma pessoa
MIN_PHOTOS = 2      # pessoa precisa aparecer em pelo menos N fotos
MIN_COVER_PX = 70   # grupo sem nenhum rosto desse tamanho é ruído (borrado/escuro)

emb = np.load(ROOT / "cache" / "emb.npy")
faces = json.loads((ROOT / "cache" / "faces.json").read_text())
photos = json.loads((ROOT / "cache" / "photos.json").read_text())

labels = AgglomerativeClustering(
    n_clusters=None, metric="cosine", linkage="average",
    distance_threshold=THRESHOLD).fit_predict(emb)

people = []
for lab in np.unique(labels):
    idx = np.where(labels == lab)[0]
    photo_ids = {faces[i]["photo"] for i in idx}
    if len(photo_ids) < MIN_PHOTOS or max(faces[i]["px"] for i in idx) < MIN_COVER_PX:
        continue
    center = emb[idx].mean(0)
    center /= np.linalg.norm(center)
    # capa: rosto grande, nítido e perto do centro do grupo
    quality = [(emb[i] @ center) * min(faces[i]["px"], 220) * faces[i]["score"] for i in idx]
    people.append({"idx": idx, "cover": int(idx[int(np.argmax(quality))]), "n": len(photo_ids)})

people.sort(key=lambda p: -p["n"])
face_person = {}
for pid, p in enumerate(people):
    for i in p["idx"]:
        face_person[int(i)] = pid

(SITE / "faces").mkdir(parents=True, exist_ok=True)
(SITE / "thumbs").mkdir(parents=True, exist_ok=True)


def face_crop(pid_face):
    pid, fi = pid_face
    f = faces[fi]
    img = cv2.imread(str(IMG / f"{f['photo']}.jpg"))
    h, w = img.shape[:2]
    x, y, bw, bh = f["box"][0] * w, f["box"][1] * h, f["box"][2] * w, f["box"][3] * h
    side = max(bw, bh) * 1.6
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = cv2.resize(img[y0:y1, x0:x1], (160, 160), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(SITE / "faces" / f"p{pid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])


def thumb(photo):
    out = SITE / "thumbs" / f"{photo['id']}.jpg"
    img = cv2.imread(str(IMG / f"{photo['id']}.jpg"))
    h, w = img.shape[:2]
    photo["w"], photo["h"] = w, h
    if not out.exists():
        s = 520 / max(w, h)
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 80])


with ThreadPoolExecutor(8) as ex:
    list(ex.map(face_crop, [(pid, p["cover"]) for pid, p in enumerate(people)]))
    list(ex.map(thumb, photos))

faces_by_photo = {}
for i, f in enumerate(faces):
    faces_by_photo.setdefault(f["photo"], []).append(
        [round(v, 4) for v in f["box"]] + [face_person.get(i, -1)])

size_of = {f["id"]: f.get("size", 0) for f in json.loads((ROOT / "data" / "files.json").read_text())}
data = {
    "people": [{"id": pid, "n": p["n"]} for pid, p in enumerate(people)],
    "photos": [{"id": p["id"], "t": p["title"], "f": p["folder"], "w": p["w"], "h": p["h"], "s": size_of.get(p["id"], 0),
                "faces": faces_by_photo.get(p["id"], [])} for p in photos],
}
(SITE / "data.js").write_text("window.DATA=" + json.dumps(data, separators=(",", ":")) + ";")
print(f"{len(people)} pessoas, {len(photos)} fotos, {len(face_person)}/{len(faces)} rostos atribuídos")
