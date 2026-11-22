# Quem está na foto

Galeria das pastas D1 e D2 do Drive com filtro por rosto.

## Abrir

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Depois abra http://localhost:8765. Clique num rosto para filtrar (vários = fotos com todos eles); clique numa foto para ampliar e clique num rosto dela para trocar o filtro. Os nomes dados às pessoas ficam salvos só neste navegador.

## Refazer (fotos novas no Drive)

1. Atualize `data/files.json` (lista `{id, title, folder}` das fotos).
2. Baixe as miniaturas: `jq -r '.[].id' data/files.json | xargs -P 16 -I{} sh -c '[ -s cache/img/{}.jpg ] || curl -sfL -o cache/img/{}.jpg "https://drive.google.com/thumbnail?id={}&sz=w1600"'`
3. `.venv/bin/python -I scripts/detect.py`: detecta os rostos (YuNet) e gera os embeddings (ArcFace, em `models/`)
4. `.venv/bin/python -I scripts/build.py`: agrupa por pessoa e gera `site/`. Ajuste `THRESHOLD` (0.6) se juntar pessoas diferentes (diminua) ou separar a mesma pessoa (aumente).

## Setup numa máquina nova

`cache/` e `models/` não vão para o git. Para reprocessar: `python3 -m venv .venv && .venv/bin/pip install opencv-python numpy scikit-learn onnxruntime`, baixe `face_detection_yunet_2023mar.onnx` (opencv_zoo) como `models/yunet.onnx` e o `buffalo_l.zip` (insightface v0.7) descompactado em `models/buffalo/`. Para só ver a galeria nada disso é preciso: a foto ampliada vem do Drive.
