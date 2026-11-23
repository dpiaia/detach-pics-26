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

5. `.venv/bin/python -I scripts/web_embed.py` e depois `.venv/bin/python -I scripts/web_export.py`: geram a busca por selfie (modelo leve MobileFaceNet + média de cada pessoa em `site/selfie-data.js`). Rodar sempre que o `build.py` rodar, senão a selfie aponta para as pessoas erradas.

## Busca por selfie

Roda inteira no navegador (`site/selfie.js`, com onnxruntime-web): YuNet acha o rosto, o alinhamento é o mesmo do Python e o MobileFaceNet (`site/models/mbf.onnx`, 13 MB, baixado só quando a pessoa usa a busca) gera o vetor comparado com cada pessoa. A foto não sai do aparelho. Validado: ~98% de acerto nos rostos da base; abaixo de 0,5 de similaridade a página pergunta em vez de afirmar.

## Setup numa máquina nova

`cache/` e `models/` não vão para o git. Para reprocessar: `python3 -m venv .venv && .venv/bin/pip install opencv-python numpy scikit-learn onnxruntime`, baixe `face_detection_yunet_2023mar.onnx` (opencv_zoo) como `models/yunet.onnx` e os `buffalo_l.zip` e `buffalo_s.zip` (insightface v0.7) descompactados em `models/buffalo/` e `models/buffalo_s/`. Para só ver a galeria nada disso é preciso: a foto ampliada vem do Drive.

## Nomes colaborativos (Supabase)

Os nomes ficam no Supabase (projeto `ydxndhawilxjvjrsdytu`); o esquema está em `supabase/schema.sql` (rodar no SQL Editor; é idempotente). As tabelas ficam fechadas e o site só usa funções: `get_names`, `get_suggestions`, `suggest_name`, `vote_name`, `request_removal`.

- Um nome aparece para todos assim que é sugerido (saldo >= 1) ou com `status = 'approved'`; com vários nomes vale o de maior saldo; saldo <= 0 tira do ar; `hidden` esconde de vez.
- "Sou eu e não quero meu nome aqui" grava em `blocked_people` e esconde na hora; apagar a linha desfaz.
- Moderação: Table Editor → `name_suggestions`.
- A pessoa 999 não existe na galeria: foi usada para testar e pode ser apagada das três tabelas.
