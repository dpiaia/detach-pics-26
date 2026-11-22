// Busca por selfie: tudo roda no navegador, a foto não sai do aparelho.
// Mesmo pipeline do scripts/web_embed.py: YuNet (detecção + 5 pontos) → alinhamento
// ArcFace 112×112 → MobileFaceNet (512 dims) → similaridade com a média de cada pessoa.
(function () {
  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
  const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const DET_SIZE = 640;
  const TEMPLATE = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

  let ready = null;

  function loadScript(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error("não carregou " + src));
      document.head.appendChild(s);
    });
  }

  function init() {
    if (ready) return ready;
    ready = (async () => {
      await Promise.all([window.ort ? 0 : loadScript(ORT_URL), window.SELFIE ? 0 : loadScript("selfie-data.js")]);
      ort.env.wasm.wasmPaths = ORT_WASM;
      const opts = { executionProviders: ["wasm"] };
      const [det, rec] = await Promise.all([
        ort.InferenceSession.create("models/yunet.onnx", opts),
        ort.InferenceSession.create("models/mbf.onnx", opts),
      ]);
      const raw = Uint8Array.from(atob(window.SELFIE.q), (c) => c.charCodeAt(0));
      const q = new Int8Array(raw.buffer);
      const { n, dim } = window.SELFIE;
      const cent = new Float32Array(n * dim);
      for (let i = 0; i < n; i++) {
        let norm = 0;
        for (let j = 0; j < dim; j++) { const v = q[i * dim + j]; cent[i * dim + j] = v; norm += v * v; }
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < dim; j++) cent[i * dim + j] /= norm;
      }
      return { det, rec, cent, n, dim };
    })();
    ready.catch(() => { ready = null; });
    return ready;
  }

  // ---------- detecção (YuNet, entrada fixa 640×640, BGR 0–255)
  async function detect(m, src, w, h) {
    const scale = Math.min(DET_SIZE / w, DET_SIZE / h);
    const cv = document.createElement("canvas");
    cv.width = cv.height = DET_SIZE;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, DET_SIZE, DET_SIZE);
    ctx.drawImage(src, 0, 0, w * scale, h * scale);
    const px = ctx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
    const N = DET_SIZE * DET_SIZE, input = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { input[i] = px[i * 4 + 2]; input[N + i] = px[i * 4 + 1]; input[2 * N + i] = px[i * 4]; }
    const out = await m.det.run({ input: new ort.Tensor("float32", input, [1, 3, DET_SIZE, DET_SIZE]) });

    const faces = [];
    for (const s of [8, 16, 32]) {
      const cols = DET_SIZE / s, cls = out["cls_" + s].data, obj = out["obj_" + s].data;
      const box = out["bbox_" + s].data, kps = out["kps_" + s].data;
      for (let i = 0; i < cols * cols; i++) {
        const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (score < 0.6) continue;
        const r = Math.floor(i / cols), c = i % cols;
        const cx = (c + box[i * 4]) * s, cy = (r + box[i * 4 + 1]) * s;
        const bw = Math.exp(box[i * 4 + 2]) * s, bh = Math.exp(box[i * 4 + 3]) * s;
        const lm = [];
        for (let k = 0; k < 5; k++) lm.push([(kps[i * 10 + 2 * k] + c) * s / scale, (kps[i * 10 + 2 * k + 1] + r) * s / scale]);
        faces.push({ score, x: (cx - bw / 2) / scale, y: (cy - bh / 2) / scale, w: bw / scale, h: bh / scale, lm });
      }
    }
    // NMS
    faces.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const f of faces) {
      if (keep.every((k) => iou(k, f) < 0.3)) keep.push(f);
    }
    return keep;
  }
  function iou(a, b) {
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  // ---------- alinhamento: similaridade (escala+rotação+translação) por mínimos quadrados
  function similarity(src, dst) {
    let sx = 0, sy = 0, dx = 0, dy = 0;
    const n = src.length;
    for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
    sx /= n; sy /= n; dx /= n; dy /= n;
    let num1 = 0, num2 = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const x = src[i][0] - sx, y = src[i][1] - sy, u = dst[i][0] - dx, v = dst[i][1] - dy;
      num1 += x * u + y * v; num2 += x * v - y * u; den += x * x + y * y;
    }
    const a = num1 / den, b = num2 / den; // a = s·cosθ, b = s·sinθ
    return { a, b, tx: dx - (a * sx - b * sy), ty: dy - (b * sx + a * sy) };
  }

  async function embed(m, src, lm) {
    const t = similarity(lm, TEMPLATE);
    const cv = document.createElement("canvas");
    cv.width = cv.height = 112;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 112, 112);
    ctx.imageSmoothingQuality = "high";
    ctx.setTransform(t.a, t.b, -t.b, t.a, t.tx, t.ty);
    ctx.drawImage(src, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const px = ctx.getImageData(0, 0, 112, 112).data, N = 112 * 112, input = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      input[i] = (px[i * 4] - 127.5) / 127.5;
      input[N + i] = (px[i * 4 + 1] - 127.5) / 127.5;
      input[2 * N + i] = (px[i * 4 + 2] - 127.5) / 127.5;
    }
    const name = m.rec.inputNames[0];
    const out = await m.rec.run({ [name]: new ort.Tensor("float32", input, [1, 3, 112, 112]) });
    const e = out[m.rec.outputNames[0]].data;
    let norm = 0;
    for (let i = 0; i < e.length; i++) norm += e[i] * e[i];
    norm = Math.sqrt(norm);
    return { vec: Float32Array.from(e, (v) => v / norm), crop: cv };
  }

  // src: <img>, <canvas> ou ImageBitmap já com a orientação certa
  async function identify(src) {
    const m = await init();
    let w = src.naturalWidth || src.videoWidth || src.width, h = src.naturalHeight || src.videoHeight || src.height;
    // fotos de celular são enormes: reduz antes (o rosto de uma selfie continua grande)
    const lim = 1600;
    if (Math.max(w, h) > lim) {
      const s = lim / Math.max(w, h), c = document.createElement("canvas");
      c.width = Math.round(w * s); c.height = Math.round(h * s);
      c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
      src = c; w = c.width; h = c.height;
    }
    const faces = await detect(m, src, w, h);
    if (!faces.length) return { faces: 0 };
    const face = faces.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a)); // o maior rosto
    const { vec, crop } = await embed(m, src, face.lm);
    const sims = [];
    for (let i = 0; i < m.n; i++) {
      let s = 0;
      for (let j = 0; j < m.dim; j++) s += vec[j] * m.cent[i * m.dim + j];
      sims.push({ pid: i, sim: s });
    }
    sims.sort((a, b) => b.sim - a.sim);
    return { faces: faces.length, vec, crop, matches: sims.slice(0, 6) };
  }

  window.SelfieSearch = { init, identify };
})();
