// Separación de pistas en el dispositivo con Demucs htdemucs_6s (Meta AI, MIT) vía onnxruntime-web.
// FFT / STFT / iSTFT adaptados de microsoft/webnn-developer-preview demos/stem-separator (MIT).
// Mensajes: {type:'separate', left, right} → progress / done / error.

const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const MODEL_REV = '953d0a9e46fd364860c9413ee4c7a9aa4fbcae4b';
// Ventana de 3,9 s: la variante de 7,8 s se desborda en WebGPU (todas las pistas saturadas).
const MODEL = { file: 'demucs_htdemucs_6s_s39_fp16w_ca.onnx', sha256: '761501fb6448709414eb4ce13d67dbd2866de165c94770c7d893d63b9ba0c2d1' };
const MODEL_CACHE = 'practica-modelos-v1';
const STEMS = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
const SR = 44100, NFFT = 4096, HOP = 1024, FB = NFFT / 2, DEMUCS_PAD = HOP / 2 * 3, CENTER_PAD = NFFT / 2;

// ---------- FFT ----------
const HANN = new Float32Array(NFFT).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT));
const COS = new Float32Array(NFFT / 2).map((_, i) => Math.cos(-2 * Math.PI * i / NFFT));
const SIN = new Float32Array(NFFT / 2).map((_, i) => Math.sin(-2 * Math.PI * i / NFFT));
const REV = new Uint32Array(NFFT).map((_, i) => { let r = 0, v = i; for (let b = 0; b < 12; b++) { r = (r << 1) | (v & 1); v >>= 1 } return r });
const RE = new Float32Array(NFFT), IM = new Float32Array(NFFT);

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) { const j = REV[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t } }
  for (let size = 2; size <= NFFT; size <<= 1) {
    const half = size >> 1, step = NFFT / size;
    for (let start = 0; start < NFFT; start += size) for (let k = 0; k < half; k++) {
      const c = COS[k * step], s = SIN[k * step], e = start + k, o = e + half;
      const pr = c * re[o] - s * im[o], pi = c * im[o] + s * re[o];
      re[o] = re[e] - pr; im[o] = im[e] - pi; re[e] += pr; im[e] += pi;
    }
  }
}

function padReflect(x, l, r) {
  const n = x.length, out = new Float32Array(l + n + r);
  out.set(x, l);
  for (let i = 0; i < l; i++) out[l - 1 - i] = x[i + 1];
  for (let i = 0; i < r; i++) out[l + n + i] = x[n - 2 - i];
  return out;
}

// demucs _spec: reflect pad, torch.stft(center=True, normalized=True), frames [2 : 2+T]. Escribe en spec [2, FB, T, 2].
function stftChannel(signal, T, spec, ch) {
  const scale = 1 / Math.sqrt(NFFT);
  const padded = padReflect(padReflect(signal, DEMUCS_PAD, DEMUCS_PAD + T * HOP - signal.length), CENTER_PAD, CENTER_PAD);
  for (let f = 0; f < T; f++) {
    const start = (f + 2) * HOP;
    for (let i = 0; i < NFFT; i++) { RE[i] = padded[start + i] * HANN[i]; IM[i] = 0 }
    fft(RE, IM);
    for (let b = 0; b < FB; b++) {
      const idx = ((ch * FB + b) * T + f) * 2;
      spec[idx] = RE[b] * scale; spec[idx + 1] = IM[b] * scale;
    }
  }
}

// demucs _ispec: añade bin Nyquist y 2 frames a cada lado, istft(center=True, normalized=True), recorta DEMUCS_PAD.
function istftChannel(freq, reOff, imOff, T, L) {
  const frames = T + 4, raw = (frames - 1) * HOP + NFFT;
  const acc = new Float32Array(raw), wsum = new Float32Array(raw), scale = Math.sqrt(NFFT);
  for (let fr = 0; fr < frames; fr++) {
    const f = fr - 2;
    RE.fill(0); IM.fill(0);
    if (f >= 0 && f < T) for (let b = 0; b < FB; b++) { RE[b] = freq[reOff + b * T + f]; IM[b] = freq[imOff + b * T + f] }
    for (let b = 1; b < NFFT / 2; b++) { RE[NFFT - b] = RE[b]; IM[NFFT - b] = -IM[b] }
    for (let i = 0; i < NFFT; i++) IM[i] = -IM[i];
    fft(RE, IM);
    const base = fr * HOP;
    for (let i = 0; i < NFFT; i++) { const w = HANN[i]; acc[base + i] += RE[i] / NFFT * w * scale; wsum[base + i] += w * w }
  }
  const out = new Float32Array(L);
  for (let i = 0; i < L; i++) { const j = CENTER_PAD + DEMUCS_PAD + i, w = wsum[j]; out[i] = w > 1e-8 ? acc[j] / w : 0 }
  return out;
}

// Un segmento estéreo [L|R] → fuentes [stem][ch][L] (Float32 planar).
async function runSegment(session, seg, L, T) {
  const spec = new Float32Array(2 * FB * T * 2);
  stftChannel(seg.subarray(0, L), T, spec, 0);
  stftChannel(seg.subarray(L), T, spec, 1);
  const out = await session.run({ mix: new ort.Tensor('float32', seg, [1, 2, L]), spec: new ort.Tensor('float32', spec, [1, 2, FB, T, 2]) });
  const freq = out.freq.data, time = out.time.data, src = new Float32Array(STEMS.length * 2 * L);
  for (let s = 0; s < STEMS.length; s++) for (let c = 0; c < 2; c++) {
    const reOff = (s * 4 + c * 2) * FB * T, wave = istftChannel(freq, reOff, reOff + FB * T, T, L), off = (s * 2 + c) * L;
    for (let i = 0; i < L; i++) src[off + i] = wave[i] + time[off + i];
  }
  out.freq.dispose?.(); out.time.dispose?.();
  return src;
}

// Solape del 25 % (como demucs.apply) con fundido triangular; cada muestra queda cubierta por 2 segmentos como máximo,
// así que tras cada segmento se finalizan (a Int16 intercalado) sus primeras H muestras y no hace falta Float32 de toda la canción.
// Las pistas deben sumar aproximadamente la mezcla (en música real ~5-10 % de error; con GPU rota, 30 % o más).
// Margen absoluto de 0,005 por muestra (≈ −46 dB, inaudible) para finales en fade y silencios.
// Si no cuadra, la GPU ha dado basura (p. ej. desbordamiento numérico) y se repite en CPU.
function segmentOk(src, seg, L) {
  let err = 0, energy = 0;
  for (let j = 0; j < 2 * L; j++) {
    const c = j < L ? 0 : 1, i = j % L;
    let sum = 0;
    for (let s = 0; s < STEMS.length; s++) sum += src[(s * 2 + c) * L + i];
    if (!Number.isFinite(sum)) return false;
    err += Math.abs(sum - seg[j]); energy += Math.abs(seg[j]);
  }
  return err <= 0.25 * energy + 0.005 * 2 * L;
}

// fallback(): devuelve una sesión en CPU; se usa una sola vez si la GPU falla la comprobación.
async function separate(session, left, right, L, T, onProgress, fallback = null) {
  const N = left.length, H = Math.round(L * 0.75), O = L - H, count = Math.max(1, Math.ceil((N - O) / H));
  const w = new Float32Array(L).map((_, i) => Math.min(1, (i + 1) / O, (L - i) / O));
  const acc = new Float32Array(STEMS.length * 2 * L), wsum = new Float32Array(L);
  const out = STEMS.map(() => new Int16Array(2 * N));
  const t0 = Date.now();
  for (let k = 0; k < count; k++) {
    const start = k * H, end = Math.min(start + L, N), seg = new Float32Array(2 * L);
    seg.set(left.subarray(start, end), 0); seg.set(right.subarray(start, end), L);
    let src = await runSegment(session, seg, L, T);
    if (!segmentOk(src, seg, L) && fallback) { session = await fallback(); fallback = null; src = await runSegment(session, seg, L, T) }
    // La CPU es la referencia: si tampoco cuadra, es un pasaje difícil y se acepta; solo NaN/∞ es fatal.
    if (!src.every(Number.isFinite)) throw new Error('nan');
    for (let j = 0; j < src.length; j++) acc[j] += src[j] * w[j % L];
    for (let i = 0; i < L; i++) wsum[i] += w[i];
    const done = k === count - 1 ? L : H;
    for (let s = 0; s < STEMS.length; s++) for (let c = 0; c < 2; c++) {
      const off = (s * 2 + c) * L, o = out[s];
      for (let i = 0; i < done && start + i < N; i++) {
        const v = acc[off + i] / wsum[i];
        o[2 * (start + i) + c] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
      }
      acc.copyWithin(off, off + H, off + L); acc.fill(0, off + O, off + L);
    }
    wsum.copyWithin(0, H, L); wsum.fill(0, O, L);
    const elapsed = (Date.now() - t0) / 1000;
    onProgress((k + 1) / count, elapsed / (k + 1) * (count - k - 1));
  }
  return out;
}

// ---------- BPM (sobre la pista de batería, Int16 intercalado) ----------
function estimateBpm(stereo, sr = SR) {
  const win = 1024, hop = 512, frames = Math.min(Math.floor((stereo.length / 2 - win) / hop), Math.round(180 * sr / hop));
  if (frames < 200) return null;
  const env = new Float32Array(frames);
  let prev = 0;
  for (let f = 0; f < frames; f++) {
    let e = 0;
    for (let i = 0; i < win; i++) { const j = 2 * (f * hop + i), v = stereo[j] + stereo[j + 1]; e += v * v }
    const le = Math.log1p(e / win / 1e4);
    env[f] = Math.max(0, le - prev); prev = le;
  }
  // Suavizado triangular: los golpes caen entre frames y sin esto la autocorrelación premia periodos casi enteros.
  const sm = new Float32Array(frames);
  for (let f = 2; f < frames - 2; f++) sm[f] = (env[f - 2] + 2 * env[f - 1] + 3 * env[f] + 2 * env[f + 1] + env[f + 2]) / 9;
  env.set(sm);
  const mean = env.reduce((a, b) => a + b, 0) / frames;
  for (let f = 0; f < frames; f++) env[f] -= mean;
  const fr = sr / hop, ac = lag => { let s = 0; for (let f = lag; f < frames; f++) s += env[f] * env[f - lag]; return s };
  let best = null;
  const acAt = x => { const lo = Math.floor(x), t = x - lo; return (1 - t) * ac(lo) + t * ac(lo + 1) };
  for (let bpm = 60; bpm <= 200; bpm += 0.5) {
    const lag = 60 * fr / bpm;
    // Peine de múltiplos del periodo (descarta 2/3 y 3/2) + preferencia suave hacia ~120 BPM para errores de octava.
    let comb = 0;
    for (let k = 1; k <= 4 && k * lag < frames / 2; k++) comb += acAt(k * lag);
    const score = comb * Math.exp(-0.5 * (Math.log2(bpm / 120) / 1.2) ** 2);
    if (!best || score > best.score) best = { bpm, score };
  }
  return Math.round(best.bpm);
}

// ---------- Modelo ----------
async function sha256Hex(buf) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', buf))].map(b => b.toString(16).padStart(2, '0')).join('');
}
async function loadModelBytes(onProgress) {
  const m = MODEL, url = `https://huggingface.co/Andremi/audioweb-models/resolve/${MODEL_REV}/${m.file}`;
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(url);
  if (cached) return cached.arrayBuffer();
  const res = await fetch(url);
  if (!res.ok) throw new Error('download');
  const total = +res.headers.get('content-length') || 0, reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; if (total) onProgress(got / total) }
  const buf = await new Blob(parts).arrayBuffer();
  if (await sha256Hex(buf) !== m.sha256) throw new Error('hash');
  await cache.put(url, new Response(buf.slice(0)));
  return buf;
}
async function createSession(bytes) {
  self.ort || importScripts(ORT_BASE + 'ort.min.js');
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1;
  const providers = [];
  try { if (self.navigator?.gpu && await navigator.gpu.requestAdapter()) providers.push('webgpu') } catch {}
  providers.push('wasm');
  for (const ep of providers) {
    try { return { session: await ort.InferenceSession.create(bytes, { executionProviders: [ep] }), ep } } catch {}
  }
  throw new Error('session');
}

if (typeof module !== 'undefined') module.exports = { estimateBpm, separate, STEMS };
else self.onmessage = async ({ data }) => {
  if (data.type !== 'separate') return;
  const post = (m, t) => self.postMessage(m, t || []);
  try {
    const bytes = await loadModelBytes(p => post({ type: 'progress', phase: 'download', value: p }));
    post({ type: 'progress', phase: 'prepare', value: 0 });
    let { session, ep } = await createSession(bytes);
    const meta = session.inputMetadata.find(x => x.name === 'mix'), L = meta.shape[2];
    const T = session.inputMetadata.find(x => x.name === 'spec').shape[3];
    const progress = (value, eta) => post({ type: 'progress', phase: 'separate', value, eta, ep });
    const toCpu = ep === 'wasm' ? null : async () => { ep = 'wasm'; return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }) };
    const stems = await separate(session, data.left, data.right, L, T, progress, toCpu);
    const bpm = estimateBpm(stems[0]);
    const payload = Object.fromEntries(STEMS.map((n, i) => [n, stems[i]]));
    post({ type: 'done', stems: payload, bpm, ep }, stems.map(s => s.buffer));
  } catch (err) {
    post({ type: 'error', message: err.message || String(err) });
  }
};
