/* Jaylights engine, browser port of the Python package (jaylights/analysis + show.py).
   Same algorithms, same constants. Exposes window.Jaylights / module.exports. */
(function (root) {
  "use strict";
  const FRAME = 2048, HOP = 512, N_COARSE = 24, ENV_DECIM = 4, PHRASE = 4;
  const BANDS = [["kick", 35, 120], ["bass", 120, 300], ["mid", 300, 2000], ["snare", 2000, 5000], ["hat", 7000, 16000]];
  const SECTIONS = ["intro", "groove", "breakdown", "build", "drop", "outro"];
  const [INTRO, GROOVE, BREAKDOWN, BUILD, DROP, OUTRO] = SECTIONS;

  // ---------------------------------------------------------------- helpers
  const mod = (a, m) => ((a % m) + m) % m;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  function mean(a, s = 0, e = a.length) { let t = 0; for (let i = s; i < e; i++) t += a[i]; return e > s ? t / (e - s) : NaN; }
  function std(a) { const m = mean(a); let t = 0; for (let i = 0; i < a.length; i++) t += (a[i] - m) ** 2; return Math.sqrt(t / a.length); }
  function argmax(a) { let j = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; }
  function percentile(a, p) {
    const s = Float64Array.from(a).sort(); const n = s.length; if (!n) return NaN;
    const idx = (p / 100) * (n - 1), lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }
  function robustNorm(x, lo = 5, hi = 98) {
    const a = percentile(x, lo), b = percentile(x, hi), out = new Float64Array(x.length);
    if (b - a < 1e-9) return out;
    for (let i = 0; i < x.length; i++) out[i] = clamp((x[i] - a) / (b - a), 0, 1);
    return out;
  }
  // np.convolve(x, ones(k)/k, mode="same")
  function boxSame(x, k) {
    const n = x.length, ps = new Float64Array(n + 1), out = new Float64Array(n), c = (k - 1) >> 1;
    for (let i = 0; i < n; i++) ps[i + 1] = ps[i] + x[i];
    for (let i = 0; i < n; i++) {
      const hiI = Math.min(n - 1, i + c), loI = Math.max(0, i + c - k + 1);
      out[i] = hiI >= loI ? (ps[hiI + 1] - ps[loI]) / k : 0;
    }
    return out;
  }
  function slope(y) {
    const n = y.length; if (n < 2) return 0;
    const mx = (n - 1) / 2, my = mean(y); let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (i - mx) * (y[i] - my); den += (i - mx) ** 2; }
    return num / den;
  }
  function roundHalfEven(x) { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; }
  function searchRight(a, v) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; } return lo; }
  function searchLeft(a, v) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; } return lo; }

  // ---------------------------------------------------------------- FFT (radix-2, in place)
  const fftCache = {};
  function fftPlan(n) {
    if (fftCache[n]) return fftCache[n];
    const rev = new Uint32Array(n), bits = Math.log2(n);
    for (let i = 0; i < n; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
    return (fftCache[n] = { rev, cos, sin });
  }
  function fft(re, im) {
    const n = re.length, { rev, cos, sin } = fftPlan(n);
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half;
          const tr = re[b] * cos[k] - im[b] * sin[k], ti = re[b] * sin[k] + im[b] * cos[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
  }

  // ---------------------------------------------------------------- features
  async function extractFeatures(x, sr, progress) {
    const n = Math.max(0, 1 + Math.floor((x.length - FRAME) / HOP));
    const nb = FRAME / 2 + 1, freqs = new Float64Array(nb);
    for (let k = 0; k < nb; k++) freqs[k] = k * sr / FRAME;
    const win = new Float64Array(FRAME);
    for (let i = 0; i < FRAME; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FRAME - 1));
    const masks = BANDS.map(([, lo, hi]) => { const idx = []; for (let k = 0; k < nb; k++) if (freqs[k] >= lo && freqs[k] < Math.min(hi, sr / 2)) idx.push(k); return idx; });
    const hiIdx = []; for (let k = 0; k < nb; k++) if (freqs[k] > 1000) hiIdx.push(k);
    const edges = []; const top = Math.min(16000, sr / 2 - 1);
    for (let i = 0; i <= N_COARSE; i++) edges.push(40 * Math.pow(top / 40, i / N_COARSE));
    const ci = edges.map(e => clamp(searchLeft(freqs, e), 1, nb - 1));

    const F = { rms: new Float64Array(n), flux: new Float64Array(n), centroid: new Float64Array(n), flatness: new Float64Array(n), coarse: new Float64Array(n * N_COARSE), n, sr, frameRate: sr / HOP };
    for (const [name] of BANDS) { F[name + "_e"] = new Float64Array(n); F[name + "_flux"] = new Float64Array(n); }
    const re = new Float64Array(FRAME), im = new Float64Array(FRAME), spec = new Float64Array(nb), pw = new Float64Array(nb), cs = new Float64Array(nb);
    let prev = null, cur = new Float64Array(nb);
    for (let f = 0; f < n; f++) {
      const o = f * HOP; let sq = 0;
      for (let i = 0; i < FRAME; i++) { const v = x[o + i]; sq += v * v; re[i] = v * win[i]; im[i] = 0; }
      fft(re, im);
      let tot = 0, cen = 0, acc = 0;
      for (let k = 0; k < nb; k++) {
        const m = Math.hypot(re[k], im[k]); spec[k] = m; const p = m * m; pw[k] = p;
        tot += p; cen += p * freqs[k]; acc += p; cs[k] = acc; cur[k] = Math.log1p(m);
      }
      if (!prev) prev = Float64Array.from(cur);
      let fluxAll = 0; const fluxK = new Float64Array(nb);
      for (let k = 0; k < nb; k++) { const d = cur[k] - prev[k]; fluxK[k] = d > 0 ? d : 0; fluxAll += fluxK[k]; }
      BANDS.forEach(([name], b) => {
        const idx = masks[b]; let s = 0, fl = 0;
        for (const k of idx) { s += pw[k]; fl += fluxK[k]; }
        F[name + "_e"][f] = Math.log10(s / idx.length + 1e-9); F[name + "_flux"][f] = fl / idx.length;
      });
      F.flux[f] = fluxAll / nb;
      F.rms[f] = Math.sqrt(sq / FRAME);
      F.centroid[f] = cen / (tot + 1e-12);
      let lg = 0, mh = 0; for (const k of hiIdx) { const p = pw[k] + 1e-12; lg += Math.log(p); mh += p; }
      F.flatness[f] = Math.exp(lg / hiIdx.length) / (mh / hiIdx.length);
      for (let b = 0; b < N_COARSE; b++) F.coarse[f * N_COARSE + b] = Math.log10((cs[ci[b + 1]] - cs[ci[b]]) / Math.max(ci[b + 1] - ci[b], 1) + 1e-9);
      const t = prev; prev = cur; cur = t;
      if (progress && f % 2000 === 0) await progress(f / n);
    }
    return F;
  }
  const frameTime = (i, sr) => (i * HOP + FRAME / 2) / sr;

  // ---------------------------------------------------------------- beats
  function onsetEnvelope(F) {
    const a = robustNorm(F.kick_flux), b = robustNorm(F.snare_flux), c = robustNorm(F.flux), env = new Float64Array(F.n);
    for (let i = 0; i < F.n; i++) env[i] = 0.6 * a[i] + 0.25 * b[i] + 0.15 * c[i];
    const loc = boxSame(env, Math.max(3, Math.floor(0.5 * F.frameRate)));
    for (let i = 0; i < F.n; i++) env[i] = Math.max(env[i] - loc[i], 0);
    return env;
  }
  function phaseEnvelope(F) {
    const a = robustNorm(F.kick_flux), b = robustNorm(F.snare_flux), env = new Float64Array(F.n);
    for (let i = 0; i < F.n; i++) env[i] = 0.5 * a[i] + 0.5 * b[i];
    const loc = boxSame(env, Math.max(3, Math.floor(0.5 * F.frameRate)));
    for (let i = 0; i < F.n; i++) env[i] = Math.max(env[i] - loc[i], 0);
    return env;
  }
  function estimateTempo(env, fr, range = [90, 190], prior = 128) {
    const n = env.length, m = mean(env);
    let N = 1; while (N < 2 * n) N <<= 1;
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < n; i++) re[i] = env[i] - m;
    fft(re, im);
    for (let i = 0; i < N; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
    fft(re, im); // real+even spectrum: forward FFT == N * inverse
    const acf = new Float64Array(n), a0 = re[0] / N + 1e-12;
    for (let i = 0; i < n; i++) acf[i] = (re[i] / N) / a0;
    const lagMin = Math.floor(60 * fr / range[1]), lagMax = Math.min(n - 1, Math.floor(60 * fr / range[0]) + 1);
    const lags = []; for (let l = Math.max(lagMin, 1); l < lagMax; l++) lags.push(l);
    if (!lags.length) return prior;
    const score = lags.map(l => {
      let s = 0; for (const k of [1, 2, 3, 4]) s += acf[Math.min(l * k, n - 1)];
      const bpm = 60 * fr / l; return (s / 4) * Math.exp(-0.5 * (Math.log2(bpm / prior) / 0.8) ** 2);
    });
    const i = argmax(score); let lag = lags[i];
    if (i > 0 && i < lags.length - 1) {
      const a = score[i - 1], b = score[i], c = score[i + 1], d = a - 2 * b + c;
      if (Math.abs(d) > 1e-12) lag += 0.5 * (a - c) / d;
    }
    return 60 * fr / lag;
  }
  function foldPhase(env, period, offset = 0) {
    const n = env.length, bins = 64, hist = new Float64Array(bins), ph = new Float64Array(n);
    for (let i = 0; i < n; i++) { ph[i] = mod((i - offset) / period, 1); hist[Math.floor(ph[i] * bins) % bins] += env[i]; }
    const h = hist.map((v, j) => v + hist[mod(j - 1, bins)] + hist[(j + 1) % bins]);
    const peak = (argmax(h) + 0.5) / bins, sharp = Math.max(...h) / (mean(h) + 1e-12);
    let wd = 0, ws = 0;
    for (let i = 0; i < n; i++) { const d = mod(ph[i] - peak + 0.5, 1) - 0.5; if (Math.abs(d) < 0.2) { wd += d * env[i]; ws += env[i]; } }
    return ws <= 0 ? [peak, sharp] : [mod(peak + wd / ws, 1), sharp];
  }
  function fitGrid(env, fr, bpm, phaseEnv, search = 0.008) {
    const n = env.length, base = 60 * fr / bpm, bins = 160; let best = [-1, base, 0];
    for (let s = 0; s < 81; s++) {
      const period = base * (1 + (-search + (2 * search) * s / 80)), hist = new Float64Array(bins);
      for (let i = 0; i < n; i++) hist[Math.floor((i % period) / period * bins) % bins] += env[i];
      const h = hist.map((v, j) => v + hist[mod(j - 1, bins)] + hist[(j + 1) % bins]);
      const j = argmax(h), sharp = h[j] / (mean(h) + 1e-12);
      if (sharp > best[0]) best = [sharp, period, (j + 0.5) / bins * period];
    }
    let [, period, phase] = best;
    if (phaseEnv) phase = foldPhase(phaseEnv, period)[0] * period;
    const out = []; for (let t = phase; t < n; t += period) out.push(clamp(Math.round(t), 0, n - 1));
    return out;
  }
  function tempoIsSteady(env, fr, bpm, range, tol = 0.012) {
    const n = env.length; if (n < fr * 60) return true;
    for (let q = 0; q < 4; q++) {
      const part = env.slice(Math.floor(q * n / 4), Math.floor((q + 1) * n / 4));
      if (std(part) < 1e-6) continue;
      const r = estimateTempo(part, fr, range, bpm) / bpm;
      if (Math.min(Math.abs(r - 1), Math.abs(r * 2 - 1), Math.abs(r / 2 - 1)) > tol) return false;
    }
    return true;
  }
  function fillGaps(beats, period, n) {
    if (!beats.length) { const o = []; for (let t = 0; t < n; t += period) o.push(Math.floor(t)); return o; }
    let out = []; let t = beats[0] - period;
    while (t >= -period * 0.25) { out.push(t); t -= period; }
    out.reverse();
    for (let i = 0; i < beats.length - 1; i++) {
      const a = beats[i], gap = beats[i + 1] - a, k = roundHalfEven(gap / period); out.push(a);
      for (let j = 1; j < k; j++) out.push(a + gap * j / k);
    }
    out.push(beats[beats.length - 1]);
    t = beats[beats.length - 1] + period; while (t < n) { out.push(t); t += period; }
    return out.map(v => clamp(Math.round(v), 0, n - 1));
  }
  function trackBeats(env, fr, bpm, tightness = 300) {
    const period = 60 * fr / bpm, n = env.length, sd = std(env) + 1e-9;
    const onset = env.map(v => v / sd), score = Float64Array.from(onset), back = new Int32Array(n).fill(-1);
    const loOff = Math.round(2 * period), hiOff = Math.round(period / 2), offs = [], cost = [];
    for (let o = hiOff; o <= loOff; o++) { offs.push(o); cost.push(-tightness * Math.log(o / period) ** 2); }
    for (let t = hiOff; t < n; t++) {
      let bj = -1, bv = -Infinity;
      for (let j = 0; j < offs.length; j++) { const p = t - offs[j]; if (p < 0) continue; const v = score[p] + cost[j]; if (v > bv) { bv = v; bj = p; } }
      if (bj >= 0 && bv > 0) { score[t] = onset[t] + bv; back[t] = bj; }
    }
    const tail = Math.max(0, n - Math.floor(period)); let t = tail + argmax(score.subarray(tail));
    const beats = []; while (t >= 0) { beats.push(t); t = back[t]; }
    return fillGaps(beats.reverse(), period, n);
  }
  function findDownbeatPhase(F, bf) {
    if (bf.length < 9) return 0;
    const sync = [];
    for (let i = 0; i < bf.length - 1; i++) {
      const a = bf[i], b = bf[i + 1]; if (b <= a) continue;
      const row = new Float64Array(N_COARSE);
      for (let f = a; f < b; f++) for (let k = 0; k < N_COARSE; k++) row[k] += F.coarse[f * N_COARSE + k];
      for (let k = 0; k < N_COARSE; k++) row[k] /= (b - a);
      sync.push(row);
    }
    const nov = [0];
    for (let i = 1; i < sync.length; i++) { let s = 0; for (let k = 0; k < N_COARSE; k++) s += (sync[i][k] - sync[i - 1][k]) ** 2; nov.push(Math.sqrt(s)); }
    const scores = [0, 1, 2, 3].map(p => { let s = 0, c = 0; for (let i = p; i < nov.length; i += 4) { s += nov[i]; c++; } return s / c; });
    return argmax(scores);
  }
  function pickOnsets(x, fr, minGap, k = 1) {
    const n = x.length, w = Math.max(3, Math.floor(fr)), m = boxSame(x, w), sq = boxSame(x.map(v => v * v), w);
    const floor = 0.1 * percentile(x, 99), out = []; let last = -1e9; const gap = Math.floor(minGap * fr);
    for (let i = 0; i < n; i++) {
      const sdv = Math.sqrt(Math.max(sq[i] - m[i] * m[i], 0)), thr = Math.max(m[i] + k * sdv, floor);
      if (x[i] > thr && x[i] >= x[mod(i - 1, n)] && x[i] >= x[(i + 1) % n] && i - last >= gap) { out.push(i); last = i; }
    }
    return out;
  }

  // ---------------------------------------------------------------- sections
  function barFeatures(F, bf, phase, onsets) {
    const n = F.n, kf = F.kick_flux, p95 = percentile(kf, 95) + 1e-9;
    const beatKick = bf.map((a, i) => {
      const b = i + 1 < bf.length ? bf[i + 1] : n; let mx = -Infinity;
      for (let f = Math.max(0, a - 3); f < Math.max(a - 2, b - 3); f++) mx = Math.max(mx, kf[f]);
      return clamp(mx / p95, 0, 1.5);
    });
    const low = robustNorm(F.kick_e), energy = robustNorm(F.rms.map(v => 20 * Math.log10(v + 1e-6)));
    const high = robustNorm(F.hat_e.map((v, i) => v * 0.6 + F.snare_e[i] * 0.4));
    const bright = robustNorm(F.centroid.map(v => Math.log(v + 1))), noise = robustNorm(F.flatness);
    const hits = onsets.snare.concat(onsets.hat).sort((a, b) => a - b);
    const barStarts = []; for (let b = phase; b < bf.length - 4; b += 4) barStarts.push(b);
    const M = { kick: [], low: [], energy: [], high: [], tension: [], density: [] };
    for (const b of barStarts) {
      const f0 = bf[b]; let f1 = bf[Math.min(b + 4, bf.length - 1)]; f1 = Math.min(Math.max(f1, f0 + 1), n);
      const ks = beatKick.slice(b, b + 4).sort((x, y) => x - y);
      M.kick.push(ks[ks.length - 2]);
      M.low.push(mean(low, f0, f1)); M.energy.push(mean(energy, f0, f1)); M.high.push(mean(high, f0, f1));
      M.tension.push((mean(high, f0, f1) + mean(bright, f0, f1) + mean(noise, f0, f1)) / 3);
      M.density.push(searchLeft(hits, f1) - searchLeft(hits, f0));
    }
    if (M.density.length) { const d = Math.max(1, percentile(M.density, 95)); M.density = M.density.map(v => v / d); }
    return [barStarts, M];
  }
  function segment(M) {
    const n = M.kick.length; if (!n) return [];
    const cols = [M.kick, M.low, M.energy, M.high, M.density].map(c => { const m = mean(c), s = std(c) + 1e-6; return c.map(v => (v - m) / s); });
    const nov = (b, w = PHRASE) => {
      if (b - 1 < 0 || b + 1 > n) return 0;
      let s = 0; for (const c of cols) { const a = mean(c, Math.max(0, b - w), b), d = mean(c, b, Math.min(n, b + w)); s += (a - d) ** 2; }
      return Math.sqrt(s);
    };
    let offset = 0, bestS = -Infinity;
    for (let o = 0; o < PHRASE; o++) { let s = 0; for (let b = o + PHRASE; b < n; b += PHRASE) s += nov(b); if (s > bestS) { bestS = s; offset = o; } }
    const bounds = new Set([0, n]);
    for (let b = offset === 0 ? offset + PHRASE : offset; b < n; b += PHRASE) if (nov(b) > 1.0) bounds.add(b);
    const bs = [...bounds].sort((a, b) => a - b), out = [];
    for (let i = 0; i < bs.length - 1; i++) if (bs[i + 1] > bs[i]) out.push([bs[i], bs[i + 1]]);
    return out;
  }
  function labelSections(M) {
    const segs = segment(M); if (!segs.length) return [];
    const kick = M.kick, lo = percentile(kick, 15), hi = percentile(kick, 90), thr = hi - lo > 0.2 ? (lo + hi) / 2 : -1;
    let info = segs.map(([a, b]) => {
      let c = 0; for (let i = a; i < b; i++) if (kick[i] > thr) c++;
      return { a, b, kick: c / (b - a) > 0.5, energy: mean(M.energy, a, b) + 0.5 * mean(M.low, a, b) };
    });
    const ke = info.filter(s => s.kick).map(s => s.energy), top = ke.length ? Math.max(...ke) : 1;
    info.forEach((s, i) => {
      if (s.kick) { const after = i > 0 && !info[i - 1].kick; s.kind = s.energy >= top - (after ? 0.2 : 0.05) ? DROP : GROOVE; }
      else s.kind = BREAKDOWN;
    });
    const runs = [];
    for (const s of info) {
      const r = runs[runs.length - 1];
      if (r && r.kind === s.kind) { const w0 = r.b - r.a, w1 = s.b - s.a; r.energy = (r.energy * w0 + s.energy * w1) / (w0 + w1); r.b = s.b; }
      else runs.push({ ...s });
    }
    info = runs;
    const out = [];
    info.forEach((s, i) => {
      const nxt = info[i + 1], { a, b, kind } = s;
      if (nxt && nxt.kind === DROP && kind !== DROP) {
        const tension = []; for (let j = a; j < b; j++) tension.push(M.tension[j] + 0.5 * M.density[j]);
        const length = b - a; let bstart;
        if (!s.kick) {
          if (length <= 8) bstart = a;
          else {
            const search = tension.slice(0, length - PHRASE), mn = Math.min(...search), floor = mn + 0.25 * (Math.max(...tension) - mn);
            let lastCalm = 0; search.forEach((v, j) => { if (v <= floor) lastCalm = j; });
            const blen = clamp(roundHalfEven((length - lastCalm - 1) / PHRASE) * PHRASE, PHRASE, 16);
            bstart = b - blen;
          }
        } else {
          const tail = Math.min(8, length);
          bstart = slope(tension.slice(-tail)) > 0.02 ? b - tail : b;
        }
        if (bstart > a) out.push({ kind, start: a, end: bstart });
        if (bstart < b) out.push({ kind: BUILD, start: bstart, end: b });
      } else out.push({ kind, start: a, end: b });
    });
    if (out.length && (out[0].kind === GROOVE || out[0].kind === BREAKDOWN)) out[0].kind = INTRO;
    let lastDrop = -1; out.forEach((s, i) => { if (s.kind === DROP) lastDrop = i; });
    if (lastDrop >= 0) {
      const nb = out[out.length - 1].end;
      for (const s of out.slice(lastDrop + 1)) if ((s.kind === GROOVE || s.kind === BREAKDOWN) && s.start >= 0.75 * nb) s.kind = OUTRO;
    }
    const merged = [];
    for (const s of out) { const m = merged[merged.length - 1]; if (m && m.kind === s.kind) m.end = s.end; else merged.push({ ...s }); }
    return merged;
  }

  // ---------------------------------------------------------------- offline timeline
  function analyzeOffline(F, range = [90, 190]) {
    const fr = F.frameRate, sr = F.sr, duration = (F.n - 1) * HOP / sr + FRAME / sr;
    const env = onsetEnvelope(F);
    let bpm = estimateTempo(env, fr, range);
    let bf = tempoIsSteady(env, fr, bpm, range) ? fitGrid(env, fr, bpm, phaseEnvelope(F)) : trackBeats(env, fr, bpm);
    bf = bf.filter(v => v < env.length);
    const beats = bf.map(f => frameTime(f, sr));
    if (beats.length > 8) bpm = 60 / slope(beats);
    const phase = findDownbeatPhase(F, bf);
    const on = { kick: pickOnsets(F.kick_flux, fr, 0.15, 1.2), snare: pickOnsets(F.snare_flux, fr, 0.06, 1.0), hat: pickOnsets(F.hat_flux, fr, 0.05, 1.0) };
    const [barStarts, M] = barFeatures(F, bf, phase, on);
    const secs = labelSections(M).map(s => {
      let sb = s.start < barStarts.length ? barStarts[s.start] : beats.length - 1;
      const eb = s.end < barStarts.length ? barStarts[s.end] : beats.length - 1;
      if (s.start === 0) sb = 0;
      return { kind: s.kind, start: sb > 0 ? beats[sb] : 0, end: eb < beats.length - 1 ? beats[eb] : duration, start_beat: sb > 0 ? sb : 0, end_beat: eb };
    });
    const dec = a => { const o = []; for (let i = 0; i < a.length; i += ENV_DECIM) o.push(a[i]); return o; };
    const curves = {
      energy: dec(robustNorm(F.rms.map(v => 20 * Math.log10(v + 1e-6)))),
      low: dec(robustNorm(F.kick_e.map((v, i) => v * 0.6 + F.bass_e[i] * 0.4))),
      mid: dec(robustNorm(F.mid_e)),
      high: dec(robustNorm(F.hat_e.map((v, i) => v * 0.6 + F.snare_e[i] * 0.4))),
    };
    return new Timeline(duration, bpm, beats, phase, secs, fr / ENV_DECIM, curves,
      { kick: on.kick.map(f => frameTime(f, sr)), snare: on.snare.map(f => frameTime(f, sr)), hat: on.hat.map(f => frameTime(f, sr)) });
  }
  class Timeline {
    constructor(duration, bpm, beats, phase, sections, envRate, curves, onsets) {
      Object.assign(this, { duration, bpm, beats, phase, sections, envRate, curves, onsets });
      this.secStarts = sections.map(s => s.start);
    }
    beatAt(t) {
      const b = this.beats; if (b.length < 2) return t * this.bpm / 60;
      if (t <= b[0]) return (t - b[0]) / (b[1] - b[0]);
      if (t >= b[b.length - 1]) return b.length - 1 + (t - b[b.length - 1]) / (b[b.length - 1] - b[b.length - 2]);
      const i = searchRight(b, t) - 1; return i + (t - b[i]) / (b[i + 1] - b[i]);
    }
    env(name, t, tau) { const on = this.onsets[name]; const i = searchRight(on, t) - 1; return i < 0 ? 0 : Math.exp(-(t - on[i]) / tau); }
    curve(name, t) {
      const a = this.curves[name], x = t * this.envRate, i = Math.floor(x);
      if (i < 0) return a[0]; if (i >= a.length - 1) return a[a.length - 1];
      const f = x - i; return a[i] * (1 - f) + a[i + 1] * f;
    }
    stateAt(t) {
      const beat = this.beatAt(t), rel = beat - this.phase, st = newState();
      st.t = t; st.beat = beat; st.beat_in_bar = mod(rel, 4); st.bar = Math.floor(rel / 4);
      const i = Math.max(0, searchLeft(this.beats, t) - 1);
      st.bpm = i > 0 && i < this.beats.length - 1 ? 60 / (this.beats[i + 1] - this.beats[i]) : this.bpm;
      if (this.sections.length) {
        const si = Math.max(0, searchRight(this.secStarts, t) - 1), s = this.sections[si];
        st.section = s.kind; st.section_index = si; st.section_beats = beat - s.start_beat;
        st.section_progress = clamp(st.section_beats / Math.max(1e-6, s.end_beat - s.start_beat), 0, 1);
        if (si + 1 < this.sections.length) { st.next_section = this.sections[si + 1].kind; st.beats_to_next = s.end_beat - beat; }
      }
      for (const k of ["energy", "low", "mid", "high"]) st[k] = this.curve(k, t);
      st.kick = this.env("kick", t, 0.12); st.snare = this.env("snare", t, 0.09); st.hat = this.env("hat", t, 0.05);
      return st;
    }
  }
  function newState() {
    return { t: 0, bpm: 128, beat: 0, beat_in_bar: 0, bar: 0, section: GROOVE, section_index: 0, section_beats: 0, section_progress: 0,
      next_section: null, beats_to_next: null, energy: 0, low: 0, mid: 0, high: 0, kick: 0, snare: 0, hat: 0, locked: true };
  }

  // ---------------------------------------------------------------- live analyzer
  class AutoGain {
    constructor(frames) { this.rate = 1 / Math.max(1, frames); this.lo = null; this.hi = null; }
    g(x) {
      if (this.lo === null) { this.lo = x; this.hi = x + 1e-6; }
      const span = Math.max(this.hi - this.lo, 1e-6);
      this.hi = Math.max(x, this.hi - span * this.rate); this.lo = Math.min(x, this.lo + span * this.rate);
      return clamp((x - this.lo) / Math.max(this.hi - this.lo, 1e-6), 0, 1);
    }
  }
  class PeakNorm {
    constructor(frames) { this.decay = Math.pow(0.5, 1 / Math.max(1, frames)); this.peak = 1e-9; }
    g(x) { this.peak = Math.max(x, this.peak * this.decay); return Math.min(1, x / this.peak); }
  }
  class OnsetDetector {
    constructor(fr, gap, k = 1.5) { this.a = 1 / (fr * 0.8); this.k = k; this.gap = Math.floor(gap * fr); this.mean = 0; this.var = 0; this.p2 = 0; this.p1 = 0; this.since = 1e6; this.peak = 1e-6; }
    g(x) {
      const p2 = this.p2, p1 = this.p1; this.p2 = p1; this.p1 = x; this.since++;
      this.peak = Math.max(x, this.peak * 0.9995); let hit = 0;
      const thr = this.mean + this.k * Math.sqrt(this.var) + 0.05 * this.peak;
      if (p1 > thr && p1 >= p2 && p1 > x && this.since >= this.gap) { hit = Math.min(1, p1 / this.peak); this.since = 0; }
      const d = x - this.mean; this.mean += this.a * d; this.var = (1 - this.a) * (this.var + this.a * d * d);
      return hit;
    }
  }
  const ratioRelated = (a, b, tol = 0.015) => [0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2].some(q => Math.abs(a / b / q - 1) < tol);

  class LiveAnalyzer {
    constructor(sr, range = [90, 190], latency = 0.03) {
      const fr = sr / HOP; this.fr = fr; this.range = range; this.latency = latency;
      this.envBuf = []; this.envMax = Math.floor(8 * fr); this.phaseBuf = []; this.phaseMax = Math.floor(6 * fr);
      this.everLocked = false; this.phaseGood = false; this.flipVotes = 0;
      this.nKick = new PeakNorm(fr * 30); this.nSnare = new PeakNorm(fr * 30);
      this.kickDet = new OnsetDetector(fr, 0.15); this.snareDet = new OnsetDetector(fr, 0.06); this.hatDet = new OnsetDetector(fr, 0.05);
      this.gE = new AutoGain(fr * 20); this.gL = new AutoGain(fr * 20); this.gM = new AutoGain(fr * 20); this.gH = new AutoGain(fr * 20);
      this.gKick = new PeakNorm(60); this.gTension = new AutoGain(fr * 30); this.gEnv = new AutoGain(fr * 10);
      this.frame = 0; this.bpm = 128; this.beatPos = 0; this.locked = false; this.bpmVotes = 0; this.pendingBpm = null;
      this.downbeat = 0; this.phaseNov = [0, 0, 0, 0]; this.lastCoarse = null; this.beatCoarse = [];
      this.acc = { kick: 0, tension: [], energy: [], low: [], snares: 0 }; this.hist = [];
      this.section = GROOVE; this.sectionIndex = 0; this.sectionStart = 0; this.noKick = 0; this.dropEnergy = 0;
      this.lowSlow = -9; this.kickRef = null; this.preDrop = null; this.preDropLow = 0; this.dropLow = null;
      this.st = newState(); this.st.locked = false; this.kEnv = 0; this.sEnv = 0; this.hEnv = 0;
      this.dec = { kick: Math.exp(-1 / (0.12 * fr)), snare: Math.exp(-1 / (0.09 * fr)), hat: Math.exp(-1 / (0.05 * fr)) };
    }
    bAt(dt) { return this.beatPos + dt * this.bpm / 60; }
    kickRunning(beats = 4) {
      if (this.hist.length < 4) return true;
      const k = this.hist.slice(-beats).map(x => x.kick).sort((a, b) => a - b);
      return k[k.length - 1] > 0.5 && k[k.length - 2] > 0.3;
    }
    frameIn(F, i) {
      const fr = this.fr; this.frame++;
      const kf = F.kick_flux[i], sf = F.snare_flux[i], kE = F.kick_e[i];
      const k = this.kickDet.g(kf), s = this.snareDet.g(sf), h = this.hatDet.g(F.hat_flux[i]);
      this.kEnv = Math.max(k, this.kEnv * this.dec.kick); this.sEnv = Math.max(s, this.sEnv * this.dec.snare); this.hEnv = Math.max(h, this.hEnv * this.dec.hat);
      this.envBuf.push(this.gEnv.g(0.6 * kf + 0.25 * sf + 0.15 * F.flux[i])); if (this.envBuf.length > this.envMax) this.envBuf.shift();
      this.phaseBuf.push(0.5 * this.nKick.g(kf) + 0.5 * this.nSnare.g(sf)); if (this.phaseBuf.length > this.phaseMax) this.phaseBuf.shift();
      if (this.frame % 16 === 0 && this.envBuf.length > fr * 3) this.updateTempo();
      let prev = this.beatPos; this.beatPos += this.bpm / 60 / fr;
      this.lowSlow += (kE - this.lowSlow) / (2 * fr);
      if (k > 0 && this.section !== BUILD && this.section !== BREAKDOWN) this.kickRef = this.kickRef === null ? kE : 0.9 * this.kickRef + 0.1 * kE;
      if (k > 0.3 && (this.section === BUILD || this.section === BREAKDOWN) && this.noKick >= 4 && kE > this.lowSlow + 0.8 && (this.kickRef === null || kE > this.kickRef - 0.4)) {
        const b = Math.round(this.bAt(-1 / fr));
        this.preDrop = [this.section, this.sectionStart, this.downbeat, b];
        this.beatPos = b + this.bpm / 60 / fr; prev = this.beatPos - 1e-6; this.downbeat = mod(b, 4);
        this.setSection(DROP); this.noKick = 0; this.dropEnergy = -120; this.preDropLow = this.lowSlow; this.kEnv = 1;
      }
      const a = this.acc, en = 20 * Math.log10(F.rms[i] + 1e-6);
      a.kick = Math.max(a.kick, kf); a.energy.push(en); a.low.push(kE);
      a.tension.push(F.hat_e[i] + 2 * Math.log(F.centroid[i] + 1) + 3 * F.flatness[i]); if (s > 0) a.snares++;
      this.beatCoarse.push(i);
      if (Math.floor(this.beatPos) > Math.floor(prev)) this.onBeat(F);
      const st = this.st;
      st.energy = 0.8 * st.energy + 0.2 * this.gE.g(en);
      st.low = 0.7 * st.low + 0.3 * this.gL.g(kE * 0.6 + F.bass_e[i] * 0.4);
      st.mid = 0.7 * st.mid + 0.3 * this.gM.g(F.mid_e[i]);
      st.high = 0.7 * st.high + 0.3 * this.gH.g(F.hat_e[i] * 0.6 + F.snare_e[i] * 0.4);
      this.publish();
    }
    updateTempo() {
      const env = this.envBuf; if (std(env) < 1e-4) { this.locked = false; return; }
      if (this.locked && (!this.kickRunning() || this.section === BUILD || this.section === BREAKDOWN)) return;
      const cand = estimateTempo(env, this.fr, this.range, this.everLocked ? this.bpm : 128);
      if (!this.everLocked) this.bpm = cand;
      else if (Math.abs(cand / this.bpm - 1) >= 0.03 && (this.section === DROP || !ratioRelated(cand, this.bpm))) {
        if (this.pendingBpm && Math.abs(cand / this.pendingBpm - 1) < 0.03) this.bpmVotes++; else { this.pendingBpm = cand; this.bpmVotes = 1; }
        if (this.bpmVotes >= 15) { this.bpm = cand; this.bpmVotes = 0; }
      } else this.bpmVotes = 0;
      const raw = this.phaseBuf, loc = boxSame(raw, 21), rec = raw.map((v, j) => Math.max(v - loc[j], 0));
      let tot = 0; for (const v of rec) tot += v; if (tot <= 0) return;
      const n = rec.length, bins = 64, base = 60 * this.fr / this.bpm;
      const periods = []; for (let s = 0; s < 41; s++) periods.push(base * (1 - 0.02 + 0.04 * s / 40));
      const sharp = [], hists = [];
      for (const P of periods) {
        const hst = new Float64Array(bins);
        for (let j = 0; j < n; j++) hst[Math.floor(mod(-(n - 1 - j) / P, 1) * bins) % bins] += rec[j];
        const sm = hst.map((v, q) => v + hst[mod(q - 1, bins)] + hst[(q + 1) % bins]);
        hists.push(sm); sharp.push(Math.max(...sm) / (mean(sm) + 1e-9));
      }
      const j = argmax(sharp); let period = periods[j];
      if (j > 0 && j < periods.length - 1) { const a = sharp[j - 1], b = sharp[j], c = sharp[j + 1], d = a - 2 * b + c; if (Math.abs(d) > 1e-9) period += 0.5 * (a - c) / d * (periods[1] - periods[0]); }
      const peak = (argmax(hists[j]) + 0.5) / bins; let wd = 0, ws = 0;
      for (let q = 0; q < n; q++) { const ph = mod(-(n - 1 - q) / periods[j], 1), d = mod(ph - peak + 0.5, 1) - 0.5; if (Math.abs(d) < 0.2) { wd += d * rec[q]; ws += rec[q]; } }
      const hitPhase = mod(peak + wd / Math.max(ws, 1e-9), 1);
      this.locked = sharp[j] > 1.6; if (!this.locked) return;
      this.everLocked = true; this.bpm += 0.3 * (60 * this.fr / period - this.bpm);
      const err = mod(mod(this.beatPos, 1) + hitPhase + 0.5, 1) - 0.5;
      if (Math.abs(err) > 0.3 && this.phaseGood) { this.flipVotes++; if (this.flipVotes < 5) return; }
      this.flipVotes = 0; this.phaseGood = true; this.beatPos -= Math.abs(err) < 0.3 ? 0.3 * err : err;
    }
    onBeat(F) {
      const a = this.acc, beatN = Math.floor(this.beatPos);
      if (this.beatCoarse.length) {
        const c = new Float64Array(N_COARSE);
        for (const i of this.beatCoarse) for (let k = 0; k < N_COARSE; k++) c[k] += F.coarse[i * N_COARSE + k];
        for (let k = 0; k < N_COARSE; k++) c[k] /= this.beatCoarse.length;
        if (this.lastCoarse) {
          let s = 0; for (let k = 0; k < N_COARSE; k++) s += (c[k] - this.lastCoarse[k]) ** 2;
          const ph = mod(beatN, 4); this.phaseNov = this.phaseNov.map(v => v * 0.995); this.phaseNov[ph] += Math.sqrt(s);
          const best = argmax(this.phaseNov);
          if (best !== this.downbeat && this.phaseNov[best] > 1.15 * this.phaseNov[this.downbeat]) this.downbeat = best;
        }
        this.lastCoarse = c; this.beatCoarse = [];
      }
      this.hist.push({ kick: this.gKick.g(a.kick), energy: a.energy.length ? mean(a.energy) : -60,
        tension: a.tension.length ? this.gTension.g(mean(a.tension)) : 0, snares: a.snares, low: a.low.length ? mean(a.low) : -9 });
      if (this.hist.length > 64) this.hist.shift();
      this.acc = { kick: 0, tension: [], energy: [], low: [], snares: 0 };
      this.sections();
    }
    sections() {
      const h = this.hist; if (h.length < 8) return;
      const last = h[h.length - 1];
      const kickNow = this.kickRunning() || (this.dropLow !== null && last.low > this.dropLow - 0.25);
      if (this.section === DROP && this.kickRunning()) this.dropLow = this.dropLow === null ? last.low : 0.95 * this.dropLow + 0.05 * last.low;
      const beatsIn = this.beatPos - this.sectionStart, energy = mean(h.slice(-4).map(x => x.energy));
      const tension = h.slice(-16).map(x => x.tension), sn = h.slice(-16).map(x => x.snares);
      const rising = tension.length >= 8 && (slope(tension) > 0.01 || mean(sn.slice(-4)) > mean(sn.slice(0, 4)) + 1);
      this.noKick = kickNow ? 0 : this.noKick + 1;
      const sec = this.section;
      if (this.preDrop) {
        if (beatsIn >= 3) {
          const lowUp = mean(h.slice(-3).map(x => x.low)) > this.preDropLow + 0.5;
          if (!(this.kickRunning(3) && lowUp)) { const [p, s, o] = this.preDrop; this.section = p; this.sectionStart = s; this.downbeat = o; this.sectionIndex++; this.noKick = 4; }
          this.preDrop = null;
        }
        return;
      }
      if (beatsIn < 8) return;
      if (kickNow) {
        if (sec === BUILD || sec === BREAKDOWN) { this.setSection(sec === BREAKDOWN && energy < this.dropEnergy - 6 ? GROOVE : DROP); if (this.section === DROP) this.dropEnergy = energy; }
        else if (sec === DROP) {
          this.dropEnergy = Math.max(this.dropEnergy - 0.02, mean(h.slice(-8).map(x => x.energy)));
          const earlier = mean(h.slice(-8, -4).map(x => x.energy));
          if (energy < this.dropEnergy - 6 && earlier < this.dropEnergy - 6 && beatsIn > 32) this.setSection(GROOVE);
        } else if (sec === GROOVE && rising && mean(sn.slice(-4)) >= 3 && beatsIn > 32) this.setSection(BUILD);
      } else {
        if ((sec === DROP || sec === GROOVE) && this.noKick >= 2) this.setSection(rising ? BUILD : BREAKDOWN);
        else if (sec === BREAKDOWN && rising) this.setSection(BUILD);
      }
    }
    setSection(k) { if (k !== this.section) { this.section = k; this.sectionIndex++; this.sectionStart = Math.round(this.beatPos); } }
    publish() {
      const st = this.st, beat = this.bAt(this.latency), rel = beat - this.downbeat;
      st.t = this.frame / this.fr; st.bpm = this.bpm; st.beat = beat; st.beat_in_bar = mod(rel, 4); st.bar = Math.floor(rel / 4);
      st.section = this.section; st.section_index = this.sectionIndex; st.section_beats = beat - this.sectionStart;
      if (this.section === BUILD) { const tn = this.hist.length ? this.hist[this.hist.length - 1].tension : 0; st.section_progress = Math.min(0.98, 0.5 * st.section_beats / 32 + 0.5 * tn); }
      else st.section_progress = Math.min(1, st.section_beats / 64);
      st.next_section = null; st.beats_to_next = null; st.locked = this.locked;
      const frac = beat - Math.floor(beat), kickOn = this.kickRunning() && this.section !== BUILD && this.section !== BREAKDOWN;
      st.kick = this.locked && kickOn ? Math.exp(-frac * 60 / this.bpm / 0.12) : this.kEnv; st.snare = this.sEnv; st.hat = this.hEnv;
    }
  }

  // ---------------------------------------------------------------- show engine
  function hsv(h, s = 1, v = 1) {
    h = mod(h, 360) / 360; const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  }
  const mix = (a, b, f) => { f = clamp(f, 0, 1); return a.map((x, i) => x + (b[i] - x) * f); };
  const wave = x => 0.5 + 0.5 * Math.sin(2 * Math.PI * x);
  const WHITE = [1, 1, 1];
  const PALETTES = {
    hot: [[0, 300], [330, 30], [280, 0], [0, 45], [300, 200]], cool: [[220, 270], [190, 240], [260, 300], [170, 210]],
    acid: [[100, 290], [80, 0], [120, 300], [0, 280]], warm: [[30, 330], [15, 45], [340, 20], [40, 280]],
    mixed: [[300, 180], [0, 220], [270, 30], [160, 320], [200, 60]],
  };
  const GENRES = {
    auto: [[90, 190], "mixed", "cool", 1.0, 0.8], house: [[115, 132], "warm", "cool", 0.6, 0.6],
    progressive: [[118, 132], "mixed", "cool", 0.7, 0.6], techno: [[120, 150], "cool", "cool", 0.9, 0.7],
    trance: [[125, 145], "mixed", "cool", 0.9, 0.7], dubstep: [[134, 156], "acid", "cool", 1.0, 1.0],
    trap: [[125, 165], "hot", "cool", 1.0, 1.0], dnb: [[160, 182], "acid", "cool", 1.0, 0.9],
  };
  function demoRig() {
    const fx = [];
    for (let i = 0; i < 8; i++) fx.push({ name: "Par " + (i + 1), kind: "wash", group: "front", y: 0, rgb: true });
    for (let i = 0; i < 4; i++) fx.push({ name: "Mover " + (i + 1), kind: "mover", group: "movers", x: 0.2 + i * 0.2, y: 0, pan_span: 0.12, tilt_span: 0.18, rgb: true });
    fx.push({ name: "Strobe L", kind: "strobe", group: "strobe", x: 0.1, y: 0 }, { name: "Strobe R", kind: "strobe", group: "strobe", x: 0.9, y: 0 });
    for (let i = 0; i < 6; i++) fx.push({ name: "Floor " + (i + 1), kind: "wash", group: "floor", y: 1, rgb: true });
    const groups = {};
    for (const f of fx) { Object.assign(f, { pan_center: 0.5, tilt_center: 0.5, tilt_up: -1, pan_span: f.pan_span || 0.15, tilt_span: f.tilt_span || 0.15 }); (groups[f.group] = groups[f.group] || []).push(f); }
    for (const m of Object.values(groups)) {
      const auto = m.filter(f => f.x === undefined); auto.forEach((f, j) => f.x = (j + 0.5) / auto.length);
      m.sort((a, b) => a.x - b.x).forEach((f, j) => { f.index = j; f.group_size = m.length; });
    }
    return fx;
  }
  class ShowEngine {
    constructor(fixtures, genre = "auto", seed = 1) {
      this.fx = fixtures; const g = GENRES[genre] || GENRES.auto;
      [, this.dropPal, this.calmPal, this.strobeAmt, this.snareAmt] = g;
      let s = seed >>> 0; this.rand = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      this.secIdx = -1; this.palette = [hsv(0), hsv(240)]; this.smooth = fixtures.map(() => ({ pan: 0.5, tilt: 0.5 })); this.first = true;
    }
    pick(kind) {
      const name = kind === DROP ? this.dropPal : kind === BREAKDOWN ? this.calmPal : "mixed", list = PALETTES[name];
      const hues = list[Math.floor(this.rand() * list.length)];
      this.palette = hues.map(h => kind === BREAKDOWN ? hsv(h, 0.9, 0.9) : hsv(h));
    }
    render(st, ctl) {
      if (st.section_index !== this.secIdx) { this.secIdx = st.section_index; if (st.section !== BUILD) this.pick(st.section); }
      const out = this.fx.map(f => ({ i: 0, c: [1, 1, 1], hz: 0, pan: f.pan_center, tilt: f.tilt_center }));
      const look = st.section === DROP ? this.drop : st.section === BUILD ? this.build : st.section === BREAKDOWN ? this.breakdown : this.groove;
      this.fx.forEach((f, i) => look.call(this, st, f, out[i]));
      return this.post(st, ctl, out);
    }
    move(f, o, dp, dt) { o.pan = f.pan_center + dp * f.pan_span; o.tilt = f.tilt_center + dt * f.tilt_span * f.tilt_up; }
    groove(st, f, o) {
      const A = this.palette[0], B = this.palette[this.palette.length - 1], calm = st.section === INTRO || st.section === OUTRO ? 0.7 : 1, swap = mod(st.bar, 2);
      if (f.kind === "wash") { o.c = (f.index + swap) % 2 === 0 ? A : B; o.i = calm * (0.2 + 0.6 * st.kick + 0.2 * st.energy); }
      else if (f.kind === "mover") {
        o.c = f.index % 2 ? B : A; o.i = calm * (0.5 + 0.3 * st.hat); const ph = f.index * 0.25;
        this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 16 + ph)) * 0.8, Math.sin(2 * Math.PI * (st.beat / 8 + ph)) * 0.5);
      } else if (f.kind === "strobe") o.i = st.energy > 0.6 ? 0.35 * st.snare * this.snareAmt : 0;
    }
    breakdown(st, f, o) {
      const A = this.palette[0], B = this.palette[this.palette.length - 1], breathe = 0.6 + 0.4 * wave(st.beat / 16), level = 0.1 + 0.5 * Math.max(st.mid, st.energy * 0.8);
      if (f.kind === "wash") { o.c = mix(A, B, wave(f.x * 0.8 + st.beat / 32)); o.i = level * breathe; }
      else if (f.kind === "mover") { o.c = mix(B, A, wave(st.beat / 32 + f.index * 0.3)); o.i = 0.35 * breathe; this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 32 + f.index * 0.37)) * 0.6, 0.6); }
      else o.i = 0;
    }
    build(st, f, o) {
      const p = st.section_progress;
      if (st.next_section === DROP && st.beats_to_next !== null && st.beats_to_next <= 1) { o.i = 0; this.move(f, o, 0, 1); return; }
      const div = p < 0.25 ? 1 : p < 0.5 ? 2 : p < 0.75 ? 4 : 8, color = mix(this.palette[0], WHITE, Math.pow(p, 1.5)), sub = st.beat * div, gate = 1 - mod(sub, 1);
      if (f.kind === "wash") {
        o.c = color;
        if (p < 0.5) { const lit = mod(Math.floor(sub), Math.max(1, f.group_size)); o.i = (0.35 + 0.65 * p) * (lit === f.index ? 1 : 0.15) * (0.4 + 0.6 * gate); }
        else o.i = (0.35 + 0.65 * p) * gate;
        o.i = Math.max(o.i, st.snare * (0.4 + 0.6 * p));
      } else if (f.kind === "mover") {
        o.c = color; o.i = 0.4 + 0.6 * p; const sp = 1 + 3 * p;
        this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 8 * sp + f.index * 0.25)) * (1 - p), 0.3 + 0.7 * p);
      } else if (f.kind === "strobe") {
        if (p > 0.6) { o.i = (p - 0.6) / 0.4 * this.strobeAmt; o.hz = st.bpm / 60 * div; } else o.i = 0.6 * st.snare * p;
      }
    }
    drop(st, f, o) {
      const sb = st.section_beats, A = this.palette[0], B = this.palette[this.palette.length - 1];
      if (sb < 1) { o.c = WHITE; o.i = f.kind !== "strobe" ? 1 - 0.5 * sb : 1 - sb; this.move(f, o, 0, -0.4); return; }
      const v = mod(Math.floor(sb / 32), 3), bi = Math.floor(st.beat), k = st.kick;
      if (f.kind === "wash") {
        if (v === 0) { o.c = mod(bi + f.index, 2) === 0 ? A : B; o.i = 0.15 + 0.85 * k; }
        else if (v === 1) {
          const n = Math.max(1, f.group_size), step = mod(Math.floor(st.beat * 2), n > 2 ? 2 * n - 2 : n), pos = step < n ? step : 2 * n - 2 - step;
          o.c = mod(Math.floor(bi / 2), 2) === 0 ? A : B; o.i = Math.max(pos === f.index ? 1 : 0.1, 0.5 * k);
        } else { const left = f.x < 0.5, on = left === (mod(bi, 2) === 0); o.c = on ? A : B; o.i = on ? 0.3 + 0.7 * k : 0.15 * k; }
        o.i = Math.max(o.i, st.snare * this.snareAmt * 0.9);
      } else if (f.kind === "mover") {
        o.c = f.index % 2 ? B : A; o.i = 0.5 + 0.5 * k; const mir = f.index % 2 ? -1 : 1;
        if (v === 0) this.move(f, o, mir * Math.sin(2 * Math.PI * st.beat / 4), -0.5 + 0.8 * k);
        else if (v === 1) { const a = 2 * Math.PI * (st.beat / 4 + f.index * 0.25); this.move(f, o, Math.sin(a), Math.cos(a) * 0.8); }
        else this.move(f, o, (mod(bi + f.index, 2) ? 1 : -1) * 0.9, 0.2);
      } else if (f.kind === "strobe") {
        o.i = v === 1 ? (st.beat_in_bar < 0.15 ? this.strobeAmt : 0) : this.strobeAmt * Math.max(st.snare ** 2 * this.snareAmt, k ** 4 * 0.5);
        o.c = WHITE;
      }
    }
    post(st, ctl, out) {
      const a = st.section === BREAKDOWN ? 0.08 : 0.3;
      this.fx.forEach((f, i) => {
        const o = out[i], sm = this.smooth[i];
        if (this.first) { sm.pan = o.pan; sm.tilt = o.tilt; }
        sm.pan += (o.pan - sm.pan) * a; sm.tilt += (o.tilt - sm.tilt) * a;
        o.pan = clamp(sm.pan, 0, 1); o.tilt = clamp(sm.tilt, 0, 1);
        o.hz = Math.min(o.hz, ctl.maxStrobe); o.i = clamp(o.i, 0, 1);
      });
      this.first = false; return out;
    }
  }
  // what a fixture looks like at time t (software/hardware strobe shown as on/off)
  function effective(f, o, t) {
    let i = o.i; if (o.hz > 0 && mod(t * o.hz, 1) > 0.3) i = 0;
    const c = f.rgb ? o.c : WHITE; return [i, c[0], c[1], c[2], o.pan, o.tilt];
  }

  // ---------------------------------------------------------------- whole pipeline
  const FPS = 30;
  function pack(rig, eng, states, fxOut, stOut, frame) {
    const ctl = { maxStrobe: 10 };
    const st = states, outs = eng.render(st, ctl), t = frame / FPS;
    rig.forEach((f, j) => {
      const e = effective(f, outs[j], t), o = (frame * rig.length + j) * 6;
      for (let q = 0; q < 6; q++) fxOut[o + q] = Math.round(clamp(e[q], 0, 1) * 255);
    });
    const so = frame * 13, bpm10 = Math.round(st.bpm * 10);
    stOut[so] = Math.max(0, SECTIONS.indexOf(st.section)); stOut[so + 1] = Math.floor(st.beat_in_bar * 60) % 240;
    stOut[so + 2] = (bpm10 >> 8) & 255; stOut[so + 3] = bpm10 & 255;
    [st.section_progress, st.kick, st.snare, st.hat, st.low, st.mid, st.high, st.energy].forEach((v, q) => stOut[so + 4 + q] = Math.round(clamp(v, 0, 1) * 255));
    stOut[so + 12] = st.locked ? 1 : 0;
  }
  async function analyzeTrack(x, sr, genre, progress = async () => {}) {
    const range = (GENRES[genre] || GENRES.auto)[0];
    const F = await extractFeatures(x, sr, p => progress("Listening", p));
    await progress("Finding the beat and the drops", 0);
    const tl = analyzeOffline(F, range);
    const rig = demoRig(), dur = x.length / sr, frames = Math.floor(dur * FPS);
    const pre = { fx: new Uint8Array(frames * rig.length * 6), st: new Uint8Array(frames * 13) };
    let eng = new ShowEngine(rig, genre);
    for (let fi = 0; fi < frames; fi++) { const st = tl.stateAt(fi / FPS + 0.03); pack(rig, eng, st, pre.fx, pre.st, fi); }
    // live: replay the features frame by frame, snapshot at each video frame
    const live = { fx: new Uint8Array(frames * rig.length * 6), st: new Uint8Array(frames * 13) }, liveSecs = [];
    const la = new LiveAnalyzer(sr, range, 0.03); eng = new ShowEngine(rig, genre); let fpos = 0, last = null;
    for (let fi = 0; fi < frames; fi++) {
      const t = fi / FPS;
      while (fpos < F.n && frameTime(fpos, sr) <= t) la.frameIn(F, fpos++);
      const st = { ...la.st };
      if (st.section !== last) { liveSecs.push({ kind: st.section, start: t }); last = st.section; }
      pack(rig, eng, st, live.fx, live.st, fi);
      if (fi % 900 === 0) await progress("Simulating live mode", fi / frames);
    }
    liveSecs.forEach((s, i) => s.end = i + 1 < liveSecs.length ? liveSecs[i + 1].start : dur);
    return {
      track: { bpm: tl.bpm, detected_bpm: Math.round(tl.bpm * 100) / 100, duration: dur, frames,
        offline_sections: tl.sections.map(s => ({ kind: s.kind, start: s.start, end: s.end })), live_sections: liveSecs, truth: null },
      pre, live, timeline: tl,
    };
  }

  const api = { extractFeatures, analyzeOffline, LiveAnalyzer, ShowEngine, demoRig, analyzeTrack, frameTime, SECTIONS, GENRES, FPS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Jaylights = api;
})(typeof self !== "undefined" ? self : globalThis);
