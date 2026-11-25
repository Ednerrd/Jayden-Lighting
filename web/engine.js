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
      let tot = 0, cen = 0, acc = 0;  // acc/cs mimic numpy's float32 cumsum (matters for near-silent bands)
      for (let k = 0; k < nb; k++) {
        const m = Math.hypot(re[k], im[k]); spec[k] = m; const p = m * m; pw[k] = p;
        tot += p; cen += p * freqs[k]; const m32 = Math.fround(m); acc = Math.fround(acc + Math.fround(m32 * m32)); cs[k] = acc; cur[k] = Math.log1p(m);
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
      for (let b = 0; b < N_COARSE; b++) F.coarse[f * N_COARSE + b] = Math.log10(Math.fround(cs[ci[b + 1]] - cs[ci[b]]) / Math.max(ci[b + 1] - ci[b], 1) + 1e-9);
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
    return rerankRatios(acf, fr, 60 * fr / lag, range);
  }
  function acfAt(acf, lag) {
    const i = Math.floor(lag); if (i + 1 >= acf.length) return 0;
    const f = lag - i; return acf[i] * (1 - f) + acf[i + 1] * f;
  }
  // bar-level re-ranking of ratio-related tempos (syncopated beats fool the beat-level ACF)
  function rerankRatios(acf, fr, bpm, range) {
    const n = acf.length;
    const cands = [1 / 2, 2 / 3, 3 / 4, 4 / 5, 1, 5 / 4, 4 / 3, 3 / 2, 2].map(r => bpm * r).filter(c => range[0] <= c && c <= range[1]);
    const score = b => {
      const L = 60 * fr / b, bars = [];
      for (const k of [4, 8, 16]) if (k * L < 0.6 * n) bars.push(acfAt(acf, k * L));
      if (!bars.length) return -1;
      const beat = (acfAt(acf, L) + acfAt(acf, 2 * L)) / 2, prior = Math.exp(-0.5 * (Math.log2(b / 135) / 1.0) ** 2);
      return (0.65 * mean(bars) + 0.35 * beat) * (0.85 + 0.15 * prior);
    };
    let best = bpm, bs = -Infinity;
    for (const c of cands) { const s = score(c); if (s > bs) { bs = s; best = c; } }
    return bs > score(bpm) + 1e-9 ? best : bpm;
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
  // per-phase evidence (4 numbers) that beat p of the list is a "1": z(novelty) + z(kick at beat) - z(snare at beat)
  function downbeatScores(coarse, kickFlux, snareFlux, n, bf) {
    if (bf.length < 9) return [0, 0, 0, 0];
    const sync = [];
    for (let i = 0; i < bf.length - 1; i++) {
      const a = bf[i], b = bf[i + 1]; if (b <= a) continue;
      const row = new Float64Array(N_COARSE);
      for (let f = a; f < b; f++) for (let k = 0; k < N_COARSE; k++) row[k] += coarse[f * N_COARSE + k];
      for (let k = 0; k < N_COARSE; k++) row[k] /= (b - a);
      sync.push(row);
    }
    const nov = [0];
    for (let i = 1; i < sync.length; i++) { let s = 0; for (let k = 0; k < N_COARSE; k++) s += (sync[i][k] - sync[i - 1][k]) ** 2; nov.push(Math.sqrt(s)); }
    const m = nov.length;
    const atBeats = x => {
      const xn = robustNorm(x), o = [];
      for (let q = 0; q < m; q++) { const f = bf[q]; let mx = -Infinity; for (let j = Math.max(0, f - 3); j < Math.min(n, f + 4); j++) mx = Math.max(mx, xn[j]); o.push(mx); }
      return o;
    };
    const z = v => { const mu = mean(v), sd = std(v) + 1e-9; return v.map(t => (t - mu) / sd); };
    const zn = z(nov), zk = z(atBeats(kickFlux)), zs = z(atBeats(snareFlux)), sc = zn.map((v, i) => v + zk[i] - zs[i]);
    return [0, 1, 2, 3].map(p => { let s = 0, c = 0; for (let i = p; i < sc.length; i += 4) { s += sc[i]; c++; } return s / c; });
  }
  function findDownbeatPhase(F, bf) {
    if (bf.length < 9) return 0;
    return argmax(downbeatScores(F.coarse, F.kick_flux, F.snare_flux, F.n, bf));
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
  // [start, end, flag] runs; runs shorter than minLen get absorbed by a neighbour
  function runsOf(flags, minLen = 2) {
    let runs = [];
    flags.forEach((f, i) => { if (runs.length && runs[runs.length - 1][2] === f) runs[runs.length - 1][1] = i + 1; else runs.push([i, i + 1, f]); });
    let changed = true;
    while (changed && runs.length > 1) {
      changed = false;
      for (let i = 0; i < runs.length; i++) {
        const r = runs[i];
        const preDropGap = !r[2] && i > 0 && i < runs.length - 1 && runs[i + 1][1] - runs[i + 1][0] >= 4;
        if (r[1] - r[0] < minLen && !preDropGap) {
          const j = i > 0 ? i - 1 : i + 1;
          runs[j][0] = Math.min(runs[j][0], r[0]); runs[j][1] = Math.max(runs[j][1], r[1]);
          runs.splice(i, 1);
          const merged = [];
          for (const q of runs) { if (merged.length && merged[merged.length - 1][2] === q[2]) merged[merged.length - 1][1] = q[1]; else merged.push(q); }
          runs = merged; changed = true; break;
        }
      }
    }
    return runs;
  }
  function labelSections(M) {
    const low = M.low, n = low.length; if (!n) return [];
    const p90 = percentile(low, 90), p10 = percentile(low, 10), thr = p90 - p10 > 0.15 ? p90 - 0.15 : null;
    const full = Array.from(low, v => thr !== null ? v > thr : true);
    const power = low.map((v, i) => M.energy[i] + 0.5 * v + 0.5 * M.high[i]);
    const tension = low.map((v, i) => M.tension[i] + 0.5 * M.density[i]);
    const pieces = [];
    for (const [a, b, f] of runsOf(full)) {
      const cuts = [a];
      if (f && b - a >= 12) {
        for (let c = a + 4; c < b - 3; c += PHRASE) {
          const before = mean(power, Math.max(cuts[cuts.length - 1], c - 8), c), after = mean(power, c, Math.min(b, c + 8));
          if (after - before > 0.2) cuts.push(c);
        }
      }
      cuts.push(b);
      for (let i = 0; i < cuts.length - 1; i++) pieces.push([cuts[i], cuts[i + 1], f, i > 0]);
    }
    const fp = pieces.filter(p => p[2]).map(p => mean(power, p[0], p[1])), top = fp.length ? Math.max(...fp) : 1;
    let groovePower = null;  // loudness of the first groove, once we've heard one
    const kinds = pieces.map(([a, b, f, jumped], i) => {
      if (!f) return BREAKDOWN;
      const afterGap = i > 0 && (!pieces[i - 1][2] || jumped), margin = i === 1 ? 0.08 : 0.2;
      const pw = mean(power, a, b);
      let isDrop = afterGap && pw >= top - margin;
      // back at groove level (closer to the groove than the biggest drop) = groove again
      // ...but only after a short dip (<= 8 bars). A real breakdown runs 16+ bars, and
      // whatever comes after one is a drop even when it's no louder than the groove
      const shortDip = i > 0 && !pieces[i - 1][2] && pieces[i - 1][1] - pieces[i - 1][0] <= 8;
      if (isDrop && shortDip && groovePower !== null && pw < (groovePower + top) / 2) isDrop = false;
      if (!isDrop && groovePower === null) groovePower = pw;
      return isDrop ? DROP : GROOVE;
    });
    // trap: the hook can be as loud as the drop (Love Sosa). The real drops land after a big
    // build (a long stretch with the sub cut). A section that only follows a short dip, isn't
    // louder than those drops and is much busier (hi-hat rolls over the hook) is the hook = groove.
    const big = [];
    for (let i = 1; i < pieces.length; i++) if (kinds[i] === DROP && !pieces[i - 1][2] && pieces[i - 1][1] - pieces[i - 1][0] >= 6) big.push(i);
    const hooks = [];
    if (big.length) {
      const refPw = Math.max(...big.map(i => mean(power, pieces[i][0], pieces[i][1])));
      const refD = Math.max(...big.map(i => mean(M.density, pieces[i][0], pieces[i][1])));
      for (let i = 1; i < pieces.length; i++) {
        const [a, b] = pieces[i], [pa, pb, pf] = pieces[i - 1];
        if (kinds[i] === DROP && !big.includes(i) && !pf && pb - pa <= 2 && mean(power, a, b) <= refPw && mean(M.density, a, b) >= refD + 0.25) {
          kinds[i] = GROOVE; hooks.push([a, b]);
        }
      }
    }
    // a 1-2 bar hole between two alike drop halves (a vocal cut) is part of the drop. If the part
    // before it was still climbing (a snare roll, busier than what follows), that was the build
    // and the hole is the silence right before the real drop: leave it.
    for (let i = 1; i < pieces.length - 1; i++) {
      if (kinds[i] === BREAKDOWN && kinds[i - 1] === DROP && kinds[i + 1] === DROP && pieces[i][1] - pieces[i][0] <= 2) {
        const [a0, a1] = pieces[i - 1], [c0, c1] = pieces[i + 1];
        if (slope(tension.slice(a0, a1)) <= 0.015 && Math.abs(mean(power, a0, a1) - mean(power, c0, c1)) < 0.1
            && Math.abs(mean(M.density, a0, a1) - mean(M.density, c0, c1)) < 0.2) kinds[i] = DROP;
      }
    }
    // the kick slamming back in after the sub was cut is the drop, even while the sub is still
    // swelling back up for that first bar
    for (let i = 1; i < pieces.length; i++) {
      const a = pieces[i][0];
      if (kinds[i] === DROP && !pieces[i - 1][2] && a - pieces[i - 1][0] >= 5 && M.kick[a - 1] >= 1.2 && low[a - 1] >= 0.5
          && low[a - 4] < 0.2 && low[a - 3] < 0.2 && low[a - 2] < 0.2) { pieces[i][0]--; pieces[i - 1][1]--; }
    }
    // a short bass dip that keeps the kick going and leads back into the groove is groove
    for (let i = 1; i < pieces.length - 1; i++) {
      if (kinds[i] === BREAKDOWN && kinds[i + 1] === GROOVE && pieces[i][1] - pieces[i][0] <= 8 && mean(M.kick, pieces[i][0], pieces[i][1]) >= 0.6) kinds[i] = GROOVE;
    }
    const out = [], last = () => out[out.length - 1];
    pieces.forEach(([a, b, f], i) => {
      const kind = kinds[i], nxt = i + 1 < kinds.length ? kinds[i + 1] : null;
      if (nxt !== DROP || kind === DROP) { out.push({ kind, start: a, end: b }); return; }
      let lo = a; if (out.length && last().kind !== DROP) lo = last().start;
      const t = tension.slice(a, b), length = b - a; let blen;
      if (!f && length <= 8) blen = length;  // short dip right before a drop = the build
      else {
        const search = length > 4 ? t.slice(0, -2) : t, mn = Math.min(...search), floor = mn + 0.25 * (Math.max(...t) - mn);
        let lastCalm = 0; search.forEach((v, j) => { if (v <= floor) lastCalm = j; });
        blen = length - lastCalm - 1;
        if (f && slope(t.slice(-Math.min(8, length))) <= 0.01) blen = 0;
      }
      if (blen > 0) blen = clamp(Math.ceil(blen / PHRASE) * PHRASE, PHRASE, 32);
      let start = Math.max(lo, b - blen);
      // drops and grooves come in whole 8-bar phrases: a build that would leave the section
      // before it on a half phrase (e.g. 12 bars) really started 4 bars earlier, kick still going
      if (out.length && (last().kind === DROP || last().kind === GROOVE) && last().end >= start) {
        const dlen = start - last().start;
        if (dlen >= 12 && dlen % 8 === 4) { start -= 4; last().end = start; }
      }
      while (out.length && last().start >= start && last().kind !== DROP) out.pop();
      if (out.length && last().end > start && last().kind !== DROP) last().end = start;
      else if (out.length && last().end > start) start = last().end;
      if (start > a && (!out.length || last().end < a)) out.push({ kind, start: a, end: start });
      else if (start > a) out.push({ kind, start: Math.max(a, last().end), end: start });
      if (start < b) out.push({ kind: BUILD, start, end: b });
    });
    let lastDrop = -1; out.forEach((s, i) => { if (s.kind === DROP) lastDrop = i; });
    if (lastDrop >= 0) {
      const d = out[lastDrop]; let bestC = null, bestGain = 0.12;
      for (let c = d.start + 8; c < d.end - 3; c += PHRASE) {
        const gain = mean(power, d.start, c) - mean(power, c, d.end);
        if (gain > bestGain) { bestC = c; bestGain = gain; }
      }
      if (bestC !== null) { out.splice(lastDrop + 1, 0, { kind: OUTRO, start: bestC, end: d.end }); d.end = bestC; }
      const nb = out[out.length - 1].end;
      for (const s of out.slice(lastDrop + 1)) if ((s.kind === GROOVE || s.kind === BREAKDOWN) && s.start >= 0.75 * nb) s.kind = OUTRO;
    }
    if (out.length && (out[0].kind === GROOVE || out[0].kind === BREAKDOWN)) out[0].kind = INTRO;
    const merged = [];
    for (const s of out) { const m = merged[merged.length - 1]; if (m && m.kind === s.kind) m.end = s.end; else merged.push({ ...s }); }
    const secs = brightDrops(merged, M), grid = phraseGrid(secs);
    if (grid === null) return secs;
    return snapToGrid(hookBuilds(secs, hooks, power, grid), grid);
  }

  // Which bar (mod 4) phrases start on, from the drops (the most reliable boundaries). Only
  // when every drop agrees: if they don't, one of them is off and the grid can't be trusted.
  function phraseGrid(secs) {
    const starts = new Set(secs.filter(s => s.kind === DROP).map(s => mod(s.start, PHRASE)));
    return starts.size === 1 ? [...starts][0] : null;
  }
  function fromLabels(labels) {
    const out = [];
    labels.forEach((lab, i) => { const l = out[out.length - 1]; if (l && l.kind === lab) l.end = i + 1; else out.push({ kind: lab, start: i, end: i + 1 }); });
    return out;
  }
  // Trap: once the hook ends, everything up to the drop is the build (Love Sosa: hook, then a
  // kick-pinned snare climb, then the sub cuts, then the drop). The build starts on the phrase
  // line where the hook's power falls away.
  function hookBuilds(secs, hooks, power, grid) {
    if (!hooks.length) return secs;
    const labels = [];
    for (const s of secs) for (let i = s.start; i < s.end; i++) labels.push(s.kind);
    const n = power.length;
    let changed = false;
    for (const [ha, hb] of hooks) {
      const drop = secs.find(s => s.kind === DROP && s.start >= hb);
      if (!drop || labels.slice(hb, drop.start).includes(DROP)) continue;
      const hookPw = mean(power, ha, hb);
      // first phrase line after which no bar is as loud as the hook any more
      const lines = [];
      for (let c = ha + 2; c < Math.min(hb + 3, drop.start); c++) {
        if (mod(c - grid, PHRASE) !== 0) continue;
        let mx = -Infinity; for (let j = c; j < Math.min(n, c + 4); j++) mx = Math.max(mx, power[j]);
        if (mx < hookPw - 0.05) lines.push(c);
      }
      if (!lines.length || drop.start - lines[0] > 28) continue;
      const start = lines[0];
      let loud = false;  // something as big as the hook in between: not one long build
      for (let c = start; c < drop.start - 3; c++) if (mean(power, c, c + 4) >= hookPw - 0.1) loud = true;
      if (loud) continue;
      for (let i = start; i < drop.start; i++) labels[i] = BUILD;
      changed = true;
    }
    return changed ? fromLabels(labels) : secs;
  }
  // A boundary one bar off the drops' phrase grid moves onto it (a sub that fades a bar early,
  // a 1-bar pause before the groove). Drop starts stay put.
  function snapToGrid(secs, grid) {
    const out = secs.map(s => ({ ...s }));
    for (let i = 1; i < out.length; i++) {
      const x = out[i].start, r = mod(x - grid, PHRASE);
      if (out[i].kind === DROP || (r !== 1 && r !== PHRASE - 1)) continue;
      const y = r === PHRASE - 1 ? x + 1 : x - 1;
      if (out[i - 1].start < y && y < out[i].end) { out[i - 1].end = y; out[i].start = y; }
    }
    return out;
  }

  // Drops that hit with the kick and the growl instead of the sub (trap / bass house:
  // Chicken Soup's groove has more sub than its drops). An 8+ bar stretch with the kick
  // pinned and much brighter than the groove, not already a drop, becomes one; its build
  // runs back to the short gap that ended the groove before it.
  function brightDrops(secs, M) {
    const g = secs.find(s => s.kind === GROOVE);
    if (!g) return secs;
    const gt = mean(M.tension, g.start, g.end), gl = mean(M.low, g.start, g.end), n = M.kick.length;
    const flags = Array.from(M.kick, (k, i) => k >= 1.4 && M.tension[i] >= gt + 0.08);
    const thr = percentile(M.low, 90) - 0.15;
    const labels = [];
    for (const s of secs) for (let i = s.start; i < s.end; i++) labels.push(s.kind);
    // bridge gaps of <= 2 bars (a wobble that lets go for a bar mid-drop)
    const stretches = [];
    flags.forEach((f, i) => {
      if (!f) return;
      const l = stretches[stretches.length - 1];
      if (l && i - l[1] <= 2) l[1] = i + 1; else stretches.push([i, i + 1]);
    });
    let changed = false;
    for (let [a, b] of stretches) {
      // a drop starts solid: 3 slammed bars in a row (kick hits inside a snare roll don't count)
      while (a + 3 <= b && !(flags[a] && flags[a + 1] && flags[a + 2])) a++;
      // a 1-2 bar dip at the very end that isn't anything else yet belongs to the drop
      let k = 0;
      while (b + k < n && k < 2 && labels[b + k] === BREAKDOWN) k++;
      if (k && (b + k >= n || labels[b + k] !== BREAKDOWN)) b += k;
      // too short, still the intro, or the run-up to a drop we already have
      if (b - a < 8 || a < g.end || labels.slice(a, Math.min(n, b + 2)).includes(DROP)) continue;
      if (mean(M.low, a, b) >= gl) continue;  // as much sub as the groove: the sub rule already had its say
      // still climbing (snare roll speeding up): that's a kick build, not a drop
      if (slope(Array.from({ length: b - a }, (_, j) => M.tension[a + j] + 0.5 * M.density[a + j])) > 0.015) continue;
      // the pause at the peak of the build: the bar where the sub JUST cut out after a heavy
      // build bar. The drop hits right after it. If the sub was already out the bar before,
      // that was the pause and this bar is the drop (some drops open on a sub-less bar).
      if (M.low[a] < thr && a > 0 && M.low[a - 1] >= thr && a + 1 < b && M.low[a + 1] >= thr) a++;
      // build: back to the last energy gap (a 1-bar dip) in the 12 bars before the drop
      let lastGap = -1;
      for (let i = Math.max(0, a - 12); i < a - 2; i++) if (M.energy[i] < 0.5) lastGap = i;
      let start = lastGap >= 0 ? lastGap + 1 : Math.max(0, a - 8);
      let afterDrop = 0; for (let i = 0; i < start; i++) if (labels[i] === DROP) afterDrop = i + 1;
      start = Math.max(start, afterDrop);
      // whatever old build/drop guess the new build/drop cuts through becomes groove
      for (const s of secs) {
        if ((s.kind === BUILD || s.kind === DROP) && s.start < b && s.end > start) for (let i = s.start; i < s.end; i++) labels[i] = GROOVE;
      }
      for (let i = start; i < a; i++) labels[i] = BUILD;
      for (let i = a; i < b; i++) labels[i] = DROP;
      changed = true;
    }
    return changed ? fromLabels(labels) : secs;
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
    const tl = new Timeline(duration, bpm, beats, phase, secs, fr / ENV_DECIM, curves,
      { kick: on.kick.map(f => frameTime(f, sr)), snare: on.snare.map(f => frameTime(f, sr)), hat: on.hat.map(f => frameTime(f, sr)) });
    // what the section labeler saw, bar by bar (for the "copy analysis" report)
    tl.bars = { start: barStarts.map(b => beats[b]), ...M };
    return tl;
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

  function median(a) { const s = Float64Array.from(a).sort(), n = s.length; return n % 2 ? s[(n - 1) >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2; }

  // ---------------------------------------------------------------- live section machine (twin of analysis/live_sections.py)
  const LS_FULL_WITHIN = 0.15, LS_EMPTY_BELOW = 0.42, LS_FLOOR = 0.35, LS_TENSION_UP = 0.10, LS_HOLD = 2;
  const meanOf = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
  class LiveSections {
    constructor() {
      this.kind = INTRO; this.run = 0; this.rows = []; this.kinds = []; this.full = []; this.power = [];
      this.lowRef = 0; this.pwRef = 0; this.gTe = null; this.gN = 0; this.gPw = null;
      this.emptyRun = 0; this.fullRun = 0; this.dipRun = 0; this.buildCut = 0; this.curCut = 0;
      this.dropRef = null; this.dropN = 0; this.sagRun = 0; this.slackRun = 0; this.sinceDrop = 99; this.kTeMin = 9;
    }
    dips4() { let n = 0; for (const f of this.full.slice(-4)) if (!f) n++; return n; }
    teUp(te) { return this.gN >= 8 && this.gTe !== null && te >= this.gTe + LS_TENSION_UP; }
    enter(kind) {
      if (kind === this.kind) return;
      this.kind = kind;
      if (kind === BREAKDOWN) this.kTeMin = Math.min(...this.rows.slice(-3).map(r => r.tension));
      if (kind === BUILD) this.buildCut = this.curCut;
      if (kind === DROP) { this.dropRef = [0, 0, 0]; this.dropN = 0; this.sagRun = 0; this.slackRun = 0; }
    }
    // feed the features of the bar that just ended; slam = the frame-level detector heard the sub slam in at its start
    push(row, slam = false) {
      const low = row.low, en = row.energy, hi = row.high, te = row.tension, kick = row.kick;
      const p = en + 0.5 * low + 0.5 * hi, i = this.rows.length;
      const prevLow = i ? this.rows[i - 1].low : low, prevP = i ? this.power[i - 1] : p;
      this.lowRef = Math.max(Math.min(low, prevLow), this.lowRef - 0.001);
      this.pwRef = Math.max(Math.min(p, prevP), this.pwRef - 0.003);
      const full = low >= Math.max(this.lowRef - LS_FULL_WITHIN, LS_FLOOR);
      const empty = this.lowRef >= 0.5 && low < this.lowRef - LS_EMPTY_BELOW;
      const erPrev = this.emptyRun, dips4 = this.dips4(), j = i - 1 - erPrev;
      const buildLike = j >= 0 && (this.kinds[j] === BUILD || !this.full[j]), heavyBefore = j >= 0 && this.full[j];
      this.emptyRun = empty ? this.emptyRun + 1 : 0; this.curCut = this.emptyRun;
      this.fullRun = full ? this.fullRun + 1 : 0; this.dipRun = full ? 0 : this.dipRun + 1;
      if (this.kind === BUILD) this.buildCut = Math.max(this.buildCut, this.emptyRun);
      this.rows.push(row); this.power.push(p); this.full.push(full);
      const was = this.kind;
      if (this.kind !== DROP && i >= 1) {
        const cut = erPrev >= 1 && (buildLike || erPrev >= 3);
        let hit = false;
        if (cut && full && this.gTe !== null && te >= this.gTe - 0.1) hit = true;
        else if (cut && low >= this.lowRef - 0.3 && kick >= 1.2 && p >= 0.82 * this.pwRef) hit = true;
        else if (erPrev >= 1 && erPrev <= 2 && buildLike && heavyBefore && kick >= 1.4 && this.gTe !== null && te >= this.gTe + 0.08) hit = true;
        else if (this.kind === BUILD && this.buildCut >= 3 && full && !this.full[i - 1] && p >= 0.9 * this.pwRef) hit = true;
        else if ((this.kind === BUILD || this.kind === BREAKDOWN) && this.run >= 2 && full && low >= prevLow + 0.15
                 && p >= meanOf(this.power.slice(-3, -1)) + 0.3) hit = true;
        else if (slam && (this.kind === BUILD || this.kind === BREAKDOWN || dips4 >= 2 || erPrev >= 1) && low >= this.lowRef - 0.3
                 && p >= 0.75 * this.pwRef) hit = true;
        if (hit) this.enter(DROP);
      }
      this.kTeMin = Math.min(this.kTeMin, te);
      if ((this.kind === INTRO || this.kind === GROOVE) && this.gPw !== null && p >= this.gPw + 0.35) { this.gTe = null; this.gPw = null; this.gN = 0; }
      if (this.kind === DROP) this.inDrop(row); else this.others();
      if (full && (this.kind === INTRO || this.kind === GROOVE)) {
        this.gTe = this.gTe === null ? te : 0.75 * this.gTe + 0.25 * te;
        this.gPw = this.gPw === null ? p : 0.75 * this.gPw + 0.25 * p;
        this.gN++;
      }
      this.run = this.kind === was ? this.run + 1 : 1;
      this.sinceDrop = this.kind === DROP ? 0 : this.sinceDrop + 1;
      this.kinds.push(this.kind);
      return this.kind;
    }
    inDrop(row) {
      if (this.dropN < 4) { this.dropRef[0] += row.low; this.dropRef[1] += row.energy; this.dropRef[2] += row.tension; this.dropN++; }
      const n = this.dropN, dEn = this.dropRef[1] / n, dTe = this.dropRef[2] / n;
      const sag = row.energy <= dEn - 0.15;
      this.sagRun = sag && this.emptyRun === 0 ? this.sagRun + 1 : 0;
      const slack = row.tension <= dTe - 0.2 && this.emptyRun === 0;
      this.slackRun = slack ? this.slackRun + 1 : 0;
      if (!(this.kinds.length && this.kinds[this.kinds.length - 1] === DROP && this.run >= LS_HOLD)) return;
      if (this.emptyRun >= 3) this.enter(this.teUp(meanOf(this.rows.slice(-3).map(r => r.tension))) ? BUILD : BREAKDOWN);
      else if (this.sagRun >= 2 || this.slackRun >= 2) this.enter(GROOVE);
    }
    others() {
      if (this.run < LS_HOLD && this.kind !== INTRO) return;
      const k = this.kind, settling = this.sinceDrop < 8;
      if (k === INTRO) { if (this.fullRun >= 3) this.enter(GROOVE); }
      else if (k === GROOVE) {
        const n = this.rows.length, tes2 = meanOf(this.rows.slice(-2).map(r => r.tension)), g = this.gTe;
        if (this.emptyRun === 1 && n >= 2 && !this.full[n - 2] && !settling && g !== null && tes2 >= g - 0.05) this.enter(BUILD);
        else if (this.emptyRun >= 3 || (this.emptyRun === 2 && this.teUp(tes2))) this.enter(this.teUp(meanOf(this.rows.slice(-3).map(r => r.tension))) ? BUILD : BREAKDOWN);
        else if (this.dipRun >= 2 && this.teUp(tes2) && !settling) this.enter(BUILD);
        else if (this.dipRun >= 2 && g !== null && tes2 <= g - 0.15) this.enter(BREAKDOWN);
        else if (n >= 8 && this.gN >= 8 && this.fullRun >= 2 && g !== null && tes2 >= g + 0.15 && !settling
                 && tes2 >= meanOf(this.rows.slice(-8, -4).map(r => r.tension)) + 0.1) this.enter(BUILD);
      } else if (k === BREAKDOWN) {
        const tes2 = meanOf(this.rows.slice(-2).map(r => r.tension));
        if (this.teUp(tes2) || tes2 >= this.kTeMin + 0.2) this.enter(BUILD);
        else if (this.fullRun >= 3) this.enter(GROOVE);
      } else if (k === BUILD) {
        const te4 = meanOf(this.rows.slice(-4).map(r => r.tension)), quiet = this.gTe !== null && te4 < this.gTe + 0.06;
        if ((this.run >= 16 && this.fullRun >= 8) || (this.run >= 8 && this.fullRun >= 8 && quiet)) this.enter(GROOVE);
      }
    }
  }
  // push a track's per-bar features through LiveSections one bar at a time -> [kind, start, end] sections
  function simulateLiveBars(M) {
    const n = M.low.length, ls = new LiveSections(), kinds = [];
    for (let i = 0; i < n; i++) kinds.push(ls.push({ kick: M.kick[i], low: M.low[i], energy: M.energy[i], high: M.high[i], tension: M.tension[i], density: M.density[i] }));
    const out = [];
    kinds.forEach((k, i) => { if (out.length && out[out.length - 1].kind === k) out[out.length - 1].end = i + 1; else out.push({ kind: k, start: i, end: i + 1 }); });
    return out;
  }

  // fixed-size window of the most recent values (order doesn't matter: only percentiles are taken from it)
  class Ring {
    constructor(n) { this.n = n; this.a = new Float64Array(n); this.i = 0; this.count = 0; }
    push(x) { this.a[this.i] = x; this.i = (this.i + 1) % this.n; if (this.count < this.n) this.count++; }
    view() { return this.a.subarray(0, this.count); }
  }
  // twin of live.BarFeatures: the six per-bar features of sections.bar_features(), normalised causally
  // (5th..98th percentile of the last ~3 minutes of frames, the bar itself included)
  const BF_KEYS = ["low", "energy", "high", "bright", "noise", "kflux"];
  class BarFeatures {
    constructor(frameRate, windowS = 180) {
      const n = Math.floor(windowS * frameRate);
      this.win = {}; for (const k of BF_KEYS) this.win[k] = new Ring(n);
      this.chunks = []; this.cur = BarFeatures.fresh(); this.hitHist = [];
    }
    static fresh() { return { low: [], energy: [], high: [], bright: [], noise: [], kmax: 0, hits: 0 }; }
    add(F, i, hits) {
      const c = this.cur;
      const vals = { low: F.kick_e[i], energy: 20 * Math.log10(F.rms[i] + 1e-6), high: F.hat_e[i] * 0.6 + F.snare_e[i] * 0.4,
        bright: Math.log(F.centroid[i] + 1), noise: F.flatness[i], kflux: F.kick_flux[i] };
      for (const k of BF_KEYS) this.win[k].push(vals[k]);
      for (const k of ["low", "energy", "high", "bright", "noise"]) c[k].push(vals[k]);
      c.kmax = Math.max(c.kmax, vals.kflux); c.hits += hits;
    }
    endBeat() { this.chunks.push(this.cur); if (this.chunks.length > 4) this.chunks.shift(); this.cur = BarFeatures.fresh(); }
    ready() { return this.chunks.length === 4; }
    row() {
      const ch = this.chunks, p95 = percentile(this.win.kflux.view(), 95) + 1e-9;
      const beatKick = ch.map(c => Math.min(Math.max(c.kmax / p95, 0), 1.5)).sort((a, b) => a - b);
      const norm = key => {
        const w = this.win[key].view(), lo = percentile(w, 5), hi = percentile(w, 98);
        let t = 0, n = 0; for (const c of ch) for (const v of c[key]) { t += clamp((v - lo) / (hi - lo), 0, 1); n++; }
        return hi - lo < 1e-9 || n === 0 ? 0 : t / n;
      };
      const high = norm("high"), bright = norm("bright"), noise = norm("noise");
      let hits = 0; for (const c of ch) hits += c.hits;
      this.hitHist.push(hits);
      const dens = hits / Math.max(1, percentile(this.hitHist, 95));
      return { kick: beatKick[2], low: norm("low"), energy: norm("energy"), high, tension: (high + bright + noise) / 3, density: dens };
    }
  }

  class LiveAnalyzer {
    constructor(sr, range = [90, 190], latency = 0.03) {
      const fr = sr / HOP; this.fr = fr; this.range = range; this.latency = latency;
      this.envBuf = []; this.envMax = Math.floor(16 * fr); this.phaseBuf = []; this.phaseMax = Math.floor(6 * fr);
      this.everLocked = false; this.phaseGood = false; this.flipVotes = 0;
      this.nKick = new PeakNorm(fr * 30); this.nSnare = new PeakNorm(fr * 30);
      this.kickDet = new OnsetDetector(fr, 0.15); this.snareDet = new OnsetDetector(fr, 0.06); this.hatDet = new OnsetDetector(fr, 0.05);
      this.gE = new AutoGain(fr * 20); this.gL = new AutoGain(fr * 20); this.gM = new AutoGain(fr * 20); this.gH = new AutoGain(fr * 20);
      this.gKick = new PeakNorm(60); this.gTension = new AutoGain(fr * 30); this.gEnv = new AutoGain(fr * 10);
      this.frame = 0; this.bpm = 128; this.beatPos = 0; this.locked = false; this.bpmVotes = 0; this.pendingBpm = null;
      this.downbeat = 0;
      this.acc = this.newAcc(); this.hist = [];
      this.section = GROOVE; this.sectionIndex = 0; this.sectionStart = 0; this.forced = null;
      this.preDrop = null;
      this.lowRef = null; this.peakRef = null; this.cooldownUntil = -1; this.kickBeat = 0; this.prevKickE = -9;
      this.full = true; this.emptyBeats = 0; this.freeze = false; this.freezeMin = 0;
      this.bars = new BarFeatures(fr); this.sections = new LiveSections(); this.barTrace = [];
      this.fbufMax = Math.floor(16 * fr); this.fbuf = []; this.beatFrames = []; this.dbScore = [0, 0, 0, 0]; this.dbHoldUntil = -1;
      this.st = newState(); this.st.locked = false; this.kEnv = 0; this.sEnv = 0; this.hEnv = 0;
      this.dec = { kick: Math.exp(-1 / (0.12 * fr)), snare: Math.exp(-1 / (0.09 * fr)), hat: Math.exp(-1 / (0.05 * fr)) };
    }
    forceSection(kind) { this.forced = kind; if (kind) this.setSection(kind); }
    tapDownbeat() {
      const b = roundHalfEven(this.beatPos); this.downbeat = mod(b, 4);
      this.beatPos = b + (this.beatPos - b); this.dbHoldUntil = this.beatPos + 64;
    }
    nudgeHalfBeat() { this.beatPos += 0.5; this.flipVotes = 0; }
    newAcc() { return { kick: 0, kickOn: 0, snareOn: 0, tension: [], energy: [], low: [], snares: 0 }; }
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
      if (this.frame % 16 === 0 && this.envBuf.length > fr * 6) this.updateTempo();
      let prev = this.beatPos; this.beatPos += this.bpm / 60 / fr;

      // THE DROP: the sub level itself crossing up (onset detectors get deafened by risers and snare rolls)
      const frac = this.beatPos - Math.floor(this.beatPos);
      const prevLow = this.prevKickE; this.prevKickE = kE;
      const h4 = this.hist.slice(-4);
      const dipping = this.section === GROOVE && (this.sections.dipRun >= 2 || this.sections.emptyRun >= 1);
      const floorOut = this.emptyBeats >= 2 && this.section !== DROP;
      if (!this.forced && (this.section === BUILD || this.section === BREAKDOWN || dipping || floorOut) && this.preDrop === null
          && this.peakRef !== null && this.hist.length >= 16 && this.beatPos > this.cooldownUntil
          && (this.section === BUILD || (this.beatPos - this.sectionStart >= 8 && mean(h4.map(x => x.tension)) >= 0.6))) {
        const hist = this.hist, recent = mean(hist.slice(-2).map(x => x.peak));
        const unheard = Math.max(...hist.slice(-16).map(x => x.peak)) + 0.25;
        const level = Math.max(this.peakRef - 0.3, recent + 0.35, unheard);
        const afterGap = this.section === BUILD && this.emptyBeats >= 2 && this.sections.run >= 2 && kE > this.peakRef - 0.3 && prevLow < this.peakRef - 0.6;
        if ((kE > level && level >= prevLow && (this.emptyBeats >= 1 || recent < this.peakRef - 0.35)) || afterGap) {
          this.kickBeat = this.bAt(-0.5 / fr);
          const b = roundHalfEven(this.kickBeat);
          this.preDrop = [this.section, this.sectionStart, this.downbeat, b];
          this.beatPos += b - this.kickBeat; prev = this.beatPos - 1e-6;
          this.downbeat = mod(b, 4); this.dbScore[mod(b, 4)] += 3.0;
          this.setSection(DROP); this.kEnv = 1;
        }
      }

      const a = this.acc, en = 20 * Math.log10(F.rms[i] + 1e-6);
      a.kick = Math.max(a.kick, kf);
      if (frac < 0.15) { a.kickOn = Math.max(a.kickOn, kf); a.snareOn = Math.max(a.snareOn, sf); }
      a.energy.push(en); a.low.push(kE);
      a.tension.push(F.hat_e[i] + 2 * Math.log(F.centroid[i] + 1) + 3 * F.flatness[i]); if (s > 0) a.snares++;
      this.bars.add(F, i, (s > 0 ? 1 : 0) + (h > 0 ? 1 : 0));
      this.fbuf.push(i); if (this.fbuf.length > this.fbufMax) this.fbuf.shift();
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
      if (this.locked && (!(this.kickRunning() || this.full) || this.freeze)) return;
      const cand = estimateTempo(env, this.fr, this.range, this.everLocked ? this.bpm : 128);
      if (!this.everLocked) this.bpm = cand;
      else if (Math.abs(cand / this.bpm - 1) >= 0.03 && (this.section === DROP || this.frame < 45 * this.fr || !ratioRelated(cand, this.bpm))) {
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
      const a = this.acc, ended = Math.floor(this.beatPos) - 1;
      // downbeat: every bar, run the offline "1" finder over the last ~16 s of frames; change only on 2 votes
      this.beatFrames.push([this.frame, ended + 1]); if (this.beatFrames.length > 48) this.beatFrames.shift();
      if (mod(ended + 1, 4) === 0 && this.beatFrames.length >= 17 && this.beatPos > this.dbHoldUntil) {
        const first = this.frame - this.fbuf.length + 1;
        const bf = this.beatFrames.filter(([fn]) => fn - first >= 0).map(([fn, bn]) => [fn - first, bn]);
        if (bf.length >= 13) {
          const i0 = this.fbuf[0], len = this.fbuf.length;
          const sc = downbeatScores(F.coarse.subarray(i0 * N_COARSE, (i0 + len) * N_COARSE), F.kick_flux.subarray(i0, i0 + len), F.snare_flux.subarray(i0, i0 + len), len, bf.map(x => x[0]));
          this.dbScore = this.dbScore.map(v => v * 0.85);
          for (let ph = 0; ph < 4; ph++) this.dbScore[mod(bf[ph][1], 4)] += sc[ph];
          const best = argmax(this.dbScore);
          if (best !== this.downbeat && this.preDrop === null && this.dbScore[best] > this.dbScore[this.downbeat] + 1.0) this.downbeat = best;
        }
      }
      const kick = this.gKick.g(a.kick), energy = a.energy.length ? mean(a.energy) : -60;
      const tension = a.tension.length ? this.gTension.g(mean(a.tension)) : 0;
      const low = a.low.length ? median(a.low) : -9, peak = a.low.length ? Math.max(...a.low) : -9;
      this.hist.push({ kick, energy, tension, snares: a.snares, low, peak }); if (this.hist.length > 64) this.hist.shift();
      this.peakRef = this.peakRef === null ? peak : Math.max(peak, this.peakRef - 0.0015);
      this.acc = this.newAcc();
      // floor level: running drop-level reference + full/empty with hysteresis
      this.lowRef = this.lowRef === null ? low : Math.max(low, this.lowRef - 0.0015);
      if (low > this.lowRef - 0.3) this.full = true; else if (low < this.lowRef - 0.8) this.full = false;
      this.emptyBeats = this.full ? 0 : this.emptyBeats + 1;
      if (this.emptyBeats >= 2) { if (!this.freeze) { this.freeze = true; this.freezeMin = this.beatPos + 6; } }
      else if (this.freeze && this.full && this.beatPos >= this.freezeMin) this.freeze = false;
      this.bars.endBeat();
      if (mod(Math.floor(this.beatPos) - this.downbeat, 4) === 0 && this.bars.ready()) this.onBar(this.bars.row());
    }
    // a bar just ended: hand its causal features to the section machine. The machine decides the section; a
    // provisional drop (the frame-level trigger that fired mid-bar on the first sub hit) is confirmed if the machine
    // agrees, taken back if it doesn't.
    onBar(row) {
      const kind = this.sections.push(row, this.preDrop !== null);
      this.barTrace.push(Object.assign({}, row, { section: kind, locked: !!this.locked, t: this.frame / this.fr }));
      if (this.forced) { this.preDrop = null; return; }
      if (this.preDrop !== null) {
        if (kind !== DROP) {
          const [p, s, o] = this.preDrop; this.section = p; this.sectionStart = s; this.downbeat = o; this.sectionIndex++;
          this.cooldownUntil = this.beatPos + 16;
        }
        this.preDrop = null;
      }
      if (kind !== this.section) this.setSection(kind);
    }
    setSection(k) { if (k !== this.section) { this.section = k; this.sectionIndex++; this.sectionStart = roundHalfEven(this.beatPos); } }
    publish() {
      const st = this.st, beat = this.bAt(this.latency), rel = beat - this.downbeat;
      st.t = this.frame / this.fr; st.bpm = this.bpm; st.beat = beat; st.beat_in_bar = mod(rel, 4); st.bar = Math.floor(rel / 4);
      st.section = this.section; st.section_index = this.sectionIndex; st.section_beats = beat - this.sectionStart;
      if (this.section === BUILD) { const tn = this.hist.length ? this.hist[this.hist.length - 1].tension : 0; st.section_progress = Math.min(0.98, 0.5 * st.section_beats / 32 + 0.5 * tn); }
      else st.section_progress = Math.min(1, st.section_beats / 64);
      st.next_section = null; st.beats_to_next = null; st.locked = this.locked;
      const frac = beat - Math.floor(beat), kickOn = this.full && (this.section === GROOVE || this.section === DROP || this.section === INTRO);
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
  // Drop styles: first beat of every drop is the white impact in all of them. "auto" rotates a genre
  // subset every 8 bars (32 beats); ctl.dropStyle locks one.
  const DROP_STYLES = ["pump", "chase", "split", "strobe_hell", "fan_blast", "rainbow_chase", "blackout_stabs", "sweep", "half_time_heavy", "crowd_sweep"];
  const DROP_ROTATION = {
    auto: ["pump", "chase", "split", "fan_blast", "crowd_sweep", "rainbow_chase", "sweep"],
    house: ["pump", "crowd_sweep", "rainbow_chase", "sweep", "chase"],
    progressive: ["sweep", "crowd_sweep", "rainbow_chase", "pump", "chase"],
    techno: ["pump", "crowd_sweep", "strobe_hell", "chase", "blackout_stabs"],
    trance: ["sweep", "crowd_sweep", "rainbow_chase", "pump", "fan_blast"],
    dubstep: ["half_time_heavy", "strobe_hell", "crowd_sweep", "blackout_stabs", "fan_blast"],
    trap: ["half_time_heavy", "blackout_stabs", "crowd_sweep", "strobe_hell", "fan_blast"],
    dnb: ["fan_blast", "chase", "strobe_hell", "pump"],
  };
  // Audience blinders are warm white whatever the engine says. Mover aiming: move(f, o, dpan, dtilt) -- dtilt > 0 aims UP
  // at the ceiling (builds), dtilt < 0 aims at the AUDIENCE (-1 = straight at the crowd); same convention as show.py.
  const BLINDER_COLOR = [1.0, 0.85, 0.6];
  function demoRig() {
    const fx = [];
    for (let i = 0; i < 8; i++) fx.push({ name: "Par " + (i + 1), kind: "wash", group: "front", y: 0, rgb: true });
    for (let i = 0; i < 4; i++) fx.push({ name: "Mover " + (i + 1), kind: "mover", group: "movers", x: 0.2 + i * 0.2, y: 0, pan_span: 0.12, tilt_span: 0.18, rgb: true });
    fx.push({ name: "Strobe L", kind: "strobe", group: "strobe", x: 0.1, y: 0 }, { name: "Strobe R", kind: "strobe", group: "strobe", x: 0.9, y: 0 });
    for (let i = 0; i < 2; i++) fx.push({ name: "Blinder " + (i + 1), kind: "blinder", group: "blinders", x: i ? 0.7 : 0.3, y: 0 });
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
      this.fx = fixtures; this.genre = GENRES[genre] ? genre : "auto"; const g = GENRES[this.genre]; this.style = "pump"; this.hzCap = 10;
      [, this.dropPal, this.calmPal, this.strobeAmt, this.snareAmt] = g;
      let s = seed >>> 0; this.rand = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      this.secIdx = -1; this.palette = [hsv(0), hsv(240)]; this.smooth = fixtures.map(() => ({ pan: 0.5, tilt: 0.5 })); this.first = true;
    }
    pick(kind) {
      const name = kind === DROP ? this.dropPal : kind === BREAKDOWN ? this.calmPal : "mixed", list = PALETTES[name];
      const hues = list[Math.floor(this.rand() * list.length)];
      this.palette = hues.map(h => kind === BREAKDOWN ? hsv(h, 0.9, 0.9) : hsv(h));
    }
    dropStyleAt(sectionBeats, setting = "auto") {
      if (DROP_STYLES.includes(setting)) return setting;
      const rot = DROP_ROTATION[this.genre]; return rot[Math.floor(sectionBeats / 32) % rot.length];
    }
    render(st, ctl) {
      this.hzCap = ctl.strobeEnabled === false ? 2 : ctl.maxStrobe;   // wash flashing obeys the strobe safety too
      if (st.section === DROP) this.style = this.dropStyleAt(st.section_beats, ctl.dropStyle);
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
      if (st.next_section === DROP && st.beats_to_next !== null && f.kind === "blinder" && st.beats_to_next <= 4) { o.i = 0.5 * (4 - st.beats_to_next) / 3; return; }   // last bar swell
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
      const sb = st.section_beats;
      if (sb < 1) { o.c = WHITE; o.i = f.kind !== "strobe" && f.kind !== "blinder" ? 1 - 0.5 * sb : 1 - sb; this.move(f, o, (f.x - 0.5) * 0.8, -1); return; }   // IMPACT, every style
      this["d_" + this.style](st, f, o);
    }
    // ---- drop styles (mirror show.py exactly). Each handles every fixture kind so any rig subset looks good.
    flashDiv(st, want) { let d = want; while (d > 1 && st.bpm / 60 * d > this.hzCap) d = Math.floor(d / 2); return d; }
    d_pump(st, f, o) {
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick;
      if (f.kind === "wash") { o.c = mod(bi + f.index, 2) === 0 ? A : B; o.i = Math.max(0.15 + 0.85 * k, st.snare * this.snareAmt * 0.9); }
      else if (f.kind === "mover") { o.c = f.index % 2 ? B : A; o.i = 0.5 + 0.5 * k; this.move(f, o, (f.index % 2 ? -1 : 1) * Math.sin(2 * Math.PI * st.beat / 4), -0.3 - 0.6 * k); }
      else if (f.kind === "strobe") { o.i = this.strobeAmt * Math.max(st.snare ** 2 * this.snareAmt, k ** 4 * 0.5); o.c = WHITE; }
    }
    d_chase(st, f, o) {
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick;
      if (f.kind === "wash") {
        const n = Math.max(1, f.group_size), step = mod(Math.floor(st.beat * 2), n > 2 ? 2 * n - 2 : n), pos = step < n ? step : 2 * n - 2 - step;
        o.c = mod(Math.floor(bi / 2), 2) === 0 ? A : B; o.i = Math.max(pos === f.index ? 1 : 0.1, 0.5 * k, st.snare * this.snareAmt * 0.9);
      } else if (f.kind === "mover") { o.c = f.index % 2 ? B : A; o.i = 0.5 + 0.5 * k; const a = 2 * Math.PI * (st.beat / 4 + f.index * 0.25); this.move(f, o, Math.sin(a), Math.cos(a) * 0.8); }
      else if (f.kind === "strobe") { o.i = st.beat_in_bar < 0.15 ? this.strobeAmt : 0; o.c = WHITE; }
    }
    d_split(st, f, o) {
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick;
      if (f.kind === "wash") { const on = (f.x < 0.5) === (mod(bi, 2) === 0); o.c = on ? A : B; o.i = Math.max(on ? 0.3 + 0.7 * k : 0.15 * k, st.snare * this.snareAmt * 0.9); }
      else if (f.kind === "mover") { o.c = f.index % 2 ? B : A; o.i = 0.5 + 0.5 * k; this.move(f, o, (mod(bi + f.index, 2) ? 1 : -1) * 0.9, 0.2); }
      else if (f.kind === "strobe") { o.i = this.strobeAmt * Math.max(st.snare ** 2 * this.snareAmt, k ** 4 * 0.5); o.c = WHITE; }
    }
    d_strobe_hell(st, f, o) {   // washes + strobes flash on every 8th (16ths on the last bar of 4), capped by the strobe-rate limit
      const A = this.palette[0], B = this.palette[this.palette.length - 1], k = st.kick;
      const div = this.flashDiv(st, mod(st.bar, 4) === 3 ? 4 : 2), sub = st.beat * div, n = Math.floor(sub), on = mod(sub, 1) < 0.5;
      if (f.kind === "wash") { o.c = n % 2 === 0 ? A : B; o.i = on ? 1 : 0.04 + 0.2 * k; }
      else if (f.kind === "mover") {
        o.c = on && n % 4 === 0 ? WHITE : (f.index % 2 ? B : A); o.i = on ? 1 : 0.2;
        this.move(f, o, ((n + f.index) % 2 ? 1 : -1) * 0.7, 0.3);
      } else if (f.kind === "strobe") { o.c = WHITE; o.i = Math.max(this.strobeAmt, 0.8) * (0.6 + 0.4 * k); o.hz = st.bpm / 60 * div; }
    }
    d_fan_blast(st, f, o) {     // movers fan out from centre on each kick then snap back; washes burst outward
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick, c = bi % 2 === 0 ? A : B;
      if (f.kind === "wash") {
        const dist = Math.abs(f.x - 0.5) * 2, fan = Math.min(1, Math.max(0, k * 1.6 - dist * 0.9 + 0.25));
        o.c = mix(c, WHITE, 0.6 * st.snare); o.i = Math.max(0.08, fan, st.snare * this.snareAmt * 0.9);
      } else if (f.kind === "mover") {
        o.c = f.index % 2 === 0 ? c : (bi % 2 === 0 ? B : A); o.i = 0.4 + 0.6 * k;
        this.move(f, o, (f.x - 0.5) * 2 * k * 1.1, -0.2 + 0.6 * k);
      } else if (f.kind === "strobe") { o.c = WHITE; o.i = this.strobeAmt * Math.max(st.snare ** 2 * this.snareAmt, k ** 6 * 0.6); }
    }
    d_rainbow_chase(st, f, o) { // hue rolls across the rig by x while a comet chases on 1/2 beats
      const k = st.kick, hue = f.x * 300 + st.beat * 40;
      if (f.kind === "wash") {
        const n = Math.max(1, f.group_size), head = mod(Math.floor(st.beat * 2), n), d = mod(head - f.index, n), tail = Math.max(0, 1 - d / Math.max(2, n / 2));
        o.c = hsv(hue); o.i = Math.max(0.12 + 0.88 * tail * (0.65 + 0.35 * k), st.snare * this.snareAmt * 0.8);
      } else if (f.kind === "mover") {
        o.c = hsv(hue + 120); o.i = 0.6 + 0.4 * k; const ph = f.x * 0.5;
        this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 8 + ph)) * 0.8, 0.1 + 0.4 * Math.sin(2 * Math.PI * (st.beat / 4 + ph)));
      } else if (f.kind === "strobe") { o.c = WHITE; o.i = this.strobeAmt * (st.beat_in_bar < 0.15 ? 1 : 0.5 * st.snare ** 2); }
    }
    d_blackout_stabs(st, f, o) { // dark except hard cut-off stabs on every kick + snare
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick, sn = st.snare;
      let v = Math.max(k, sn); v = v > 0.4 ? v : 0;
      if (f.kind === "wash") { o.c = sn > k ? mix(B, WHITE, 0.7) : A; o.i = v; }
      else if (f.kind === "mover") { o.c = sn > k ? WHITE : (f.index % 2 ? B : A); o.i = v; this.move(f, o, (mod(bi + f.index, 2) ? 1 : -1) * 0.9, sn > k ? -0.9 : 0.3); }
      else if (f.kind === "strobe") { o.c = WHITE; o.i = sn > 0.5 ? this.strobeAmt : 0; }
      else if (f.kind === "blinder") o.i = sn > 0.6 && sn > k ? 0.8 * sn : 0;
    }
    d_sweep(st, f, o) {         // smooth + slow: wide mover arcs, washes crossfade colour in a travelling wave
      const A = this.palette[0], B = this.palette[this.palette.length - 1], k = st.kick;
      if (f.kind === "wash") { o.c = mix(A, B, wave(f.x * 0.6 + st.beat / 16)); o.i = Math.max(0.4 + 0.35 * wave(st.beat / 8 - f.x * 0.5) + 0.25 * k, st.snare * 0.5); }
      else if (f.kind === "mover") {
        o.c = mix(B, A, wave(st.beat / 16 + f.index * 0.25)); o.i = 0.6 + 0.3 * k;
        this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 16 + f.index * 0.125)) * 0.9, 0.1 + 0.5 * Math.sin(2 * Math.PI * (st.beat / 32 + f.index * 0.2)));
      } else if (f.kind === "strobe") { o.c = WHITE; o.i = this.strobeAmt * 0.6 * Math.max(st.snare ** 2, k ** 4); }
    }
    d_half_time_heavy(st, f, o) { // kick = solid hit, snare = huge slam + strobes; slow movers snap up on the snare
      const A = this.palette[0], B = this.palette[this.palette.length - 1], k = st.kick, sn = st.snare;
      if (f.kind === "wash") { o.c = sn > k ? mix(B, WHITE, 0.5 * sn) : A; o.i = Math.max(0.12 + 0.5 * k ** 1.5, sn * this.snareAmt); }
      else if (f.kind === "mover") { o.c = f.index % 2 ? B : A; o.i = 0.4 + 0.6 * Math.max(k, sn); this.move(f, o, Math.sin(2 * Math.PI * (st.beat / 16 + f.index * 0.25)) * 0.7, 0.2 - 1.2 * sn); }
      else if (f.kind === "strobe") { o.c = WHITE; o.i = this.strobeAmt * sn ** 2; o.hz = sn > 0.5 ? 14 : 0; }
      else if (f.kind === "blinder") o.i = sn > 0.6 ? 0.8 * sn : 0;
    }
    d_crowd_sweep(st, f, o) {   // movers aimed into the crowd, mirrored sweeps (1 bar/cycle); blinders flash on every downbeat
      const A = this.palette[0], B = this.palette[this.palette.length - 1], bi = Math.floor(st.beat), k = st.kick;
      if (f.kind === "wash") { o.c = mod(bi + f.index, 2) === 0 ? A : B; o.i = Math.max(0.15 + 0.85 * k, st.snare * this.snareAmt * 0.9); }
      else if (f.kind === "mover") { o.c = f.index % 2 ? B : A; o.i = 0.6 + 0.4 * k; this.move(f, o, (f.index % 2 ? -1 : 1) * Math.sin(2 * Math.PI * st.beat / 4) * 0.9, -0.8 - 0.2 * k); }
      else if (f.kind === "strobe") { o.c = WHITE; o.i = this.strobeAmt * st.snare ** 2 * this.snareAmt; }
      else if (f.kind === "blinder") o.i = st.beat_in_bar < 0.15 ? 0.6 * (1 - st.beat_in_bar / 0.15) : 0;
    }
    post(st, ctl, out) {
      let a = st.section === BREAKDOWN ? 0.08 : 0.3;
      if (st.section === DROP && st.section_beats < 1) a = 0.55;   // impact: movers snap round to the crowd
      this.fx.forEach((f, i) => {
        const o = out[i], sm = this.smooth[i];
        if (this.first) { sm.pan = o.pan; sm.tilt = o.tilt; }
        sm.pan += (o.pan - sm.pan) * a; sm.tilt += (o.tilt - sm.tilt) * a;
        o.pan = clamp(sm.pan, 0, 1); o.tilt = clamp(sm.tilt, 0, 1);
        o.hz = Math.min(o.hz, ctl.maxStrobe);
        if (f.kind === "blinder") { o.c = BLINDER_COLOR; o.hz = 0; if (ctl.blinders === false) o.i = 0; }
        o.i = ctl.blackout ? 0 : clamp(o.i, 0, 1) * (ctl.master === undefined ? 1 : ctl.master);
      });
      this.first = false; return out;
    }
  }
  // what a fixture looks like at time t (software/hardware strobe shown as on/off)
  function effective(f, o, t) {
    let i = o.i; if (o.hz > 0 && mod(t * o.hz, 1) > 0.3) i = 0;
    const c = f.kind === "blinder" ? BLINDER_COLOR : f.rgb ? o.c : WHITE; return [i, c[0], c[1], c[2], o.pan, o.tilt];
  }

  // ---------------------------------------------------------------- whole pipeline
  const FPS = 30;
  // Writes one frame of fixture bytes (6 per fixture: i r g b pan tilt) for music state `st`.
  function packFx(rig, outs, frame, t, fxOut) {
    rig.forEach((f, j) => {
      const e = effective(f, outs[j], t), o = (frame * rig.length + j) * 6;
      for (let q = 0; q < 6; q++) fxOut[o + q] = Math.round(clamp(e[q], 0, 1) * 255);
    });
  }
  // Writes one frame of music-state bytes (13 per frame).
  function packSt(st, frame, stOut) {
    const so = frame * 13, bpm10 = Math.round(st.bpm * 10);
    stOut[so] = Math.max(0, SECTIONS.indexOf(st.section)); stOut[so + 1] = Math.floor(st.beat_in_bar * 60) % 240;
    stOut[so + 2] = (bpm10 >> 8) & 255; stOut[so + 3] = bpm10 & 255;
    [st.section_progress, st.kick, st.snare, st.hat, st.low, st.mid, st.high, st.energy].forEach((v, q) => stOut[so + 4 + q] = Math.round(clamp(v, 0, 1) * 255));
    stOut[so + 12] = st.locked ? 1 : 0;
  }
  const STATE_KEYS = ["bpm", "beat", "beat_in_bar", "bar", "section_index", "section_beats", "section_progress", "beats_to_next", "energy", "low", "mid", "high", "kick", "snare", "hat"];
  function snapshot(st) {   // a plain copy of the music state (what the show engine reads)
    const o = { section: st.section, next_section: st.next_section, locked: st.locked };
    for (const k of STATE_KEYS) o[k] = st[k]; return o;
  }
  // Re-renders the show for a sequence of per-frame music states (30 fps). Returns {fx, st} in the same byte layout
  // the page decodes from the demo JSON: fx = frames*fixtures*6 bytes, st = frames*13 bytes.
  function renderShow(states, genre, dropStyle = "auto", opts = {}) {
    const rig = opts.rig || demoRig(), frames = states.length, ctl = { maxStrobe: opts.maxStrobe || 10, dropStyle };
    const out = { fx: new Uint8Array(frames * rig.length * 6), st: new Uint8Array(frames * 13) }, eng = new ShowEngine(rig, genre);
    for (let fi = 0; fi < frames; fi++) { const st = states[fi]; packFx(rig, eng.render(st, ctl), fi, fi / FPS, out.fx); packSt(st, fi, out.st); }
    return out;
  }
  async function analyzeTrack(x, sr, genre, progress = async () => {}) {
    const range = (GENRES[genre] || GENRES.auto)[0];
    const F = await extractFeatures(x, sr, p => progress("Listening", p));
    await progress("Finding the beat and the drops", 0);
    const tl = analyzeOffline(F, range);
    const dur = x.length / sr, frames = Math.floor(dur * FPS);
    const preStates = [];
    for (let fi = 0; fi < frames; fi++) preStates.push(snapshot(tl.stateAt(fi / FPS + 0.03)));
    // live: replay the features frame by frame, snapshot at each video frame
    const liveStates = [], liveSecs = [];
    const la = new LiveAnalyzer(sr, range, 0.03); let fpos = 0, last = null;
    for (let fi = 0; fi < frames; fi++) {
      const t = fi / FPS;
      while (fpos < F.n && frameTime(fpos, sr) <= t) la.frameIn(F, fpos++);
      const st = snapshot(la.st);
      if (st.section !== last) { liveSecs.push({ kind: st.section, start: t }); last = st.section; }
      liveStates.push(st);
      if (fi % 900 === 0) await progress("Simulating live mode", fi / frames);
    }
    liveSecs.forEach((s, i) => s.end = i + 1 < liveSecs.length ? liveSecs[i + 1].start : dur);
    return {
      track: { bpm: tl.bpm, detected_bpm: Math.round(tl.bpm * 100) / 100, duration: dur, frames,
        offline_sections: tl.sections.map(s => ({ kind: s.kind, start: s.start, end: s.end })), live_sections: liveSecs,
        live_bars: la.barTrace, truth: null },   // live_bars: per bar, what the live analyzer saw (its 6 causal features, section, lock)
      states: { pre: preStates, live: liveStates },     // keep these: renderShow(states, genre, style) re-renders cheaply
      timeline: tl,
    };
  }

  const api = { BLINDER_COLOR, extractFeatures, analyzeOffline, labelSections, LiveSections, simulateLiveBars, LiveAnalyzer, ShowEngine, demoRig, analyzeTrack, renderShow, frameTime, DROP_STYLES, DROP_ROTATION, SECTIONS, GENRES, FPS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Jaylights = api;
})(typeof self !== "undefined" ? self : globalThis);
