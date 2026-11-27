/* ADSIR retrieval pipeline, re-implemented for the browser.
   Mirrors src/pipeline.py and Code_series/adsir_common.py of RegulationQA_dataset:
   BM25 (rank_bm25 BM25Okapi, k1=1.5, b=0.75, eps=0.25) + BGE inner product, top 100 each,
   RRF (k=60, 0-based rank -> 1/(k+rank+1)) keyed by (doc, chunk_idx), pool of 20,
   candidates ranked by the best cross-encoder score among their OWN sentences,
   answer span = best-scoring sentence in the top candidate's +-1 window.
   Models are injected (embed, score) so this file has no dependencies and can be tested in Node. */
(function (root) {
  const PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
  const PYWS = /[ \t\n\r\x0b\x0c\x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/; // Python str.split() whitespace
  function tokenize(text) {
    let s = String(text).toLowerCase(), out = '';
    for (const ch of s) if (!PUNCT.has(ch)) out += ch;
    return out.split(PYWS).filter(Boolean);
  }

  function buildBM25(texts, k1 = 1.5, b = 0.75, epsilon = 0.25) {
    const N = texts.length, tfs = [], len = new Float64Array(N), df = new Map();
    let total = 0;
    texts.forEach((t, i) => {
      const toks = tokenize(t), tf = new Map();
      for (const w of toks) tf.set(w, (tf.get(w) || 0) + 1);
      tfs.push(tf); len[i] = toks.length; total += toks.length;
      for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1);
    });
    const avgdl = total / N, idf = new Map(), neg = [];
    let idfSum = 0;
    for (const [w, f] of df) {
      const v = Math.log(N - f + 0.5) - Math.log(f + 0.5);
      idf.set(w, v); idfSum += v; if (v < 0) neg.push(w);
    }
    const eps = epsilon * (idfSum / idf.size);
    for (const w of neg) idf.set(w, eps);
    return {
      scores(query) {
        const q = tokenize(query), out = new Float64Array(N);
        for (const w of q) {                       // duplicates counted, as in rank_bm25
          const iw = idf.get(w); if (iw === undefined) continue;
          for (let i = 0; i < N; i++) {
            const f = tfs[i].get(w); if (!f) continue;
            out[i] += iw * (f * (k1 + 1) / (f + k1 * (1 - b + b * len[i] / avgdl)));
          }
        }
        return out;
      }
    };
  }

  function topK(scores, k) {
    const idx = Array.from(scores.keys());
    idx.sort((a, b) => scores[b] - scores[a] || a - b);
    return idx.slice(0, k);
  }

  function f16ToF32(buf) {
    const h = new Uint16Array(buf), f = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 31, m = x & 1023;
      f[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return f;
  }

  function splitSentences(text) {
    if (typeof Intl !== 'undefined' && Intl.Segmenter) {
      const seg = new Intl.Segmenter('en', { granularity: 'sentence' });
      return Array.from(seg.segment(text), s => s.segment.trim()).filter(Boolean);
    }
    return text.split(/(?<=[.!?])\s+(?=[A-Z(0-9])/).map(s => s.trim()).filter(Boolean);
  }

  /* passages: [{id, doc, ci, text}] in index order; vectors: Float32Array (N*dim, L2-normalised) */
  function createPipeline(passages, vectors, dim = 768) {
    const N = passages.length, bm25 = buildBM25(passages.map(p => p.text));
    const byDoc = new Map();
    passages.forEach((p, i) => { if (!byDoc.has(p.doc)) byDoc.set(p.doc, []); byDoc.get(p.doc).push(i); });
    const pos = new Map(); for (const ids of byDoc.values()) ids.forEach((i, k) => pos.set(i, [ids, k]));
    function windowIdx(i) { const [ids, k] = pos.get(i); return [ids[k - 1], i, ids[k + 1]].filter(x => x !== undefined); }
    function dense(q) {
      const out = new Float64Array(N);
      for (let i = 0; i < N; i++) { let s = 0; const o = i * dim; for (let d = 0; d < dim; d++) s += q[d] * vectors[o + d]; out[i] = s; }
      return out;
    }

    /* Runs every stage and returns a trace for the visualisation.
       embed(text) -> Float32Array(dim) normalised, or null to skip the dense retriever.
       score(query, sentences[]) -> Promise<number[]> cross-encoder logits, or null to stop after fusion. */
    async function run(query, { embed, score, nRet = 100, pool = 20, rrfK = 60, onStage } = {}) {
      const T = { query };
      const bmS = bm25.scores(query);
      T.bm25 = topK(bmS, nRet).map(i => ({ i, s: bmS[i] }));
      onStage && onStage('bm25', T);
      const q = embed ? await embed(query) : null;
      if (q) {
        const dS = dense(q);
        T.dense = topK(dS, nRet).map(i => ({ i, s: dS[i] }));
      } else T.dense = [];
      onStage && onStage('dense', T);
      const acc = new Map(), first = new Map();
      for (const list of [T.dense, T.bm25]) list.forEach(({ i }, r) => {
        const key = passages[i].doc + '\u0000' + passages[i].ci;
        if (!first.has(key)) first.set(key, i);
        acc.set(key, (acc.get(key) || 0) + 1 / (rrfK + r + 1));
      });
      T.fused = [...acc].sort((a, b) => b[1] - a[1]).map(([k, s]) => ({ i: first.get(k), s }));
      T.pool = T.fused.slice(0, pool);
      T.bmRank = new Map(T.bm25.map((x, r) => [x.i, r + 1]));
      T.denseRank = new Map(T.dense.map((x, r) => [x.i, r + 1]));
      onStage && onStage('fusion', T);
      T.windows = T.pool.map(c => windowIdx(c.i));
      onStage && onStage('window', T);
      if (!score) return T;
      const cache = new Map();
      async function scoreAll(sents) {
        const todo = [...new Set(sents.filter(s => !cache.has(s)))];
        if (todo.length) { const v = await score(query, todo); if (!v) return null; todo.forEach((s, k) => cache.set(s, v[k])); }
        return sents.map(s => cache.get(s));
      }
      const own = T.pool.map(c => splitSentences(passages[c.i].text));
      const flat = await scoreAll(own.flat());
      if (!flat) return T;                         // no cross-encoder available: stop after the window stage
      let o = 0;
      T.cands = T.pool.map((c, k) => {
        const sc = flat.slice(o, o + own[k].length); o += own[k].length;
        let best = 0; sc.forEach((v, j) => { if (v > sc[best]) best = j; });
        return { i: c.i, rrf: c.s, rrfRank: k + 1, ce: sc.length ? sc[best] : -Infinity, best: own[k][best], nSent: sc.length };
      });
      T.ranked = [...T.cands].sort((a, b) => b.ce - a.ce);
      T.ranked.forEach((c, r) => c.ceRank = r + 1);
      onStage && onStage('score', T);
      const top = T.ranked[0], w = windowIdx(top.i);
      const wText = w.map(i => passages[i].text).join(' ');
      const wSents = splitSentences(wText), wSc = await scoreAll(wSents);
      let b = 0; wSc.forEach((v, j) => { if (v > wSc[b]) b = j; });
      const span = wSents[b];
      const from = w.find(i => passages[i].text.includes(span.slice(0, 80))) ?? top.i;
      T.answer = { i: top.i, from, window: w, span, ce: wSc[b], nWin: wSents.length, words: span.split(/\s+/).length };
      T.scored = cache.size;
      onStage && onStage('answer', T);
      return T;
    }
    return { run, bm25, tokenize, splitSentences, windowIdx };
  }

  const API = { createPipeline, tokenize, buildBM25, f16ToF32, splitSentences, topK };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.ADSIR = API;
})(this);
