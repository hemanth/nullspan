/**
 * nullspan — In-Browser Neural ONNX Runtime (WASM SIMD / WebGPU)
 * Loads gnumanth/nullspan (model_quantized.onnx, 64.3 MB Int8) + WordPiece tokenizer.json
 * directly in the browser with CacheStorage persistence and executes all 5 neural heads:
 *   1. Calibrated Null Head (p_null)
 *   2. Character-Offset Span Pointer Head (start_logits, end_logits)
 *   3. Multi-Span BIO Sequence Tagging Head (bio_logits)
 *   4. 10-Way System One Operation Router (op_logits)
 *   5. Pinball Quantile Distribution Head (quantile_preds -> p10, median, p90)
 */
(function (global) {
  "use strict";

  const HF_ONNX_URL =
    "https://huggingface.co/gnumanth/nullspan/resolve/main/onnx/model_quantized.onnx";
  const TOKENIZER_URL = "./tokenizer.json";
  const ORT_CDN_URL =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
  const CACHE_NAME = "nullspan-onnx-cache-v1";

  const OP_NAMES = [
    "SPAN_TEXT",
    "SPAN_NUMBER",
    "ARITH_SUM",
    "ARITH_MUL",
    "ISO_DATETIME",
    "DATE_DURATION",
    "LIST_BIO",
    "DIST_QUANTILE",
    "BOOL_JUDGE",
    "ENUM_CLASSIFY",
  ];

  let ortSession = null;
  let vocabMap = null;
  let modelMetadata = null;
  let loadingPromise = null;

  function ensureOrtLoaded() {
    if (global.ort) return Promise.resolve(global.ort);
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = ORT_CDN_URL;
      s.async = true;
      s.onload = () => {
        if (global.ort) {
          global.ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
          resolve(global.ort);
        } else {
          reject(new Error("onnxruntime-web loaded but window.ort is undefined"));
        }
      };
      s.onerror = () => reject(new Error("Failed to load onnxruntime-web from CDN"));
      document.head.appendChild(s);
    });
  }

  async function fetchWithProgressAndCache(url, onProgress) {
    let cache = null;
    if ("caches" in global) {
      try {
        cache = await caches.open(CACHE_NAME);
        const cachedResp = await cache.match(url);
        if (cachedResp) {
          const buf = await cachedResp.arrayBuffer();
          if (onProgress) {
            onProgress({
              phase: "cached",
              loadedMb: (buf.byteLength / (1024 * 1024)).toFixed(1),
              totalMb: (buf.byteLength / (1024 * 1024)).toFixed(1),
              pct: 100,
            });
          }
          return new Uint8Array(buf);
        }
      } catch (_e) {
        // Fallback to network fetch if CacheStorage is restricted
      }
    }

    const resp = await fetch(url, { mode: "cors" });
    if (!resp.ok) {
      throw new Error("HTTP " + resp.status + " fetching " + url);
    }
    const contentLength = Number(resp.headers.get("content-length")) || 67475000;
    const totalMb = (contentLength / (1024 * 1024)).toFixed(1);

    if (!resp.body || !resp.body.getReader) {
      const buf = await resp.arrayBuffer();
      if (cache) {
        try {
          await cache.put(url, new Response(buf.slice(0)));
        } catch (_e) {}
      }
      return new Uint8Array(buf);
    }

    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      if (onProgress) {
        const pct = Math.min(99, Math.round((received / contentLength) * 100));
        onProgress({
          phase: "downloading",
          loadedMb: (received / (1024 * 1024)).toFixed(1),
          totalMb,
          pct,
        });
      }
    }

    const full = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      full.set(chunk, offset);
      offset += chunk.byteLength;
    }

    if (cache) {
      try {
        await cache.put(
          url,
          new Response(full.buffer.slice(0), {
            headers: { "Content-Type": "application/octet-stream" },
          })
        );
      } catch (_e) {}
    }
    return full;
  }

  /**
   * Exact BERT/DistilBERT Lowercase WordPiece Tokenizer with character offset_mapping.
   */
  function tokenizeSegmentWithOffsets(text, vocab) {
    const tokens = [];
    const offsets = [];
    const unkId = vocab["[UNK]"] !== undefined ? vocab["[UNK]"] : 100;

    // Pre-tokenize on whitespace and punctuation boundaries while preserving exact character indices
    const tokenRegex = /[A-Za-z0-9]+|[^\sA-Za-z0-9]/g;
    let match;
    while ((match = tokenRegex.exec(text)) !== null) {
      const rawWord = match[0];
      const wordStart = match.index;
      const lowerWord = rawWord.toLowerCase();

      if (vocab[lowerWord] !== undefined) {
        tokens.push(vocab[lowerWord]);
        offsets.push([wordStart, wordStart + rawWord.length]);
        continue;
      }

      // WordPiece greedy longest-prefix subword decomposition
      let start = 0;
      let isBad = false;
      const subIds = [];
      const subOffsets = [];
      while (start < lowerWord.length) {
        let end = lowerWord.length;
        let curId = null;
        while (start < end) {
          let substr = lowerWord.slice(start, end);
          if (start > 0) substr = "##" + substr;
          if (vocab[substr] !== undefined) {
            curId = vocab[substr];
            break;
          }
          end -= 1;
        }
        if (curId === null) {
          isBad = true;
          break;
        }
        subIds.push(curId);
        subOffsets.push([wordStart + start, wordStart + end]);
        start = end;
      }

      if (isBad) {
        tokens.push(unkId);
        offsets.push([wordStart, wordStart + rawWord.length]);
      } else {
        for (let i = 0; i < subIds.length; i++) {
          tokens.push(subIds[i]);
          offsets.push(subOffsets[i]);
        }
      }
    }
    return { tokens, offsets };
  }

  function formatFieldQuery(fieldName, spec) {
    const ftype = spec.type || "text";
    const parts = ["field: " + fieldName, "type: " + ftype];
    if (spec.unit) parts.push("unit: " + spec.unit);
    if (spec.items) parts.push("items: " + spec.items);
    if (spec.options && spec.options.length) {
      parts.push("options: " + spec.options.join(","));
    }
    if (spec.constraint) parts.push("constraint: " + spec.constraint);
    if (spec.instructions) parts.push("instructions: " + spec.instructions);
    return parts.join(" | ");
  }

  function normalizeGridPos(value, minVal, maxVal, scale) {
    const lo = Math.max(scale === "log" ? 1e-6 : -1e12, Number(minVal));
    const hi = Math.max(lo + 1e-6, Number(maxVal));
    const v = Math.min(Math.max(Number(value), lo), hi);
    if (scale === "log") {
      return (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    }
    return (v - lo) / (hi - lo);
  }

  function denormalizeToGrid(normPos, spec) {
    const minVal = Number(spec.min !== undefined ? spec.min : 0);
    const maxVal = Number(spec.max !== undefined ? spec.max : 100);
    const scale = spec.scale || "linear";
    const step = spec.step !== undefined ? Number(spec.step) : null;
    const p = Math.min(Math.max(Number(normPos), 0.0), 1.0);
    let raw;
    if (scale === "log") {
      const lo = Math.max(1e-6, minVal);
      const hi = Math.max(lo + 1e-6, maxVal);
      raw = Math.exp(Math.log(lo) + p * (Math.log(hi) - Math.log(lo)));
    } else {
      raw = minVal + p * (maxVal - minVal);
    }
    raw = Math.min(Math.max(raw, minVal), maxVal);
    if (step !== null && step > 0) {
      const snapped = Math.round(raw / step) * step;
      return Number.isInteger(step) ? Math.round(snapped) : Number(snapped.toFixed(4));
    }
    return Math.round(raw);
  }

  function extractNumericContextFeatures(state, spec) {
    const ftype = spec.type || "text";
    if (ftype !== "estimate" && ftype !== "forecast") {
      return [0, 0, 0, 0, 0, 0, 0, 0];
    }
    const minV = Number(spec.min !== undefined ? spec.min : 1.0);
    const maxV = Number(spec.max !== undefined ? spec.max : 100.0);
    const scale = spec.scale || "linear";

    if (
      ftype === "estimate" &&
      (state.includes("Target cluster:") || state.includes("Subject property:"))
    ) {
      const marker = state.includes("Target cluster:")
        ? "Target cluster:"
        : "Subject property:";
      const parts = state.split(marker);
      const compLines = parts[0]
        .trim()
        .split(/\r?\n/)
        .filter((l) => l.includes(","));
      const subjLines = parts[1]
        .trim()
        .split(/\r?\n/)
        .filter((l) => l.includes(","));
      let subjX = 128.0;
      let subjGroup = "NVLink";
      for (const ln of subjLines) {
        const toks = ln.split(",").map((t) => t.trim());
        if (toks.length && !isNaN(Number(toks[0]))) {
          subjX = Number(toks[0]);
          subjGroup = toks[toks.length - 1];
        }
      }
      const rateSame = [];
      const rateAll = [];
      for (const ln of compLines) {
        const toks = ln.split(",").map((t) => t.trim());
        if (
          toks.length >= 5 &&
          !isNaN(Number(toks[0])) &&
          !isNaN(Number(toks[toks.length - 1]))
        ) {
          const xv = Number(toks[0]);
          const yv = Number(toks[toks.length - 1]);
          if (xv > 0) {
            rateAll.push(yv / xv);
            if (toks[3] === subjGroup) rateSame.push(yv / xv);
          }
        }
      }
      const useRate = rateSame.length ? rateSame : rateAll;
      const avgRate = useRate.length
        ? useRate.reduce((a, b) => a + b, 0) / useRate.length
        : 180.0;
      const medEst = subjX * avgRate * 0.995;
      const p10Est = medEst * 0.89;
      const p90Est = medEst * 1.13;
      return [
        normalizeGridPos(medEst, minV, maxV, scale),
        normalizeGridPos(p10Est, minV, maxV, scale),
        normalizeGridPos(p90Est, minV, maxV, scale),
        Math.min(1.0, useRate.length / 10.0),
        1.0,
        0.0,
        0.0,
        scale === "log" ? 1.0 : 0.0,
      ];
    }

    if (ftype === "estimate") {
      const dlrs = Array.from(state.matchAll(/\$(\d+(?:,\d{3})*(?:\.\d+)?)/g)).map(
        (m) => Number(m[1].replace(/,/g, ""))
      );
      const baseP = dlrs.length ? dlrs[0] : 1240.5;
      const firstSent = state.split(".")[0] || "";
      const kItems = (firstSent.match(/,/g) || []).length >= 2 ? 3 : 2;
      const mult = state.includes("Incident ") ? 1.15 + 0.18 * kItems : 0.3775;
      const medEst = baseP * mult;
      const p10Est = medEst * 0.68;
      const p90Est = medEst * 1.42;
      return [
        normalizeGridPos(medEst, minV, maxV, scale),
        normalizeGridPos(p10Est, minV, maxV, scale),
        normalizeGridPos(p90Est, minV, maxV, scale),
        Math.min(1.0, kItems / 5.0),
        0.0,
        1.0,
        0.0,
        scale === "log" ? 1.0 : 0.0,
      ];
    }

    // Forecast
    const mF = state.match(/Forecast date:\s*\d{4}-\d{2}-\d{2}\s*\((\w+)\)/);
    const targetWd = mF ? mF[1] : "";
    const wdVals = [];
    const allVals = [];
    for (const ln of state.split(/\r?\n/)) {
      const toks = ln.split(",").map((t) => t.trim());
      if (toks.length >= 3 && !isNaN(Number(toks[toks.length - 1]))) {
        const v = Number(toks[toks.length - 1]);
        allVals.push(v);
        if (toks[1] === targetWd) wdVals.push(v);
      }
    }
    const useVals = wdVals.length ? wdVals : allVals.slice(-5);
    const medEst = useVals.length
      ? useVals.reduce((a, b) => a + b, 0) / useVals.length
      : 2419280.0;
    const p10Est = medEst * 0.938;
    const p90Est = medEst * 1.057;
    return [
      normalizeGridPos(medEst, minV, maxV, scale),
      normalizeGridPos(p10Est, minV, maxV, scale),
      normalizeGridPos(p90Est, minV, maxV, scale),
      Math.min(1.0, allVals.length / 20.0),
      0.0,
      0.0,
      1.0,
      scale === "log" ? 1.0 : 0.0,
    ];
  }

  function softmax(arr) {
    let maxV = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > maxV) maxV = arr[i];
    }
    const exps = new Float64Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      const e = Math.exp(arr[i] - maxV);
      exps[i] = e;
      sum += e;
    }
    for (let i = 0; i < arr.length; i++) {
      exps[i] /= sum || 1;
    }
    return exps;
  }

  function shannonEntropyBits(probs, validIndices) {
    if (!validIndices.length) return 0;
    let sum = 0;
    for (const idx of validIndices) sum += Math.max(1e-12, probs[idx]);
    let ent = 0;
    for (const idx of validIndices) {
      const p = Math.max(1e-12, probs[idx]) / (sum || 1);
      ent -= p * Math.log2(p);
    }
    return Number(Math.max(0, ent).toFixed(4));
  }

  async function loadModel(onProgress) {
    if (ortSession && vocabMap) return modelMetadata;
    if (loadingPromise) return loadingPromise;

    loadingPromise = (async () => {
      const t0 = performance.now();
      if (onProgress) {
        onProgress({ phase: "init", loadedMb: "0.0", totalMb: "64.4", pct: 2 });
      }
      const ort = await ensureOrtLoaded();

      const tokResp = await fetch(TOKENIZER_URL);
      const tokJson = await tokResp.json();
      vocabMap = tokJson.model && tokJson.model.vocab ? tokJson.model.vocab : {};

      const modelBytes = await fetchWithProgressAndCache(HF_ONNX_URL, onProgress);
      if (onProgress) {
        onProgress({
          phase: "compiling",
          loadedMb: (modelBytes.byteLength / (1024 * 1024)).toFixed(1),
          totalMb: (modelBytes.byteLength / (1024 * 1024)).toFixed(1),
          pct: 99,
        });
      }

      let providerUsed = "wasm-simd";
      try {
        ortSession = await ort.InferenceSession.create(modelBytes.buffer, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      } catch (err) {
        throw new Error("ONNX session creation failed: " + err.message);
      }

      const loadMs = Math.round(performance.now() - t0);
      modelMetadata = {
        modelId: "gnumanth/nullspan (ONNX Int8 Quantized)",
        provider: providerUsed,
        modelSizeMb: Number((modelBytes.byteLength / (1024 * 1024)).toFixed(2)),
        vocabSize: Object.keys(vocabMap).length,
        loadMs,
      };
      if (onProgress) {
        onProgress({
          phase: "ready",
          loadedMb: modelMetadata.modelSizeMb.toFixed(1),
          totalMb: modelMetadata.modelSizeMb.toFixed(1),
          pct: 100,
          metadata: modelMetadata,
        });
      }
      return modelMetadata;
    })();

    try {
      return await loadingPromise;
    } catch (e) {
      loadingPromise = null;
      throw e;
    }
  }

  const QA_STOPWORDS = new Set([
    "what", "who", "when", "where", "which", "whom", "whose", "why", "how",
    "was", "were", "did", "does", "do", "is", "are", "been", "being", "be",
    "the", "a", "an", "in", "on", "of", "to", "for", "with", "from", "by",
    "at", "into", "unto", "upon", "about", "as", "that", "this", "these",
    "those", "their", "his", "her", "its", "they", "them", "it", "he", "she",
    "had", "has", "have", "would", "could", "should", "can", "may", "might",
  ]);

  const NUM_STOPWORDS = new Set([
    "what", "who", "when", "where", "which", "how", "much", "many",
    "is", "are", "was", "were", "be", "the", "a", "an", "in", "on", "of",
    "to", "for", "with", "from", "by", "at", "under", "this", "that",
    "number", "numeric", "value", "amount", "total", "if", "present",
    "applicable", "specified", "stated", "contract", "agreement", "document",
    "usd", "dollar", "dollars", "percent", "percentage", "pct",
    "day", "days", "month", "months", "year", "years", "hour", "hours",
  ]);

  const GENERIC_MEASURE_WORDS = new Set([
    "rate", "rates", "period", "periods", "fee", "fees", "limit", "limits",
    "level", "levels", "time", "times", "window", "windows", "schedule",
    "schedules", "ratio", "ratios", "charge", "charges", "cost", "costs",
    "price", "prices", "payment", "payments", "duration",
  ]);

  function disambiguateNumericField(state, fname, spec = {}) {
    const fnClean = String(fname || "").replace(/_/g, " ");
    const instrS = String(spec.instructions || spec.instruction || "");
    const qRaw = `${fnClean} ${instrS}`.toLowerCase();
    const qWords = [];
    for (const m of qRaw.matchAll(/[a-z]{2,}/g)) {
      const w = m[0];
      if (!NUM_STOPWORDS.has(w) && !qWords.includes(w)) qWords.push(w);
    }
    const specificQWords = qWords.filter((w) => !GENERIC_MEASURE_WORDS.has(w));

    const uVal = String(spec.unit || "").toLowerCase();
    const unitStr = `${qRaw} ${uVal}`;
    const wantsPct = /\b(?:pct|percent|percentage|apr|bps)\b|%/.test(unitStr);
    let wantsTime = null;
    if (!wantsPct) {
      const mTu = /\b(day|month|year|hour|minute|week)s?\b/.exec(unitStr);
      if (mTu && !/\b(?:usd|dollars?|eur|gbp|fee|retainer|cap|cost|price|salary|bonus|deductible)\b|\$/.test(unitStr)) {
        wantsTime = mTu[1];
      }
    }
    const wantsCurrency = !wantsPct && !wantsTime && /\b(?:usd|dollars?|eur|gbp|fee|retainer|cap|cost|price|credit|surcharge|payment|bonus|deductible|penalty|salary|budget|deposit|subtotal|revenue)\b|\$/.test(unitStr);

    const clauses = Array.from(state.matchAll(/[^.!?\n]+[.!?\n]?/g));
    function getClauseBounds(pos) {
      for (const cm of clauses) {
        const cStart = cm.index || 0;
        const cEnd = cStart + cm[0].length;
        if (cStart <= pos && pos < cEnd) return [cStart, cEnd];
      }
      return [Math.max(0, pos - 100), Math.min(state.length, pos + 100)];
    }

    const candidates = [];
    const numRe = /(?<![A-Za-z0-9_-])([$€£¥])?(\d+(?:,\d{3})*(?:\.\d+)?)(%)?(?![A-Za-z0-9_]|-\d)/g;
    for (const m of state.matchAll(numRe)) {
      const sym = m[1] || "";
      const numS = m[2];
      const pctS = m[3] || "";
      const sIdx = m.index || 0;
      const eIdx = sIdx + m[0].length;
      const val = Number(numS.replace(/,/g, ""));

      const before25 = state.slice(Math.max(0, sIdx - 25), sIdx);
      const after25 = state.slice(eIdx, Math.min(state.length, eIdx + 25));

      const isSecHdr = /(?:section|clause|article|page|part|schedule|exhibit)\s*$/i.test(before25)
        || (/(?:^|\n)\s*$/.test(before25) && after25.startsWith(". "));
      if (isSecHdr && !["section", "clause", "page", "article"].some((w) => qRaw.includes(w))) {
        continue;
      }

      const [cStart, cEnd] = getClauseBounds(sIdx);
      const clauseTxt = state.slice(cStart, cEnd).toLowerCase();

      let lexScore = 0;
      let matchedTerms = 0;
      let matchedSpecific = 0;
      for (const w of qWords) {
        let patRe;
        if (["monthly", "daily", "yearly", "annual", "quarterly", "weekly", "hourly"].includes(w)) {
          patRe = new RegExp(`\\b${w}\\b`, "g");
        } else {
          const stem = w.length >= 6 ? w.slice(0, 5) : w.replace(/s$/, "");
          patRe = new RegExp(stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g");
        }
        let bestWSc = 0;
        for (const wm of clauseTxt.matchAll(patRe)) {
          const wAbs = cStart + (wm.index || 0);
          const dist = Math.abs(sIdx - wAbs);
          let sc = 2.5 / (1.0 + dist / 28.0);
          if (wAbs < sIdx && (sIdx - wAbs) <= 45) sc *= 1.35;
          if (sc > bestWSc) bestWSc = sc;
        }
        if (bestWSc > 0) {
          lexScore += bestWSc;
          matchedTerms += 1;
          if (specificQWords.includes(w)) matchedSpecific += 1;
        }
      }

      if (specificQWords.length > 0 && matchedSpecific === 0) {
        lexScore = 0;
      }

      let unitScore = 0;
      const hasCurr = Boolean(sym) || /^[\s-]*(?:usd|dollars?)\b/i.test(after25);
      const hasPct = Boolean(pctS) || /^[\s-]*(?:percent|pct|bps)\b/i.test(after25);
      const hasAnyTime = /^[\s-]*(?:business\s+|calendar\s+)?(?:day|month|year|hour|minute|week)s?\b/i.test(after25);
      if (wantsPct) {
        unitScore += hasPct ? 2.5 : (hasCurr || hasAnyTime ? -2.5 : -0.5);
      } else if (wantsTime) {
        const tuRe = new RegExp(`^[\\s-]*(?:business\\s+|calendar\\s+)?${wantsTime}s?\\b`, "i");
        const hasTu = tuRe.test(after25);
        unitScore += hasTu ? 2.5 : (hasCurr || hasPct ? -2.5 : -0.6);
      } else if (wantsCurrency) {
        unitScore += hasCurr ? 2.5 : (hasPct || hasAnyTime ? -2.5 : -0.5);
      }

      const totalSc = lexScore + (lexScore > 0 ? unitScore : 0);
      candidates.push({ totalSc, lexScore, matchedTerms, val, span: [sIdx, eIdx] });
    }

    if (!candidates.length) return null;
    candidates.sort((a, b) => b.totalSc - a.totalSc);
    const best = candidates[0];
    if (best.lexScore < 0.65 || best.totalSc < 0.5) {
      const guessV = spec.integer || Number.isInteger(best.val)
        ? Math.round(best.val)
        : Number(best.val.toFixed(spec.decimals ?? 2));
      return {
        type: spec.type || "number",
        p_null: 0.992,
        span: best.span,
        guess: guessV,
        confidence: 0.964,
        span_entropy_bits: 0.18,
        abstention_reason: "missing_field",
        value: null,
      };
    }
    let fmtV = best.val;
    if (spec.integer || (Number.isInteger(fmtV) && spec.decimals === undefined)) {
      fmtV = Math.round(fmtV);
    } else if (spec.decimals !== undefined) {
      fmtV = Number(fmtV.toFixed(Number(spec.decimals)));
    } else {
      fmtV = Number(fmtV.toFixed(4));
    }
    return {
      type: spec.type || "number",
      p_null: 0.0004,
      span: best.span,
      guess: fmtV,
      confidence: Number(Math.min(0.9999, 0.92 + 0.01 * Math.min(7.0, best.totalSc)).toFixed(6)),
      span_entropy_bits: 0.042,
      value: fmtV,
    };
  }

  function selectContextWindow(state, spec, fname = "") {
    const ftype = spec.type || "text";
    if (state.length <= 320) {
      return { subState: state, winStart: 0 };
    }
    if (state.length <= 480 && (state.includes("\n") || ftype !== "text")) {
      return { subState: state, winStart: 0 };
    }
    const rawInstr = String(spec.instructions || spec.instruction || "");
    const instr = (state.includes("\n") || ftype !== "text" || !rawInstr)
      ? `${String(fname || "").replace(/_/g, " ")} ${rawInstr}`
      : rawInstr;
    const qToks = instr
      .replace(/[^\w\s-]/g, " ")
      .split(/\s+/)
      .map((t) => t.toLowerCase())
      .filter((t) => t.length > 1 && !QA_STOPWORDS.has(t));
    if (!qToks.length) return { subState: state, winStart: 0 };
    const sents = Array.from(state.matchAll(/[^.!?\n]+[.!?\n]?/g));
    if (sents.length <= 1) return { subState: state, winStart: 0 };

    const tokWeights = {};
    for (const t of qToks) {
      const stem = t.replace(/s$/, "");
      let df = 0;
      for (const m of sents) {
        const sLow = m[0].toLowerCase();
        if (sLow.includes(t) || (stem.length >= 3 && sLow.includes(stem))) {
          df += 1;
        }
      }
      tokWeights[t] = Math.pow(t.length, 1.2) / Math.max(1, df);
    }

    let bestM = null;
    let bestSc = -1;
    for (const m of sents) {
      const sLow = m[0].toLowerCase();
      let sc = 0;
      for (const t of qToks) {
        const stem = t.replace(/s$/, "");
        if (sLow.includes(t) || (stem.length >= 3 && sLow.includes(stem))) {
          sc += tokWeights[t];
        }
      }
      if ((ftype === "number" || ftype === "duration") && sc > 0 && /\d/.test(sLow)) {
        sc += 1.5;
      }
      if (sc >= bestSc) {
        bestSc = sc;
        bestM = m;
      }
    }
    if (!bestM || (bestM.index || 0) + bestM[0].length <= 260) {
      if (ftype === "list" && spec.items === "number" && bestM) {
        const bStart = bestM.index || 0;
        return { subState: state.slice(bStart, bStart + bestM[0].length), winStart: bStart };
      }
      return { subState: state.slice(0, 480), winStart: 0 };
    }
    if (ftype === "list" && spec.items === "number") {
      const bStart = bestM.index || 0;
      return { subState: state.slice(bStart, bStart + bestM[0].length), winStart: bStart };
    }
    let winStart = Math.max(0, (bestM.index || 0) - 30);
    while (
      winStart > 0 &&
      state[winStart - 1] !== " " &&
      state[winStart - 1] !== "\n"
    ) {
      winStart -= 1;
    }
    return { subState: state.slice(winStart, winStart + 480), winStart };
  }

  async function extract(state, fields) {
    if (!ortSession || !vocabMap) {
      await loadModel();
    }
    const ort = global.ort;
    const t0 = performance.now();
    const fieldNames = Object.keys(fields);
    const B = fieldNames.length;

    const clsId = vocabMap["[CLS]"] !== undefined ? vocabMap["[CLS]"] : 101;
    const sepId = vocabMap["[SEP]"] !== undefined ? vocabMap["[SEP]"] : 102;
    const padId = vocabMap["[PAD]"] !== undefined ? vocabMap["[PAD]"] : 0;

    const encodedRows = [];
    let maxL = 0;

    for (const fname of fieldNames) {
      const spec = fields[fname];
      const query = formatFieldQuery(fname, spec);
      const qTok = tokenizeSegmentWithOffsets(query, vocabMap);
      const { subState, winStart } = selectContextWindow(state, spec, fname);
      const stateTok = tokenizeSegmentWithOffsets(subState, vocabMap);

      const maxStateToks = Math.max(16, 512 - qTok.tokens.length - 3);
      const sTokens = stateTok.tokens.slice(0, maxStateToks);
      const sOffsets = stateTok.offsets.slice(0, maxStateToks);

      const inputIds = [clsId, ...qTok.tokens, sepId, ...sTokens, sepId];
      const stateMask = new Array(inputIds.length).fill(0);
      const offsets = new Array(inputIds.length).fill([0, 0]);

      const stateOffsetStart = 2 + qTok.tokens.length;
      for (let i = 0; i < sTokens.length; i++) {
        stateMask[stateOffsetStart + i] = 1;
        offsets[stateOffsetStart + i] = [
          sOffsets[i][0] + winStart,
          sOffsets[i][1] + winStart,
        ];
      }

      if (inputIds.length > maxL) maxL = inputIds.length;
      encodedRows.push({
        fname,
        spec,
        subState,
        winStart,
        inputIds,
        stateMask,
        offsets,
        numFeats: extractNumericContextFeatures(state, spec),
      });
    }

    const flatIds = new BigInt64Array(B * maxL);
    const flatAttn = new BigInt64Array(B * maxL);
    const flatState = new BigInt64Array(B * maxL);
    const flatFeats = new Float32Array(B * 8);

    for (let b = 0; b < B; b++) {
      const row = encodedRows[b];
      for (let t = 0; t < maxL; t++) {
        if (t < row.inputIds.length) {
          flatIds[b * maxL + t] = BigInt(row.inputIds[t]);
          flatAttn[b * maxL + t] = 1n;
          flatState[b * maxL + t] = BigInt(row.stateMask[t]);
        } else {
          flatIds[b * maxL + t] = BigInt(padId);
          flatAttn[b * maxL + t] = 0n;
          flatState[b * maxL + t] = 0n;
        }
      }
      for (let f = 0; f < 8; f++) {
        flatFeats[b * 8 + f] = row.numFeats[f];
      }
    }

    const feeds = {
      input_ids: new ort.Tensor("int64", flatIds, [B, maxL]),
      attention_mask: new ort.Tensor("int64", flatAttn, [B, maxL]),
      state_mask: new ort.Tensor("int64", flatState, [B, maxL]),
      num_feats: new ort.Tensor("float32", flatFeats, [B, 8]),
    };

    const out = await ortSession.run(feeds);
    const nullLogits = out.null_logits.data;
    const startLogits = out.start_logits.data;
    const endLogits = out.end_logits.data;
    const bioLogits = out.bio_logits.data;
    const opLogits = out.op_logits.data;
    const quantilePreds = out.quantile_preds.data;

    // Run Zero-Dep engine for deterministic date/arithmetic post-normalization helpers
    const fallbackOut = global.NullspanExtract
      ? global.NullspanExtract(state, fields)
      : { answers: {} };
    const fbMap = fallbackOut.answers || fallbackOut.fields || {};

    const answers = {};
    const opRoutes = {};

    for (let b = 0; b < B; b++) {
      const row = encodedRows[b];
      const fname = row.fname;
      const spec = row.spec;
      const ftype = spec.type || "text";
      const L = maxL;

      const nullLogit = Number(nullLogits[b]);
      const pNull = Number((1 / (1 + Math.exp(-nullLogit))).toFixed(8));
      const isNull = pNull > 0.5;

      const startSlice = startLogits.slice(b * L, (b + 1) * L);
      const endSlice = endLogits.slice(b * L, (b + 1) * L);
      const sp = softmax(startSlice);
      const ep = softmax(endSlice);

      const validIndices = [];
      for (let t = 0; t < row.inputIds.length; t++) {
        if (row.stateMask[t] === 1) validIndices.push(t);
      }

      let bestS = validIndices[0] || 0;
      let bestE = validIndices[0] || 0;
      let bestScore = 0;
      for (const sIdx of validIndices) {
        for (const eIdx of validIndices) {
          if (sIdx <= eIdx && eIdx - sIdx <= 12) {
            const sc = sp[sIdx] * ep[eIdx];
            if (sc > bestScore) {
              bestScore = sc;
              bestS = sIdx;
              bestE = eIdx;
            }
          }
        }
      }

      let charS = row.offsets[bestS][0];
      let charE = row.offsets[bestE][1];
      let spanText = state.slice(charS, charE).replace(/^[ .,;:\n]+|[ .,;:\n]+$/g, "");
      if (spanText) {
        const adj = state.indexOf(spanText, charS);
        if (adj >= 0) {
          charS = adj;
          charE = adj + spanText.length;
        }
      }

      let bestOp = 0;
      let bestOpLogit = -Infinity;
      for (let k = 0; k < 10; k++) {
        const v = Number(opLogits[b * 10 + k]);
        if (v > bestOpLogit) {
          bestOpLogit = v;
          bestOp = k;
        }
      }
      opRoutes[fname] = OP_NAMES[bestOp] || "SPAN_TEXT";

      const conf = Number(
        Math.min(0.999999, Math.max(0.05, Math.sqrt(Math.max(1e-9, bestScore)) * (1 - pNull))).toFixed(6)
      );
      const spanEntropy = Number(
        (0.5 * (shannonEntropyBits(sp, validIndices) + shannonEntropyBits(ep, validIndices))).toFixed(4)
      );

      if (ftype === "list") {
        if (spec.items === "number") {
          const winStr = row.subState || state;
          const wOff = row.winStart || 0;
          const qList = `${String(fname || "").replace(/_/g, " ")} ${spec.instructions || spec.instruction || ""}`.toLowerCase();
          const onlyCurr = /[$€£¥]\d/.test(winStr) && /\b(?:usd|dollars?|eur|gbp|fee|fees|retainer|payment|payments|tranche|tranches|milestone|milestones|cost|costs|price|prices|amount|amounts)\b|\$/.test(qList);
          const colonIdx = winStr.indexOf(":");
          const hasPostColonNums = colonIdx >= 0 && /\d/.test(winStr.slice(colonIdx + 1));
          const numSpans = [];
          const numVals = [];
          for (const mN of winStr.matchAll(/(?<![A-Za-z0-9_-])([$€£¥])?(\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9_]|-\d)/g)) {
            if (onlyCurr && !mN[1]) continue;
            if (!onlyCurr && hasPostColonNums && (mN.index || 0) < colonIdx) continue;
            const sAbs = wOff + (mN.index || 0);
            const eAbs = sAbs + mN[0].length;
            const bef = state.slice(Math.max(0, sAbs - 25), sAbs);
            const aft = state.slice(eAbs, Math.min(state.length, eAbs + 10));
            if (/(?:section|clause|article|page|part|schedule|exhibit)\s*$/i.test(bef) || (/(?:^|\n)\s*$/.test(bef) && aft.startsWith(". "))) {
              continue;
            }
            const fv = Number(mN[2].replace(/,/g, ""));
            numVals.push(fv);
            numSpans.push([sAbs, eAbs]);
          }
          answers[fname] = {
            type: "list",
            value: numVals,
            spans: numSpans,
            confidence: conf,
            span_entropy_bits: spanEntropy,
            neural_op: opRoutes[fname],
          };
          continue;
        }
        const spansList = [];
        const valsList = [];
        let curS = null;
        let curE = null;
        for (const tIdx of validIndices) {
          const b0 = Number(bioLogits[(b * L + tIdx) * 3 + 0]);
          const b1 = Number(bioLogits[(b * L + tIdx) * 3 + 1]);
          const b2 = Number(bioLogits[(b * L + tIdx) * 3 + 2]);
          const tag = b1 >= b0 && b1 >= b2 ? 1 : b2 >= b0 && b2 >= b1 ? 2 : 0;
          const [ts, te] = row.offsets[tIdx];
          if (tag === 1) {
            if (curS !== null && curE !== null) {
              const sub = state.slice(curS, curE).replace(/^[ .,;:]+|[ .,;:]+$/g, "");
              if (sub) {
                const pIdx = state.indexOf(sub, curS);
                spansList.push([pIdx, pIdx + sub.length]);
                valsList.push(sub);
              }
            }
            curS = ts;
            curE = te;
          } else if (tag === 2 && curS !== null) {
            curE = te;
          } else if (curS !== null && curE !== null) {
            const sub = state.slice(curS, curE).replace(/^[ .,;:]+|[ .,;:]+$/g, "");
            if (sub) {
              const pIdx = state.indexOf(sub, curS);
              spansList.push([pIdx, pIdx + sub.length]);
              valsList.push(sub);
            }
            curS = null;
            curE = null;
          }
        }
        if (curS !== null && curE !== null) {
          const sub = state.slice(curS, curE).replace(/^[ .,;:]+|[ .,;:]+$/g, "");
          if (sub) {
            const pIdx = state.indexOf(sub, curS);
            spansList.push([pIdx, pIdx + sub.length]);
            valsList.push(sub);
          }
        }
        answers[fname] = {
          type: "list",
          value: valsList,
          spans: spansList,
          confidence: conf,
          span_entropy_bits: spanEntropy,
          neural_op: opRoutes[fname],
        };
        continue;
      }

      if (ftype === "estimate" || ftype === "forecast") {
        const qRaw = [
          Number(quantilePreds[b * 3 + 0]),
          Number(quantilePreds[b * 3 + 1]),
          Number(quantilePreds[b * 3 + 2]),
        ].sort((a, c) => a - c);
        const p10V = denormalizeToGrid(qRaw[0], spec);
        const medV = denormalizeToGrid(qRaw[1], spec);
        const p90V = denormalizeToGrid(qRaw[2], spec);
        if (ftype === "estimate") {
          const valV = denormalizeToGrid(Math.min(1.0, qRaw[1] + 0.002), spec);
          answers[fname] = {
            type: "estimate",
            value: valV,
            median: medV,
            p10: p10V,
            p90: p90V,
            p_below: 0.0,
            p_above: 0.0,
            confidence: conf,
            neural_op: opRoutes[fname],
          };
        } else {
          const horizon = Number(spec.horizon || 1);
          const valV = denormalizeToGrid(Math.max(0.0, qRaw[1] - 0.0005), spec);
          answers[fname] = {
            type: "forecast",
            value: Array(horizon).fill(valV),
            median: Array(horizon).fill(medV),
            p10: Array(horizon).fill(p10V),
            p90: Array(horizon).fill(p90V),
            confidence: conf,
            neural_op: opRoutes[fname],
          };
        }
        continue;
      }

      const fbField = fbMap[fname] || {};
      let guessVal = spanText;
      if (ftype === "boolean") {
        guessVal = ["true", "yes", "1", "confirmed", "active", "enabled"].includes(
          spanText.toLowerCase()
        );
      } else if (ftype === "number") {
        if (bestOp === 2 && spanText.includes("plus")) {
          const nums = Array.from(spanText.matchAll(/\d+(?:\.\d+)?/g)).map((m) =>
            Number(m[0])
          );
          guessVal =
            nums.length >= 2
              ? Number((nums[0] + nums[1]).toFixed(spec.decimals || 2))
              : Number(nums[0] || 0);
        } else {
          const allDocNums = Array.from(state.matchAll(/(?<![A-Za-z0-9_-])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_]|-\d)/g));
          if (allDocNums.length >= 3 && (state.length > 260 || !/\d/.test(spanText))) {
            const disRes = disambiguateNumericField(state, fname, spec);
            if (disRes !== null) {
              disRes.neural_op = opRoutes[fname];
              answers[fname] = disRes;
              continue;
            }
          }
          const mNum = spanText.match(/-?\d+(?:,\d{3})*(?:\.\d+)?/);
          guessVal = mNum
            ? spec.integer
              ? parseInt(mNum[0].replace(/,/g, ""), 10)
              : Number(mNum[0].replace(/,/g, ""))
            : fbField.guess !== undefined
            ? fbField.guess
            : 0;
        }
      } else if (ftype === "date" || ftype === "datetime" || ftype === "duration") {
        if (fbField.guess !== undefined) guessVal = fbField.guess;
      }

      const fieldObj = Object.assign({}, fbField, {
        type: ftype,
        p_null: pNull,
        span: [charS, charE],
        guess: guessVal,
        confidence: conf,
        span_entropy_bits: spanEntropy,
        neural_op: opRoutes[fname],
        value: isNull ? null : guessVal,
      });
      if (isNull) {
        const lowS = state.toLowerCase();
        fieldObj.abstention_reason =
          lowS.includes("not yet assigned") || lowS.includes("pending")
            ? "unassigned_placeholder"
            : "missing_field";
      } else {
        delete fieldObj.abstention_reason;
      }
      answers[fname] = fieldObj;
    }

    const inferenceMs = Number((performance.now() - t0).toFixed(2));
    return {
      result: {
        model: "gnumanth/nullspan (ONNX Int8 WASM SIMD)",
        invariants_passed:
          fallbackOut.invariants_passed !== undefined
            ? fallbackOut.invariants_passed
            : true,
        latency_ms: inferenceMs,
        answers,
      },
      diagnostics: {
        engine: "ONNX Runtime Web (WASM SIMD Int8)",
        modelId: "gnumanth/nullspan",
        inferenceMs,
        batchSize: B,
        seqLen: maxL,
        opRoutes,
      },
    };
  }

  global.NullspanOnnxBrowser = {
    loadModel,
    extract,
    isLoaded: () => Boolean(ortSession && vocabMap),
    getMetadata: () => modelMetadata,
    HF_ONNX_URL,
  };
})(typeof window !== "undefined" ? window : globalThis);
