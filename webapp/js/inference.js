/**
 * ONNX Runtime WASM inference + embedding ranking.
 */
import { parseNpy } from './npy.js';

const MAX_LEN = 200;

export class Recommender {
  constructor() {
    this.session = null;
    this.embeddings = null;
    this.embShape = null;
  }

  async load(modelUrl, embeddingsUrl, onProgress) {
    onProgress?.('Loading ONNX model...');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    const dataUrl = modelUrl + '.data';
    const [modelBuf, dataBuf] = await Promise.all([
      fetch(modelUrl).then(r => r.arrayBuffer()),
      fetch(dataUrl).then(r => r.arrayBuffer()),
    ]);
    this.session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ['wasm'],
      externalData: [{
        path: 'model_fp32.onnx.data',
        data: new Uint8Array(dataBuf),
      }],
    });
    onProgress?.('Loading embeddings...');
    const resp = await fetch(embeddingsUrl);
    const buf = await resp.arrayBuffer();
    const npy = parseNpy(buf);
    this.embeddings = npy.data;
    this.embShape = npy.shape;
    onProgress?.('Ready');
  }

  async predict(sequence, seenSet, k = 50, allowedMask = null) {
    const input = new BigInt64Array(MAX_LEN);
    const windowStart = Math.max(0, sequence.length - MAX_LEN);
    const windowLen = Math.min(sequence.length, MAX_LEN);
    const padStart = MAX_LEN - windowLen;
    for (let i = 0; i < windowLen; i++) {
      input[padStart + i] = BigInt(sequence[windowStart + i]);
    }

    const tensor = new ort.Tensor('int64', input, [1, MAX_LEN]);
    const result = await this.session.run({ seq: tensor });
    const outputKey = Object.keys(result)[0];
    const outputTensor = result[outputKey];
    const hidden = outputTensor.data;

    const d = this.embShape[1];
    // Model outputs [1, 256] (last hidden state) not [1, 200, 256]
    const lastHidden = outputTensor.dims.length === 3
      ? hidden.slice((MAX_LEN - 1) * d, MAX_LEN * d)
      : hidden.slice(0, d);

    const nItems = this.embShape[0];
    const scores = new Float32Array(nItems);
    for (let i = 0; i < nItems; i++) {
      let dot = 0;
      const offset = i * d;
      for (let j = 0; j < d; j++) {
        dot += this.embeddings[offset + j] * lastHidden[j];
      }
      scores[i] = dot;
    }

    scores[0] = -Infinity;
    for (const idx of seenSet) {
      if (idx >= 0 && idx < nItems) scores[idx] = -Infinity;
    }
    if (allowedMask) {
      for (let i = 0; i < nItems; i++) {
        if (!allowedMask[i]) scores[i] = -Infinity;
      }
    }

    const indices = Array.from({ length: nItems }, (_, i) => i);
    indices.sort((a, b) => scores[b] - scores[a]);

    const topK = [];
    for (let i = 0; i < Math.min(k, nItems); i++) {
      const idx = indices[i];
      if (scores[idx] === -Infinity) break;
      topK.push({ index: idx, score: scores[idx] });
    }

    const validScores = scores.filter(s => s > -Infinity);
    validScores.sort((a, b) => a - b);
    for (const item of topK) {
      const rank = validScores.filter(s => s < item.score).length;
      item.percentile = (rank / validScores.length) * 100;
    }

    return topK;
  }
}
