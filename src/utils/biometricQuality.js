/**
 * Biometric quality assurance and multi-frame temporal consensus utilities.
 */

/**
 * Validates whether a frame meets minimum sharpness and exposure standards.
 * @param {Object} metrics - Frame statistics
 * @param {number} metrics.sharpness - Laplacian variance (focus indicator)
 * @param {number} metrics.glarePercentage - Percentage of saturated highlight pixels
 * @param {number} metrics.meanLuminance - Average grayscale intensity (0-255)
 * @returns {{ acceptable: boolean, reason?: string }}
 */
export function evaluateFrameQuality(metrics) {
  const MIN_SHARPNESS = 120.0;
  const MAX_GLARE_RATIO = 4.5;
  const MIN_LUMINANCE = 55;
  const MAX_LUMINANCE = 205;

  if (metrics.sharpness < MIN_SHARPNESS) {
    return {
      acceptable: false,
      reason: `Frame blurred (Sharpness ${metrics.sharpness.toFixed(1)} < ${MIN_SHARPNESS})`,
    };
  }

  if (metrics.glarePercentage > MAX_GLARE_RATIO) {
    return {
      acceptable: false,
      reason: `Excessive specular glare (${metrics.glarePercentage.toFixed(1)}% > ${MAX_GLARE_RATIO}%)`,
    };
  }

  if (metrics.meanLuminance < MIN_LUMINANCE || metrics.meanLuminance > MAX_LUMINANCE) {
    return {
      acceptable: false,
      reason: `Sub-optimal illumination (${metrics.meanLuminance.toFixed(0)} not in [${MIN_LUMINANCE}, ${MAX_LUMINANCE}])`,
    };
  }

  return { acceptable: true };
}

/**
 * Computes a confidence-weighted consensus embedding from a window of K embeddings.
 * @param {Array<{ vector: number[], weight: number }>} frameEmbeddings
 * @returns {number[]} L2-normalized 512-D consensus embedding
 */
export function computeTemporalConsensusEmbedding(frameEmbeddings) {
  if (!frameEmbeddings || frameEmbeddings.length === 0) {
    throw new Error('No frame embeddings provided for temporal consensus');
  }

  const dim = frameEmbeddings[0].vector.length;
  const consensus = new Float32Array(dim);
  let totalWeight = 0;

  for (const item of frameEmbeddings) {
    const w = Math.max(0.1, item.weight);
    totalWeight += w;
    for (let i = 0; i < dim; i++) {
      consensus[i] += item.vector[i] * w;
    }
  }

  // L2 normalize
  let sumSq = 0;
  for (let i = 0; i < dim; i++) {
    consensus[i] /= totalWeight;
    sumSq += consensus[i] * consensus[i];
  }

  const norm = Math.sqrt(sumSq) || 1e-9;
  const normalizedVector = new Array(dim);
  for (let i = 0; i < dim; i++) {
    normalizedVector[i] = consensus[i] / norm;
  }

  return normalizedVector;
}
