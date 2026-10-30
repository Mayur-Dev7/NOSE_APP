/**
 * Configuration and hardware acceleration profiles for On-Device ONNX Runtime.
 */

export const ONNX_EXECUTION_PROVIDERS = {
  NNAPI: {
    id: 'nnapi',
    label: 'Android NNAPI (NPU / DSP Acceleration)',
    description: 'Hardware accelerated via Qualcomm Hexagon, MediaTek APU, or Google TPU',
    useFp16: true,
    intraOpThreads: 4,
    recommendedFor: 'Production live camera verification',
  },
  XNNPACK: {
    id: 'xnnpack',
    label: 'ARM NEON SIMD (XNNPACK)',
    description: 'Multi-threaded SIMD acceleration on ARMv8/v9 CPU cores',
    useFp16: false,
    intraOpThreads: 4,
    recommendedFor: 'Devices with uncalibrated NNAPI HAL drivers',
  },
  DEFAULT_CPU: {
    id: 'cpu',
    label: 'Standard Multi-Core CPU',
    description: 'Universal fallback ensuring compatibility across all Android devices',
    useFp16: false,
    intraOpThreads: 2,
    recommendedFor: 'Legacy Android OS or virtualized environments',
  },
};

export const BIOMETRIC_MODEL_SPECS = {
  name: 'ConvNeXt-Tiny Muzzle Extractor',
  version: '1.0.0-quantized',
  inputShape: [1, 3, 224, 224],
  embeddingDim: 512,
  matchThreshold: 0.42, // Cosine distance threshold
  uncertaintyMargin: 0.05,
  assetPath: 'models/muometric_convnext_tiny_quantized.onnx',
};
