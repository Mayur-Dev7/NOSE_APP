#!/usr/bin/env python3
"""
export_mobile_onnx.py
=====================
Export and quantize the cattle biometric muzzle embedding model for mobile edge deployment.
Reproducibility script located in NOSE_MOBILE_APP/models.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Add nose_lines to sys.path
NOSE_LINES_ROOT = Path("/home/adis_mayur/F_CODES/nose_lines")
if str(NOSE_LINES_ROOT) not in sys.path:
    sys.path.insert(0, str(NOSE_LINES_ROOT))

import numpy as np
import torch
import onnx
from onnxconverter_common import float16
import onnxruntime as ort

from dl_muzzle_embedder import MuzzleBiometricModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ONNX-Export] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("export_mobile_onnx")


def export_and_quantize(
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    weights_path: str = None,
    target_dim: int = 512,
    opset_version: int = 17,
    output_dir: str = "/home/adis_mayur/F_CODES/NOSE_MOBILE_APP/models",
    android_assets_dir: str = "/home/adis_mayur/F_CODES/NOSE_MOBILE_APP/android/app/src/main/assets/models",
    fp32_filename: str = "muometric_convnext_tiny.onnx",
    quantized_filename: str = "muometric_convnext_tiny_quantized.onnx",
    keep_io_types: bool = True,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = out_path / fp32_filename
    quantized_onnx_path = out_path / quantized_filename

    logger.info("Initializing MuzzleBiometricModel...")
    logger.info(f"Backbone: {backbone}, Pretrained: {pretrained}, Target Dim: {target_dim}")
    model = MuzzleBiometricModel(
        backbone=backbone,
        pretrained=pretrained,
        weights_path=weights_path,
        target_dim=target_dim,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pt_sample_out = model(dummy_input)
    logger.info(f"PyTorch dummy inference successful. Output shape: {pt_sample_out.shape}, Norm: {torch.norm(pt_sample_out).item():.6f}")

    # Step 1: Export FP32 ONNX
    logger.info(f"Exporting FP32 ONNX to {fp32_onnx_path} (Opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["muzzle_input"],
        output_names=["biometric_embedding"],
        dynamic_axes={
            "muzzle_input": {0: "batch_size"},
            "biometric_embedding": {0: "batch_size"},
        },
    )

    # Validate ONNX graph structure
    onnx_fp32_model = onnx.load(str(fp32_onnx_path))
    onnx.checker.check_model(onnx_fp32_model)
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    logger.info(f"FP32 ONNX model exported and verified. Size: {fp32_size_mb:.2f} MB")

    # Step 2: Quantize to FP16
    logger.info(f"Quantizing to FP16 precision (keep_io_types={keep_io_types})...")
    onnx_fp16_model = float16.convert_float_to_float16(
        onnx_fp32_model,
        keep_io_types=keep_io_types,
    )
    onnx.save(onnx_fp16_model, str(quantized_onnx_path))
    quant_size_mb = os.path.getsize(quantized_onnx_path) / (1024 * 1024)
    logger.info(f"Quantized FP16 ONNX model saved to {quantized_onnx_path}. Size: {quant_size_mb:.2f} MB")

    # Step 3: Copy to Android Assets directory if specified
    if android_assets_dir:
        assets_path = Path(android_assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)
        android_target = assets_path / quantized_filename
        shutil.copy2(quantized_onnx_path, android_target)
        logger.info(f"Copied quantized model to Android assets: {android_target}")

    # Step 4: Parity and Validation Testing
    logger.info("--- Starting Parity and Validation Tests ---")
    np.random.seed(42)
    test_tensor_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    test_tensor_pt = torch.from_numpy(test_tensor_np)

    with torch.no_grad():
        pt_out = model(test_tensor_pt).cpu().numpy()[0]

    # Run FP32 ONNX inference
    session_fp32 = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    ort_fp32_out = session_fp32.run(None, {"muzzle_input": test_tensor_np})[0][0]

    # Run Quantized FP16 ONNX inference
    session_quant = ort.InferenceSession(str(quantized_onnx_path), providers=["CPUExecutionProvider"])
    input_quant = test_tensor_np if keep_io_types else test_tensor_np.astype(np.float16)
    ort_quant_out = session_quant.run(None, {"muzzle_input": input_quant})[0][0].astype(np.float32)

    # Calculate metrics
    cos_sim_fp32 = float(np.dot(pt_out, ort_fp32_out) / (np.linalg.norm(pt_out) * np.linalg.norm(ort_fp32_out)))
    cos_sim_quant = float(np.dot(pt_out, ort_quant_out) / (np.linalg.norm(pt_out) * np.linalg.norm(ort_quant_out)))
    max_diff_fp32 = float(np.max(np.abs(pt_out - ort_fp32_out)))
    max_diff_quant = float(np.max(np.abs(pt_out - ort_quant_out)))

    logger.info(f"PyTorch vs FP32 ONNX Cosine Similarity: {cos_sim_fp32:.8f} (Max Diff: {max_diff_fp32:.2e})")
    logger.info(f"PyTorch vs Quantized ONNX Cosine Similarity: {cos_sim_quant:.8f} (Max Diff: {max_diff_quant:.2e})")

    assert cos_sim_fp32 > 0.9999, f"FP32 Cosine similarity {cos_sim_fp32} < 0.9999"
    assert cos_sim_quant > 0.999, f"Quantized Cosine similarity {cos_sim_quant} < 0.999"

    # Step 5: Test dynamic batching
    logger.info("Testing dynamic batch inference (batch_size=4)...")
    batch_test_np = np.random.randn(4, 3, 224, 224).astype(np.float32)
    batch_input_quant = batch_test_np if keep_io_types else batch_test_np.astype(np.float16)
    batch_out = session_quant.run(None, {"muzzle_input": batch_input_quant})[0]
    logger.info(f"Dynamic batch output shape: {batch_out.shape}")
    assert batch_out.shape == (4, target_dim), f"Unexpected shape {batch_out.shape}"
    for i in range(4):
        norm = np.linalg.norm(batch_out[i])
        assert abs(norm - 1.0) < 1e-3, f"Batch item {i} not normalized: {norm}"

    logger.info("=== ONNX Export and Quantization Complete ===")
    logger.info(f"FP32 Model:      {fp32_onnx_path} ({fp32_size_mb:.2f} MB)")
    logger.info(f"Quantized Model: {quantized_onnx_path} ({quant_size_mb:.2f} MB)")
    if android_assets_dir:
        logger.info(f"Android Asset:   {android_target}")
    logger.info(f"Cosine Similarity (Parity): {cos_sim_quant:.6f} (> 0.999 threshold PASSED)")

    return {
        "fp32_path": str(fp32_onnx_path),
        "fp32_size_mb": fp32_size_mb,
        "quantized_path": str(quantized_onnx_path),
        "quantized_size_mb": quant_size_mb,
        "android_asset_path": str(android_target) if android_assets_dir else None,
        "cosine_similarity": cos_sim_quant,
    }


def main():
    parser = argparse.ArgumentParser(description="Export and quantize MuzzleBiometricModel to ONNX")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", type=str, default="/home/adis_mayur/F_CODES/NOSE_MOBILE_APP/models")
    parser.add_argument("--android-assets-dir", type=str, default="/home/adis_mayur/F_CODES/NOSE_MOBILE_APP/android/app/src/main/assets/models")
    parser.add_argument("--keep-io-types", action="store_true", default=True)
    args = parser.parse_args()

    export_and_quantize(
        backbone=args.backbone,
        pretrained=args.pretrained,
        weights_path=args.weights,
        target_dim=args.target_dim,
        opset_version=args.opset,
        output_dir=args.output_dir,
        android_assets_dir=args.android_assets_dir,
        keep_io_types=args.keep_io_types,
    )


if __name__ == "__main__":
    main()
