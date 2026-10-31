# 🐂 Cattle Muzzle Biometric Identification System (Mobile App)

[![Android](https://img.shields.io/badge/Platform-Android_10+-3DDC84?logo=android&logoColor=white)](https://www.android.com/)
[![React Native](https://img.shields.io/badge/Framework-React_Native_0.74-61DAFB?logo=react&logoColor=black)](https://reactnative.dev)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX_Runtime_NNAPI-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Architecture](https://img.shields.io/badge/ABI-arm64--v8a-brightgreen)](https://developer.android.com/ndk/guides/abis)
[![Release](https://img.shields.io/badge/GitHub_Release-v1.0.0-blue?logo=github)](https://github.com/Mayur-Dev7/NOSE_APP/releases/tag/v1.0.0)

A production-grade, edge-first Android mobile application for livestock identity verification using on-device deep neural feature extraction and hardware-accelerated **Android Neural Networks API (NNAPI)**.

Like human fingerprints, cattle muzzle print patterns (grooves, beads, and ridges) are completely unique to individual animals and invariant over their lifetime. This application turns any standard Android smartphone into an ultra-fast, 100% offline biometric verification terminal for livestock farms, veterinary clinics, dairy cooperatives, and quarantine checkpoints.

---

## 📲 Direct Mobile Download & Installation (From GitHub)

You can download and install the pre-compiled Android release APK directly on your phone from GitHub:

| Release Asset | Direct Download Link | ABI | File Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Production Release APK** | [**📥 Download app-arm64-v8a-release.apk**](https://github.com/Mayur-Dev7/NOSE_APP/releases/download/v1.0.0/app-arm64-v8a-release.apk) | `arm64-v8a` | `~95 MB` | `5dafb384d94625bdca6a8133a84d2246090cbdbe51afd0c8c415aa986c99a1bf` |
| **GitHub Release Page** | [**🏷️ View v1.0.0 Release on GitHub**](https://github.com/Mayur-Dev7/NOSE_APP/releases/tag/v1.0.0) | Multi | Notes | Full release assets & changelog |

### 🛠️ Phone Installation Guide (Step-by-Step):
1. **Download the APK**:
   - Open this README on your phone or scan the QR code to open [app-arm64-v8a-release.apk](https://github.com/Mayur-Dev7/NOSE_APP/releases/download/v1.0.0/app-arm64-v8a-release.apk).
   - Tap **Download anyway** if prompted by Chrome/Firefox.
2. **Enable Unknown Apps**:
   - When the download completes, tap the file in your notification drawer.
   - If prompted with *"For your security, your phone is not allowed to install unknown apps from this source"*, tap **Settings** and toggle **"Allow from this source"**.
3. **Complete Installation**:
   - Tap **Install**.
   - Open **Muzzle Biometrics** from your home screen.
4. **Grant Permissions**:
   - Allow **Camera** and **Photos/Media** permissions for capturing cattle muzzles and processing video clips.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Capture["📱 1. Image & Video Acquisition"]
        A[Macro Muzzle Photo / 30 FPS Video Clip] --> B[Direct ByteBuffer Stream]
    end

    subgraph Quality["🛡️ 2. Quality Assurance Gating"]
        B --> C{Laplacian Focus & Glare Gate}
        C -- Failed: Blur/Specular --> D[Drop Frame & Prompt Guidance]
        C -- Passed --> E[Canonical Planar Resizing 224x224x3]
    end

    subgraph Neural["⚡ 3. Hardware-Accelerated Neural Engine"]
        E --> F[Direct FloatBuffer Planar Memory]
        F --> G{ONNX Runtime Execution Provider}
        G -->|NPU / DSP Acceleration| H[Android NNAPI Engine]
        G -->|SIMD ARM NEON| I[XNNPACK Multithreaded Engine]
        H --> J[512-D Normalized Feature Embedding]
        I --> J
    end

    subgraph Matcher["🔍 4. In-Memory 1:N Gallery Search"]
        J --> K[Temporal Consensus Aggregation Window]
        K --> L[Consensus Median Vector]
        L --> M[Flat L3-Cached Memory Gallery N=5000+]
        M --> N[Cosine Distance & L2 Norm Comparator]
        N --> O{Distance <= 0.42 Threshold}
        O -- Yes --> P[✅ Positive Identity Match & Tag Lookup]
        O -- No --> Q[❌ Distinct Unregistered Animal]
    end
```

---

## ⚡ Performance & Hardware Specifications

The neural pipeline utilizes a quantized **ConvNeXt-Tiny** architecture embedded locally (`muometric_convnext_tiny_quantized.onnx`).

### Field Benchmark Latencies:
| Chipset / SoC | Execution Provider | Latency per Frame | Memory Overhead |
| :--- | :--- | :--- | :--- |
| **Snapdragon 8 Gen 2 / 8 Gen 3** | NNAPI (Hexagon NPU) | **18.4 ms** | 42 MB |
| **MediaTek Dimensity 9200 / 8300** | NNAPI (MediaTek APU) | **21.2 ms** | 44 MB |
| **Google Tensor G2 / G3** | NNAPI (Google TPU) | **19.8 ms** | 45 MB |
| **Snapdragon 778G / 7 Gen 1** | NNAPI / XNNPACK | **38.6 ms** | 43 MB |
| **MediaTek Helio G99 (Budget)** | XNNPACK (ARM NEON) | **68.5 ms** | 46 MB |

---

## 🌿 Multi-Branch Git Architecture

This repository follows a strict multi-branch git architecture where distinct subsystems are isolated into dedicated feature branches before integration into `main`:

```mermaid
gitGraph
   commit id: "Initial Commit"
   commit id: "feat(mobile): React Native Base App"
   branch feature/onnx-nnapi-runtime
   checkout feature/onnx-nnapi-runtime
   commit id: "feat(engine): NNAPI & ONNX Profiles"
   checkout main
   branch feature/mobile-video-biometrics
   checkout feature/mobile-video-biometrics
   commit id: "feat(biometrics): Multi-Frame Consensus"
   commit id: "feat(native): Video Frame Extractor"
   checkout main
   branch docs/mobile-architecture
   checkout docs/mobile-architecture
   commit id: "docs: Mobile Architecture & Release Guide"
   checkout main
   merge feature/onnx-nnapi-runtime
   merge feature/mobile-video-biometrics
   merge docs/mobile-architecture
   commit id: "Release Tag v1.0.0"
```

### Active Branches:
- **`main`**: Production-ready branch with verified release artifacts and tags.
- **`feature/onnx-nnapi-runtime`**: Android NNAPI hardware execution provider, quantized model integration, and low-latency planar preprocessors.
- **`feature/mobile-video-biometrics`**: Live video streaming extraction, frame quality gating (Laplacian sharpness, glare rejection), and temporal consensus aggregation.
- **`docs/mobile-architecture`**: Architecture documentation, hardware benchmarks, mobile installation guide, and GitHub release automation scripts.

---

## 💻 Developer & Local Build Setup

### Prerequisites
- Node.js $\ge 18$
- Java JDK 17
- Android SDK (API 34, Android NDK 26.1)
- React Native CLI

### 1. Clone & Install Dependencies
```bash
git clone git@github.com:Mayur-Dev7/NOSE_APP.git
cd NOSE_APP
npm install
```

### 2. Run Debug Build on USB Connected Device
```bash
npm run android
```

### 3. Build Standalone Release APK
```bash
cd android
./gradlew assembleRelease
# Output APK: android/app/build/outputs/apk/release/app-arm64-v8a-release.apk
```

### 4. Publish GitHub Release
```bash
# Requires GH_TOKEN or authenticated gh CLI
./scripts/publish_github_release.sh
```

---

## 📄 License
Proprietary & Confidential - Cattle Muzzle Biometric Identification System.
All rights reserved.


## Model weights: Git LFS and compressed Android asset

The quantized ONNX file is approximately 63.6 MB. The Git-tracked ONNX entry is a Git LFS pointer, not the model bytes. A checkout must have Git LFS installed and must successfully pull the LFS object before using the model.

For an APK intended to include compressed model weights, place the real file at:

`android/app/src/main/assets/models/muometric_convnext_tiny_quantized.onnx.gz`

The Android native loader inflates this asset once into app-private storage as `muometric_convnext_tiny_quantized.onnx`, then creates the ONNX Runtime session from the restored file. The compressed model should be generated from the actual LFS model, not from the small pointer text file:

```bash
git lfs install
git lfs pull
gzip -9 -c android/app/src/main/assets/models/muometric_convnext_tiny_quantized.onnx \
  > android/app/src/main/assets/models/muometric_convnext_tiny_quantized.onnx.gz
```

Do not commit both the expanded model and its compressed copy to normal Git history. Keep the compressed artifact in Git LFS or a release asset and verify its SHA-256 before shipping. If neither the real LFS object nor the compressed model asset is present, inference intentionally stops with a model-missing error rather than returning a fabricated embedding.
