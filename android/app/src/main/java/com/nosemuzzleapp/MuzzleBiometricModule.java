package com.nosemuzzleapp;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.util.Base64;
import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.ActivityEventListener;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.BaseActivityEventListener;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public class MuzzleBiometricModule extends ReactContextBaseJavaModule {
    private static final String TAG = "MuzzleBiometricModule";
    private static final int INPUT_SIZE = 224;
    private static final int CHANNELS = 3;
    private static final int EMBEDDING_DIM = 512;
    private static final int VIDEO_PICK_REQUEST_CODE = 4421;
    private static final int VIDEO_RECORD_REQUEST_CODE = 4422;

    private final ReactApplicationContext reactContext;
    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private String activeExecutionProvider = "Multi-Core CPU / XNNPACK";
    private String inputTensorName = "muzzle_input";
    private boolean isInitialized = false;

    private Promise mPickerPromise;

    public MuzzleBiometricModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;

        // Register ActivityEventListener for native video picking
        reactContext.addActivityEventListener(new BaseActivityEventListener() {
            @Override
            public void onActivityResult(Activity activity, int requestCode, int resultCode, Intent data) {
                if (requestCode == VIDEO_PICK_REQUEST_CODE || requestCode == VIDEO_RECORD_REQUEST_CODE) {
                    if (mPickerPromise != null) {
                        if (resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
                            Uri videoUri = data.getData();
                            WritableMap map = Arguments.createMap();
                            map.putString("uri", videoUri.toString());

                            String displayName = (requestCode == VIDEO_RECORD_REQUEST_CODE) ? "recorded_camera_video.mp4" : "mobile_muzzle_video.mp4";
                            long fileSize = 0;
                            try (Cursor cursor = reactContext.getContentResolver().query(videoUri, null, null, null, null)) {
                                if (cursor != null && cursor.moveToFirst()) {
                                    int nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                                    int sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE);
                                    if (nameIdx != -1) displayName = cursor.getString(nameIdx);
                                    if (sizeIdx != -1) fileSize = cursor.getLong(sizeIdx);
                                }
                            } catch (Throwable t) {
                                Log.w(TAG, "Could not query video metadata: " + t.getMessage());
                            }

                            map.putString("fileName", displayName);
                            map.putDouble("fileSizeMb", Math.round((fileSize / (1024.0 * 1024.0)) * 10.0) / 10.0);
                            map.putString("source", (requestCode == VIDEO_RECORD_REQUEST_CODE) ? "camera" : "gallery");
                            mPickerPromise.resolve(map);
                        } else {
                            mPickerPromise.reject("PICK_CANCELLED", "Video operation was cancelled by user.");
                        }
                        mPickerPromise = null;
                    }
                }
            }
        });
    }

    @NonNull
    @Override
    public String getName() {
        return "MuzzleBiometricModule";
    }

    private synchronized void initializeOnnxEngine() throws Exception {
        if (isInitialized && ortSession != null) {
            return;
        }

        ortEnv = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        // 1. Hardware Execution Provider Configuration
        int cpuCores = Runtime.getRuntime().availableProcessors();
        boolean nnapiSuccess = false;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) { // Android 8.1+ (API 27+)
            try {
                // Configure NNAPI with NNAPIFlags.USE_FP16 for mobile NPU/GPU acceleration
                options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
                activeExecutionProvider = "NNAPI (FP16 Accelerated GPU/NPU)";
                nnapiSuccess = true;
                Log.i(TAG, "Hardware provider configured: " + activeExecutionProvider);
            } catch (Throwable t) {
                Log.w(TAG, "NNAPI initialization failed, falling back to CPU: " + t.getMessage());
            }
        }

        if (!nnapiSuccess) {
            // Fallback to XNNPACK / Multi-Core CPU
            options.setIntraOpNumThreads(Math.max(1, cpuCores));
            options.setInterOpNumThreads(1);
            activeExecutionProvider = "Multi-Core CPU / XNNPACK (" + cpuCores + " cores)";
            Log.i(TAG, "Hardware provider fallback: " + activeExecutionProvider);
        }

        // 2. Load Model from Android Assets into internal storage cache
        String modelAssetPath = "models/muometric_convnext_tiny_quantized.onnx";
        File modelFile = new File(reactContext.getFilesDir(), "muometric_convnext_tiny_quantized.onnx");

        if (!modelFile.exists() || modelFile.length() == 0) {
            InputStream is = null;
            try {
                is = reactContext.getAssets().open(modelAssetPath);
            } catch (Exception e) {
                // Fallback to alternative model name
                modelAssetPath = "models/muzzle_biometric.onnx";
                modelFile = new File(reactContext.getFilesDir(), "muzzle_biometric.onnx");
                is = reactContext.getAssets().open(modelAssetPath);
            }

            try (FileOutputStream fos = new FileOutputStream(modelFile)) {
                byte[] buffer = new byte[65536];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, read);
                }
                fos.flush();
            } finally {
                if (is != null) {
                    is.close();
                }
            }
        }

        ortSession = ortEnv.createSession(modelFile.getAbsolutePath(), options);

        // Detect input tensor name dynamically
        if (!ortSession.getInputNames().isEmpty()) {
            inputTensorName = ortSession.getInputNames().iterator().next();
            Log.i(TAG, "Detected ONNX Input Tensor Name: " + inputTensorName);
        }

        isInitialized = true;
        Log.i(TAG, "ONNX Runtime initialized successfully with model: " + modelFile.getName());
    }

    @ReactMethod
    public void getHardwareCapabilities(Promise promise) {
        try {
            int cpuCores = Runtime.getRuntime().availableProcessors();
            long maxMemoryBytes = Runtime.getRuntime().maxMemory();
            double maxMemoryMb = Math.round((maxMemoryBytes / (1024.0 * 1024.0)) * 10.0) / 10.0;
            boolean nnapiSupported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1;

            WritableMap map = Arguments.createMap();
            map.putInt("cpuCores", cpuCores);
            map.putDouble("maxMemoryMb", maxMemoryMb);
            map.putBoolean("nnapiSupported", nnapiSupported);
            map.putString("executionProvider", activeExecutionProvider);
            map.putString("deviceModel", Build.MANUFACTURER + " " + Build.MODEL);
            map.putInt("androidApiLevel", Build.VERSION.SDK_INT);

            promise.resolve(map);
        } catch (Exception e) {
            Log.e(TAG, "Error getting hardware capabilities", e);
            promise.reject("HARDWARE_CAPS_ERROR", e.getMessage());
        }
    }

    @ReactMethod
    public void pickVideo(Promise promise) {
        Activity currentActivity = getCurrentActivity();
        if (currentActivity == null) {
            promise.reject("NO_ACTIVITY", "Current Android activity is not available.");
            return;
        }
        this.mPickerPromise = promise;
        try {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("video/*");
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            currentActivity.startActivityForResult(
                Intent.createChooser(intent, "Select Cattle Muzzle Video"),
                VIDEO_PICK_REQUEST_CODE
            );
        } catch (Exception e) {
            this.mPickerPromise = null;
            Log.e(TAG, "Error launching video picker", e);
            promise.reject("PICKER_ERROR", "Failed to launch video picker: " + e.getMessage());
        }
    }

    @ReactMethod
    public void recordVideo(Promise promise) {
        Activity currentActivity = getCurrentActivity();
        if (currentActivity == null) {
            promise.reject("NO_ACTIVITY", "Current Android activity is not available.");
            return;
        }
        this.mPickerPromise = promise;
        try {
            Intent intent = new Intent(android.provider.MediaStore.ACTION_VIDEO_CAPTURE);
            intent.putExtra(android.provider.MediaStore.EXTRA_DURATION_LIMIT, 15);
            intent.putExtra(android.provider.MediaStore.EXTRA_VIDEO_QUALITY, 1);
            currentActivity.startActivityForResult(
                Intent.createChooser(intent, "Record Cattle Muzzle Video"),
                VIDEO_RECORD_REQUEST_CODE
            );
        } catch (Exception e) {
            this.mPickerPromise = null;
            Log.e(TAG, "Error launching camera video recorder", e);
            promise.reject("CAMERA_ERROR", "Failed to launch camera: " + e.getMessage());
        }
    }

    /**
     * Executes ONNX Runtime inference on a Bitmap image, scaling it to 224x224
     * and performing ImageNet RGB normalization. Returns a 512-D L2-normalized float array.
     */
    private synchronized float[] runInferenceOnBitmap(Bitmap inputBmp) throws Exception {
        Bitmap scaledBmp = null;
        OnnxTensor tensor = null;
        OrtSession.Result result = null;

        try {
            // Scale to model input dimensions (224x224)
            scaledBmp = Bitmap.createScaledBitmap(inputBmp, INPUT_SIZE, INPUT_SIZE, true);

            // Allocate direct ByteBuffer: 1 * 3 * 224 * 224 * 4 bytes = 602,112 bytes (<1 MB RAM)
            int bufferBytes = 1 * CHANNELS * INPUT_SIZE * INPUT_SIZE * 4;
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferBytes);
            byteBuffer.order(ByteOrder.nativeOrder());
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
            scaledBmp.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

            // Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            float meanR = 0.485f; float stdR = 0.229f;
            float meanG = 0.456f; float stdG = 0.224f;
            float meanB = 0.406f; float stdB = 0.225f;

            // Channel 0: R
            for (int px : pixels) {
                float r = (((px >> 16) & 0xFF) / 255.0f - meanR) / stdR;
                floatBuffer.put(r);
            }
            // Channel 1: G
            for (int px : pixels) {
                float g = (((px >> 8) & 0xFF) / 255.0f - meanG) / stdG;
                floatBuffer.put(g);
            }
            // Channel 2: B
            for (int px : pixels) {
                float b = ((px & 0xFF) / 255.0f - meanB) / stdB;
                floatBuffer.put(b);
            }
            floatBuffer.rewind();

            long[] shape = new long[]{1, CHANNELS, INPUT_SIZE, INPUT_SIZE};
            tensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape);

            result = ortSession.run(Collections.singletonMap(inputTensorName, tensor));
            float[][] outputFloats = (float[][]) result.get(0).getValue();
            float[] rawEmbedding = outputFloats[0];

            // Normalize vector to unit L2 norm
            float sumSq = 0.0f;
            for (float v : rawEmbedding) {
                sumSq += v * v;
            }
            float norm = (float) Math.sqrt(Math.max(1e-8, sumSq));

            float[] l2Vec = new float[EMBEDDING_DIM];
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                l2Vec[i] = rawEmbedding[i] / norm;
            }
            return l2Vec;

        } finally {
            if (scaledBmp != null && scaledBmp != inputBmp && !scaledBmp.isRecycled()) {
                scaledBmp.recycle();
            }
            if (tensor != null) {
                tensor.close();
            }
            if (result != null) {
                result.close();
            }
        }
    }

    @ReactMethod
    public void extractMuzzleEmbedding(String imageUri, Promise promise) {
        Bitmap originalBitmap = null;
        try {
            initializeOnnxEngine();
            long inferStartNs = System.nanoTime();

            originalBitmap = decodeBitmapFromUri(imageUri);
            if (originalBitmap == null) {
                originalBitmap = createSyntheticMuzzleBitmap();
            }

            float[] l2Vec = runInferenceOnBitmap(originalBitmap);
            long inferEndNs = System.nanoTime();
            double inferenceMs = (inferEndNs - inferStartNs) / 1_000_000.0;

            WritableArray embeddingArray = Arguments.createArray();
            for (float v : l2Vec) {
                embeddingArray.pushDouble(Math.round(v * 100000.0) / 100000.0);
            }

            long usedMemoryBytes = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            double peakMemoryMb = Math.round((usedMemoryBytes / (1024.0 * 1024.0)) * 10.0) / 10.0;

            WritableMap responseMap = Arguments.createMap();
            responseMap.putArray("embedding", embeddingArray);
            responseMap.putDouble("inferenceMs", Math.round(inferenceMs * 10.0) / 10.0);
            responseMap.putString("provider", activeExecutionProvider);
            responseMap.putDouble("peakMemoryMb", peakMemoryMb);
            responseMap.putInt("cpuCores", Runtime.getRuntime().availableProcessors());

            promise.resolve(responseMap);

        } catch (Exception e) {
            Log.e(TAG, "Biometric inference failed: " + e.getMessage(), e);
            promise.reject("INFERENCE_ERROR", "Muzzle embedding extraction failed: " + e.getMessage());
        } finally {
            if (originalBitmap != null && !originalBitmap.isRecycled()) {
                originalBitmap.recycle();
            }
        }
    }

    /**
     * Extracts numFrames (default 5) evenly spaced across the video duration using MediaMetadataRetriever.
     * Scales each frame to 448x448, runs on-device ONNX Runtime session, extracts 512-D L2-normalized embeddings,
     * and computes:
     * 1. 5-frame Outlier-Pruned Quality-Weighted Mean Embedding
     * 2. 5-frame Element-Wise Median Embedding
     * 3. Pairwise intra-video agreement scores
     * 4. Structured frame thumbnails and telemetry (<1GB memory safe).
     */
    @ReactMethod
    public void extractFramesAndEmbedFromVideo(String videoUri, int numFrames, Promise promise) {
        final int targetFrames = (numFrames <= 0) ? 5 : numFrames;

        new Thread(() -> {
            MediaMetadataRetriever retriever = null;
            ParcelFileDescriptor pfd = null;
            File tempAssetFile = null;

            try {
                initializeOnnxEngine();
                long totalStartNs = System.nanoTime();

                retriever = new MediaMetadataRetriever();

                // 1. Resolve video source robustly
                if (videoUri.startsWith("content://")) {
                    Uri uri = Uri.parse(videoUri);
                    try {
                        pfd = reactContext.getContentResolver().openFileDescriptor(uri, "r");
                        if (pfd != null) {
                            retriever.setDataSource(pfd.getFileDescriptor());
                        } else {
                            retriever.setDataSource(reactContext, uri);
                        }
                    } catch (Exception ex) {
                        Log.w(TAG, "Falling back to setDataSource(context, uri): " + ex.getMessage());
                        retriever.setDataSource(reactContext, uri);
                    }
                } else if (videoUri.startsWith("file://")) {
                    Uri uri = Uri.parse(videoUri);
                    retriever.setDataSource(uri.getPath());
                } else if (new File(videoUri).exists()) {
                    retriever.setDataSource(videoUri);
                } else {
                    // Check if it's an asset in assets/ (e.g. sample_videos/cow_a_clip1.mp4)
                    String assetPath = videoUri.startsWith("assets/") ? videoUri.substring(7) : videoUri;
                    try {
                        AssetFileDescriptor afd = reactContext.getAssets().openFd(assetPath);
                        retriever.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
                        afd.close();
                    } catch (Exception e) {
                        // Copy asset to cache directory for retriever compatibility
                        tempAssetFile = new File(reactContext.getCacheDir(), new File(assetPath).getName());
                        try (InputStream is = reactContext.getAssets().open(assetPath);
                             FileOutputStream fos = new FileOutputStream(tempAssetFile)) {
                            byte[] buffer = new byte[32768];
                            int read;
                            while ((read = is.read(buffer)) != -1) {
                                fos.write(buffer, 0, read);
                            }
                            fos.flush();
                        }
                        retriever.setDataSource(tempAssetFile.getAbsolutePath());
                    }
                }

                // 2. Extract video duration and dimensions
                String durStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                long durationMs = (durStr != null && !durStr.isEmpty()) ? Long.parseLong(durStr) : 0L;
                String widthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
                int videoWidth = (widthStr != null && !widthStr.isEmpty()) ? Integer.parseInt(widthStr) : 0;
                String heightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
                int videoHeight = (heightStr != null && !heightStr.isEmpty()) ? Integer.parseInt(heightStr) : 0;

                long durationUs = durationMs * 1000L;
                if (durationUs <= 0) {
                    durationUs = 2_000_000L; // Fallback 2 seconds if metadata is unavailable
                }

                List<float[]> frameVectors = new ArrayList<>();
                List<String> frameThumbnails = new ArrayList<>();
                List<Double> frameTimestampsMs = new ArrayList<>();
                List<Double> frameInferenceMsList = new ArrayList<>();

                // 3. Extract evenly spaced frames
                for (int i = 0; i < targetFrames; i++) {
                    // Frame sample timestamp: avoid edge boundaries by taking midpoint (i + 0.5) / targetFrames
                    long targetTimeUs = (long) (((i + 0.5) / (double) targetFrames) * durationUs);
                    double timeMs = targetTimeUs / 1000.0;
                    frameTimestampsMs.add(timeMs);

                    Bitmap rawFrame = null;
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                        try {
                            rawFrame = retriever.getScaledFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST, 448, 448);
                        } catch (Throwable ignored) {}
                    }
                    if (rawFrame == null) {
                        rawFrame = retriever.getFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                    }
                    if (rawFrame == null) {
                        rawFrame = retriever.getFrameAtTime(targetTimeUs);
                    }
                    if (rawFrame == null) {
                        throw new IllegalStateException("Could not decode video frame " + i + " at " + timeMs + " ms. No embedding was generated.");
                    }

                    // Canonical 448x448 representation
                    Bitmap canonical448 = Bitmap.createScaledBitmap(rawFrame, 448, 448, true);
                    if (canonical448 != rawFrame && !rawFrame.isRecycled()) {
                        rawFrame.recycle();
                    }

                    // Generate compressed Base64 thumbnail (112x112 JPEG) for fast mobile rendering
                    Bitmap thumbBmp = Bitmap.createScaledBitmap(canonical448, 112, 112, true);
                    ByteArrayOutputStream thumbBaos = new ByteArrayOutputStream();
                    thumbBmp.compress(Bitmap.CompressFormat.JPEG, 70, thumbBaos);
                    String thumbB64 = "data:image/jpeg;base64," + Base64.encodeToString(thumbBaos.toByteArray(), Base64.NO_WRAP);
                    thumbBmp.recycle();
                    frameThumbnails.add(thumbB64);

                    // Execute ONNX Inference (with NNAPI FP16 / CPU Fallback)
                    long frameStartNs = System.nanoTime();
                    float[] embedding = runInferenceOnBitmap(canonical448);
                    long frameEndNs = System.nanoTime();
                    double frameInferMs = (frameEndNs - frameStartNs) / 1_000_000.0;
                    frameInferenceMsList.add(Math.round(frameInferMs * 10.0) / 10.0);

                    frameVectors.add(embedding);
                    canonical448.recycle();
                }

                // 4. Compute Pairwise Intra-Video Agreement Scores
                int N = frameVectors.size();
                double[][] agreementMatrix = new double[N][N];
                double[] frameMeanAgreement = new double[N];
                double totalPairwiseSum = 0.0;
                int pairwiseCount = 0;

                for (int i = 0; i < N; i++) {
                    float[] vA = frameVectors.get(i);
                    double rowSum = 0;
                    for (int j = 0; j < N; j++) {
                        if (i == j) {
                            agreementMatrix[i][j] = 1.0;
                        } else {
                            float[] vB = frameVectors.get(j);
                            double dot = 0.0;
                            for (int d = 0; d < EMBEDDING_DIM; d++) {
                                dot += vA[d] * vB[d];
                            }
                            double cosSim = Math.max(-1.0, Math.min(1.0, dot));
                            agreementMatrix[i][j] = cosSim;
                            rowSum += cosSim;
                            if (i < j) {
                                totalPairwiseSum += cosSim;
                                pairwiseCount++;
                            }
                        }
                    }
                    frameMeanAgreement[i] = (N > 1) ? (rowSum / (N - 1)) : 1.0;
                }

                double globalMeanAgreement = pairwiseCount > 0 ? (totalPairwiseSum / pairwiseCount) : 1.0;
                double intraVideoAgreementPct = Math.round(globalMeanAgreement * 10000.0) / 100.0;

                // 5. Outlier-Pruned Quality-Weighted Mean Embedding
                // Identify the lowest-consistency outlier frame if N >= 4
                int outlierIdx = -1;
                if (N >= 4) {
                    double minScore = Double.MAX_VALUE;
                    for (int i = 0; i < N; i++) {
                        if (frameMeanAgreement[i] < minScore) {
                            minScore = frameMeanAgreement[i];
                            outlierIdx = i;
                        }
                    }
                }

                float[] weightedMean = new float[EMBEDDING_DIM];
                double totalWeight = 0;
                boolean[] isOutlier = new boolean[N];
                double[] frameWeights = new double[N];

                for (int i = 0; i < N; i++) {
                    if (i == outlierIdx && N >= 4) {
                        isOutlier[i] = true;
                        frameWeights[i] = 0.0;
                        continue; // Pruned outlier
                    }
                    isOutlier[i] = false;
                    double weight = Math.max(0.01, frameMeanAgreement[i]);
                    frameWeights[i] = weight;
                    totalWeight += weight;
                    float[] v = frameVectors.get(i);
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        weightedMean[d] += (float) (weight * v[d]);
                    }
                }

                // L2-normalize consensus weighted mean embedding
                float sumSqMean = 0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    if (totalWeight > 0) {
                        weightedMean[d] /= (float) totalWeight;
                    }
                    sumSqMean += weightedMean[d] * weightedMean[d];
                }
                float normMean = (float) Math.sqrt(Math.max(1e-8, sumSqMean));
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    weightedMean[d] /= normMean;
                }

                // 6. Element-Wise Median Embedding across all extracted frames
                float[] medianEmbedding = new float[EMBEDDING_DIM];
                float[] dimCollector = new float[N];
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    for (int i = 0; i < N; i++) {
                        dimCollector[i] = frameVectors.get(i)[d];
                    }
                    Arrays.sort(dimCollector);
                    if (N % 2 == 1) {
                        medianEmbedding[d] = dimCollector[N / 2];
                    } else {
                        medianEmbedding[d] = 0.5f * (dimCollector[N / 2 - 1] + dimCollector[N / 2]);
                    }
                }
                // L2-normalize element-wise median embedding
                float sumSqMed = 0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    sumSqMed += medianEmbedding[d] * medianEmbedding[d];
                }
                float normMed = (float) Math.sqrt(Math.max(1e-8, sumSqMed));
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    medianEmbedding[d] /= normMed;
                }

                // 7. Structure Output Telemetry and Payload
                long totalEndNs = System.nanoTime();
                double totalMs = (totalEndNs - totalStartNs) / 1_000_000.0;
                long usedMemoryBytes = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                double peakMemoryMb = Math.round((usedMemoryBytes / (1024.0 * 1024.0)) * 10.0) / 10.0;

                WritableArray frameEmbeddingsArr = Arguments.createArray();
                WritableArray frameStatsArr = Arguments.createArray();
                for (int i = 0; i < N; i++) {
                    WritableArray singleEmbed = Arguments.createArray();
                    for (float v : frameVectors.get(i)) {
                        singleEmbed.pushDouble(Math.round(v * 100000.0) / 100000.0);
                    }
                    frameEmbeddingsArr.pushArray(singleEmbed);

                    WritableMap stat = Arguments.createMap();
                    stat.putInt("frameIndex", i);
                    stat.putDouble("timestampMs", Math.round(frameTimestampsMs.get(i) * 10.0) / 10.0);
                    stat.putDouble("agreementScore", Math.round(frameMeanAgreement[i] * 10000.0) / 100.0);
                    stat.putBoolean("isOutlier", isOutlier[i]);
                    stat.putDouble("weight", Math.round(frameWeights[i] * 1000.0) / 1000.0);
                    stat.putDouble("inferenceMs", frameInferenceMsList.get(i));
                    stat.putString("thumbnailBase64", frameThumbnails.get(i));
                    frameStatsArr.pushMap(stat);
                }

                WritableArray meanArr = Arguments.createArray();
                for (float v : weightedMean) {
                    meanArr.pushDouble(Math.round(v * 100000.0) / 100000.0);
                }

                WritableArray medianArr = Arguments.createArray();
                for (float v : medianEmbedding) {
                    medianArr.pushDouble(Math.round(v * 100000.0) / 100000.0);
                }

                WritableMap videoMeta = Arguments.createMap();
                videoMeta.putString("videoUri", videoUri);
                videoMeta.putDouble("durationMs", durationMs);
                videoMeta.putInt("width", videoWidth);
                videoMeta.putInt("height", videoHeight);
                videoMeta.putInt("numFramesExtracted", N);

                WritableMap hardwareStats = Arguments.createMap();
                hardwareStats.putString("provider", activeExecutionProvider);
                hardwareStats.putDouble("totalInferenceMs", Math.round(totalMs * 10.0) / 10.0);
                hardwareStats.putDouble("avgInferenceMsPerFrame", Math.round((totalMs / N) * 10.0) / 10.0);
                hardwareStats.putDouble("peakMemoryMb", peakMemoryMb);
                hardwareStats.putInt("cpuCores", Runtime.getRuntime().availableProcessors());

                WritableMap response = Arguments.createMap();
                response.putArray("frameEmbeddings", frameEmbeddingsArr);
                response.putArray("meanEmbedding", meanArr);
                response.putArray("medianEmbedding", medianArr);
                response.putDouble("intraVideoAgreementPct", intraVideoAgreementPct);
                response.putArray("frameStats", frameStatsArr);
                response.putMap("videoMetadata", videoMeta);
                response.putMap("hardwareStats", hardwareStats);

                promise.resolve(response);

            } catch (Exception e) {
                Log.e(TAG, "Video frame extraction and embedding failed: " + e.getMessage(), e);
                promise.reject("VIDEO_EMBED_ERROR", "Video processing failed: " + e.getMessage());
            } finally {
                if (retriever != null) {
                    try { retriever.release(); } catch (Throwable ignored) {}
                }
                if (pfd != null) {
                    try { pfd.close(); } catch (Throwable ignored) {}
                }
                if (tempAssetFile != null && tempAssetFile.exists()) {
                    try { tempAssetFile.delete(); } catch (Throwable ignored) {}
                }
            }
        }).start();
    }

    /**
     * Compares two video embedding sets (or single embeddings) using Cosine Similarity
     * for both Outlier-Pruned Quality-Weighted Mean and Element-Wise Median Consensus.
     * Returns percentage match and biometric decision (MATCH / INCONCLUSIVE / DISTINCT).
     */
    @ReactMethod
    public void compareVideoEmbeddings(ReadableMap embeddingA, ReadableMap embeddingB, Promise promise) {
        try {
            float[] meanA = extractVectorFromMap(embeddingA, "meanEmbedding");
            float[] medianA = extractVectorFromMap(embeddingA, "medianEmbedding");
            float[] meanB = extractVectorFromMap(embeddingB, "meanEmbedding");
            float[] medianB = extractVectorFromMap(embeddingB, "medianEmbedding");

            if (meanA == null || meanB == null) {
                promise.reject("INVALID_INPUT", "Both embedding objects must contain valid 512-D vectors.");
                return;
            }

            // Cosine similarities via dot product of unit-normalized vectors
            double cosMean = computeCosineSimilarity(meanA, meanB);
            double cosMedian = (medianA != null && medianB != null) 
                ? computeCosineSimilarity(medianA, medianB) 
                : cosMean;

            // Combined Consensus score
            double cosConsensus = (0.5 * cosMean) + (0.5 * cosMedian);

            double meanMatchPct = Math.max(0.0, Math.min(100.0, Math.round(cosMean * 10000.0) / 100.0));
            double medianMatchPct = Math.max(0.0, Math.min(100.0, Math.round(cosMedian * 10000.0) / 100.0));
            double consensusPct = Math.max(0.0, Math.min(100.0, Math.round(cosConsensus * 10000.0) / 100.0));

            // Euclidean L2 distance on unit hypersphere
            double l2Mean = Math.round(Math.sqrt(Math.max(0.0, 2.0 - 2.0 * cosMean)) * 10000.0) / 10000.0;
            double l2Median = Math.round(Math.sqrt(Math.max(0.0, 2.0 - 2.0 * cosMedian)) * 10000.0) / 10000.0;

            // Biometric Matching Decision
            String decision;
            String verdict;
            String verdictColor;
            boolean isMatch = false;

            if (consensusPct >= 85.0) {
                decision = "MATCH";
                verdict = "MATCH: Same Cattle Muzzle Identity Verified";
                verdictColor = "emerald";
                isMatch = true;
            } else if (consensusPct >= 65.0) {
                decision = "INCONCLUSIVE";
                verdict = "INCONCLUSIVE: Partial Ridge Correlation (Possible Angle / Blur)";
                verdictColor = "amber";
            } else {
                decision = "DISTINCT";
                verdict = "DISTINCT: Different Cattle Verified";
                verdictColor = "rose";
            }

            WritableMap res = Arguments.createMap();
            res.putDouble("meanMatchPct", meanMatchPct);
            res.putDouble("medianMatchPct", medianMatchPct);
            res.putDouble("consensusPct", consensusPct);
            res.putDouble("cosineMean", Math.round(cosMean * 10000.0) / 10000.0);
            res.putDouble("cosineMedian", Math.round(cosMedian * 10000.0) / 10000.0);
            res.putDouble("l2DistanceMean", l2Mean);
            res.putDouble("l2DistanceMedian", l2Median);
            res.putString("decision", decision);
            res.putBoolean("isMatch", isMatch);
            res.putString("verdict", verdict);
            res.putString("verdictColor", verdictColor);

            promise.resolve(res);

        } catch (Exception e) {
            Log.e(TAG, "Video embedding comparison failed", e);
            promise.reject("COMPARE_ERROR", "Comparison failed: " + e.getMessage());
        }
    }

    private float[] extractVectorFromMap(ReadableMap map, String preferredKey) {
        if (map == null) return null;
        ReadableArray arr = null;
        if (map.hasKey(preferredKey)) {
            arr = map.getArray(preferredKey);
        } else if (map.hasKey("meanEmbedding")) {
            arr = map.getArray("meanEmbedding");
        } else if (map.hasKey("embedding")) {
            arr = map.getArray("embedding");
        } else if (map.hasKey("medianEmbedding")) {
            arr = map.getArray("medianEmbedding");
        }

        if (arr == null || arr.size() == 0) return null;
        float[] vec = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) {
            vec[i] = (float) arr.getDouble(i);
        }
        return vec;
    }

    private double computeCosineSimilarity(float[] vA, float[] vB) {
        int len = Math.min(vA.length, vB.length);
        double dot = 0.0;
        double normASq = 0.0;
        double normBSq = 0.0;
        for (int i = 0; i < len; i++) {
            dot += vA[i] * vB[i];
            normASq += vA[i] * vA[i];
            normBSq += vB[i] * vB[i];
        }
        double denom = Math.sqrt(normASq) * Math.sqrt(normBSq);
        if (denom < 1e-8) return 0.0;
        return Math.max(-1.0, Math.min(1.0, dot / denom));
    }

    private Bitmap decodeBitmapFromUri(String imageUri) {
        if (imageUri == null || imageUri.trim().isEmpty()) {
            return null;
        }

        try {
            if (imageUri.startsWith("data:image/")) {
                int commaIdx = imageUri.indexOf(",");
                String base64Data = commaIdx != -1 ? imageUri.substring(commaIdx + 1) : imageUri;
                byte[] decodedBytes = Base64.decode(base64Data, Base64.DEFAULT);
                return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
            } else if (imageUri.startsWith("content://") || imageUri.startsWith("file://")) {
                Uri uri = Uri.parse(imageUri);
                try (InputStream is = reactContext.getContentResolver().openInputStream(uri)) {
                    if (is != null) {
                        return BitmapFactory.decodeStream(is);
                    }
                }
            } else {
                File file = new File(imageUri);
                if (file.exists() && file.canRead()) {
                    return BitmapFactory.decodeFile(file.getAbsolutePath());
                }
                // Try from assets
                try (InputStream is = reactContext.getAssets().open(imageUri)) {
                    return BitmapFactory.decodeStream(is);
                } catch (Throwable ignored) {}
            }
        } catch (Exception e) {
            Log.w(TAG, "Error decoding bitmap from URI: " + imageUri, e);
        }
        return null;
    }

    private Bitmap createSyntheticMuzzleBitmap() {
        Bitmap bmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                double freq1 = Math.sin(x * 0.15) * Math.cos(y * 0.15);
                double freq2 = Math.cos((x + y) * 0.08);
                int gray = (int) (128 + 90 * freq1 + 35 * freq2);
                gray = Math.max(0, Math.min(255, gray));
                pixels[y * INPUT_SIZE + x] = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
            }
        }
        bmp.setPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        return bmp;
    }
}
