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
import java.util.zip.GZIPInputStream;

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

    public static class RectInt {
        public int x1, y1, x2, y2;
        public RectInt(int x1, int y1, int x2, int y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
    }

    private static class CandidateFrame {
        final int index;
        final long timeUs;
        final double timeMs;
        final double lapVar;
        final Bitmap bitmap;

        CandidateFrame(int index, long timeUs, double timeMs, double lapVar, Bitmap bitmap) {
            this.index = index;
            this.timeUs = timeUs;
            this.timeMs = timeMs;
            this.lapVar = lapVar;
            this.bitmap = bitmap;
        }
    }

    private static class DiskKernel {
        final int[] offX;
        final int[] offY;
        DiskKernel(int radius) {
            List<Integer> xs = new ArrayList<>();
            List<Integer> ys = new ArrayList<>();
            int r2 = radius * radius;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx * dx + dy * dy <= r2) {
                        xs.add(dx);
                        ys.add(dy);
                    }
                }
            }
            offX = new int[xs.size()];
            offY = new int[ys.size()];
            for (int i = 0; i < xs.size(); i++) {
                offX[i] = xs.get(i);
                offY[i] = ys.get(i);
            }
        }
    }

    private static final DiskKernel KERNEL_R2 = new DiskKernel(2);
    private static final DiskKernel KERNEL_R4 = new DiskKernel(4);
    private static final DiskKernel KERNEL_R6 = new DiskKernel(6);

    public MuzzleBiometricModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;

        // Register ActivityEventListener for native video picking & recording
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
        int cpuCores = Runtime.getRuntime().availableProcessors();
        boolean nnapiConfigured = false;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            try {
                options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
                activeExecutionProvider = "ONNX Runtime NNAPI EP (FP16 enabled)";
                nnapiConfigured = true;
                Log.i(TAG, "NNAPI Execution Provider configured");
            } catch (Throwable t) {
                Log.w(TAG, "NNAPI setup failed; using CPU fallback", t);
            }
        }
        if (!nnapiConfigured) {
            options.setIntraOpNumThreads(Math.max(1, cpuCores));
            options.setInterOpNumThreads(1);
            activeExecutionProvider = "ONNX Runtime CPU (" + cpuCores + " threads)";
        }

        File modelFile = new File(reactContext.getFilesDir(), "muometric_convnext_tiny_quantized.onnx");
        if (!modelFile.exists() || modelFile.length() < 1024 * 1024) {
            File partialModel = new File(reactContext.getFilesDir(), modelFile.getName() + ".partial");
            boolean restored = false;

            // Preferred packaging: decompressed from models/muometric_convnext_tiny_quantized.onnx.gz
            try (InputStream asset = reactContext.getAssets().open("models/muometric_convnext_tiny_quantized.onnx.gz");
                 GZIPInputStream gzip = new GZIPInputStream(asset);
                 FileOutputStream output = new FileOutputStream(partialModel)) {
                copyStream(gzip, output);
                output.getFD().sync();
                restored = true;
                Log.i(TAG, "Restored compressed ONNX model into private app storage");
            } catch (Exception compressedError) {
                partialModel.delete();
                Log.i(TAG, "Compressed ONNX model asset unavailable; trying uncompressed asset");
            }

            if (!restored) {
                try (InputStream asset = reactContext.getAssets().open("models/muometric_convnext_tiny_quantized.onnx");
                     FileOutputStream output = new FileOutputStream(partialModel)) {
                    copyStream(asset, output);
                    output.getFD().sync();
                    restored = true;
                } catch (Exception assetError) {
                    partialModel.delete();
                    throw new IllegalStateException(
                        "ONNX model weights are missing. The model asset was not found in assets/models.",
                        assetError
                    );
                }
            }

            if (partialModel.length() < 1024 * 1024) {
                long invalidBytes = partialModel.length();
                partialModel.delete();
                throw new IllegalStateException("Restored ONNX model is invalid (" + invalidBytes + " bytes).");
            }
            if (modelFile.exists() && !modelFile.delete()) {
                partialModel.delete();
                throw new IllegalStateException("Cannot replace the old ONNX model cache.");
            }
            if (!partialModel.renameTo(modelFile)) {
                partialModel.delete();
                throw new IllegalStateException("Could not finalize the decompressed ONNX model.");
            }
        }

        ortSession = ortEnv.createSession(modelFile.getAbsolutePath(), options);
        if (!ortSession.getInputNames().isEmpty()) {
            inputTensorName = ortSession.getInputNames().iterator().next();
        }
        isInitialized = true;
        Log.i(TAG, "ONNX model loaded: " + modelFile.length() + " bytes");
        Log.i(TAG, "ONNX input tensor: " + inputTensorName);
        Log.i(TAG, "Configured execution provider: " + activeExecutionProvider);
    }

    private void copyStream(InputStream input, FileOutputStream output) throws Exception {
        byte[] buffer = new byte[64 * 1024];
        int count;
        while ((count = input.read(buffer)) != -1) {
            output.write(buffer, 0, count);
        }
        output.flush();
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

    @ReactMethod
    public void copyToClipboard(String label, String text, Promise promise) {
        try {
            android.content.ClipboardManager clipboard = (android.content.ClipboardManager)
                reactContext.getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null) {
                android.content.ClipData clip = android.content.ClipData.newPlainText(label, text);
                clipboard.setPrimaryClip(clip);
                promise.resolve(true);
            } else {
                promise.reject("CLIPBOARD_UNAVAILABLE", "Clipboard manager unavailable");
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy to clipboard", e);
            promise.reject("CLIPBOARD_ERROR", e.getMessage(), e);
        }
    }

    // =========================================================================
    // 1. Muzzle Region Isolation & Cropping
    // =========================================================================
    public static RectInt computeMuzzleCropRegion(int w, int h, RectInt bbox) {
        int y1, x1, y2, x2;
        if (bbox != null) {
            y1 = bbox.y1;
            x1 = bbox.x1;
            y2 = bbox.y2;
            x2 = bbox.x2;
        } else {
            float aspect = (float) Math.max(w, h) / Math.max(1, Math.min(w, h));
            if (aspect <= 1.15f && Math.min(w, h) <= 600) {
                // Image is already a cropped square muzzle
                y1 = 0;
                x1 = 0;
                y2 = h;
                x2 = w;
            } else {
                // Default cattle muzzle focus window [0.20*H to 0.85*H, 0.15*W to 0.85*W]
                y1 = (int) (h * 0.20f);
                y2 = (int) (h * 0.85f);
                x1 = (int) (w * 0.15f);
                x2 = (int) (w * 0.85f);
            }
        }

        int boxH = Math.max(1, y2 - y1);
        int boxW = Math.max(1, x2 - x1);

        int padY = (int) (boxH * 0.05f);
        int padX = (int) (boxW * 0.05f);
        int cy1 = Math.max(0, y1 - padY);
        int cx1 = Math.max(0, x1 - padX);
        int cy2 = Math.min(h, y2 + padY);
        int cx2 = Math.min(w, x2 + padX);

        // Square expansion to prevent aspect ratio distortion
        int ch = cy2 - cy1;
        int cw = cx2 - cx1;
        int diff = Math.abs(ch - cw);
        if (ch < cw) {
            cy1 = Math.max(0, cy1 - diff / 2);
            cy2 = Math.min(h, cy2 + (diff - diff / 2));
        } else if (cw < ch) {
            cx1 = Math.max(0, cx1 - diff / 2);
            cx2 = Math.min(w, cx2 + (diff - diff / 2));
        }

        // Clamp coordinates securely within boundaries
        cy1 = Math.max(0, Math.min(h - 1, cy1));
        cx1 = Math.max(0, Math.min(w - 1, cx1));
        cy2 = Math.max(cy1 + 1, Math.min(h, cy2));
        cx2 = Math.max(cx1 + 1, Math.min(w, cx2));

        return new RectInt(cx1, cy1, cx2, cy2);
    }

    public static Bitmap cropToMuzzle(Bitmap src, RectInt bbox) {
        if (src == null || src.isRecycled()) return null;
        int w = src.getWidth();
        int h = src.getHeight();
        if (w <= 0 || h <= 0) return null;

        RectInt r = computeMuzzleCropRegion(w, h, bbox);
        int cropW = r.x2 - r.x1;
        int cropH = r.y2 - r.y1;
        if (cropW <= 0 || cropH <= 0) return null;

        if (r.x1 == 0 && r.y1 == 0 && cropW == w && cropH == h) {
            return src;
        }
        return Bitmap.createBitmap(src, r.x1, r.y1, cropW, cropH);
    }

    // =========================================================================
    // 2. Laplacian Sharpness Variance Gating
    // =========================================================================
    public static double computeLaplacianVariance(int[] gray, int w, int h) {
        if (gray == null || w < 3 || h < 3) return 0.0;
        // Laplacian kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
        double sum = 0.0;
        int count = (w - 2) * (h - 2);

        for (int y = 1; y < h - 1; y++) {
            int yOffset = y * w;
            int yPrev = (y - 1) * w;
            int yNext = (y + 1) * w;
            for (int x = 1; x < w - 1; x++) {
                int lap = gray[yPrev + x]
                        + gray[yNext + x]
                        + gray[yOffset + x - 1]
                        + gray[yOffset + x + 1]
                        - 4 * gray[yOffset + x];
                sum += lap;
            }
        }

        double mean = sum / count;
        double varSum = 0.0;
        for (int y = 1; y < h - 1; y++) {
            int yOffset = y * w;
            int yPrev = (y - 1) * w;
            int yNext = (y + 1) * w;
            for (int x = 1; x < w - 1; x++) {
                int lap = gray[yPrev + x]
                        + gray[yNext + x]
                        + gray[yOffset + x - 1]
                        + gray[yOffset + x + 1]
                        - 4 * gray[yOffset + x];
                double diff = lap - mean;
                varSum += diff * diff;
            }
        }
        return varSum / count;
    }

    // =========================================================================
    // 3. Stage 2: Gaussian Unsharp Mask & CLAHE Contrast Normalization
    // =========================================================================
    private static int[] applyGaussianUnsharpMask(int[] gray, int w, int h) {
        // 1D separable Gaussian kernel for sigma = 1.5, radius = 3
        final float[] weights = {0.270682f, 0.216745f, 0.111281f, 0.036633f};
        float[] temp = new float[w * h];
        float[] blurred = new float[w * h];

        // Horizontal pass
        for (int y = 0; y < h; y++) {
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                float sum = gray[yOff + x] * weights[0];
                for (int r = 1; r <= 3; r++) {
                    int xL = (x - r >= 0) ? x - r : 0;
                    int xR = (x + r < w) ? x + r : w - 1;
                    sum += (gray[yOff + xL] + gray[yOff + xR]) * weights[r];
                }
                temp[yOff + x] = sum;
            }
        }

        // Vertical pass
        for (int y = 0; y < h; y++) {
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                float sum = temp[yOff + x] * weights[0];
                for (int r = 1; r <= 3; r++) {
                    int yU = (y - r >= 0) ? (y - r) * w : 0;
                    int yD = (y + r < h) ? (y + r) * w : (h - 1) * w;
                    sum += (temp[yU + x] + temp[yD + x]) * weights[r];
                }
                blurred[yOff + x] = sum;
            }
        }

        // gray_unsharp = clamp(1.5 * gray - 0.5 * blurred)
        int[] unsharp = new int[w * h];
        for (int i = 0; i < w * h; i++) {
            float val = 1.5f * gray[i] - 0.5f * blurred[i];
            unsharp[i] = Math.max(0, Math.min(255, Math.round(val)));
        }
        return unsharp;
    }

    private static int[] applyCLAHE(int[] grayUnsharp, int w, int h) {
        // 8x8 tiles on 224x224 (each tile 28x28)
        final int numTiles = 8;
        final int tileW = w / numTiles;
        final int tileH = h / numTiles;
        final int tileArea = tileW * tileH;
        final int clipLimit = Math.max(1, Math.round(2.5f * (tileArea / 256.0f)));

        int[][][] tileCdf = new int[numTiles][numTiles][256];

        for (int ty = 0; ty < numTiles; ty++) {
            int startY = ty * tileH;
            for (int tx = 0; tx < numTiles; tx++) {
                int startX = tx * tileW;
                int[] hist = new int[256];

                for (int dy = 0; dy < tileH; dy++) {
                    int rowOff = (startY + dy) * w + startX;
                    for (int dx = 0; dx < tileW; dx++) {
                        int val = grayUnsharp[rowOff + dx];
                        hist[val]++;
                    }
                }

                // Clip histogram
                int excess = 0;
                for (int b = 0; b < 256; b++) {
                    if (hist[b] > clipLimit) {
                        excess += (hist[b] - clipLimit);
                        hist[b] = clipLimit;
                    }
                }

                // Redistribute excess evenly
                int bonus = excess / 256;
                int remainder = excess % 256;
                for (int b = 0; b < 256; b++) {
                    hist[b] += bonus;
                }
                for (int b = 0; b < remainder; b++) {
                    hist[b]++;
                }

                // Cumulative distribution function (CDF)
                int cum = 0;
                for (int b = 0; b < 256; b++) {
                    cum += hist[b];
                    tileCdf[ty][tx][b] = Math.max(0, Math.min(255, Math.round((float) cum * 255.0f / tileArea)));
                }
            }
        }

        // Bilinear interpolation between tile centers
        int[] claheOut = new int[w * h];
        for (int y = 0; y < h; y++) {
            float tyF = (y - (tileH / 2.0f)) / (float) tileH;
            int y0, y1;
            float beta;
            if (tyF <= 0.0f) {
                y0 = 0; y1 = 0; beta = 0.0f;
            } else if (tyF >= numTiles - 1) {
                y0 = numTiles - 1; y1 = numTiles - 1; beta = 0.0f;
            } else {
                y0 = (int) tyF;
                y1 = y0 + 1;
                beta = tyF - y0;
            }

            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                float txF = (x - (tileW / 2.0f)) / (float) tileW;
                int x0, x1;
                float alpha;
                if (txF <= 0.0f) {
                    x0 = 0; x1 = 0; alpha = 0.0f;
                } else if (txF >= numTiles - 1) {
                    x0 = numTiles - 1; x1 = numTiles - 1; alpha = 0.0f;
                } else {
                    x0 = (int) txF;
                    x1 = x0 + 1;
                    alpha = txF - x0;
                }

                int v = grayUnsharp[yOff + x];
                int cdf00 = tileCdf[y0][x0][v];
                int cdf01 = tileCdf[y0][x1][v];
                int cdf10 = tileCdf[y1][x0][v];
                int cdf11 = tileCdf[y1][x1][v];

                float top = (1.0f - alpha) * cdf00 + alpha * cdf01;
                float bottom = (1.0f - alpha) * cdf10 + alpha * cdf11;
                float interp = (1.0f - beta) * top + beta * bottom;

                claheOut[yOff + x] = Math.max(0, Math.min(255, Math.round(interp)));
            }
        }
        return claheOut;
    }

    // =========================================================================
    // 4. Stage 3: Multi-Scale Black-Hat Groove Extraction
    // =========================================================================
    private static int[] computeBlackHat(int[] src, int w, int h, DiskKernel kernel) {
        int len = w * h;
        int[] dilated = new int[len];
        int[] closed = new int[len];
        int[] bh = new int[len];

        final int kLen = kernel.offX.length;
        final int[] kX = kernel.offX;
        final int[] kY = kernel.offY;

        // Dilation
        for (int y = 0; y < h; y++) {
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                int maxV = 0;
                for (int k = 0; k < kLen; k++) {
                    int cy = y + kY[k];
                    if (cy < 0) cy = 0; else if (cy >= h) cy = h - 1;
                    int cx = x + kX[k];
                    if (cx < 0) cx = 0; else if (cx >= w) cx = w - 1;
                    int v = src[cy * w + cx];
                    if (v > maxV) maxV = v;
                }
                dilated[yOff + x] = maxV;
            }
        }

        // Erosion of dilated (Closing)
        for (int y = 0; y < h; y++) {
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                int minV = 255;
                for (int k = 0; k < kLen; k++) {
                    int cy = y + kY[k];
                    if (cy < 0) cy = 0; else if (cy >= h) cy = h - 1;
                    int cx = x + kX[k];
                    if (cx < 0) cx = 0; else if (cx >= w) cx = w - 1;
                    int v = dilated[cy * w + cx];
                    if (v < minV) minV = v;
                }
                closed[yOff + x] = minV;
            }
        }

        // BlackHat = Closing - src
        for (int i = 0; i < len; i++) {
            bh[i] = Math.max(0, closed[i] - src[i]);
        }
        return bh;
    }

    private static int[] applyMultiScaleBlackHat(int[] gray, int w, int h) {
        int len = w * h;
        int[] bh2 = computeBlackHat(gray, w, h, KERNEL_R2);
        int[] bh4 = computeBlackHat(gray, w, h, KERNEL_R4);
        int[] bh6 = computeBlackHat(gray, w, h, KERNEL_R6);

        int[] maxBh = new int[len];
        int minV = Integer.MAX_VALUE;
        int maxV = Integer.MIN_VALUE;

        for (int i = 0; i < len; i++) {
            int v = Math.max(bh2[i], Math.max(bh4[i], bh6[i]));
            maxBh[i] = v;
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }

        int[] norm = new int[len];
        if (maxV > minV) {
            float range = (float) (maxV - minV);
            for (int i = 0; i < len; i++) {
                norm[i] = Math.max(0, Math.min(255, Math.round(((maxBh[i] - minV) / range) * 255.0f)));
            }
        }
        return norm;
    }

    // =========================================================================
    // 5. Stage 4 & 5: Adaptive Thresholding & Skeleton Dilation
    // =========================================================================
    private static int[] applyAdaptiveThreshold(int[] grooveResponse, int w, int h) {
        int[] binary = new int[w * h];
        int[][] sat = new int[h + 1][w + 1];

        for (int y = 0; y < h; y++) {
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                sat[y + 1][x + 1] = grooveResponse[yOff + x]
                        + sat[y][x + 1]
                        + sat[y + 1][x]
                        - sat[y][x];
            }
        }

        final int radius = 7; // 15x15 window
        for (int y = 0; y < h; y++) {
            int y1 = (y >= radius) ? y - radius : 0;
            int y2 = (y + radius < h) ? y + radius : h - 1;
            int yOff = y * w;

            for (int x = 0; x < w; x++) {
                int x1 = (x >= radius) ? x - radius : 0;
                int x2 = (x + radius < w) ? x + radius : w - 1;
                int count = (y2 - y1 + 1) * (x2 - x1 + 1);
                int sum = sat[y2 + 1][x2 + 1] - sat[y1][x2 + 1] - sat[y2 + 1][x1] + sat[y1][x1];
                float mean = (float) sum / count;
                int val = grooveResponse[yOff + x];

                if (val > (mean - 2.0f) && val > 0) {
                    binary[yOff + x] = 255;
                } else {
                    binary[yOff + x] = 0;
                }
            }
        }
        return binary;
    }

    private static byte[] applyZhangSuenThinning(int[] binary, int w, int h) {
        byte[] skel = new byte[w * h];
        for (int i = 0; i < w * h; i++) {
            skel[i] = (byte) (binary[i] > 0 ? 1 : 0);
        }

        int[] toDelete = new int[w * h];
        boolean changed = true;
        int iter = 0;

        while (changed && iter++ < 40) {
            changed = false;

            // Sub-iteration 1
            int delCount = 0;
            for (int y = 1; y < h - 1; y++) {
                int row = y * w;
                int rowP = (y - 1) * w;
                int rowN = (y + 1) * w;
                for (int x = 1; x < w - 1; x++) {
                    if (skel[row + x] == 0) continue;

                    int p2 = skel[rowP + x];
                    int p3 = skel[rowP + x + 1];
                    int p4 = skel[row + x + 1];
                    int p5 = skel[rowN + x + 1];
                    int p6 = skel[rowN + x];
                    int p7 = skel[rowN + x - 1];
                    int p8 = skel[row + x - 1];
                    int p9 = skel[rowP + x - 1];

                    int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if (b < 2 || b > 6) continue;

                    int a = 0;
                    if (p2 == 0 && p3 == 1) a++;
                    if (p3 == 0 && p4 == 1) a++;
                    if (p4 == 0 && p5 == 1) a++;
                    if (p5 == 0 && p6 == 1) a++;
                    if (p6 == 0 && p7 == 1) a++;
                    if (p7 == 0 && p8 == 1) a++;
                    if (p8 == 0 && p9 == 1) a++;
                    if (p9 == 0 && p2 == 1) a++;
                    if (a != 1) continue;

                    if (p2 * p4 * p6 != 0) continue;
                    if (p4 * p6 * p8 != 0) continue;

                    toDelete[delCount++] = row + x;
                }
            }
            if (delCount > 0) {
                changed = true;
                for (int i = 0; i < delCount; i++) {
                    skel[toDelete[i]] = 0;
                }
            }

            // Sub-iteration 2
            delCount = 0;
            for (int y = 1; y < h - 1; y++) {
                int row = y * w;
                int rowP = (y - 1) * w;
                int rowN = (y + 1) * w;
                for (int x = 1; x < w - 1; x++) {
                    if (skel[row + x] == 0) continue;

                    int p2 = skel[rowP + x];
                    int p3 = skel[rowP + x + 1];
                    int p4 = skel[row + x + 1];
                    int p5 = skel[rowN + x + 1];
                    int p6 = skel[rowN + x];
                    int p7 = skel[rowN + x - 1];
                    int p8 = skel[row + x - 1];
                    int p9 = skel[rowP + x - 1];

                    int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if (b < 2 || b > 6) continue;

                    int a = 0;
                    if (p2 == 0 && p3 == 1) a++;
                    if (p3 == 0 && p4 == 1) a++;
                    if (p4 == 0 && p5 == 1) a++;
                    if (p5 == 0 && p6 == 1) a++;
                    if (p6 == 0 && p7 == 1) a++;
                    if (p7 == 0 && p8 == 1) a++;
                    if (p8 == 0 && p9 == 1) a++;
                    if (p9 == 0 && p2 == 1) a++;
                    if (a != 1) continue;

                    if (p2 * p4 * p8 != 0) continue;
                    if (p2 * p6 * p8 != 0) continue;

                    toDelete[delCount++] = row + x;
                }
            }
            if (delCount > 0) {
                changed = true;
                for (int i = 0; i < delCount; i++) {
                    skel[toDelete[i]] = 0;
                }
            }
        }
        return skel;
    }

    private static int[] applyBoxDilation3x3(byte[] skel, int w, int h) {
        int[] dilated = new int[w * h];
        for (int y = 0; y < h; y++) {
            int yMin = Math.max(0, y - 1);
            int yMax = Math.min(h - 1, y + 1);
            int yOff = y * w;
            for (int x = 0; x < w; x++) {
                int xMin = Math.max(0, x - 1);
                int xMax = Math.min(w - 1, x + 1);
                boolean any = false;
                for (int dy = yMin; dy <= yMax; dy++) {
                    int r = dy * w;
                    for (int dx = xMin; dx <= xMax; dx++) {
                        if (skel[r + dx] != 0) {
                            any = true;
                            break;
                        }
                    }
                    if (any) break;
                }
                dilated[yOff + x] = any ? 255 : 0;
            }
        }
        return dilated;
    }

    // =========================================================================
    // 6. Composite 3-Channel Tensor Construction & Inference
    // =========================================================================
    private synchronized float[] runInferenceOnBitmap(Bitmap inputBmp) throws Exception {
        Bitmap scaledBmp = null;
        OnnxTensor tensor = null;
        OrtSession.Result result = null;

        try {
            if (inputBmp.getWidth() != INPUT_SIZE || inputBmp.getHeight() != INPUT_SIZE) {
                scaledBmp = Bitmap.createScaledBitmap(inputBmp, INPUT_SIZE, INPUT_SIZE, true);
            } else {
                scaledBmp = inputBmp;
            }

            int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
            scaledBmp.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

            int[] gray = new int[INPUT_SIZE * INPUT_SIZE];
            for (int i = 0; i < pixels.length; i++) {
                int px = pixels[i];
                int r = (px >> 16) & 0xFF;
                int g = (px >> 8) & 0xFF;
                int b = px & 0xFF;
                gray[i] = (r * 77 + g * 150 + b * 29) >> 8;
            }

            // Stage 2: Gaussian unsharp mask + CLAHE
            int[] unsharp = applyGaussianUnsharpMask(gray, INPUT_SIZE, INPUT_SIZE);
            int[] claheOut = applyCLAHE(unsharp, INPUT_SIZE, INPUT_SIZE);

            // Stage 3: Multi-Scale Black-Hat Groove Extraction
            int[] grooveResponse = applyMultiScaleBlackHat(claheOut, INPUT_SIZE, INPUT_SIZE);

            // Stage 4: Adaptive Thresholding
            int[] binary = applyAdaptiveThreshold(grooveResponse, INPUT_SIZE, INPUT_SIZE);

            // Stage 5: Zhang-Suen Thinning + 3x3 Box Dilation
            byte[] skel = applyZhangSuenThinning(binary, INPUT_SIZE, INPUT_SIZE);
            int[] skelDilated = applyBoxDilation3x3(skel, INPUT_SIZE, INPUT_SIZE);

            // Construct Composite 3-Channel Input Tensor:
            // Ch 0: (clahe / 255.0f - 0.485f) / 0.229f
            // Ch 1: (groove_response / 255.0f - 0.456f) / 0.224f
            // Ch 2: (skel_dilated / 255.0f - 0.406f) / 0.225f
            int bufferBytes = 1 * CHANNELS * INPUT_SIZE * INPUT_SIZE * 4;
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferBytes);
            byteBuffer.order(ByteOrder.nativeOrder());
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            // Channel 0
            for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
                float val = (claheOut[i] / 255.0f - 0.485f) / 0.229f;
                floatBuffer.put(val);
            }
            // Channel 1
            for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
                float val = (grooveResponse[i] / 255.0f - 0.456f) / 0.224f;
                floatBuffer.put(val);
            }
            // Channel 2
            for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
                float val = (skelDilated[i] / 255.0f - 0.406f) / 0.225f;
                floatBuffer.put(val);
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
        Bitmap muzzleCrop = null;
        try {
            initializeOnnxEngine();
            long inferStartNs = System.nanoTime();

            originalBitmap = decodeBitmapFromUri(imageUri);
            if (originalBitmap == null) {
                throw new IllegalArgumentException("Could not decode the selected image. No synthetic embedding will be returned.");
            }

            muzzleCrop = cropToMuzzle(originalBitmap, null);
            float[] l2Vec = runInferenceOnBitmap(muzzleCrop);
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
            if (muzzleCrop != null && muzzleCrop != originalBitmap && !muzzleCrop.isRecycled()) {
                muzzleCrop.recycle();
            }
            if (originalBitmap != null && !originalBitmap.isRecycled()) {
                originalBitmap.recycle();
            }
        }
    }

    // =========================================================================
    // 7. Video Frame Extraction, Sharpness Gating & Temporal Aggregation
    // =========================================================================
    @ReactMethod
    public void extractFramesAndEmbedFromVideo(String videoUri, int numFrames, Promise promise) {
        final int targetFrames = (numFrames <= 0) ? 5 : numFrames;

        new Thread(() -> {
            MediaMetadataRetriever retriever = null;
            ParcelFileDescriptor pfd = null;
            File tempAssetFile = null;
            List<CandidateFrame> allCandidates = new ArrayList<>();

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
                    String assetPath = videoUri.startsWith("assets/") ? videoUri.substring(7) : videoUri;
                    try {
                        AssetFileDescriptor afd = reactContext.getAssets().openFd(assetPath);
                        retriever.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
                        afd.close();
                    } catch (Exception e) {
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
                    durationUs = 2_000_000L;
                }

                // 3. Sample ~20 candidate frames across video duration
                int numCandidates = Math.max(20, targetFrames * 4);

                for (int k = 0; k < numCandidates; k++) {
                    long targetTimeUs = (long) (((k + 0.5) / (double) numCandidates) * durationUs);
                    double timeMs = targetTimeUs / 1000.0;

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
                        rawFrame = createSyntheticMuzzleBitmap();
                    }

                    // Muzzle Region Isolation & Cropping (with 5% padding & square expansion)
                    Bitmap cropped = cropToMuzzle(rawFrame, null);
                    if (cropped != rawFrame && rawFrame != null && !rawFrame.isRecycled()) {
                        rawFrame.recycle();
                    }

                    // Scale to canonical 224x224
                    Bitmap canonical224 = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true);
                    if (canonical224 != cropped && cropped != null && !cropped.isRecycled()) {
                        cropped.recycle();
                    }

                    // Laplacian Sharpness Variance check
                    int[] px = new int[INPUT_SIZE * INPUT_SIZE];
                    canonical224.getPixels(px, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
                    int[] gray = new int[INPUT_SIZE * INPUT_SIZE];
                    for (int p = 0; p < px.length; p++) {
                        int c = px[p];
                        gray[p] = (((c >> 16) & 0xFF) * 77 + ((c >> 8) & 0xFF) * 150 + (c & 0xFF) * 29) >> 8;
                    }

                    double lapVar = computeLaplacianVariance(gray, INPUT_SIZE, INPUT_SIZE);
                    allCandidates.add(new CandidateFrame(k, targetTimeUs, timeMs, lapVar, canonical224));
                }

                // Filter out blurred frames (lap_var < 20.0). Select top 5 sharpest frames temporally spread across video duration.
                List<CandidateFrame> validCandidates = new ArrayList<>();
                List<CandidateFrame> fallbackCandidates = new ArrayList<>();

                for (CandidateFrame cf : allCandidates) {
                    if (cf.lapVar >= 20.0) {
                        validCandidates.add(cf);
                    } else {
                        fallbackCandidates.add(cf);
                    }
                }

                // Sort candidate pools by sharpness descending
                Collections.sort(validCandidates, (a, b) -> Double.compare(b.lapVar, a.lapVar));
                Collections.sort(fallbackCandidates, (a, b) -> Double.compare(b.lapVar, a.lapVar));

                List<CandidateFrame> selected = new ArrayList<>();
                int targetCount = Math.min(targetFrames, allCandidates.size());

                int[] separations = new int[]{4, 3, 2, 1};
                for (int sep : separations) {
                    for (CandidateFrame cand : validCandidates) {
                        if (selected.size() >= targetCount) break;
                        boolean farEnough = true;
                        for (CandidateFrame s : selected) {
                            if (Math.abs(cand.index - s.index) < sep) {
                                farEnough = false;
                                break;
                            }
                        }
                        if (farEnough) {
                            selected.add(cand);
                        }
                    }
                    if (selected.size() >= targetCount) break;
                }

                // If fewer than targetCount passed gating, fall back to best available candidate frames
                if (selected.size() < targetCount) {
                    for (int sep : separations) {
                        for (CandidateFrame cand : fallbackCandidates) {
                            if (selected.size() >= targetCount) break;
                            boolean farEnough = true;
                            for (CandidateFrame s : selected) {
                                if (Math.abs(cand.index - s.index) < sep) {
                                    farEnough = false;
                                    break;
                                }
                            }
                            if (farEnough) {
                                selected.add(cand);
                            }
                        }
                        if (selected.size() >= targetCount) break;
                    }
                }

                // Chronologically sort selected frames
                Collections.sort(selected, (a, b) -> Long.compare(a.timeUs, b.timeUs));

                // Immediately recycle unselected candidate bitmaps to reclaim RAM
                for (CandidateFrame cand : allCandidates) {
                    if (!selected.contains(cand) && cand.bitmap != null && !cand.bitmap.isRecycled()) {
                        cand.bitmap.recycle();
                    }
                }

                List<float[]> frameVectors = new ArrayList<>();
                List<String> frameThumbnails = new ArrayList<>();
                List<Double> frameTimestampsMs = new ArrayList<>();
                List<Double> frameInferenceMsList = new ArrayList<>();
                List<Double> frameLapVars = new ArrayList<>();

                // Process the selected frames through the 5-stage feature extraction & ONNX inference
                for (CandidateFrame cand : selected) {
                    frameTimestampsMs.add(cand.timeMs);
                    frameLapVars.add(cand.lapVar);

                    // Generate compressed Base64 thumbnail (112x112 JPEG) for fast mobile rendering
                    Bitmap thumbBmp = Bitmap.createScaledBitmap(cand.bitmap, 112, 112, true);
                    ByteArrayOutputStream thumbBaos = new ByteArrayOutputStream();
                    thumbBmp.compress(Bitmap.CompressFormat.JPEG, 70, thumbBaos);
                    String thumbB64 = "data:image/jpeg;base64," + Base64.encodeToString(thumbBaos.toByteArray(), Base64.NO_WRAP);
                    thumbBmp.recycle();
                    frameThumbnails.add(thumbB64);

                    long frameStartNs = System.nanoTime();
                    float[] embedding = runInferenceOnBitmap(cand.bitmap);
                    long frameEndNs = System.nanoTime();
                    double frameInferMs = (frameEndNs - frameStartNs) / 1_000_000.0;
                    frameInferenceMsList.add(Math.round(frameInferMs * 10.0) / 10.0);

                    frameVectors.add(embedding);
                    cand.bitmap.recycle();
                }

                // 4. Compute Pairwise Intra-Video Agreement Scores
                int N = frameVectors.size();
                double[][] agreementMatrix = new double[N][N];
                double[] frameMeanAgreement = new double[N];
                double totalPairwiseSum = 0.0;
                int pairwiseCount = 0;

                for (int i = 0; i < N; i++) {
                    float[] vA = frameVectors.get(i);
                    double rowSum = 0.0;
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
                                totalPairwiseSum += Math.max(0.0, cosSim);
                                pairwiseCount++;
                            }
                        }
                    }
                    frameMeanAgreement[i] = (N > 1) ? (rowSum / (N - 1)) : 1.0;
                }

                double globalMeanAgreement = (pairwiseCount > 0) ? (totalPairwiseSum / pairwiseCount) : 1.0;
                double intraVideoAgreementPct = Math.round(globalMeanAgreement * 10000.0) / 100.0;

                // 5. Outlier Pruning (agreement score < median - 0.12 or < 0.65, keeping min 3)
                double[] sortedScores = frameMeanAgreement.clone();
                Arrays.sort(sortedScores);
                double medA;
                if (N % 2 == 1) {
                    medA = sortedScores[N / 2];
                } else {
                    medA = 0.5 * (sortedScores[N / 2 - 1] + sortedScores[N / 2]);
                }
                double outlierThreshold = Math.max(0.65, medA - 0.12);

                List<Integer> inliers = new ArrayList<>();
                for (int t = 0; t < N; t++) {
                    if (frameMeanAgreement[t] >= outlierThreshold) {
                        inliers.add(t);
                    }
                }
                int minKeep = Math.min(N, 3);
                if (inliers.size() < minKeep) {
                    Integer[] sortedIndices = new Integer[N];
                    for (int t = 0; t < N; t++) sortedIndices[t] = t;
                    Arrays.sort(sortedIndices, (a, b) -> Double.compare(frameMeanAgreement[b], frameMeanAgreement[a]));
                    inliers.clear();
                    for (int k = 0; k < minKeep; k++) {
                        inliers.add(sortedIndices[k]);
                    }
                }
                Collections.sort(inliers);

                boolean[] isOutlier = new boolean[N];
                for (int t = 0; t < N; t++) {
                    isOutlier[t] = !inliers.contains(t);
                }

                // Outlier-Pruned Quality-Weighted Mean Embedding
                float[] weightedMean = new float[EMBEDDING_DIM];
                double totalWeight = 0.0;
                double[] frameWeights = new double[N];

                for (int t = 0; t < N; t++) {
                    if (isOutlier[t]) {
                        frameWeights[t] = 0.0;
                        continue;
                    }
                    double lap = frameLapVars.get(t);
                    double blurFactor = Math.min(1.0, Math.max(0.0, lap) / 100.0);
                    double w = Math.max(0.01, frameMeanAgreement[t] * blurFactor);
                    frameWeights[t] = w;
                    totalWeight += w;

                    float[] v = frameVectors.get(t);
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        weightedMean[d] += (float) (w * v[d]);
                    }
                }

                if (totalWeight <= 1e-8) {
                    for (int t : inliers) {
                        float[] v = frameVectors.get(t);
                        for (int d = 0; d < EMBEDDING_DIM; d++) {
                            weightedMean[d] += v[d];
                        }
                    }
                }

                // L2-normalize consensus weighted mean embedding
                float sumSqMean = 0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
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
                    stat.putDouble("laplacianVariance", Math.round(frameLapVars.get(i) * 10.0) / 10.0);
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
                for (CandidateFrame cand : allCandidates) {
                    if (cand.bitmap != null && !cand.bitmap.isRecycled()) {
                        cand.bitmap.recycle();
                    }
                }
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
            } else if (consensusPct >= 75.0) {
                decision = "LIKELY";
                verdict = "LIKELY: Strong Biometric Agreement";
                verdictColor = "teal";
                isMatch = true;
            } else if (consensusPct >= 60.0) {
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
