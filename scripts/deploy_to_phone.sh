#!/usr/bin/env bash
# ==============================================================================
# deploy_to_phone.sh - Automates APK deployment to Android phone over USB-C
# Supports both ADB install and GVfs/MTP direct file copy to Download folder.
# ==============================================================================
set -euo pipefail

APK_PATH="/home/adis_mayur/F_CODES/NOSE_MOBILE_APP/android/app/build/outputs/apk/release/app-arm64-v8a-release.apk"
TARGET_APK_NAME="CattleMuzzleBiometric.apk"

echo "======================================================================"
echo " 📱 Cattle Muzzle Biometric - Android USB-C Deployment Helper"
echo "======================================================================"

if [ ! -f "${APK_PATH}" ]; then
    echo "[-] Error: Release APK not found at: ${APK_PATH}"
    echo "    Please run gradle assembleRelease first."
    exit 1
fi

APK_SIZE_MB=$(du -m "${APK_PATH}" | cut -f1)
echo "[+] Found Release APK: ${APK_PATH} (${APK_SIZE_MB} MB)"

# Method 1: Check ADB
echo "[*] Checking for ADB authorized device..."
ADB_DEVICE=$(adb devices | grep -v "List of" | grep "device$" | head -n 1 | awk '{print $1}' || true)

if [ -n "${ADB_DEVICE}" ]; then
    echo "[+] Authorized ADB device found: ${ADB_DEVICE}"
    echo "[*] Installing APK directly via ADB..."
    adb -s "${ADB_DEVICE}" install -r "${APK_PATH}"
    echo "[✓] Successfully installed APK on ${ADB_DEVICE} via ADB!"
    exit 0
fi

UNAUTHORIZED_DEVICE=$(adb devices | grep "unauthorized$" | head -n 1 | awk '{print $1}' || true)
if [ -n "${UNAUTHORIZED_DEVICE}" ]; then
    echo "[!] Notice: Device ${UNAUTHORIZED_DEVICE} is connected but UNAUTHORIZED for ADB."
    echo "    Please look at your phone screen and tap 'Allow USB debugging'."
fi

# Method 2: Check MTP / GVfs
echo "[*] Checking for MTP device via GIO / GVfs..."
MTP_MOUNTS=$(gio mount -l | grep -i "mtp://" | sed -e 's/^[ \t]*//' || true)

if [ -n "${MTP_MOUNTS}" ]; then
    echo "[+] Detected MTP mount(s):"
    echo "${MTP_MOUNTS}"
    
    # Try Samsung or generic MTP URI
    FIRST_MTP_URI=$(echo "${MTP_MOUNTS}" | head -n 1 | awk '{print $NF}')
    TARGET_DEST="${FIRST_MTP_URI}Internal storage/Download/${TARGET_APK_NAME}"
    
    echo "[*] Transferring APK to: ${TARGET_DEST} ..."
    if gio copy "${APK_PATH}" "${TARGET_DEST}"; then
        echo "[✓] Successfully copied APK to phone Download folder!"
        echo "    Open 'My Files' -> 'Downloads' on your phone and tap '${TARGET_APK_NAME}' to install."
        exit 0
    else
        echo "[-] gio copy failed, trying GVfs filesystem path..."
    fi
fi

# Check GVfs filesystem mount point
GVFS_DIR="/run/user/$(id -u)/gvfs"
if [ -d "${GVFS_DIR}" ]; then
    MTP_DIRS=$(find "${GVFS_DIR}" -maxdepth 1 -name "mtp:*" || true)
    for mdir in ${MTP_DIRS}; do
        DOWNLOAD_DIR="${mdir}/Internal storage/Download"
        if [ -d "${DOWNLOAD_DIR}" ]; then
            echo "[+] Found phone Download directory at: ${DOWNLOAD_DIR}"
            echo "[*] Copying APK..."
            cp -v "${APK_PATH}" "${DOWNLOAD_DIR}/${TARGET_APK_NAME}"
            echo "[✓] Successfully copied APK to phone Download folder via GVfs!"
            echo "    Open 'My Files' -> 'Downloads' on your phone and tap '${TARGET_APK_NAME}' to install."
            exit 0
        fi
    done
fi

echo "======================================================================"
echo "[!] No connected Android device detected via ADB or MTP."
echo "    Please connect your phone to your laptop using your USB-C cable."
echo "    On your phone screen, select: 'Transferring files / Android Auto' (MTP mode)."
echo "    Once connected, run this script again: bash scripts/deploy_to_phone.sh"
echo "======================================================================"
exit 2
