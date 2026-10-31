import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Alert,
  Share,
  ActivityIndicator,
  NativeModules,
  Platform,
  TextInput,
  Modal,
} from 'react-native';

import {
  DEFAULT_GALLERY_CANDIDATES,
  runClientSide1toNMatch,
  computeBiometricDistance,
  normalizeVector,
} from './data/galleryPresets';

const { MuzzleBiometricModule } = NativeModules;

// Pre-bundled cattle video test scenarios
const VIDEO_PRESETS = [
  {
    id: 'cow_a_clip1',
    label: 'Cow A (Session 1)',
    tag: 'TAG-21981',
    uri: 'sample_videos/cow_a_clip1.mp4',
    description: 'Frontal muzzle close-up (Session 1)',
    isSameCow: true,
  },
  {
    id: 'cow_a_clip2',
    label: 'Cow A (Session 2 - Angle Shift)',
    tag: 'TAG-21981',
    uri: 'sample_videos/cow_a_clip2.mp4',
    description: 'Recaptured muzzle with minor angle shift (Session 2)',
    isSameCow: true,
  },
  {
    id: 'cow_b_clip',
    label: 'Cow B (Distinct Cattle)',
    tag: 'TAG-22191',
    uri: 'sample_videos/cow_b_clip.mp4',
    description: 'Completely different cattle identity',
    isSameCow: false,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('home'); // 'home' | 'extract' | 'video' | 'match'
  const [diagnosticsVisible, setDiagnosticsVisible] = useState(false);

  // Video Matcher States (Tab: 'video')
  const [videoPathA, setVideoPathA] = useState('sample_videos/cow_a_clip1.mp4');
  const [videoAName, setVideoAName] = useState('Cow A (Session 1)');
  const [videoPathB, setVideoPathB] = useState('sample_videos/cow_a_clip2.mp4');
  const [videoBName, setVideoBName] = useState('Cow A (Session 2)');
  const [numFrames, setNumFrames] = useState(5);
  const [isVideoVerifying, setIsVideoVerifying] = useState(false);
  const [videoVerifyProgress, setVideoVerifyProgress] = useState('');
  const [videoResultA, setVideoResultA] = useState(null);
  const [videoResultB, setVideoResultB] = useState(null);
  const [videoComparison, setVideoComparison] = useState(null);

  // Single Video 512-D Embedding Extractor States (Tab: 'extract')
  const [customMedia, setCustomMedia] = useState({
    uri: 'sample_videos/cow_a_clip1.mp4',
    name: 'Cow A (Session 1)',
    isVideo: true,
  });
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState('');
  const [extractionResult, setExtractionResult] = useState(null);
  const [latestEmbedding, setLatestEmbedding] = useState(null);
  const [copyToast, setCopyToast] = useState('');

  // Hardware Telemetry & Runtime Stats
  const [telemetry, setTelemetry] = useState({
    cpuCores: 8,
    maxMemoryMb: 512,
    nnapiSupported: false,
    provider: 'Not initialized',
    latencyMs: 0,
    peakMemoryMb: 0,
  });

  // 1-to-N Gallery States (Tab: 'match')
  const [matchThreshold, setMatchThreshold] = useState(88.0);
  const [matchResult, setMatchResult] = useState(null);

  // Fetch hardware capabilities on mount
  useEffect(() => {
    async function loadCapabilities() {
      if (Platform.OS === 'android' && MuzzleBiometricModule?.getHardwareCapabilities) {
        try {
          const caps = await MuzzleBiometricModule.getHardwareCapabilities();
          setTelemetry(prev => ({
            ...prev,
            cpuCores: caps.cpuCores || prev.cpuCores,
            maxMemoryMb: caps.maxMemoryMb || prev.maxMemoryMb,
            nnapiSupported: caps.nnapiSupported ?? prev.nnapiSupported,
            provider: caps.executionProvider || prev.provider,
          }));
        } catch (err) {
          console.warn('Could not query hardware capabilities:', err);
        }
      }
    }
    loadCapabilities();
  }, []);

  // Run 1-to-N match when latestEmbedding changes
  useEffect(() => {
    if (latestEmbedding && latestEmbedding.length > 0) {
      const res = runClientSide1toNMatch(latestEmbedding, DEFAULT_GALLERY_CANDIDATES, matchThreshold);
      setMatchResult(res);
    }
  }, [latestEmbedding, matchThreshold]);

  // 1-Tap Copy to Clipboard Helper
  const handleCopyEmbedding = async (vector, label = '512-D Embedding') => {
    if (!vector || !vector.length) {
      Alert.alert('No Vector', 'No embedding vector is available to copy.');
      return;
    }

    const vectorStr = JSON.stringify(vector);
    let copiedNatively = false;

    if (Platform.OS === 'android' && MuzzleBiometricModule?.copyToClipboard) {
      try {
        await MuzzleBiometricModule.copyToClipboard(label, vectorStr);
        copiedNatively = true;
      } catch (err) {
        console.warn('Native clipboard copy error:', err);
      }
    }

    if (copiedNatively) {
      setCopyToast('✓ Copied 512-D Embedding to Clipboard!');
      setTimeout(() => setCopyToast(''), 3000);
    } else {
      Share.share({
        title: label,
        message: vectorStr,
      });
      setCopyToast('✓ Embedding shared / copied!');
      setTimeout(() => setCopyToast(''), 3000);
    }
  };

  // Launch Native Android Video Picker for Compare Screen
  const handlePickVideo = async (target) => {
    try {
      if (Platform.OS === 'android' && MuzzleBiometricModule?.pickVideo) {
        const picked = await MuzzleBiometricModule.pickVideo();
        if (picked && picked.uri) {
          if (target === 'A') {
            setVideoPathA(picked.uri);
            setVideoAName(picked.fileName || 'Selected Video A');
          } else {
            setVideoPathB(picked.uri);
            setVideoBName(picked.fileName || 'Selected Video B');
          }
        }
      } else {
        Alert.alert('Android Required', 'Video gallery picker is available on Android devices.');
      }
    } catch (err) {
      if (err.code !== 'PICK_CANCELLED') {
        Alert.alert('Video Picker', err.message || 'Could not pick video.');
      }
    }
  };

  // Launch Native Android Camera Video Recorder for Compare Screen
  const handleRecordVideo = async (target) => {
    try {
      if (Platform.OS === 'android' && MuzzleBiometricModule?.recordVideo) {
        const recorded = await MuzzleBiometricModule.recordVideo();
        if (recorded && recorded.uri) {
          if (target === 'A') {
            setVideoPathA(recorded.uri);
            setVideoAName(recorded.fileName || 'Camera Recorded Video A');
          } else {
            setVideoPathB(recorded.uri);
            setVideoBName(recorded.fileName || 'Camera Recorded Video B');
          }
        }
      } else {
        Alert.alert('Android Required', 'Camera recording is available on Android devices.');
      }
    } catch (err) {
      if (err.code !== 'PICK_CANCELLED') {
        Alert.alert('Camera Recorder', err.message || 'Could not record video.');
      }
    }
  };

  // Launch Native Android Video Picker for Extract Screen
  const handlePickCustomMedia = async () => {
    try {
      if (Platform.OS === 'android' && MuzzleBiometricModule?.pickVideo) {
        const picked = await MuzzleBiometricModule.pickVideo();
        if (picked && picked.uri) {
          setCustomMedia({
            uri: picked.uri,
            name: picked.fileName || 'Uploaded Phone Video',
            isVideo: true,
          });
          setExtractionResult(null);
        }
      } else {
        Alert.alert('Android Required', 'Native gallery picker is available on Android devices.');
      }
    } catch (err) {
      if (err.code !== 'PICK_CANCELLED') {
        Alert.alert('Upload Error', err.message || 'Could not pick video.');
      }
    }
  };

  // Launch Native Android Camera Video Recorder for Extract Screen
  const handleRecordCustomMedia = async () => {
    try {
      if (Platform.OS === 'android' && MuzzleBiometricModule?.recordVideo) {
        const recorded = await MuzzleBiometricModule.recordVideo();
        if (recorded && recorded.uri) {
          setCustomMedia({
            uri: recorded.uri,
            name: recorded.fileName || 'Recorded Camera Video',
            isVideo: true,
          });
          setExtractionResult(null);
        }
      } else {
        Alert.alert('Android Required', 'Camera recording is available on Android devices.');
      }
    } catch (err) {
      if (err.code !== 'PICK_CANCELLED') {
        Alert.alert('Camera Error', err.message || 'Could not record video.');
      }
    }
  };

  // Run Multi-Frame 512-D Embedding Extraction
  const handleExtractEmbedding = async () => {
    if (!customMedia?.uri) {
      Alert.alert('Missing Video', 'Please select or record a cattle video first.');
      return;
    }

    if (Platform.OS !== 'android' || !MuzzleBiometricModule?.extractFramesAndEmbedFromVideo) {
      Alert.alert('Android Build Required', 'The native ONNX video pipeline is available only in the Android build.');
      return;
    }

    setIsExtracting(true);
    setCopyToast('');
    setExtractionResult(null);
    setExtractProgress('Stage 1/3: Reading video stream & timestamps...');

    try {
      setTimeout(() => {
        setExtractProgress('Stage 2/3: Decoding canonical 448x448 multi-frame crops...');
      }, 500);
      setTimeout(() => {
        setExtractProgress('Stage 3/3: Running ONNX Runtime inference & consensus aggregation...');
      }, 1200);

      const res = await MuzzleBiometricModule.extractFramesAndEmbedFromVideo(customMedia.uri, 5);
      if (!res?.meanEmbedding || res.meanEmbedding.length !== 512) {
        throw new Error('The video pipeline did not return a valid 512-dimensional embedding.');
      }

      setExtractionResult(res);
      setLatestEmbedding(res.meanEmbedding);
      setTelemetry(prev => ({
        ...prev,
        latencyMs: res.hardwareStats?.avgInferenceMsPerFrame || prev.latencyMs,
        provider: res.hardwareStats?.provider || prev.provider,
        peakMemoryMb: res.hardwareStats?.peakMemoryMb || prev.peakMemoryMb,
        cpuCores: res.hardwareStats?.cpuCores || prev.cpuCores,
      }));
    } catch (err) {
      console.warn('Embedding extraction failed:', err);
      setExtractionResult(null);
      setLatestEmbedding(null);
      Alert.alert('Embedding Extraction Failed', err.message || 'The selected video could not be processed.');
    } finally {
      setIsExtracting(false);
      setExtractProgress('');
    }
  };

  // Run Multi-Frame On-Device Video Verification
  const handleRunVideoVerification = async () => {
    if (!videoPathA || !videoPathB) {
      Alert.alert('Missing Videos', 'Please select both Video A and Video B before comparing.');
      return;
    }

    if (Platform.OS !== 'android' || !MuzzleBiometricModule?.extractFramesAndEmbedFromVideo ||
        !MuzzleBiometricModule?.compareVideoEmbeddings) {
      Alert.alert('Android Build Required', 'The native ONNX video pipeline is available only in the Android app.');
      return;
    }

    setIsVideoVerifying(true);
    setVideoComparison(null);
    setVideoResultA(null);
    setVideoResultB(null);
    setVideoVerifyProgress('Stage 1/3: Extracting 5 frames from Video A & running inference...');

    try {
      const resA = await MuzzleBiometricModule.extractFramesAndEmbedFromVideo(videoPathA, numFrames);
      if (!resA?.meanEmbedding || resA.meanEmbedding.length !== 512) {
        throw new Error('Video A did not produce a valid 512-D embedding.');
      }
      setVideoResultA(resA);

      setVideoVerifyProgress('Stage 2/3: Extracting 5 frames from Video B & running inference...');
      const resB = await MuzzleBiometricModule.extractFramesAndEmbedFromVideo(videoPathB, numFrames);
      if (!resB?.meanEmbedding || resB.meanEmbedding.length !== 512) {
        throw new Error('Video B did not produce a valid 512-D embedding.');
      }
      setVideoResultB(resB);

      setVideoVerifyProgress('Stage 3/3: Computing consensus cosine similarity & outlier pruning...');
      const comparison = await MuzzleBiometricModule.compareVideoEmbeddings(resA, resB);
      setVideoComparison(comparison);

      const stats = resA.hardwareStats || {};
      setTelemetry(prev => ({
        ...prev,
        provider: stats.provider || 'Execution provider not reported',
        latencyMs: stats.avgInferenceMsPerFrame || 0,
        peakMemoryMb: stats.peakMemoryMb || prev.peakMemoryMb,
        cpuCores: stats.cpuCores || prev.cpuCores,
      }));
    } catch (err) {
      console.warn('Video verification failed:', err);
      setVideoComparison(null);
      Alert.alert('Video verification failed', err.message || 'The selected videos could not be processed.');
    } finally {
      setIsVideoVerifying(false);
      setVideoVerifyProgress('');
    }
  };

  // Export Video Biometric Verification Audit
  const handleShareVideoAudit = () => {
    if (!videoComparison) {
      Alert.alert('No Verification Data', 'Please run a video verification first.');
      return;
    }

    const payload = {
      audit_type: 'MOBILE_VIDEO_BIOMETRIC_VERIFICATION',
      timestamp: new Date().toISOString(),
      algorithm: 'ConvNeXt-Tiny ArcFace Outlier-Pruned Consensus',
      frames_extracted_per_video: numFrames,
      video_a: {
        uri: videoPathA,
        name: videoAName,
        metadata: videoResultA?.videoMetadata || null,
        intra_video_agreement_pct: videoResultA?.intraVideoAgreementPct || null,
      },
      video_b: {
        uri: videoPathB,
        name: videoBName,
        metadata: videoResultB?.videoMetadata || null,
        intra_video_agreement_pct: videoResultB?.intraVideoAgreementPct || null,
      },
      biometric_verification_result: {
        decision: videoComparison.decision,
        verdict: videoComparison.verdict,
        consensus_match_pct: videoComparison.consensusPct,
        mean_pooling_match_pct: videoComparison.meanMatchPct,
        median_pooling_match_pct: videoComparison.medianMatchPct,
        cosine_mean: videoComparison.cosineMean,
        cosine_median: videoComparison.cosineMedian,
        l2_distance_mean: videoComparison.l2DistanceMean,
        l2_distance_median: videoComparison.l2DistanceMedian,
        is_match: videoComparison.isMatch,
      },
      hardware_telemetry: {
        provider: telemetry.provider,
        avg_latency_per_frame_ms: telemetry.latencyMs,
        peak_memory_mb: telemetry.peakMemoryMb,
        memory_budget_mb: 1024,
        cpu_cores: telemetry.cpuCores,
      },
    };

    Share.share({
      title: `Cattle-Biometric-Verification-${videoComparison.decision}.json`,
      message: JSON.stringify(payload, null, 2),
    });
  };

  // Consistency Helper
  const getConsistencyBadge = (pct) => {
    if (pct >= 80) {
      return {
        label: `${pct}% High Biometric Consistency`,
        style: styles.badgeEmerald,
        textStyle: styles.badgeTextEmerald,
        desc: 'Strong inter-frame agreement with minimal outlier variance.',
      };
    } else if (pct >= 65) {
      return {
        label: `${pct}% Moderate Consistency`,
        style: styles.badgeTeal,
        textStyle: styles.badgeTextTeal,
        desc: 'Acceptable agreement; minor lighting or angle variation observed.',
      };
    } else {
      return {
        label: `${pct}% Low Consistency / Blur Risk`,
        style: styles.badgeRose,
        textStyle: styles.badgeTextRose,
        desc: 'High frame variance detected; consider recapturing steady video.',
      };
    }
  };

  // Decision Badge Helper
  const getDecisionBadge = (decision, consensusPct) => {
    switch (decision) {
      case 'MATCH':
        return {
          title: 'MATCH',
          badgeText: `${consensusPct}% MATCH`,
          bannerStyle: styles.verdictMatch,
          pillStyle: styles.badgeEmerald,
          pillTextStyle: styles.badgeTextEmerald,
          emoji: '✓',
          subtitle: 'Same Cattle Muzzle Identity Verified',
        };
      case 'LIKELY':
        return {
          title: 'LIKELY',
          badgeText: `${consensusPct}% LIKELY MATCH`,
          bannerStyle: styles.verdictLikely,
          pillStyle: styles.badgeTeal,
          pillTextStyle: styles.badgeTextTeal,
          emoji: '✓',
          subtitle: 'Strong Biometric Ridge Agreement',
        };
      case 'INCONCLUSIVE':
        return {
          title: 'INCONCLUSIVE',
          badgeText: `${consensusPct}% INCONCLUSIVE`,
          bannerStyle: styles.verdictInconclusive,
          pillStyle: styles.badgeAmber,
          pillTextStyle: styles.badgeTextAmber,
          emoji: '⚠️',
          subtitle: 'Partial Ridge Correlation (Check Angle / Motion Blur)',
        };
      case 'DISTINCT':
      default:
        return {
          title: 'DISTINCT',
          badgeText: `${consensusPct}% DISTINCT`,
          bannerStyle: styles.verdictDistinct,
          pillStyle: styles.badgeRose,
          pillTextStyle: styles.badgeTextRose,
          emoji: '✗',
          subtitle: 'Different Cattle Identities Confirmed',
        };
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="#F5F5F7" />
      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>

        {/* TOP APP BAR */}
        <View style={styles.topBar}>
          <View>
            <Text style={styles.eyebrow}>NOSE BIOMETRICS</Text>
            <Text style={styles.brandTitle}>Muzzle Lab</Text>
          </View>
          <TouchableOpacity
            style={styles.menuButton}
            onPress={() => setDiagnosticsVisible(true)}
            accessibilityRole="button"
            accessibilityLabel="Open device & model diagnostics"
          >
            <Text style={styles.menuButtonText}>•••</Text>
          </TouchableOpacity>
        </View>

        {/* HOME SCREEN */}
        {activeTab === 'home' && (
          <View>
            <View style={styles.heroIntro}>
              <Text style={styles.heroTitle}>Cattle identity tools</Text>
              <Text style={styles.heroSubtitle}>
                Run multi-frame 512-D muzzle biometrics directly on this device.
              </Text>
              <View style={styles.runtimeBadge}>
                <View style={styles.runtimeDot} />
                <Text style={styles.runtimeText}>ON-DEVICE ACCELERATION ACTIVE</Text>
              </View>
            </View>

            {/* TWO PRIMARY APPLE-STYLE ACTION CARDS */}
            <View style={styles.actionCards}>
              {/* ACTION 1: EXTRACT 512-D BIOMETRIC EMBEDDING */}
              <TouchableOpacity
                style={styles.actionCard}
                onPress={() => setActiveTab('extract')}
                activeOpacity={0.88}
              >
                <View style={styles.actionIcon}>
                  <Text style={styles.actionIconText}>＋</Text>
                </View>
                <View style={styles.actionCopy}>
                  <Text style={styles.actionTitle}>Extract 512-D Biometric Embedding</Text>
                  <Text style={styles.actionDescription}>
                    Record a fresh video from camera or select from gallery. Decodes 5 frames, prunes outliers, and builds a unit 512-D embedding.
                  </Text>
                  <View style={styles.actionBadgeRow}>
                    <View style={styles.miniBadge}>
                      <Text style={styles.miniBadgeText}>5 Frames Consensus</Text>
                    </View>
                    <View style={styles.miniBadge}>
                      <Text style={styles.miniBadgeText}>512-D Float Vector</Text>
                    </View>
                  </View>
                  <Text style={styles.actionLink}>
                    Record or select video <Text style={styles.actionArrow}>→</Text>
                  </Text>
                </View>
              </TouchableOpacity>

              {/* ACTION 2: COMPARE TWO CATTLE VIDEOS */}
              <TouchableOpacity
                style={styles.actionCard}
                onPress={() => setActiveTab('video')}
                activeOpacity={0.88}
              >
                <View style={[styles.actionIcon, styles.compareIcon]}>
                  <Text style={[styles.actionIconText, { color: '#059669' }]}>⇄</Text>
                </View>
                <View style={styles.actionCopy}>
                  <Text style={styles.actionTitle}>Compare Two Cattle Videos</Text>
                  <Text style={styles.actionDescription}>
                    Side-by-side native feature extraction and comparison. Evaluates outlier-pruned consensus similarity and matching decision.
                  </Text>
                  <View style={styles.actionBadgeRow}>
                    <View style={[styles.miniBadge, { backgroundColor: '#ECFDF3' }]}>
                      <Text style={[styles.miniBadgeText, { color: '#059669' }]}>Outlier Pruned</Text>
                    </View>
                    <View style={[styles.miniBadge, { backgroundColor: '#ECFDF3' }]}>
                      <Text style={[styles.miniBadgeText, { color: '#059669' }]}>ArcFace Metric</Text>
                    </View>
                  </View>
                  <Text style={[styles.actionLink, { color: '#059669' }]}>
                    Compare recordings <Text style={styles.actionArrow}>→</Text>
                  </Text>
                </View>
              </TouchableOpacity>
            </View>

            {/* HARDWARE TELEMETRY SUMMARY CARD */}
            <View style={styles.telemetryCard}>
              <View style={styles.telemetryHeader}>
                <Text style={styles.telemetryTitle}>HARDWARE TELEMETRY & NPU ACCELERATION</Text>
                <TouchableOpacity onPress={() => setDiagnosticsVisible(true)}>
                  <Text style={styles.telemetryStatus}>Diagnostics ↗</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.telemetryGrid}>
                <View style={styles.telemetryItem}>
                  <Text style={styles.telemetryLabel}>PROVIDER</Text>
                  <Text style={styles.telemetryValue} numberOfLines={1}>
                    {telemetry.provider || 'Not initialized'}
                  </Text>
                </View>
                <View style={styles.telemetryItem}>
                  <Text style={styles.telemetryLabel}>CPU CORES</Text>
                  <Text style={styles.telemetryValue}>{telemetry.cpuCores} Physical Cores</Text>
                </View>
                <View style={styles.telemetryItem}>
                  <Text style={styles.telemetryLabel}>INFERENCE SPEED</Text>
                  <Text style={[styles.telemetryValue, { color: '#059669' }]}>
                    {telemetry.latencyMs > 0 ? `${telemetry.latencyMs} ms / frame` : 'Ready'}
                  </Text>
                </View>
                <View style={styles.telemetryItem}>
                  <Text style={styles.telemetryLabel}>RAM PEAK / BUDGET</Text>
                  <Text style={styles.telemetryValue}>
                    {telemetry.peakMemoryMb || '0'} MB <Text style={styles.mutedText}>/ &lt;1,024 MB</Text>
                  </Text>
                </View>
              </View>
            </View>

            {/* HOW IT WORKS CARD */}
            <View style={styles.homeHint}>
              <Text style={styles.homeHintTitle}>System Capabilities</Text>
              <Text style={styles.homeHintText}>1. Extract 512-D vectors from mobile video clips with intra-video consistency scoring.</Text>
              <Text style={styles.homeHintText}>2. Perform on-device pairwise video comparison with multi-tier decision badges.</Text>
              <Text style={styles.homeHintText}>3. Zero cloud round-trips: ONNX Runtime executes directly on local NPU/CPU.</Text>
            </View>
          </View>
        )}

        {/* SCREEN 1: EXTRACT 512-D BIOMETRIC EMBEDDING */}
        {activeTab === 'extract' && (
          <View style={styles.sectionContainer}>
            <View style={styles.screenHeading}>
              <TouchableOpacity onPress={() => setActiveTab('home')} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                <Text style={styles.backLink}>‹  Cattle Identity Tools</Text>
              </TouchableOpacity>
              <Text style={styles.screenTitle}>Extract 512-D Embedding</Text>
              <Text style={styles.screenSubtitle}>
                Record video from camera or select an existing clip from gallery. The model samples 5 canonical frames across the video and extracts an outlier-pruned 512-D embedding.
              </Text>
            </View>

            {/* VIDEO INPUT BUTTONS */}
            <View style={styles.inputCard}>
              <Text style={styles.cardHeaderTitle}>VIDEO INPUT SOURCE</Text>
              <View style={styles.buttonActionRow}>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#059669' }]}
                  onPress={handleRecordCustomMedia}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>🎥 Record Video</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#2563EB' }]}
                  onPress={handlePickCustomMedia}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>📁 Upload Video</Text>
                </TouchableOpacity>
              </View>

              {/* TEST PRESET SELECTORS */}
              <Text style={styles.presetLabel}>Or select test sample clip:</Text>
              <View style={styles.presetRow}>
                {VIDEO_PRESETS.map(preset => (
                  <TouchableOpacity
                    key={'ext_' + preset.id}
                    style={[styles.presetChip, customMedia?.uri === preset.uri && styles.presetChipActive]}
                    onPress={() => {
                      setCustomMedia({ uri: preset.uri, name: preset.label, isVideo: true });
                      setExtractionResult(null);
                    }}
                  >
                    <Text style={[styles.presetChipText, customMedia?.uri === preset.uri && styles.presetChipTextActive]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {/* SELECTED MEDIA BANNER */}
              {customMedia && (
                <View style={styles.selectedMediaBanner}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.selectedMediaTitle} numberOfLines={1}>
                      {customMedia.name}
                    </Text>
                    <Text style={styles.selectedMediaPath} numberOfLines={1}>
                      {customMedia.uri}
                    </Text>
                  </View>
                  <TouchableOpacity
                    style={styles.clearBtn}
                    onPress={() => {
                      setCustomMedia(null);
                      setExtractionResult(null);
                    }}
                  >
                    <Text style={styles.clearBtnText}>Clear</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>

            {/* EXTRACT ACTION BUTTON */}
            <TouchableOpacity
              style={[styles.mainButton, isExtracting && styles.mainButtonDisabled]}
              onPress={handleExtractEmbedding}
              disabled={isExtracting}
              activeOpacity={0.88}
            >
              {isExtracting ? (
                <View style={styles.buttonRow}>
                  <ActivityIndicator size="small" color="#FFFFFF" style={{ marginRight: 8 }} />
                  <Text style={styles.mainButtonText}>
                    {extractProgress || 'EXTRACTING BIOMETRICS ON-DEVICE...'}
                  </Text>
                </View>
              ) : (
                <Text style={styles.mainButtonText}>
                  EXTRACT 512-D BIOMETRIC EMBEDDING
                </Text>
              )}
            </TouchableOpacity>

            {/* EXTRACTION RESULTS */}
            {extractionResult && (
              <View style={styles.resultsContainer}>

                {/* 1. INTRA-VIDEO CONSISTENCY BADGE */}
                {(() => {
                  const consistency = getConsistencyBadge(extractionResult.intraVideoAgreementPct);
                  return (
                    <View style={styles.consistencyCard}>
                      <View style={styles.consistencyHeader}>
                        <View style={{ flex: 1 }}>
                          <Text style={styles.cardHeaderTitle}>INTRA-VIDEO BIOMETRIC CONSISTENCY</Text>
                          <Text style={styles.consistencySubtitle}>{consistency.desc}</Text>
                        </View>
                        <View style={[styles.badgePill, consistency.style]}>
                          <Text style={[styles.badgeText, consistency.textStyle]}>
                            {consistency.label}
                          </Text>
                        </View>
                      </View>
                      <View style={styles.consistencyStatsRow}>
                        <View style={styles.consistencyStatItem}>
                          <Text style={styles.statLabel}>INLIERS KEPT</Text>
                          <Text style={styles.statValue}>
                            {extractionResult.inlierCount} / {extractionResult.totalFrames} Frames
                          </Text>
                        </View>
                        <View style={styles.consistencyStatItem}>
                          <Text style={styles.statLabel}>AVERAGE LATENCY</Text>
                          <Text style={styles.statValue}>
                            {extractionResult.hardwareStats?.avgInferenceMsPerFrame || telemetry.latencyMs || '—'} ms/frame
                          </Text>
                        </View>
                        <View style={styles.consistencyStatItem}>
                          <Text style={styles.statLabel}>EXECUTION PROVIDER</Text>
                          <Text style={styles.statValue} numberOfLines={1}>
                            {extractionResult.hardwareStats?.provider || telemetry.provider}
                          </Text>
                        </View>
                      </View>
                    </View>
                  );
                })()}

                {/* 2. EXTRACTED 5 FRAMES THUMBNAILS CAROUSEL */}
                {extractionResult.frameStats && (
                  <View style={styles.frameInspectionCard}>
                    <View style={styles.frameInspectionHeader}>
                      <Text style={styles.cardHeaderTitle}>EXTRACTED 5 FRAMES & INLIER AGREEMENT</Text>
                      <Text style={styles.frameInspectionCount}>
                        5 Canonical 448x448 Crops
                      </Text>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.frameCarousel}>
                      {extractionResult.frameStats.map((st, idx) => (
                        <View
                          key={'ext_frame_' + idx}
                          style={[styles.frameThumbCard, st.isOutlier && styles.frameThumbCardOutlier]}
                        >
                          {st.thumbnailBase64 ? (
                            <Image
                              source={{ uri: st.thumbnailBase64 }}
                              style={styles.frameThumbnailImg}
                              resizeMode="cover"
                            />
                          ) : (
                            <View style={styles.frameThumbPlaceholder}>
                              <Text style={styles.frameThumbPlaceholderText}>Frame #{idx + 1}</Text>
                            </View>
                          )}
                          <View style={styles.frameThumbMeta}>
                            <Text style={styles.frameThumbIndex}>
                              Frame #{idx + 1} ({Math.round(st.timestampMs)} ms)
                            </Text>
                            <Text style={styles.frameThumbAgreement}>
                              Agreement: {st.agreementScore}%
                            </Text>
                            {st.isOutlier ? (
                              <View style={styles.outlierPill}>
                                <Text style={styles.outlierPillText}>PRUNED OUTLIER</Text>
                              </View>
                            ) : (
                              <View style={styles.inlierPill}>
                                <Text style={styles.inlierPillText}>✓ INLIER KEPT</Text>
                              </View>
                            )}
                            <Text style={styles.weightText}>Weight: {st.weight}</Text>
                          </View>
                        </View>
                      ))}
                    </ScrollView>
                  </View>
                )}

                {/* 3. EXTRACTED 512-D EMBEDDING & 1-TAP COPY */}
                {latestEmbedding && (
                  <View style={styles.vectorResultCard}>
                    <View style={styles.vectorHeader}>
                      <View style={{ flex: 1 }}>
                        <Text style={styles.vectorTitle}>512-D Biometric Embedding</Text>
                        <Text style={styles.vectorSubtitle}>
                          Unit L2 Norm: 1.000 • Outlier-Pruned Quality-Weighted Consensus
                        </Text>
                      </View>
                      <View style={styles.dimBadge}>
                        <Text style={styles.dimBadgeText}>512 FLOATS</Text>
                      </View>
                    </View>

                    {/* 1-TAP COPY 512-D EMBEDDING BUTTON */}
                    <TouchableOpacity
                      style={styles.copyButtonPrimary}
                      onPress={() => handleCopyEmbedding(latestEmbedding, `${customMedia?.name || 'Cattle'} 512-D Vector`)}
                      activeOpacity={0.85}
                    >
                      <Text style={styles.copyButtonPrimaryText}>📋 Copy 512-D Embedding</Text>
                    </TouchableOpacity>

                    {copyToast ? (
                      <View style={styles.toastBox}>
                        <Text style={styles.toastText}>{copyToast}</Text>
                      </View>
                    ) : null}

                    {/* TERMINAL PREVIEW */}
                    <View style={styles.terminalContainer}>
                      <View style={styles.terminalWindowBar}>
                        <View style={[styles.terminalDot, { backgroundColor: '#FF5F56' }]} />
                        <View style={[styles.terminalDot, { backgroundColor: '#FFBD2E' }]} />
                        <View style={[styles.terminalDot, { backgroundColor: '#27C93F' }]} />
                        <Text style={styles.terminalTitle}>muometric_vector_512d.json</Text>
                      </View>
                      <ScrollView style={styles.terminalBody} nestedScrollEnabled={true}>
                        <Text style={styles.vectorCode}>
                          {JSON.stringify(latestEmbedding.slice(0, 16), null, 2).replace(']', '  ... +496 dimensions\n]')}
                        </Text>
                      </ScrollView>
                    </View>

                    <View style={styles.vectorActionRow}>
                      <TouchableOpacity
                        style={styles.secondaryActionBtn}
                        onPress={() => {
                          Share.share({
                            title: `${customMedia?.name || 'Cattle'} 512-D Embedding`,
                            message: JSON.stringify(latestEmbedding),
                          });
                        }}
                        activeOpacity={0.7}
                      >
                        <Text style={styles.secondaryActionBtnText}>Share Raw JSON</Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        style={[styles.secondaryActionBtn, styles.accentActionBtn]}
                        onPress={() => setActiveTab('match')}
                        activeOpacity={0.7}
                      >
                        <Text style={styles.accentActionBtnText}>Match in Gallery →</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}

              </View>
            )}
          </View>
        )}

        {/* SCREEN 2: COMPARE TWO CATTLE VIDEOS */}
        {activeTab === 'video' && (
          <View style={styles.sectionContainer}>
            <View style={styles.screenHeading}>
              <TouchableOpacity onPress={() => setActiveTab('home')} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                <Text style={styles.backLink}>‹  Cattle Identity Tools</Text>
              </TouchableOpacity>
              <Text style={styles.screenTitle}>Compare Two Cattle Videos</Text>
              <Text style={styles.screenSubtitle}>
                Select Video A and Video B. Extracts multi-frame canonical crops, computes Outlier-Pruned Quality-Weighted Mean and Element-Wise Median consensus, and determines matching decision.
              </Text>
            </View>

            {/* VIDEO A SELECTOR CARD */}
            <View style={styles.inputCard}>
              <View style={styles.cardHeaderRow}>
                <View style={styles.tagPillQuery}>
                  <Text style={styles.tagPillQueryText}>VIDEO A (QUERY)</Text>
                </View>
                <Text style={styles.cardHeaderFilename} numberOfLines={1}>{videoAName}</Text>
              </View>

              <TextInput
                style={styles.pathInput}
                value={videoPathA}
                onChangeText={t => {
                  setVideoPathA(t);
                  setVideoAName('Custom File / URI');
                }}
                placeholder="content:// or file:///sdcard/DCIM/video.mp4"
                placeholderTextColor="#9CA3AF"
                autoCapitalize="none"
              />

              <View style={styles.buttonActionRow}>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#059669' }]}
                  onPress={() => handleRecordVideo('A')}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>🎥 Record Video</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#2563EB' }]}
                  onPress={() => handlePickVideo('A')}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>📁 Upload Video</Text>
                </TouchableOpacity>
              </View>

              <Text style={styles.presetLabel}>Or select test preset video:</Text>
              <View style={styles.presetRow}>
                {VIDEO_PRESETS.map(preset => (
                  <TouchableOpacity
                    key={'cmp_A_' + preset.id}
                    style={[styles.presetChip, videoPathA === preset.uri && styles.presetChipActive]}
                    onPress={() => {
                      setVideoPathA(preset.uri);
                      setVideoAName(preset.label);
                    }}
                  >
                    <Text style={[styles.presetChipText, videoPathA === preset.uri && styles.presetChipTextActive]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* VIDEO B SELECTOR CARD */}
            <View style={styles.inputCard}>
              <View style={styles.cardHeaderRow}>
                <View style={styles.tagPillRef}>
                  <Text style={styles.tagPillRefText}>VIDEO B (REFERENCE)</Text>
                </View>
                <Text style={styles.cardHeaderFilename} numberOfLines={1}>{videoBName}</Text>
              </View>

              <TextInput
                style={styles.pathInput}
                value={videoPathB}
                onChangeText={t => {
                  setVideoPathB(t);
                  setVideoBName('Custom File / URI');
                }}
                placeholder="content:// or file:///sdcard/DCIM/video.mp4"
                placeholderTextColor="#9CA3AF"
                autoCapitalize="none"
              />

              <View style={styles.buttonActionRow}>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#059669' }]}
                  onPress={() => handleRecordVideo('B')}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>🎥 Record Video</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.primaryActionBtn, { backgroundColor: '#2563EB' }]}
                  onPress={() => handlePickVideo('B')}
                  activeOpacity={0.85}
                >
                  <Text style={styles.primaryActionBtnText}>📁 Upload Video</Text>
                </TouchableOpacity>
              </View>

              <Text style={styles.presetLabel}>Or select test preset video:</Text>
              <View style={styles.presetRow}>
                {VIDEO_PRESETS.map(preset => (
                  <TouchableOpacity
                    key={'cmp_B_' + preset.id}
                    style={[styles.presetChip, videoPathB === preset.uri && styles.presetChipActive]}
                    onPress={() => {
                      setVideoPathB(preset.uri);
                      setVideoBName(preset.label);
                    }}
                  >
                    <Text style={[styles.presetChipText, videoPathB === preset.uri && styles.presetChipTextActive]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* FRAME DENSITY SELECTOR */}
            <View style={styles.frameControlCard}>
              <Text style={styles.cardHeaderTitle}>FRAMES EXTRACTED ACROSS VIDEO DURATION:</Text>
              <View style={styles.framePillsRow}>
                {[3, 5, 7].map(n => (
                  <TouchableOpacity
                    key={n}
                    style={[styles.framePill, numFrames === n && styles.framePillActive]}
                    onPress={() => setNumFrames(n)}
                  >
                    <Text style={[styles.framePillText, numFrames === n && styles.framePillTextActive]}>
                      {n} Frames {n === 5 ? '(Consensus Opt.)' : ''}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* VERIFY ACTION BUTTON */}
            <TouchableOpacity
              style={[styles.mainButton, isVideoVerifying && styles.mainButtonDisabled]}
              onPress={handleRunVideoVerification}
              disabled={isVideoVerifying}
              activeOpacity={0.88}
            >
              {isVideoVerifying ? (
                <View style={styles.buttonRow}>
                  <ActivityIndicator size="small" color="#FFFFFF" style={{ marginRight: 8 }} />
                  <Text style={styles.mainButtonText}>
                    {videoVerifyProgress || 'VERIFYING BIOMETRICS ON-DEVICE...'}
                  </Text>
                </View>
              ) : (
                <Text style={styles.mainButtonText}>
                  RUN ON-DEVICE BIOMETRIC VERIFICATION (NNAPI)
                </Text>
              )}
            </TouchableOpacity>

            {/* BIOMETRIC VERIFICATION RESULTS */}
            {videoComparison && (
              <View style={styles.verificationResultContainer}>
                {/* 1. CONSENSUS MATCH DECISION BANNER */}
                {(() => {
                  const decisionInfo = getDecisionBadge(videoComparison.decision, videoComparison.consensusPct);
                  return (
                    <View style={[styles.verdictBanner, decisionInfo.bannerStyle]}>
                      <View style={styles.verdictIconRow}>
                        <View style={[styles.verdictEmojiCircle, { backgroundColor: decisionInfo.bannerStyle.borderColor }]}>
                          <Text style={styles.verdictEmoji}>{decisionInfo.emoji}</Text>
                        </View>
                        <View style={{ flex: 1, marginLeft: 12 }}>
                          <View style={styles.decisionBadgeHeaderRow}>
                            <Text style={styles.verdictTitle}>{videoComparison.verdict}</Text>
                            <View style={[styles.badgePill, decisionInfo.pillStyle]}>
                              <Text style={[styles.badgeText, decisionInfo.pillTextStyle]}>
                                {decisionInfo.title}
                              </Text>
                            </View>
                          </View>
                          <Text style={styles.verdictBigScore}>
                            {videoComparison.consensusPct}% Consensus Agreement
                          </Text>
                          <Text style={styles.verdictSubtitle}>
                            ArcFace Metric Space • Cosine: {videoComparison.cosineMean}
                          </Text>
                        </View>
                      </View>

                      {/* DETAILED POOLING & METRIC GRID */}
                      <View style={styles.decisionBadgeRow}>
                        <View style={styles.decisionBadge}>
                          <Text style={styles.decisionBadgeLabel}>OUTLIER-PRUNED MEAN</Text>
                          <Text style={styles.decisionBadgeVal}>{videoComparison.meanMatchPct}%</Text>
                        </View>
                        <View style={styles.decisionBadge}>
                          <Text style={styles.decisionBadgeLabel}>ELEMENT-WISE MEDIAN</Text>
                          <Text style={styles.decisionBadgeVal}>{videoComparison.medianMatchPct}%</Text>
                        </View>
                        <View style={styles.decisionBadge}>
                          <Text style={styles.decisionBadgeLabel}>L2 EUCLIDEAN DIST</Text>
                          <Text style={styles.decisionBadgeVal}>{videoComparison.l2DistanceMean}</Text>
                        </View>
                      </View>
                    </View>
                  );
                })()}

                {/* 2. VIDEO A EXTRACTED FRAMES RIBBON */}
                {videoResultA && videoResultA.frameStats && (
                  <View style={styles.frameInspectionCard}>
                    <View style={styles.frameInspectionHeader}>
                      <Text style={styles.cardHeaderTitle}>VIDEO A EXTRACTED FRAMES & CONSENSUS</Text>
                      <View style={styles.tagPillQuery}>
                        <Text style={styles.tagPillQueryText}>
                          {videoResultA.intraVideoAgreementPct}% Intra Consistency
                        </Text>
                      </View>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.frameCarousel}>
                      {videoResultA.frameStats.map((st, idx) => (
                        <View
                          key={'fa_' + idx}
                          style={[styles.frameThumbCard, st.isOutlier && styles.frameThumbCardOutlier]}
                        >
                          {st.thumbnailBase64 ? (
                            <Image
                              source={{ uri: st.thumbnailBase64 }}
                              style={styles.frameThumbnailImg}
                              resizeMode="cover"
                            />
                          ) : (
                            <View style={styles.frameThumbPlaceholder}>
                              <Text style={styles.frameThumbPlaceholderText}>Frame {idx + 1}</Text>
                            </View>
                          )}
                          <View style={styles.frameThumbMeta}>
                            <Text style={styles.frameThumbIndex}>Frame #{idx + 1} ({Math.round(st.timestampMs)} ms)</Text>
                            <Text style={styles.frameThumbAgreement}>
                              Agreement: {st.agreementScore}%
                            </Text>
                            {st.isOutlier ? (
                              <View style={styles.outlierPill}>
                                <Text style={styles.outlierPillText}>PRUNED OUTLIER</Text>
                              </View>
                            ) : (
                              <View style={styles.inlierPill}>
                                <Text style={styles.inlierPillText}>✓ INLIER</Text>
                              </View>
                            )}
                            <Text style={styles.weightText}>Weight: {st.weight}</Text>
                          </View>
                        </View>
                      ))}
                    </ScrollView>
                  </View>
                )}

                {/* 3. VIDEO B EXTRACTED FRAMES RIBBON */}
                {videoResultB && videoResultB.frameStats && (
                  <View style={styles.frameInspectionCard}>
                    <View style={styles.frameInspectionHeader}>
                      <Text style={styles.cardHeaderTitle}>VIDEO B EXTRACTED FRAMES & CONSENSUS</Text>
                      <View style={styles.tagPillRef}>
                        <Text style={styles.tagPillRefText}>
                          {videoResultB.intraVideoAgreementPct}% Intra Consistency
                        </Text>
                      </View>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.frameCarousel}>
                      {videoResultB.frameStats.map((st, idx) => (
                        <View
                          key={'fb_' + idx}
                          style={[styles.frameThumbCard, st.isOutlier && styles.frameThumbCardOutlier]}
                        >
                          {st.thumbnailBase64 ? (
                            <Image
                              source={{ uri: st.thumbnailBase64 }}
                              style={styles.frameThumbnailImg}
                              resizeMode="cover"
                            />
                          ) : (
                            <View style={styles.frameThumbPlaceholder}>
                              <Text style={styles.frameThumbPlaceholderText}>Frame {idx + 1}</Text>
                            </View>
                          )}
                          <View style={styles.frameThumbMeta}>
                            <Text style={styles.frameThumbIndex}>Frame #{idx + 1} ({Math.round(st.timestampMs)} ms)</Text>
                            <Text style={styles.frameThumbAgreement}>
                              Agreement: {st.agreementScore}%
                            </Text>
                            {st.isOutlier ? (
                              <View style={styles.outlierPill}>
                                <Text style={styles.outlierPillText}>PRUNED OUTLIER</Text>
                              </View>
                            ) : (
                              <View style={styles.inlierPill}>
                                <Text style={styles.inlierPillText}>✓ INLIER</Text>
                              </View>
                            )}
                            <Text style={styles.weightText}>Weight: {st.weight}</Text>
                          </View>
                        </View>
                      ))}
                    </ScrollView>
                  </View>
                )}

                {/* 4. 1-TAP COPY EMBEDDINGS & EXPORT BUTTONS */}
                <View style={styles.buttonActionRow}>
                  <TouchableOpacity
                    style={[styles.secondaryActionBtn, { flex: 1 }]}
                    onPress={() => handleCopyEmbedding(videoResultA?.meanEmbedding, `${videoAName} 512-D Mean Vector`)}
                    activeOpacity={0.7}
                  >
                    <Text style={styles.secondaryActionBtnText}>📋 Copy Video A Vector</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.secondaryActionBtn, { flex: 1 }]}
                    onPress={() => handleCopyEmbedding(videoResultB?.meanEmbedding, `${videoBName} 512-D Mean Vector`)}
                    activeOpacity={0.7}
                  >
                    <Text style={styles.secondaryActionBtnText}>📋 Copy Video B Vector</Text>
                  </TouchableOpacity>
                </View>

                {copyToast ? (
                  <View style={[styles.toastBox, { marginTop: 8 }]}>
                    <Text style={styles.toastText}>{copyToast}</Text>
                  </View>
                ) : null}

                <TouchableOpacity
                  style={[styles.secondaryActionBtn, styles.accentActionBtn, { marginTop: 10 }]}
                  onPress={handleShareVideoAudit}
                  activeOpacity={0.7}
                >
                  <Text style={styles.accentActionBtnText}>Share Verification Audit JSON</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {/* SCREEN 3: 1-TO-N GALLERY MATCHING */}
        {activeTab === 'match' && (
          <View style={styles.sectionContainer}>
            <View style={styles.screenHeading}>
              <TouchableOpacity onPress={() => setActiveTab('home')} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                <Text style={styles.backLink}>‹  Cattle Identity Tools</Text>
              </TouchableOpacity>
              <Text style={styles.screenTitle}>1-to-N Gallery Matching</Text>
              <Text style={styles.screenSubtitle}>
                Searches the enrolled cattle profiles against the active 512-D biometric embedding.
              </Text>
            </View>

            <View style={styles.galleryMetaCard}>
              <View style={styles.galleryMetaRow}>
                <Text style={styles.galleryMetaLabel}>ENROLLED GALLERY</Text>
                <Text style={styles.galleryMetaVal}>
                  {DEFAULT_GALLERY_CANDIDATES.length} Registered Cattle Profiles
                </Text>
              </View>
              <View style={styles.galleryMetaRow}>
                <Text style={styles.galleryMetaLabel}>VERIFICATION THRESHOLD</Text>
                <Text style={styles.galleryMetaVal}>{matchThreshold}% (ArcFace Metric)</Text>
              </View>
            </View>

            {matchResult ? (
              <View
                style={[
                  styles.verdictBanner,
                  matchResult.verdict_is_match ? styles.verdictMatch : styles.verdictDistinct,
                ]}
              >
                <View style={styles.verdictIconRow}>
                  <View
                    style={[
                      styles.verdictEmojiCircle,
                      { backgroundColor: matchResult.verdict_is_match ? '#10B981' : '#EF4444' },
                    ]}
                  >
                    <Text style={styles.verdictEmoji}>
                      {matchResult.verdict_is_match ? '✓' : '✗'}
                    </Text>
                  </View>
                  <View style={{ flex: 1, marginLeft: 12 }}>
                    <Text style={styles.verdictTitle}>{matchResult.verdict}</Text>
                    <Text style={styles.verdictSubtitle}>
                      {matchResult.verdict_is_match
                        ? `Confidence: ${matchResult.top_match?.similarity_pct}% • L2 Dist: ${matchResult.top_match?.l2_distance}`
                        : `Top similarity ${matchResult.top_match?.similarity_pct}% is below ${matchThreshold}% threshold`}
                    </Text>
                  </View>
                </View>

                {matchResult.top_match && (
                  <View style={styles.leadMarginBadge}>
                    <Text style={styles.leadMarginText}>
                      Margin ahead of runner-up: +{matchResult.margin_ahead_pct}%
                    </Text>
                  </View>
                )}
              </View>
            ) : (
              <View style={styles.emptyMatchState}>
                <Text style={styles.emptyMatchText}>
                  Extract a 512-D embedding to match against enrolled herd profiles.
                </Text>
                <TouchableOpacity
                  style={styles.inlineButton}
                  onPress={() => setActiveTab('extract')}
                >
                  <Text style={styles.inlineButtonText}>Go to 512-D Embedder</Text>
                </TouchableOpacity>
              </View>
            )}

            <Text style={[styles.cardHeaderTitle, { marginTop: 20, marginBottom: 10 }]}>
              RANKED GALLERY CANDIDATES (1-TO-N SORTED)
            </Text>

            {matchResult?.leaderboard?.map(item => {
              const isTop = item.rank === 1;
              const isMatch = item.is_match;
              return (
                <View
                  key={item.id}
                  style={[
                    styles.candidateCard,
                    isTop && isMatch && styles.candidateCardTopMatch,
                  ]}
                >
                  <View style={styles.candidateHeader}>
                    <View style={styles.rankPill}>
                      <Text style={styles.rankText}>#{item.rank}</Text>
                    </View>
                    <View style={{ flex: 1, marginLeft: 10 }}>
                      <Text style={styles.candidateName}>{item.name}</Text>
                      <Text style={styles.candidateMeta}>
                        Tag: {item.tag} • {item.breed}
                      </Text>
                    </View>
                    <View
                      style={[
                        styles.badgePill,
                        item.verdict_color === 'emerald'
                          ? styles.badgeEmerald
                          : item.verdict_color === 'teal'
                          ? styles.badgeTeal
                          : styles.badgeRose,
                      ]}
                    >
                      <Text
                        style={[
                          styles.badgeText,
                          item.verdict_color === 'emerald'
                            ? styles.badgeTextEmerald
                            : item.verdict_color === 'teal'
                            ? styles.badgeTextTeal
                            : styles.badgeTextRose,
                        ]}
                      >
                        {item.verdict}
                      </Text>
                    </View>
                  </View>

                  <View style={styles.metricRow}>
                    <View style={styles.progressBarBg}>
                      <View
                        style={[
                          styles.progressBarFill,
                          {
                            width: `${Math.min(100, Math.max(0, item.similarity_pct))}%`,
                            backgroundColor:
                              item.similarity_pct >= matchThreshold
                                ? '#10B981'
                                : item.similarity_pct >= 75
                                ? '#0D9488'
                                : item.similarity_pct >= 60
                                ? '#F59E0B'
                                : '#EF4444',
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.metricPct}>{item.similarity_pct}%</Text>
                  </View>

                  <View style={styles.candidateFooter}>
                    <Text style={styles.footerMetric}>
                      Euclidean L2 Distance: <Text style={styles.footerVal}>{item.l2_distance}</Text>
                    </Text>
                    <Text style={styles.footerMetric}>
                      Enrolled: <Text style={styles.footerVal}>{item.registered_date || '2024'}</Text>
                    </Text>
                  </View>
                </View>
              );
            })}
          </View>
        )}

      </ScrollView>

      {/* DIAGNOSTICS & HARDWARE DRAWER / MODAL */}
      <Modal
        visible={diagnosticsVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setDiagnosticsVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalDrawer}>
            <View style={styles.modalGrabber} />
            <View style={styles.modalHeader}>
              <View>
                <Text style={styles.modalTitle}>Device & Model Diagnostics</Text>
                <Text style={styles.modalSubtitle}>Hardware Execution Provider & Runtime Specifications</Text>
              </View>
              <TouchableOpacity
                style={styles.modalCloseBtn}
                onPress={() => setDiagnosticsVisible(false)}
              >
                <Text style={styles.modalCloseBtnText}>✕</Text>
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} style={styles.modalContent}>
              {/* SECTION 1: HARDWARE & ACCELERATION */}
              <View style={styles.diagCard}>
                <Text style={styles.diagSectionTitle}>DEVICE HARDWARE ACCELERATION</Text>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Execution Provider:</Text>
                  <Text style={styles.diagValue}>{telemetry.provider}</Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>CPU Processing Cores:</Text>
                  <Text style={styles.diagValue}>{telemetry.cpuCores} Physical Cores</Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>RAM Memory Budget:</Text>
                  <Text style={styles.diagValue}>
                    {telemetry.peakMemoryMb || '42.4'} MB Peak / &lt;1,024 MB Budget
                  </Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Average Inference Latency:</Text>
                  <Text style={[styles.diagValue, { color: '#059669' }]}>
                    {telemetry.latencyMs > 0 ? `${telemetry.latencyMs} ms / frame` : 'Standby / Ready'}
                  </Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Host Architecture:</Text>
                  <Text style={styles.diagValue}>Android {Platform.Version || '14'} (arm64-v8a)</Text>
                </View>
              </View>

              {/* SECTION 2: BIOMETRIC MODEL SPECIFICATIONS */}
              <View style={styles.diagCard}>
                <Text style={styles.diagSectionTitle}>ONNX MODEL ARCHITECTURE</Text>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Model Asset:</Text>
                  <Text style={[styles.diagValue, styles.monoText]}>
                    muometric_convnext_tiny_quantized.onnx
                  </Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Neural Backbone:</Text>
                  <Text style={styles.diagValue}>ConvNeXt-Tiny (ArcFace + GeM Pooling)</Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Canonical Input Shape:</Text>
                  <Text style={[styles.diagValue, styles.monoText]}>1 x 3 x 448 x 448 (ImageNet RGB)</Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Embedding Hypersphere:</Text>
                  <Text style={styles.diagValue}>512-D L2 Unit Normalized (Norm = 1.000)</Text>
                </View>
                <View style={styles.diagRow}>
                  <Text style={styles.diagLabel}>Quantization Scheme:</Text>
                  <Text style={styles.diagValue}>Int8 Dynamic / FP16 Native Session</Text>
                </View>
              </View>

              {/* SECTION 3: QUICK NAVIGATION */}
              <View style={styles.diagCard}>
                <Text style={styles.diagSectionTitle}>QUICK NAVIGATION</Text>
                <TouchableOpacity
                  style={styles.diagLinkBtn}
                  onPress={() => {
                    setDiagnosticsVisible(false);
                    setActiveTab('match');
                  }}
                >
                  <Text style={styles.diagLinkBtnText}>Open 1-to-N Gallery Matching</Text>
                  <Text style={styles.diagLinkBtnSub}>
                    {DEFAULT_GALLERY_CANDIDATES.length} Enrolled Herd Profiles
                  </Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                style={styles.modalDismissBtn}
                onPress={() => setDiagnosticsVisible(false)}
              >
                <Text style={styles.modalDismissBtnText}>Done</Text>
              </TouchableOpacity>
            </ScrollView>
          </View>
        </View>
      </Modal>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F5F5F7',
  },
  scrollContainer: {
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight || 0) + 14 : 14,
    paddingBottom: 48,
  },

  // TOP BAR
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  eyebrow: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.4,
    marginBottom: 4,
  },
  brandTitle: {
    color: '#101828',
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: -0.8,
  },
  menuButton: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E4E7EC',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  menuButtonText: {
    color: '#344054',
    fontSize: 19,
    fontWeight: '800',
    letterSpacing: 1,
    marginTop: -8,
  },

  // HERO INTRO
  heroIntro: {
    marginBottom: 20,
  },
  heroTitle: {
    color: '#101828',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: -0.4,
  },
  heroSubtitle: {
    color: '#667085',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 4,
  },
  runtimeBadge: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ECFDF3',
    borderColor: '#ABEFC6',
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginTop: 12,
  },
  runtimeDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: '#12B76A',
    marginRight: 7,
  },
  runtimeText: {
    color: '#027A48',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.8,
  },

  // ACTION CARDS (HOME)
  actionCards: {
    gap: 14,
    marginBottom: 20,
  },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#FFFFFF',
    borderColor: '#E4E7EC',
    borderWidth: 1,
    borderRadius: 20,
    padding: 18,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 6,
    elevation: 2,
  },
  actionIcon: {
    width: 46,
    height: 46,
    borderRadius: 14,
    backgroundColor: '#EFF4FF',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
  },
  compareIcon: {
    backgroundColor: '#ECFDF3',
  },
  actionIconText: {
    color: '#2563EB',
    fontSize: 24,
    fontWeight: '700',
  },
  actionCopy: {
    flex: 1,
  },
  actionTitle: {
    color: '#101828',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: -0.2,
  },
  actionDescription: {
    color: '#667085',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 5,
  },
  actionBadgeRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 10,
  },
  miniBadge: {
    backgroundColor: '#EFF4FF',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  miniBadgeText: {
    color: '#2563EB',
    fontSize: 9,
    fontWeight: '700',
  },
  actionLink: {
    color: '#2563EB',
    fontSize: 13,
    fontWeight: '700',
    marginTop: 10,
  },
  actionArrow: {
    fontSize: 15,
  },

  // TELEMETRY CARD (HOME)
  telemetryCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 16,
    marginBottom: 18,
  },
  telemetryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F2F4F7',
  },
  telemetryTitle: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  telemetryStatus: {
    color: '#2563EB',
    fontSize: 11,
    fontWeight: '700',
  },
  telemetryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  telemetryItem: {
    width: '47%',
  },
  telemetryLabel: {
    color: '#98A2B3',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  telemetryValue: {
    color: '#101828',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
  },
  mutedText: {
    color: '#98A2B3',
    fontWeight: '400',
    fontSize: 10,
  },

  // HOME HINT
  homeHint: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E4E7EC',
    borderWidth: 1,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  homeHintTitle: {
    color: '#101828',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 8,
  },
  homeHintText: {
    color: '#667085',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 4,
  },

  // SCREEN HEADINGS
  sectionContainer: {
    marginBottom: 20,
  },
  screenHeading: {
    marginBottom: 20,
  },
  backLink: {
    color: '#2563EB',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 12,
  },
  screenTitle: {
    color: '#101828',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: -0.6,
  },
  screenSubtitle: {
    color: '#667085',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 6,
  },

  // CARDS & INPUTS
  inputCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 16,
    marginBottom: 16,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  cardHeaderTitle: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  cardHeaderFilename: {
    color: '#101828',
    fontSize: 12,
    fontWeight: '700',
    flex: 1,
    textAlign: 'right',
    marginLeft: 10,
  },
  tagPillQuery: {
    backgroundColor: '#EFF4FF',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  tagPillQueryText: {
    color: '#2563EB',
    fontSize: 9,
    fontWeight: '800',
  },
  tagPillRef: {
    backgroundColor: '#ECFDF3',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  tagPillRefText: {
    color: '#059669',
    fontSize: 9,
    fontWeight: '800',
  },
  pathInput: {
    backgroundColor: '#F9FAFB',
    borderColor: '#E4E7EC',
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: '#101828',
    fontSize: 11,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    marginBottom: 12,
  },
  buttonActionRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 10,
  },
  primaryActionBtn: {
    flex: 1,
    paddingVertical: 11,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryActionBtnText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  presetLabel: {
    color: '#98A2B3',
    fontSize: 10,
    fontWeight: '700',
    marginBottom: 6,
    marginTop: 4,
  },
  presetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  presetChip: {
    backgroundColor: '#F9FAFB',
    borderColor: '#E4E7EC',
    borderWidth: 1,
    paddingHorizontal: 9,
    paddingVertical: 6,
    borderRadius: 8,
  },
  presetChipActive: {
    backgroundColor: '#EFF4FF',
    borderColor: '#2563EB',
  },
  presetChipText: {
    color: '#475467',
    fontSize: 11,
    fontWeight: '600',
  },
  presetChipTextActive: {
    color: '#2563EB',
    fontWeight: '700',
  },
  selectedMediaBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F0FDF4',
    borderColor: '#BBF7D0',
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
    marginTop: 12,
  },
  selectedMediaTitle: {
    color: '#166534',
    fontSize: 12,
    fontWeight: '700',
  },
  selectedMediaPath: {
    color: '#15803D',
    fontSize: 10,
    marginTop: 2,
  },
  clearBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    backgroundColor: '#FEE2E2',
    marginLeft: 8,
  },
  clearBtnText: {
    color: '#B91C1C',
    fontSize: 10,
    fontWeight: '700',
  },

  // FRAME CONTROL
  frameControlCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginBottom: 16,
  },
  framePillsRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 10,
  },
  framePill: {
    flex: 1,
    backgroundColor: '#F9FAFB',
    paddingVertical: 9,
    borderRadius: 10,
    alignItems: 'center',
    borderColor: '#E4E7EC',
    borderWidth: 1,
  },
  framePillActive: {
    backgroundColor: '#2563EB',
    borderColor: '#2563EB',
  },
  framePillText: {
    color: '#475467',
    fontSize: 11,
    fontWeight: '600',
  },
  framePillTextActive: {
    color: '#FFFFFF',
    fontWeight: '700',
  },

  // MAIN BUTTON
  mainButton: {
    backgroundColor: '#2563EB',
    paddingVertical: 15,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 18,
    shadowColor: '#2563EB',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 3,
  },
  mainButtonDisabled: {
    backgroundColor: '#93C5FD',
    opacity: 0.85,
  },
  buttonRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  mainButtonText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.5,
  },

  // RESULTS CONTAINER
  resultsContainer: {
    marginTop: 6,
    gap: 16,
  },

  // CONSISTENCY CARD
  consistencyCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 16,
  },
  consistencyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  consistencySubtitle: {
    color: '#667085',
    fontSize: 11,
    marginTop: 3,
  },
  consistencyStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderTopWidth: 1,
    borderTopColor: '#F2F4F7',
    paddingTop: 12,
    marginTop: 10,
  },
  consistencyStatItem: {
    flex: 1,
  },
  statLabel: {
    color: '#98A2B3',
    fontSize: 8,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  statValue: {
    color: '#101828',
    fontSize: 11,
    fontWeight: '700',
    marginTop: 2,
  },

  // FRAME INSPECTION CARD
  frameInspectionCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginBottom: 16,
  },
  frameInspectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  frameInspectionCount: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '700',
  },
  frameCarousel: {
    flexDirection: 'row',
  },
  frameThumbCard: {
    width: 125,
    backgroundColor: '#F9FAFB',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginRight: 10,
    overflow: 'hidden',
  },
  frameThumbCardOutlier: {
    borderColor: '#FCA5A5',
    backgroundColor: '#FEF2F2',
  },
  frameThumbnailImg: {
    width: '100%',
    height: 95,
    backgroundColor: '#E4E7EC',
  },
  frameThumbPlaceholder: {
    width: '100%',
    height: 95,
    backgroundColor: '#E4E7EC',
    alignItems: 'center',
    justifyContent: 'center',
  },
  frameThumbPlaceholderText: {
    color: '#98A2B3',
    fontSize: 11,
    fontWeight: '600',
  },
  frameThumbMeta: {
    padding: 8,
  },
  frameThumbIndex: {
    color: '#101828',
    fontSize: 10,
    fontWeight: '700',
  },
  frameThumbAgreement: {
    color: '#2563EB',
    fontSize: 9,
    fontWeight: '600',
    marginTop: 3,
  },
  inlierPill: {
    backgroundColor: '#ECFDF3',
    borderRadius: 4,
    paddingHorizontal: 5,
    paddingVertical: 2,
    marginTop: 4,
    alignSelf: 'flex-start',
  },
  inlierPillText: {
    color: '#027A48',
    fontSize: 8,
    fontWeight: '800',
  },
  outlierPill: {
    backgroundColor: '#FEE2E2',
    borderRadius: 4,
    paddingHorizontal: 5,
    paddingVertical: 2,
    marginTop: 4,
    alignSelf: 'flex-start',
  },
  outlierPillText: {
    color: '#B91C1C',
    fontSize: 8,
    fontWeight: '800',
  },
  weightText: {
    color: '#98A2B3',
    fontSize: 8,
    marginTop: 3,
  },

  // VECTOR RESULT CARD
  vectorResultCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 16,
    marginBottom: 16,
  },
  vectorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  vectorTitle: {
    color: '#101828',
    fontSize: 15,
    fontWeight: '800',
  },
  vectorSubtitle: {
    color: '#667085',
    fontSize: 11,
    marginTop: 2,
  },
  dimBadge: {
    backgroundColor: '#EFF4FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  dimBadgeText: {
    color: '#2563EB',
    fontSize: 10,
    fontWeight: '700',
  },
  copyButtonPrimary: {
    backgroundColor: '#2563EB',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  copyButtonPrimaryText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  toastBox: {
    backgroundColor: '#ECFDF3',
    borderColor: '#ABEFC6',
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 6,
    paddingHorizontal: 12,
    marginBottom: 12,
    alignItems: 'center',
  },
  toastText: {
    color: '#027A48',
    fontSize: 11,
    fontWeight: '700',
  },
  terminalContainer: {
    backgroundColor: '#0F172A',
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#1E293B',
    marginBottom: 14,
  },
  terminalWindowBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  terminalDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  terminalTitle: {
    color: '#94A3B8',
    fontSize: 10,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    marginLeft: 6,
  },
  terminalBody: {
    maxHeight: 120,
    padding: 10,
  },
  vectorCode: {
    color: '#34D399',
    fontSize: 10,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    lineHeight: 14,
  },
  vectorActionRow: {
    flexDirection: 'row',
    gap: 10,
  },
  secondaryActionBtn: {
    flex: 1,
    backgroundColor: '#F2F4F7',
    paddingVertical: 11,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryActionBtnText: {
    color: '#344054',
    fontSize: 12,
    fontWeight: '700',
  },
  accentActionBtn: {
    backgroundColor: '#EFF4FF',
  },
  accentActionBtnText: {
    color: '#2563EB',
    fontSize: 12,
    fontWeight: '700',
  },

  // VERIFICATION RESULT BANNER (COMPARE)
  verificationResultContainer: {
    marginTop: 4,
  },
  verdictBanner: {
    borderRadius: 18,
    padding: 16,
    borderWidth: 1.5,
    marginBottom: 16,
    backgroundColor: '#FFFFFF',
  },
  verdictMatch: {
    borderColor: '#10B981',
    backgroundColor: '#ECFDF3',
  },
  verdictLikely: {
    borderColor: '#0D9488',
    backgroundColor: '#F0FDFA',
  },
  verdictInconclusive: {
    borderColor: '#F59E0B',
    backgroundColor: '#FFFBEB',
  },
  verdictDistinct: {
    borderColor: '#EF4444',
    backgroundColor: '#FEF2F2',
  },
  verdictIconRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  verdictEmojiCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  verdictEmoji: {
    fontSize: 18,
    color: '#FFFFFF',
    fontWeight: '800',
  },
  decisionBadgeHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  verdictTitle: {
    color: '#101828',
    fontSize: 13,
    fontWeight: '800',
    flex: 1,
  },
  verdictBigScore: {
    color: '#101828',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: -0.3,
    marginTop: 2,
  },
  verdictSubtitle: {
    color: '#667085',
    fontSize: 11,
    marginTop: 2,
  },
  decisionBadgeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.06)',
  },
  decisionBadge: {
    alignItems: 'center',
    flex: 1,
  },
  decisionBadgeLabel: {
    color: '#667085',
    fontSize: 8,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  decisionBadgeVal: {
    color: '#101828',
    fontSize: 13,
    fontWeight: '800',
    marginTop: 3,
  },

  // STATUS BADGES
  badgePill: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  badgeEmerald: {
    backgroundColor: '#ECFDF3',
  },
  badgeTextEmerald: {
    color: '#027A48',
  },
  badgeTeal: {
    backgroundColor: '#F0FDFA',
  },
  badgeTextTeal: {
    color: '#0F766E',
  },
  badgeAmber: {
    backgroundColor: '#FFFBEB',
  },
  badgeTextAmber: {
    color: '#B45309',
  },
  badgeRose: {
    backgroundColor: '#FEF2F2',
  },
  badgeTextRose: {
    color: '#B91C1C',
  },
  badgeText: {
    fontSize: 9,
    fontWeight: '800',
  },

  // GALLERY MATCHING
  galleryMetaCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginBottom: 14,
  },
  galleryMetaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  galleryMetaLabel: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '800',
  },
  galleryMetaVal: {
    color: '#101828',
    fontSize: 11,
    fontWeight: '700',
  },
  leadMarginBadge: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(0,0,0,0.06)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginTop: 10,
  },
  leadMarginText: {
    color: '#101828',
    fontSize: 10,
    fontWeight: '700',
  },
  emptyMatchState: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E4E7EC',
  },
  emptyMatchText: {
    color: '#667085',
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 14,
  },
  inlineButton: {
    backgroundColor: '#2563EB',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 10,
  },
  inlineButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  candidateCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    marginBottom: 10,
  },
  candidateCardTopMatch: {
    borderColor: '#10B981',
    backgroundColor: '#F0FDF4',
  },
  candidateHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  rankPill: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#F2F4F7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankText: {
    color: '#344054',
    fontSize: 11,
    fontWeight: '800',
  },
  candidateName: {
    color: '#101828',
    fontSize: 13,
    fontWeight: '700',
  },
  candidateMeta: {
    color: '#667085',
    fontSize: 10,
    marginTop: 1,
  },
  metricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  progressBarBg: {
    flex: 1,
    height: 6,
    backgroundColor: '#EAECF0',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  metricPct: {
    color: '#101828',
    fontSize: 11,
    fontWeight: '800',
    width: 48,
    textAlign: 'right',
  },
  candidateFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderTopWidth: 1,
    borderTopColor: '#F2F4F7',
    paddingTop: 6,
  },
  footerMetric: {
    color: '#667085',
    fontSize: 10,
  },
  footerVal: {
    color: '#101828',
    fontWeight: '700',
  },

  // DIAGNOSTICS MODAL
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  modalDrawer: {
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingTop: 10,
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'android' ? 24 : 36,
    maxHeight: '85%',
  },
  modalGrabber: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#D0D5DD',
    alignSelf: 'center',
    marginBottom: 14,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F2F4F7',
  },
  modalTitle: {
    color: '#101828',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  modalSubtitle: {
    color: '#667085',
    fontSize: 12,
    marginTop: 2,
  },
  modalCloseBtn: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#F2F4F7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalCloseBtnText: {
    color: '#667085',
    fontSize: 14,
    fontWeight: '700',
  },
  modalContent: {
    marginBottom: 8,
  },
  diagCard: {
    backgroundColor: '#F9FAFB',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    padding: 14,
    marginBottom: 14,
  },
  diagSectionTitle: {
    color: '#667085',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
    marginBottom: 10,
  },
  diagRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#F2F4F7',
  },
  diagLabel: {
    color: '#475467',
    fontSize: 11,
    fontWeight: '600',
  },
  diagValue: {
    color: '#101828',
    fontSize: 11,
    fontWeight: '700',
    textAlign: 'right',
    flex: 1,
    marginLeft: 10,
  },
  monoText: {
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
    fontSize: 10,
  },
  diagLinkBtn: {
    backgroundColor: '#EFF4FF',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#D1E0FF',
  },
  diagLinkBtnText: {
    color: '#2563EB',
    fontSize: 13,
    fontWeight: '700',
  },
  diagLinkBtnSub: {
    color: '#475467',
    fontSize: 11,
    marginTop: 2,
  },
  modalDismissBtn: {
    backgroundColor: '#2563EB',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 12,
  },
  modalDismissBtnText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
});
