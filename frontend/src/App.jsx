import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Video,
  VideoOff,
  Camera,
  Play,
  Square,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Maximize2,
  Volume2,
  VolumeX,
  Activity,
  HardHat,
  Cpu
} from 'lucide-react';
import './App.css';

export default function App() {
  const [cameraMode, setCameraMode] = useState('native'); // 'native' (OpenCV) or 'browser' (Webcam)
  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [fps, setFps] = useState(0);
  const [confThreshold, setConfThreshold] = useState(0.45);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [telemetry, setTelemetry] = useState({
    status: 'STANDBY',
    counts: { total: 0, violations: 0, compliant: 0 },
    detections: []
  });

  // Native Stream URL state to force refresh on start/stop
  const [streamUrl, setStreamUrl] = useState('');

  // Browser camera refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameId = useRef(null);
  const isLoopRunning = useRef(false);
  const audioCtxRef = useRef(null);
  const lastAlertTime = useRef(0);

  // Play alert beep for violations
  const triggerAudioAlert = () => {
    if (!soundEnabled) return;
    const now = Date.now();
    if (now - lastAlertTime.current < 2500) return; // limit to once per 2.5s
    lastAlertTime.current = now;

    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(650, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(320, ctx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
    } catch (e) {
      console.warn('Audio alert error:', e);
    }
  };

  // Sync confidence threshold with backend
  const handleConfidenceChange = async (e) => {
    const val = parseFloat(e.target.value);
    setConfThreshold(val);
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conf: val })
      });
    } catch (err) {
      console.error('Failed to update config:', err);
    }
  };

  // --------------------------------------------------------------------------
  // Native OpenCV Camera Handlers
  // --------------------------------------------------------------------------
  const startNativeCamera = async () => {
    setIsStarting(true);
    try {
      const res = await fetch(`/api/start?conf=${confThreshold}`, { method: 'POST' });
      const data = await res.json();
      if (data.success || data.active) {
        setIsRunning(true);
        setStreamUrl(`/api/video_feed?t=${Date.now()}`);
      } else {
        alert(`Failed to start camera: ${data.message}`);
      }
    } catch (err) {
      console.error('Error starting camera:', err);
      alert('Could not connect to detection server. Ensure python backend is running.');
    } finally {
      setIsStarting(false);
    }
  };

  const stopNativeCamera = async () => {
    try {
      await fetch('/api/stop', { method: 'POST' });
    } catch (err) {
      console.error('Error stopping camera:', err);
    }
    setIsRunning(false);
    setStreamUrl('');
    setFps(0);
    setTelemetry({
      status: 'STANDBY',
      counts: { total: 0, violations: 0, compliant: 0 },
      detections: []
    });
  };

  // Poll status when Native mode is active
  useEffect(() => {
    if (cameraMode !== 'native' || !isRunning) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        setFps(data.fps || 0);
        if (data.telemetry) {
          setTelemetry(data.telemetry);
          if (data.telemetry.counts?.violations > 0) {
            triggerAudioAlert();
          }
        }
      } catch (e) {
        console.error('Telemetry poll error:', e);
      }
    }, 400);

    return () => clearInterval(interval);
  }, [cameraMode, isRunning, soundEnabled]);

  // --------------------------------------------------------------------------
  // Browser Webcam Handlers
  // --------------------------------------------------------------------------
  const startBrowserCamera = async () => {
    setIsStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsRunning(true);
      isLoopRunning.current = true;
      runBrowserDetectionLoop();
    } catch (err) {
      console.error('Browser camera access denied:', err);
      alert(`Could not access browser webcam: ${err.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  const stopBrowserCamera = () => {
    isLoopRunning.current = false;
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setFps(0);
    setTelemetry({
      status: 'STANDBY',
      counts: { total: 0, violations: 0, compliant: 0 },
      detections: []
    });
  };

  const runBrowserDetectionLoop = async () => {
    const offscreenCanvas = document.createElement('canvas');
    let lastFrameTime = performance.now();
    let frameCounter = 0;
    let fpsTimer = performance.now();

    const processFrame = async () => {
      if (!isLoopRunning.current || !videoRef.current) return;

      const video = videoRef.current;
      if (video.readyState >= 2 && video.videoWidth > 0) {
        offscreenCanvas.width = video.videoWidth;
        offscreenCanvas.height = video.videoHeight;
        const offCtx = offscreenCanvas.getContext('2d');
        offCtx.drawImage(video, 0, 0);

        offscreenCanvas.toBlob(async (blob) => {
          if (!blob || !isLoopRunning.current) return;

          try {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            formData.append('conf', confThreshold.toString());
            formData.append('return_annotated', 'true');

            const res = await fetch('/api/detect_frame', {
              method: 'POST',
              body: formData
            });

            if (res.ok) {
              const data = await res.json();
              setTelemetry(data.telemetry);
              if (data.telemetry.counts?.violations > 0) {
                triggerAudioAlert();
              }

              // Draw annotated frame onto visible canvas
              if (data.annotated_image && canvasRef.current) {
                const img = new Image();
                img.onload = () => {
                  const canvas = canvasRef.current;
                  if (canvas) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                  }
                };
                img.src = data.annotated_image;
              }

              // Calculate client FPS
              frameCounter++;
              const now = performance.now();
              if (now - fpsTimer >= 1000) {
                setFps(Math.round((frameCounter * 1000) / (now - fpsTimer)));
                frameCounter = 0;
                fpsTimer = now;
              }
            }
          } catch (e) {
            console.error('Frame detection upload error:', e);
          }

          if (isLoopRunning.current) {
            setTimeout(() => {
              animationFrameId.current = requestAnimationFrame(processFrame);
            }, 60); // ~15 FPS throttle for smooth inference
          }
        }, 'image/jpeg', 0.85);
      } else {
        if (isLoopRunning.current) {
          animationFrameId.current = requestAnimationFrame(processFrame);
        }
      }
    };

    animationFrameId.current = requestAnimationFrame(processFrame);
  };

  // Toggle Start / Stop
  const handleToggleDetection = () => {
    if (isRunning) {
      if (cameraMode === 'native') stopNativeCamera();
      else stopBrowserCamera();
    } else {
      if (cameraMode === 'native') startNativeCamera();
      else startBrowserCamera();
    }
  };

  // Switch between Native and Browser mode
  const handleModeSwitch = (newMode) => {
    if (isRunning) {
      if (cameraMode === 'native') stopNativeCamera();
      else stopBrowserCamera();
    }
    setCameraMode(newMode);
  };

  // Capture snapshot of the detected frame
  const handleTakeSnapshot = () => {
    if (!isRunning) return;

    if (cameraMode === 'native') {
      const link = document.createElement('a');
      link.href = streamUrl;
      link.download = `mineguard_capture_${Date.now()}.jpg`;
      link.target = '_blank';
      link.click();
    } else if (canvasRef.current) {
      const link = document.createElement('a');
      link.download = `mineguard_capture_${Date.now()}.png`;
      link.href = canvasRef.current.toDataURL('image/png');
      link.click();
    }
  };

  // Helper check for compliance categories
  const hasHardHat = telemetry.detections.some(d => d.class_name.toLowerCase().includes('hard hat'));
  const hasBareHead = telemetry.detections.some(d => d.class_name.toLowerCase().includes('bare') || d.class_name.toLowerCase().includes('no helmet'));
  
  const hasSuit = telemetry.detections.some(d => d.class_name.toLowerCase().includes('suit on'));
  const hasNoSuit = telemetry.detections.some(d => d.class_name.toLowerCase().includes('no suit'));

  const hasBoots = telemetry.detections.some(d => d.class_name.toLowerCase().includes('boots on'));
  const hasNoBoots = telemetry.detections.some(d => d.class_name.toLowerCase().includes('no boots'));

  const hasGloves = telemetry.detections.some(d => d.class_name.toLowerCase().includes('gloves on'));
  const hasNoGloves = telemetry.detections.some(d => d.class_name.toLowerCase().includes('no gloves'));

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="logo-badge">
            <HardHat size={28} />
          </div>
          <div>
            <h1 className="brand-title">MineGuard Detection AI</h1>
            <p className="brand-subtitle">Real-time PPE Compliance & Safety Monitor</p>
          </div>
        </div>

        <div className="header-status-group">
          <div className="status-chip">
            <span className={`pulse-dot ${isRunning ? (telemetry.counts.violations > 0 ? 'violation' : 'online') : 'offline'}`} />
            <span>{isRunning ? (telemetry.counts.violations > 0 ? 'VIOLATION DETECTED' : 'CAMERA ACTIVE') : 'SYSTEM STANDBY'}</span>
          </div>

          <div className="status-chip">
            <Activity size={14} color="#38bdf8" />
            <span>{fps} FPS</span>
          </div>

          <button
            className="btn-secondary"
            onClick={() => setSoundEnabled(!soundEnabled)}
            title={soundEnabled ? "Mute audio alarms" : "Enable audible safety alarms"}
          >
            {soundEnabled ? <Volume2 size={16} color="#10b981" /> : <VolumeX size={16} color="#94a3b8" />}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Camera Viewport & Controls */}
        <div className="camera-panel">
          <div className="viewport-card">
            {/* Native Stream Image */}
            {cameraMode === 'native' && isRunning && (
              <img
                src={streamUrl}
                alt="Live AI Detection Stream"
                className="video-stream-img"
              />
            )}

            {/* Browser Webcam Canvas & Hidden Video */}
            {cameraMode === 'browser' && (
              <>
                <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
                <canvas
                  ref={canvasRef}
                  className="video-stream-canvas"
                  style={{ display: isRunning ? 'block' : 'none' }}
                />
              </>
            )}

            {/* Standby Placeholder */}
            {!isRunning && (
              <div className="standby-state">
                <div className="standby-icon-circle">
                  <VideoOff size={40} />
                </div>
                <div>
                  <h3 className="standby-title">Live Detection Camera is Inactive</h3>
                  <p className="standby-desc">
                    Click the <strong>Start Detection</strong> button below to activate the live camera feed and see real-time AI bounding boxes.
                  </p>
                </div>
              </div>
            )}

            {/* HUD Overlay Bar */}
            {isRunning && (
              <div className="hud-overlay">
                <div className="hud-top-bar">
                  <div className={`hud-pill ${telemetry.counts.violations > 0 ? 'alert' : 'safe'}`}>
                    <span className="pulse-dot online" />
                    <span>LIVE AI FEED • {cameraMode.toUpperCase()}</span>
                  </div>

                  <div className="hud-pill">
                    <Cpu size={14} />
                    <span>YOLOv8 PPE MODEL</span>
                  </div>
                </div>

                <div className="hud-bottom-bar">
                  <div className="hud-pill">
                    <span>{telemetry.counts.total} Objects Tracked</span>
                  </div>
                  <div className="hud-pill">
                    <span>Threshold: {Math.round(confThreshold * 100)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div className="controls-bar">
            <div className="controls-left-group">
              <button
                className={isRunning ? 'btn-stop' : 'btn-start'}
                onClick={handleToggleDetection}
                disabled={isStarting}
              >
                {isStarting ? (
                  <span>Initializing...</span>
                ) : isRunning ? (
                  <>
                    <Square size={18} />
                    <span>Stop Detection</span>
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    <span>Start Detection</span>
                  </>
                )}
              </button>

              <button
                className="btn-secondary"
                onClick={handleTakeSnapshot}
                disabled={!isRunning}
                title="Capture snapshot with bounding boxes"
              >
                <Camera size={16} />
                <span>Capture Frame</span>
              </button>
            </div>

            {/* Mode Switcher */}
            <div className="mode-selector">
              <button
                className={`mode-option-btn ${cameraMode === 'native' ? 'active' : ''}`}
                onClick={() => handleModeSwitch('native')}
              >
                Local Camera (OpenCV)
              </button>
              <button
                className={`mode-option-btn ${cameraMode === 'browser' ? 'active' : ''}`}
                onClick={() => handleModeSwitch('browser')}
              >
                Browser Webcam
              </button>
            </div>
          </div>

          {/* Quick Metrics Row */}
          <div className="metrics-row">
            <div className="metric-card">
              <div>
                <div className="metric-title">Total Detected</div>
                <div className="metric-value">{telemetry.counts.total}</div>
              </div>
              <div className="metric-icon-box blue">
                <Video size={20} />
              </div>
            </div>

            <div className="metric-card">
              <div>
                <div className="metric-title">Compliant Items</div>
                <div className="metric-value">{telemetry.counts.compliant}</div>
              </div>
              <div className="metric-icon-box green">
                <CheckCircle2 size={20} />
              </div>
            </div>

            <div className="metric-card">
              <div>
                <div className="metric-title">Safety Violations</div>
                <div className="metric-value" style={{ color: telemetry.counts.violations > 0 ? '#ef4444' : '#fff' }}>
                  {telemetry.counts.violations}
                </div>
              </div>
              <div className="metric-icon-box red">
                <AlertTriangle size={20} />
              </div>
            </div>

            <div className="metric-card">
              <div>
                <div className="metric-title">Engine FPS</div>
                <div className="metric-value">{fps}</div>
              </div>
              <div className="metric-icon-box amber">
                <Activity size={20} />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Telemetry & Safety Dashboard */}
        <aside className="sidebar-panel">
          {/* Safety Compliance Banner */}
          <div className={`safety-banner ${!isRunning ? 'standby' : (telemetry.counts.violations > 0 ? 'violation' : (telemetry.counts.compliant > 0 ? 'safe' : 'standby'))}`}>
            {!isRunning ? (
              <>
                <VideoOff size={24} />
                <div>
                  <div className="safety-banner-title">Standby Mode</div>
                  <div className="safety-banner-desc">Waiting for detection camera to start</div>
                </div>
              </>
            ) : telemetry.counts.violations > 0 ? (
              <>
                <ShieldAlert size={28} />
                <div>
                  <div className="safety-banner-title">SAFETY VIOLATION ALERT</div>
                  <div className="safety-banner-desc">Non-compliant safety equipment detected in frame!</div>
                </div>
              </>
            ) : telemetry.counts.compliant > 0 ? (
              <>
                <ShieldCheck size={28} />
                <div>
                  <div className="safety-banner-title">ALL PPE COMPLIANT</div>
                  <div className="safety-banner-desc">Personnel is wearing approved safety gear</div>
                </div>
              </>
            ) : (
              <>
                <ShieldCheck size={28} />
                <div>
                  <div className="safety-banner-title">SCANNING AREA</div>
                  <div className="safety-banner-desc">No workers currently detected in camera view</div>
                </div>
              </>
            )}
          </div>

          {/* PPE Gear Matrix Checklist */}
          <div className="sidebar-card">
            <div className="card-title-row">
              <span className="card-title">
                <HardHat size={18} color="#38bdf8" />
                <span>PPE Compliance Matrix</span>
              </span>
            </div>

            <div className="ppe-matrix-grid">
              <div className="ppe-matrix-item">
                <span className="ppe-item-label">Hard Hat</span>
                <span className={`ppe-badge ${hasBareHead ? 'fail' : (hasHardHat ? 'pass' : 'neutral')}`}>
                  {hasBareHead ? 'MISSING' : (hasHardHat ? 'PASSED' : 'N/A')}
                </span>
              </div>

              <div className="ppe-matrix-item">
                <span className="ppe-item-label">Safety Suit</span>
                <span className={`ppe-badge ${hasNoSuit ? 'fail' : (hasSuit ? 'pass' : 'neutral')}`}>
                  {hasNoSuit ? 'MISSING' : (hasSuit ? 'PASSED' : 'N/A')}
                </span>
              </div>

              <div className="ppe-matrix-item">
                <span className="ppe-item-label">Safety Boots</span>
                <span className={`ppe-badge ${hasNoBoots ? 'fail' : (hasBoots ? 'pass' : 'neutral')}`}>
                  {hasNoBoots ? 'MISSING' : (hasBoots ? 'PASSED' : 'N/A')}
                </span>
              </div>

              <div className="ppe-matrix-item">
                <span className="ppe-item-label">Gloves</span>
                <span className={`ppe-badge ${hasNoGloves ? 'fail' : (hasGloves ? 'pass' : 'neutral')}`}>
                  {hasNoGloves ? 'MISSING' : (hasGloves ? 'PASSED' : 'N/A')}
                </span>
              </div>
            </div>
          </div>

          {/* Live Detections Feed */}
          <div className="sidebar-card">
            <div className="card-title-row">
              <span className="card-title">
                <Sliders size={18} color="#38bdf8" />
                <span>Active Detections ({telemetry.detections.length})</span>
              </span>
            </div>

            <div className="detections-scroll">
              {telemetry.detections.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#64748b', padding: '1.5rem', fontSize: '0.85rem' }}>
                  {isRunning ? 'No PPE objects detected in frame.' : 'Start camera to see live detections.'}
                </div>
              ) : (
                telemetry.detections.map((det) => (
                  <div
                    key={det.id}
                    className={`detection-card ${det.is_violation ? 'violation' : 'compliant'}`}
                  >
                    <div className="detection-info">
                      <span className="detection-name">{det.class_name}</span>
                      <div className="detection-conf-bar-container">
                        <div
                          className={`detection-conf-bar ${det.is_violation ? 'violation' : 'compliant'}`}
                          style={{ width: `${det.confidence_pct}%` }}
                        />
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span className="detection-conf-text">{det.confidence_pct}% conf</span>
                      <div style={{ fontSize: '0.68rem', color: det.is_violation ? '#f87171' : '#34d399', fontWeight: 700 }}>
                        {det.is_violation ? 'VIOLATION' : 'COMPLIANT'}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Model Controls */}
          <div className="sidebar-card">
            <span className="card-title">
              <Sliders size={18} color="#38bdf8" />
              <span>Model Detection Settings</span>
            </span>

            <div className="setting-item">
              <div className="setting-label-row">
                <span>Confidence Threshold</span>
                <span style={{ color: '#38bdf8', fontWeight: 700 }}>{Math.round(confThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.95"
                step="0.05"
                value={confThreshold}
                onChange={handleConfidenceChange}
                className="slider-input"
              />
              <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                Higher values reduce false positives; lower values detect subtle objects.
              </span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
