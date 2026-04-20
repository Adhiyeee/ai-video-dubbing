import React, { useState } from "react";
import "./App.css";

const TARGET_LANGS = [
  { code: "ta", label: "Tamil" },
  { code: "hi", label: "Hindi" },
  { code: "ml", label: "Malayalam" },
  { code: "en", label: "English" },
  { code: "fr", label: "French" },
];

const SOURCE_LANGS = [
  { code: "auto", label: "Auto-detect" },
  { code: "en",   label: "English" },
  { code: "hi",   label: "Hindi" },
  { code: "ta",   label: "Tamil" },
  { code: "ml",   label: "Malayalam" },
  { code: "fr",   label: "French" },
];

const STAGES = [
  { icon: "📁", label: "Uploading video..." },
  { icon: "🎧", label: "Extracting audio..." },
  { icon: "🧠", label: "Transcribing speech..." },
  { icon: "🌍", label: "Translating segments..." },
  { icon: "✍️",  label: "Fixing grammar..." },
  { icon: "🔊", label: "Generating dubbed voice..." },
  { icon: "⏱️",  label: "Syncing to timestamps..." },
  { icon: "🎬", label: "Mixing final video..." },
  { icon: "💋", label: "Lip syncing with Wav2Lip..." },
];

function App() {
  const [file,       setFile]       = useState(null);
  const [targetLang, setTargetLang] = useState("ta");
  const [sourceLang, setSourceLang] = useState("auto");
  const [loading,    setLoading]    = useState(false);
  const [videoUrl,   setVideoUrl]   = useState("");
  const [stageIdx,   setStageIdx]   = useState(0);
  const [error,      setError]      = useState("");
  const [stats,      setStats]      = useState(null);

  const startTicker = () => {
    let idx = 0;
    setStageIdx(0);
    // Each stage ≈ how long it takes in the pipeline
    const durations = [2000, 3000, 12000, 6000, 8000, 6000, 4000, 3000];
    let total = 0;
    const timers = durations.map((d, i) => {
      total += d;
      return setTimeout(() => setStageIdx(i), total - d);
    });
    return () => timers.forEach(clearTimeout);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please upload a video first!");
    setError(""); setVideoUrl(""); setStats(null);

    const fd = new FormData();
    fd.append("file", file);

    setLoading(true);
    const stopTicker = startTicker();

    try {
      const res = await fetch(
        `http://127.0.0.1:8000/upload?target_lang=${targetLang}&source_lang=${sourceLang}`,
        { method: "POST", body: fd }
      );
      stopTicker();

      let data;
      try { data = await res.json(); }
      catch { throw new Error("Backend returned invalid JSON — check terminal."); }

      if (!res.ok || data.error) {
        setError(`Backend error: ${data.error || res.statusText}`);
        setLoading(false);
        return;
      }

      setStageIdx(STAGES.length - 1);
      setVideoUrl(`http://127.0.0.1:8000/${data.video}`);
      setStats({ segments: data.segments, duration: data.duration, lipSync: data.lip_sync });
    } catch (err) {
      stopTicker();
      setError(`Connection error: ${err.message}`);
    }
    setLoading(false);
  };

  const currentStage = STAGES[Math.min(stageIdx, STAGES.length - 1)];

  return (
    <div className="container">
      <div className="card">
        <h1>🎬 AI Video Dubber</h1>
        <p className="subtitle">Upload · Translate · Sync · Done</p>

        {/* Drop zone */}
        <label
          className={`upload-box ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {file ? (
            <>
              <span className="file-icon">🎞️</span>
              <span className="file-name">{file.name}</span>
              <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
            </>
          ) : (
            <>
              <span className="upload-icon">📁</span>
              <span>Drag & drop or click to upload</span>
              <span className="hint">MP4, MOV, AVI, MKV supported</span>
            </>
          )}
          <input type="file" accept="video/*"
            onChange={(e) => setFile(e.target.files[0])} hidden />
        </label>

        {/* Language selectors */}
        <div className="lang-row">
          <span>Video language:</span>
          <select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)}>
            {SOURCE_LANGS.map(l => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>

        <div className="lang-row">
          <span>Translate to:</span>
          <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
            {TARGET_LANGS.map(l => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>

        <button onClick={handleUpload} className="btn" disabled={loading || !file}>
          {loading ? "Processing…" : "🚀 Translate & Dub"}
        </button>

        {/* Progress */}
        {loading && (
          <div className="progress-box">
            <div className="spinner" />
            <p className="stage-text">
              {currentStage.icon} {currentStage.label}
            </p>
            <div className="stage-bar">
              {STAGES.map((s, i) => (
                <div
                  key={i}
                  className={`stage-dot ${i < stageIdx ? "done" : i === stageIdx ? "active" : ""}`}
                  title={s.label}
                />
              ))}
            </div>
            <p className="stage-hint">
              Step {stageIdx + 1} of {STAGES.length}
            </p>
          </div>
        )}

        {error && <div className="error-box">❌ {error}</div>}

        {/* Result */}
        {videoUrl && (
          <div className="result">
            <p className="success-label">✅ Dubbed video ready!</p>
            {stats && (
              <p className="stats-line">
                {stats.segments} segments · {stats.duration}s ·{" "}
                {stats.lipSync
                  ? "💋 lip sync applied"
                  : "⚠️ no lip sync (check terminal)"}
              </p>
            )}
            <video controls width="100%">
              <source src={videoUrl} type="video/mp4" />
            </video>
            <a href={videoUrl} download className="download-btn">
              ⬇️ Download Video
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;