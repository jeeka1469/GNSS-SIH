// src/pages/VideoBackgroundPage.jsx
import React, { useRef } from "react";
import { Link } from "react-router-dom";

export default function VideoBackgroundPage() {
  const vidRef = useRef(null);

  // optional play/pause helper (you can remove if not needed)
  function togglePlay() {
    const v = vidRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  }

  return (
    <div className="min-h-screen relative text-white">
      {/* background video (served from public/videobackground.mp4) */}
      <video
        ref={vidRef}
        className="video-bg"
        autoPlay
        muted         /* must be muted for autoplay to work in most browsers */
        loop
        playsInline
      >
        <source src="/videobackground.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {/* dark overlay for readability */}
      <div className="video-overlay" />

      {/* content above the video */}
      <div className="page-content app-content mx-auto py-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold">Cinematic Space — Graphs & Analysis</h2>
          <div className="flex gap-2">
            <button
              onClick={togglePlay}
              className="px-3 py-1 bg-black/30 rounded text-sm"
            >
              Play / Pause
            </button>
            <Link to="/" className="px-3 py-1 bg-black/30 rounded text-sm">Back</Link>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white/8 backdrop-blur-sm p-4 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-2">Prediction Timeline</h3>
            <div className="h-56 flex items-center justify-center text-slate-200">(Chart Placeholder)</div>
          </div>

          <div className="bg-white/8 backdrop-blur-sm p-4 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-2">Residual Distribution</h3>
            <div className="h-56 flex items-center justify-center text-slate-200">(Chart Placeholder)</div>
          </div>
        </div>
      </div>
    </div>
  );
}
