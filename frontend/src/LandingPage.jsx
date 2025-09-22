// src/LandingPage.jsx
import React from "react";
import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="min-h-screen relative text-white">
      {/* background video */}
<video
  className="video-bg"
  autoPlay
  muted
  loop
  playsInline
  poster="/space-poster.jpg"   // optional
>
  <source src="/videobackground.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

      {/* dark overlay for readability */}
      <div className="video-overlay"></div>

      {/* content on top */}
      <div className="page-content h-screen flex flex-col items-center justify-center text-center px-6">
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 text-yellow-400">
  GNSS ERROR PREDICTION
</h1>
        <p className="text-lg text-slate-200 max-w-2xl mb-8">
          Explore real time satellite error prediction
        </p>
        <Link
          to="/dashboard"
          className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow-lg font-semibold transition"
        >
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
}
