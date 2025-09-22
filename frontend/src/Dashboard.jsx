import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import testData from "./data/testData.json";

const Dashboard = () => {
  const [satellite, setSatellite] = useState("");
  const [filteredData, setFilteredData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:5000");

  const satellites = Array.from(new Set(testData.map((d) => d.satellite_id)));

  useEffect(() => {
    let data = testData;
    if (satellite) data = data.filter((d) => d.satellite_id === satellite);
    setFilteredData(data);
  }, [satellite]);

  const fetchPrediction = async () => {
    if (filteredData.length < 36) {
      alert("Need at least 36 data points for prediction");
      return;
    }

    setLoading(true);
    try {
      const sampleData = filteredData.slice(-50).map(d => ({
        timestamp: d.timestamp,
        satellite_id: d.satellite_id,
        orbit_error_m: d.pos_error_m || 0,
        clock_error_ns: d.clock_error_ns || 0,
        radial_error_m: d.pos_error_m * 0.8 || 0,
        ephemeris_age_hours: 2.0
      }));

      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: sampleData }),
      });

      if (response.ok) {
        const result = await response.json();
        setPrediction(result);
      } else {
        const error = await response.json();
        alert(`Prediction failed: ${error.error}`);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to get prediction. Make sure backend is running.');
    }
    setLoading(false);
  };

  const latest = filteredData[filteredData.length - 1] || {};
  const orbitError = latest.pos_error_m || 0;
  const clockError = latest.clock_error_ns || 0;

  const orbitData = {
    labels: filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: "True Orbit Error",
        data: filteredData.map((d) => d.pos_error_m),
        borderColor: "#facc15",
        backgroundColor: "#facc15",
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
      ...(prediction ? [{
        label: "Predicted Orbit Error",
        data: [...Array(filteredData.length - 1).fill(null), prediction.predictions.orbit_error_m],
        borderColor: "#ef4444",
        backgroundColor: "#ef4444",
        pointRadius: 8,
        pointHoverRadius: 10,
        borderDash: [5, 5],
      }] : [])
    ],
  };

  const clockData = {
    labels: filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: "True Clock Error",
        data: filteredData.map((d) => d.clock_error_ns),
        borderColor: "#3b82f6",
        backgroundColor: "#3b82f6",
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
      ...(prediction ? [{
        label: "Predicted Clock Error", 
        data: [...Array(filteredData.length - 1).fill(null), prediction.predictions.clock_error_ns],
        borderColor: "#10b981",
        backgroundColor: "#10b981",
        pointRadius: 8,
        pointHoverRadius: 10,
        borderDash: [5, 5],
      }] : [])
    ],
  };

  return (
    <div className="min-h-screen text-white relative overflow-hidden">
      <video
        className="video-bg"
        autoPlay
        muted
        loop
        playsInline
        poster="/space-poster.jpg"
      >
        <source src="/dashboard4.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      <div className="video-overlay"></div>

      {Array.from({ length: 50 }).map((_, i) => (
        <div
          key={i}
          className="star"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            width: `${Math.random() * 2 + 1}px`,
            height: `${Math.random() * 2 + 1}px`,
            animationDuration: `${Math.random() * 3 + 2}s`,
          }}
        ></div>
      ))}

      <div className="container mx-auto p-8 relative z-10">
        <h1 className="text-5xl font-extrabold mb-8 text-yellow-400 text-center drop-shadow-[0_0_15px_rgba(250,204,21,0.8)]">
          GNSS Dashboard
        </h1>

        <div className="flex gap-4 mb-8 flex-wrap justify-center">
          <select
            className="p-3 rounded bg-gray-800 text-white border border-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-400"
            value={satellite}
            onChange={(e) => setSatellite(e.target.value)}
          >
            <option value="">Select Satellite</option>
            {satellites.map((sat, idx) => (
              <option key={idx} value={sat}>
                {sat}
              </option>
            ))}
          </select>
          
          <input
            type="text"
            placeholder="Backend URL"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="p-3 rounded bg-gray-800 text-white border border-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-400"
          />
          
          <button
            onClick={fetchPrediction}
            disabled={loading || !satellite || filteredData.length < 36}
            className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-600 rounded-lg shadow-lg font-semibold transition"
          >
            {loading ? "Predicting..." : "Get LSTM Prediction"}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl text-center hover:scale-105 hover:shadow-yellow-400/40 transition-transform duration-300">
            <p className="font-semibold text-yellow-400">Satellite</p>
            <p className="text-lg">{satellite || "-"}</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl text-center hover:scale-105 hover:shadow-yellow-400/40 transition-transform duration-300">
            <p className="font-semibold text-yellow-400">Current Orbit Error</p>
            <p className="text-lg">{orbitError.toFixed(3)} m</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl text-center hover:scale-105 hover:shadow-yellow-400/40 transition-transform duration-300">
            <p className="font-semibold text-yellow-400">Current Clock Error</p>
            <p className="text-lg">{clockError.toFixed(3)} ns</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl text-center hover:scale-105 hover:shadow-yellow-400/40 transition-transform duration-300">
            <p className="font-semibold text-yellow-400">Data Points</p>
            <p className="text-lg">{filteredData.length}</p>
          </div>
        </div>

        {prediction && (
          <div className="bg-gradient-to-r from-purple-800 to-indigo-800 p-6 rounded-lg shadow-xl mb-8">
            <h2 className="text-2xl font-bold text-yellow-400 mb-4">🤖 LSTM Prediction (Next Hour)</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-red-400 font-semibold">Predicted Orbit Error</p>
                <p className="text-xl">{prediction.predictions.orbit_error_m.toFixed(3)} m</p>
              </div>
              <div className="text-center">
                <p className="text-green-400 font-semibold">Predicted Clock Error</p>
                <p className="text-xl">{prediction.predictions.clock_error_ns.toFixed(3)} ns</p>
              </div>
              <div className="text-center">
                <p className="text-blue-400 font-semibold">Predicted Radial Error</p>
                <p className="text-xl">{prediction.predictions.radial_error_m.toFixed(3)} m</p>
              </div>
              <div className="text-center">
                <p className="text-purple-400 font-semibold">Confidence</p>
                <p className="text-xl">{prediction.model_confidence}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h2 className="font-semibold mb-3 text-yellow-400">Orbit Error: True vs Predicted</h2>
            <Line data={orbitData} options={{
              plugins: {
                legend: {
                  labels: {
                    color: 'white'
                  }
                }
              },
              scales: {
                x: {
                  ticks: { color: 'white' }
                },
                y: {
                  ticks: { color: 'white' }
                }
              }
            }} />
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h2 className="font-semibold mb-3 text-yellow-400">Clock Error: True vs Predicted</h2>
            <Line data={clockData} options={{
              plugins: {
                legend: {
                  labels: {
                    color: 'white'
                  }
                }
              },
              scales: {
                x: {
                  ticks: { color: 'white' }
                },
                y: {
                  ticks: { color: 'white' }
                }
              }
            }} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
