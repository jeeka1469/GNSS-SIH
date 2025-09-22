import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import testData from "./data/testData.json";

const Dashboard = () => {
  const [satellite, setSatellite] = useState("");
  const [filteredData, setFilteredData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState(
    process.env.REACT_APP_API_URL || "http://localhost:5000"
  );

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
    labels: [
      ...filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
      ...(prediction ? prediction.prediction_horizons.flatMap(horizon => 
        prediction.predictions[horizon].map((_, idx) => `${horizon}+${idx * 15}min`)
      ) : [])
    ],
    datasets: [
      {
        label: "True Orbit Error",
        data: [
          ...filteredData.map((d) => d.pos_error_m),
          ...Array(prediction ? prediction.prediction_horizons.reduce((sum, horizon) => 
            sum + prediction.predictions[horizon].length, 0) : 0).fill(null)
        ],
        borderColor: "#facc15",
        backgroundColor: "#facc15",
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
      ...(prediction ? prediction.prediction_horizons.map((horizon, horizonIdx) => ({
        label: `Predicted Orbit Error (${horizon})`,
        data: [
          ...Array(filteredData.length).fill(null),
          ...prediction.prediction_horizons.slice(0, horizonIdx).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), []),
          ...prediction.predictions[horizon].map(p => p.orbit_error_m),
          ...prediction.prediction_horizons.slice(horizonIdx + 1).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), [])
        ],
        borderColor: [`#ef4444`, `#f97316`, `#eab308`, `#22c55e`, `#3b82f6`, `#8b5cf6`, `#ec4899`][horizonIdx % 7],
        backgroundColor: [`#ef4444`, `#f97316`, `#eab308`, `#22c55e`, `#3b82f6`, `#8b5cf6`, `#ec4899`][horizonIdx % 7],
        pointRadius: 3,
        pointHoverRadius: 6,
        borderDash: [5, 5],
      })) : [])
    ],
  };

  const clockData = {
    labels: [
      ...filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
      ...(prediction ? prediction.prediction_horizons.flatMap(horizon => 
        prediction.predictions[horizon].map((_, idx) => `${horizon}+${idx * 15}min`)
      ) : [])
    ],
    datasets: [
      {
        label: "True Clock Error",
        data: [
          ...filteredData.map((d) => d.clock_error_ns),
          ...Array(prediction ? prediction.prediction_horizons.reduce((sum, horizon) => 
            sum + prediction.predictions[horizon].length, 0) : 0).fill(null)
        ],
        borderColor: "#3b82f6",
        backgroundColor: "#3b82f6",
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
      ...(prediction ? prediction.prediction_horizons.map((horizon, horizonIdx) => ({
        label: `Predicted Clock Error (${horizon})`,
        data: [
          ...Array(filteredData.length).fill(null),
          ...prediction.prediction_horizons.slice(0, horizonIdx).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), []),
          ...prediction.predictions[horizon].map(p => p.clock_error_ns),
          ...prediction.prediction_horizons.slice(horizonIdx + 1).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), [])
        ],
        borderColor: [`#10b981`, `#06b6d4`, `#8b5cf6`, `#f59e0b`, `#ef4444`, `#ec4899`, `#6366f1`][horizonIdx % 7],
        backgroundColor: [`#10b981`, `#06b6d4`, `#8b5cf6`, `#f59e0b`, `#ef4444`, `#ec4899`, `#6366f1`][horizonIdx % 7],
        pointRadius: 3,
        pointHoverRadius: 6,
        borderDash: [5, 5],
      })) : [])
    ],
  };

  // Add radial error chart
  const radialData = {
    labels: [
      ...filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
      ...(prediction ? prediction.prediction_horizons.flatMap(horizon => 
        prediction.predictions[horizon].map((_, idx) => `${horizon}+${idx * 15}min`)
      ) : [])
    ],
    datasets: [
      {
        label: "True Radial Error",
        data: [
          ...filteredData.map((d) => d.pos_error_m * 0.8), // Approximate radial component
          ...Array(prediction ? prediction.prediction_horizons.reduce((sum, horizon) => 
            sum + prediction.predictions[horizon].length, 0) : 0).fill(null)
        ],
        borderColor: "#f59e0b",
        backgroundColor: "#f59e0b",
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 4,
      },
      ...(prediction ? prediction.prediction_horizons.map((horizon, horizonIdx) => ({
        label: `Predicted Radial Error (${horizon})`,
        data: [
          ...Array(filteredData.length).fill(null),
          ...prediction.prediction_horizons.slice(0, horizonIdx).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), []),
          ...prediction.predictions[horizon].map(p => p.radial_error_m),
          ...prediction.prediction_horizons.slice(horizonIdx + 1).reduce((acc, h) => 
            acc.concat(Array(prediction.predictions[h].length).fill(null)), [])
        ],
        borderColor: [`#dc2626`, `#ea580c`, `#ca8a04`, `#16a34a`, `#2563eb`, `#7c3aed`, `#be185d`][horizonIdx % 7],
        backgroundColor: [`#dc2626`, `#ea580c`, `#ca8a04`, `#16a34a`, `#2563eb`, `#7c3aed`, `#be185d`][horizonIdx % 7],
        pointRadius: 3,
        pointHoverRadius: 6,
        borderDash: [5, 5],
      })) : [])
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
            <h2 className="text-2xl font-bold text-yellow-400 mb-4">🤖 LSTM Multi-Horizon Predictions</h2>
            <div className="mb-4">
              <p className="text-sm text-gray-300">
                Confidence: <span className="text-green-400 font-semibold">{prediction.model_confidence}</span> 
                ({(prediction.confidence_score * 100).toFixed(1)}%) | 
                Data Points: {prediction.data_points_used}
              </p>
            </div>
            
            {/* Prediction horizons grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {prediction.prediction_horizons.map((horizon, idx) => (
                <div key={horizon} className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold text-yellow-400 mb-3">{horizon} Prediction</h3>
                  {prediction.predictions[horizon].slice(-1).map((pred, predIdx) => (
                    <div key={predIdx} className="space-y-2">
                      <div className="text-center">
                        <p className="text-red-400 font-semibold text-sm">Orbit Error</p>
                        <p className="text-lg">{pred.orbit_error_m.toFixed(3)} m</p>
                      </div>
                      <div className="text-center">
                        <p className="text-green-400 font-semibold text-sm">Clock Error</p>
                        <p className="text-lg">{pred.clock_error_ns.toFixed(3)} ns</p>
                      </div>
                      <div className="text-center">
                        <p className="text-blue-400 font-semibold text-sm">Radial Error</p>
                        <p className="text-lg">{pred.radial_error_m.toFixed(3)} m</p>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            
            {/* 24hr detailed prediction summary */}
            {prediction.predictions["24hr"] && (
              <div className="mt-6 bg-gray-900 p-4 rounded-lg border border-yellow-400">
                <h3 className="text-xl font-bold text-yellow-400 mb-3">📈 Full Day (24hr) Prediction Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <p className="text-red-400 font-semibold">Orbit Error Range</p>
                    <p className="text-sm">
                      {Math.min(...prediction.predictions["24hr"].map(p => p.orbit_error_m)).toFixed(3)} - 
                      {Math.max(...prediction.predictions["24hr"].map(p => p.orbit_error_m)).toFixed(3)} m
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-green-400 font-semibold">Clock Error Range</p>
                    <p className="text-sm">
                      {Math.min(...prediction.predictions["24hr"].map(p => p.clock_error_ns)).toFixed(3)} - 
                      {Math.max(...prediction.predictions["24hr"].map(p => p.clock_error_ns)).toFixed(3)} ns
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-blue-400 font-semibold">Prediction Points</p>
                    <p className="text-lg">{prediction.predictions["24hr"].length}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h2 className="font-semibold mb-3 text-yellow-400">Orbit Error: True vs Predicted</h2>
            <Line data={orbitData} options={{
              responsive: true,
              plugins: {
                legend: {
                  labels: {
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              scales: {
                x: {
                  ticks: { 
                    color: 'white',
                    maxTicksLimit: 10,
                    font: { size: 10 }
                  }
                },
                y: {
                  ticks: { 
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              elements: {
                point: {
                  radius: 2,
                  hoverRadius: 4
                }
              }
            }} />
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h2 className="font-semibold mb-3 text-yellow-400">Clock Error: True vs Predicted</h2>
            <Line data={clockData} options={{
              responsive: true,
              plugins: {
                legend: {
                  labels: {
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              scales: {
                x: {
                  ticks: { 
                    color: 'white',
                    maxTicksLimit: 10,
                    font: { size: 10 }
                  }
                },
                y: {
                  ticks: { 
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              elements: {
                point: {
                  radius: 2,
                  hoverRadius: 4
                }
              }
            }} />
          </div>
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
            <h2 className="font-semibold mb-3 text-yellow-400">Radial Error: True vs Predicted</h2>
            <Line data={radialData} options={{
              responsive: true,
              plugins: {
                legend: {
                  labels: {
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              scales: {
                x: {
                  ticks: { 
                    color: 'white',
                    maxTicksLimit: 10,
                    font: { size: 10 }
                  }
                },
                y: {
                  ticks: { 
                    color: 'white',
                    font: { size: 10 }
                  }
                }
              },
              elements: {
                point: {
                  radius: 2,
                  hoverRadius: 4
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
