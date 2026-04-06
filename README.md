# GNSSS Satellite Error Prediction Project

This project is focused on predicting GNSS satellite positioning errors using machine learning and time-series forecasting.

GNSS systems are used in GPS navigation, mapping, transportation, aviation, and many other real-world applications. One of the biggest problems with GNSS data is that satellite signals are not always perfectly accurate. Errors can happen because of atmospheric conditions, satellite clock drift, ephemeris issues, and signal delays. 
The goal of this project is to study those errors and build models that can predict them before they affect positioning accuracy.

The project includes:
- Data parsing and preprocessing for GNSS satellite files
- Error calculation pipelines
- Time-series forecasting models
- Backend APIs for predictions
- Frontend dashboard for visualization
- Model comparison notebooks and analysis

Different notebooks in the project explore different forecasting approaches including:
- LSTM models
- GRU models
- Transformer-based models
- Classical regression baselines

The workflow starts by parsing GNSS data from satellite files and extracting important features related to orbit positions, clock drift, timing, and signal errors. After preprocessing, the data is used to train forecasting models that can predict future GNSS errors across multiple time horizons. The project also includes detailed error analysis and statistical summaries to understand which sources of error contribute the most to poor positioning accuracy. A frontend dashboard is included so predictions and trends can be visualized in a more interactive way.

The project tracks metrics such as:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Forecasting accuracy over different horizons
- Error distributions
- Statistical summaries
This project is still being improved further.

Current areas of work include:
- Comparing LSTM, GRU, and Transformer performance
- Improving multi-step forecasting accuracy
- Adding more real-world GNSS datasets
- Reducing inference latency for real-time use
- Building stronger backend APIs for deployment
- Improving dashboard visualizations

The main idea behind this project is to create a system that can predict and reduce GNSS positioning errors in real time, making navigation and satellite-based systems more accurate and reliable.
