# GNSS Error Prediction - Railway Deployment Guide

## 🚀 Railway Deployment Setup

### 1. Prepare Backend for Railway

The backend is ready for Railway deployment with:
- ✅ Flask API with LSTM model integration
- ✅ requirements.txt with all dependencies
- ✅ Procfile for Railway deployment
- ✅ Model file: `best_trained_lstm_model.keras`

### 2. Railway Deployment Steps

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Deploy Backend**
   ```bash
   cd backend
   railway init
   railway deploy
   ```

4. **Set Environment Variables** (Optional)
   ```bash
   railway variables set PORT=8080
   ```

### 3. Frontend Configuration

Update the API URL in Dashboard.jsx to your Railway backend URL:
```javascript
const [apiUrl, setApiUrl] = useState("https://your-app-name.railway.app");
```

### 4. API Endpoints

Your deployed API will have these endpoints:

- `GET /` - Health check
- `POST /predict` - LSTM predictions
- `GET /model/info` - Model information
- `GET /generate_sample_data` - Generate test data

### 5. Testing Deployment

1. **Test API Health**
   ```bash
   curl https://your-app-name.railway.app/
   ```

2. **Test Prediction**
   ```bash
   curl -X POST https://your-app-name.railway.app/generate_sample_data
   ```

### 6. Frontend Deployment

Deploy frontend to Vercel, Netlify, or Railway:

**For Vercel:**
```bash
cd frontend
npm run build
vercel --prod
```

**For Railway:**
```bash
cd frontend
railway init
railway deploy
```

## 🎯 Key Features

- **Real-time LSTM Predictions**: 1-hour ahead satellite error forecasting
- **Interactive Dashboard**: Truth vs Predicted visualization
- **Multiple Satellites**: Support for different satellite data
- **Production Ready**: Optimized for cloud deployment

## 🔧 Local Testing

1. **Start Backend**
   ```bash
   cd backend
   python app.py
   ```

2. **Start Frontend**
   ```bash
   cd frontend
   npm start
   ```

3. **Test API**
   ```bash
   cd backend
   python test_api.py
   ```

## 📊 Model Performance

- **Architecture**: 3-layer LSTM (256→128→64 units)
- **Features**: 15 engineered features
- **Sequence Length**: 36 time steps (9 hours)
- **Prediction Horizon**: 4 steps (1 hour ahead)
- **Accuracy**: High-performance time series prediction

Your GNSS Error Prediction system is ready for production deployment! 🛰️