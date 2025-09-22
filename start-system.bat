@echo off
echo Starting GNSS Error Prediction System...
echo.

echo Starting Backend API...
start "Backend" cmd /c "cd backend && python app.py"

timeout /t 5 /nobreak > nul

echo Starting Frontend...
start "Frontend" cmd /c "cd frontend && npm start"

echo.
echo System started!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
pause