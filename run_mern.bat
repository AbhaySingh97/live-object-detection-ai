@echo off
echo ===================================================
echo  Starting Smart Object Detection MERN Application
echo ===================================================

echo [1/3] Starting Python YOLO AI Microservice (Port 8000)...
start "AI Inference Service (FastAPI)" cmd /k "python -m uvicorn api.inference_server:app --port 8000 --host 127.0.0.1"

timeout /t 3 /nobreak > nul

echo [2/3] Starting Express Backend (Port 5000)...
start "Node.js Express Backend" cmd /k "cd server && node server.js"

timeout /t 2 /nobreak > nul

echo [3/3] Starting React Vite Frontend (Port 3000)...
start "React Frontend" cmd /k "cd client && npm run dev"

echo ===================================================
echo  All services started!
echo  Open React Frontend: http://localhost:3000
echo ===================================================
pause
