@echo off
title MineGuard AI - Launcher
echo ============================================================
echo         MINEGUARD AI - LIVE PPE DETECTION SYSTEM
echo ============================================================
echo.
echo [1/2] Starting Python FastAPI Backend Server (Port 8000)...
start "MineGuard Backend" cmd /k "uv run python server.py"

echo [2/2] Starting React Vite Frontend (Port 5173)...
start "MineGuard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Waiting for servers to initialize...
timeout /t 4 >nul

echo Opening dashboard in your browser...
start http://localhost:5173

echo.
echo ============================================================
echo Application is live!
echo React Dashboard: http://localhost:5173
echo Backend API / Build: http://localhost:8000
echo ============================================================
echo Press any key to exit this launcher window (servers stay running).
pause >nul

