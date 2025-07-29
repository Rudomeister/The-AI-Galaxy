@echo off
echo Starting AI-Galaxy Infrastructure Services...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Starting Redis and ChromaDB services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 >nul

echo.
echo Checking service status...
docker-compose ps

echo.
echo Services are starting up. Use 'docker-compose logs' to view logs.
echo Use 'docker-compose down' to stop all services.
echo.
pause