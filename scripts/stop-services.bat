@echo off
echo Stopping AI-Galaxy Infrastructure Services...
echo.

docker-compose down

echo.
echo All services stopped. Data is preserved in Docker volumes.
echo Use 'start-services.bat' to restart services.
echo.
pause