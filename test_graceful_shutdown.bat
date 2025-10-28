@echo off
REM Test script for graceful shutdown
REM This will start a quick collection and you should press Ctrl+C after 30 seconds

echo ========================================
echo Graceful Shutdown Test
echo ========================================
echo. 
echo This test will start a quick data collection.
echo.
echo INSTRUCTIONS:
echo   1. Wait for collection to start (about 10 seconds)
echo   2. Press Ctrl+C to trigger graceful shutdown
echo   3. Verify partial results are saved
echo.
echo Starting in 5 seconds...
timeout /t 5 /nobreak > nul

echo.
echo Starting collection...
echo.

cd /d "%~dp0"
python training\scripts\collection\run_full_collection.py --collectors synthetic --synthetic-samples 10000 --no-dashboard

echo.
echo ========================================
echo Test Complete
echo ========================================
echo.
echo Check for partial results:
dir data\raw\collection_partial_*.json 2>nul
echo.
pause
