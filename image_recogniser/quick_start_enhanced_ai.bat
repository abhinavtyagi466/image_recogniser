@echo off
echo.
echo ===============================================
echo 🚀 QUICK START - ENHANCED AI SYSTEM
echo ===============================================
echo.

echo 📋 CHECKING SYSTEM STATUS...
echo.

REM Check if enhanced AI files exist
if exist "scripts\enhanced_rag_system.py" (
    echo ✅ Enhanced RAG System found
) else (
    echo ❌ Enhanced RAG System not found
    echo Please run: install_enhanced_ai.bat
    pause
    exit /b 1
)

if exist "scripts\enhanced_gemini_engine.py" (
    echo ✅ Enhanced Gemini Engine found
) else (
    echo ❌ Enhanced Gemini Engine not found
    echo Please run: install_enhanced_ai.bat
    pause
    exit /b 1
)

REM Check for API key
if exist "models\google_api_key.txt" (
    echo ✅ Google API key found
) else (
    echo ⚠️ Google API key not found
    echo.
    echo 🔑 SETUP REQUIRED:
    echo 1. Go to: https://makersuite.google.com/app/apikey
    echo 2. Create a FREE API key
    echo 3. Save it in: models\google_api_key.txt
    echo.
    echo Press any key to continue anyway (will use fallback responses)...
    pause
)

echo.
echo 🚀 STARTING ENHANCED AI SYSTEM...
echo.

cd scripts
python mega_runengine.py

echo.
echo 👋 Enhanced AI System closed.
pause
