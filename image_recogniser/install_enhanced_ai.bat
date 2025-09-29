@echo off
echo.
echo ===============================================
echo 🚀 ENHANCED AI SYSTEM INSTALLATION
echo ===============================================
echo.
echo Installing Gemini API + Instruction Fine-tuning + RAG
echo.

echo 📦 Installing additional dependencies...
pip install google-generativeai
pip install sentence-transformers
pip install scikit-learn
pip install requests

echo.
echo 🧠 Testing Enhanced RAG System...
python -c "from scripts.enhanced_rag_system import EnhancedRAGSystem; rag = EnhancedRAGSystem(); print('✅ RAG System initialized successfully')"

if %errorlevel% neq 0 (
    echo ❌ RAG System test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo 🤖 Testing Enhanced Gemini Engine...
python -c "from scripts.enhanced_gemini_engine import EnhancedGeminiEngine; gemini = EnhancedGeminiEngine(); print('✅ Gemini Engine initialized successfully')"

if %errorlevel% neq 0 (
    echo ❌ Gemini Engine test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo 🎯 Testing Enhanced Mega Run Engine...
python -c "import sys; sys.path.append('scripts'); from mega_runengine import MegaImageRecognizer; print('✅ Mega Run Engine updated successfully')"

if %errorlevel% neq 0 (
    echo ❌ Mega Run Engine test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo 🎉 ENHANCED AI SYSTEM INSTALLATION COMPLETE!
echo ===============================================
echo.
echo 📋 NEXT STEPS:
echo.
echo 1. 🔑 SETUP GOOGLE API KEY:
echo    - Go to: https://makersuite.google.com/app/apikey
echo    - Create a FREE API key
echo    - Save it in: models/google_api_key.txt
echo.
echo 2. 🚀 RUN THE ENHANCED SYSTEM:
echo    - cd scripts
echo    - python mega_runengine.py
echo.
echo 3. 🧠 FEATURES NOW AVAILABLE:
echo    ✅ Instruction Fine-tuning via Prompt Engineering
echo    ✅ RAG (Retrieval-Augmented Generation)
echo    ✅ Enhanced Context Understanding
echo    ✅ Learning from Interactions
echo    ✅ Structured Knowledge Base
echo    ✅ Advanced Face/Object Analysis
echo.
echo 4. 💡 USAGE EXAMPLES:
echo    - "Who is in this image?" → Enhanced face analysis
echo    - "What objects do you see?" → Detailed object detection
echo    - "Describe this scene" → Comprehensive image understanding
echo    - "Tell me about the laptop" → Context-aware responses
echo.
echo 🌟 The system will now provide much more intelligent
echo    and context-aware responses using Gemini API!
echo.
pause
