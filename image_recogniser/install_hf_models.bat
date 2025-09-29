@echo off
echo 🚀 INSTALLING ULTRA ADVANCED HUGGING FACE MODELS
echo ================================================

echo.
echo 📦 Activating virtual environment...
call venv\Scripts\activate

echo.
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip
git push -u origin main
echo.
echo 🤖 Installing Hugging Face Transformers (better models)...
pip install transformers==4.30.2

echo.
echo 🔧 Installing model dependencies...
pip install tokenizers==0.13.3
pip install accelerate==0.20.3
pip install datasets==2.12.0
pip install safetensors==0.3.1
pip install sentencepiece==0.1.99
pip install protobuf==4.25.3
pip install "typing-extensions>=3.6.3,<4.6.0"
pip install huggingface-hub==0.15.1

echo.
echo 🚀 Installing better model support...
pip install bitsandbytes==0.41.0
pip install peft==0.4.0

echo.
echo ✅ Installation complete! Testing models...
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('✅ Better models ready!')"

echo.
echo 🎉 ULTRA ADVANCED MODELS INSTALLED!
echo Will try to load: Llama 2, Mistral 7B, Phi-2, Gemma 2B
echo Fallback: DialoGPT + GPT-2 + DistilBERT
echo.
echo 💡 Note: Advanced models need more RAM (4-8GB recommended)
echo 💡 Models will be downloaded automatically on first use
echo.
pause
