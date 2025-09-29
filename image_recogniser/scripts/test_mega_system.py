#!/usr/bin/env python3
"""
Test script for the Mega Image Recognizer with Hugging Face Integration
"""

import sys
import os

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import tkinter as tk
        print("✅ tkinter imported")
        
        import cv2
        print("✅ opencv imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        import PIL
        print("✅ PIL imported")
        
        # Test Hugging Face imports
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            print("✅ Hugging Face Transformers imported")
            HF_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️ Hugging Face Transformers not available: {e}")
            HF_AVAILABLE = False
        
        # Test our custom modules
        from super_face_detect import SuperFaceDetector
        print("✅ SuperFaceDetector imported")
        
        from advanced_product_detect import AdvancedProductDetector
        print("✅ AdvancedProductDetector imported")
        
        from data_mediator import DataMediator
        print("✅ DataMediator imported")
        
        from ai_chat_engine import AIChatEngine
        print("✅ AIChatEngine imported")
        
        from global_face_encodings import GlobalFaceEncodings
        print("✅ GlobalFaceEncodings imported")
        
        from live_camera_detector import LiveCameraDetector
        print("✅ LiveCameraDetector imported")
        
        return True, HF_AVAILABLE
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False, False

def test_hf_models():
    """Test Hugging Face model loading"""
    print("\n🤖 Testing Hugging Face models...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        # Test GPT-2 loading
        print("Loading GPT-2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("✅ GPT-2 loaded successfully")
        
        # Test simple generation
        prompt = "Hello, I am an AI assistant."
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=inputs.shape[1] + 20, do_sample=True, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generated response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 MEGA IMAGE RECOGNIZER - SYSTEM TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok, hf_available = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed!")
        return False
    
    # Test Hugging Face if available
    if hf_available:
        hf_ok = test_hf_models()
        if not hf_ok:
            print("\n⚠️ Hugging Face models failed, but system will work in basic mode")
    else:
        print("\n⚠️ Hugging Face not available, system will work in basic mode")
    
    print("\n🎉 SYSTEM TEST COMPLETE!")
    print("✅ All core components are working")
    print("🚀 Ready to run mega_runengine.py!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
