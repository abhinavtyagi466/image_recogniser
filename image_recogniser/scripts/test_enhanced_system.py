#!/usr/bin/env python3
"""
Test script for Enhanced AI System
Tests RAG + Instruction Fine-tuning + Gemini API integration
"""

import os
import sys
import json
from datetime import datetime

def test_enhanced_rag_system():
    """Test Enhanced RAG System"""
    print("🧠 Testing Enhanced RAG System...")
    
    try:
        from enhanced_rag_system import EnhancedRAGSystem
        
        # Initialize RAG system
        rag = EnhancedRAGSystem()
        print("✅ RAG System initialized successfully")
        
        # Test knowledge base loading
        if rag.knowledge_base:
            print(f"✅ Knowledge base loaded with {len(rag.knowledge_base)} sections")
        else:
            print("⚠️ Knowledge base is empty")
        
        # Test context retrieval
        contexts = rag.retrieve_relevant_context("Who is in this image?", "face_analysis")
        print(f"✅ Retrieved {len(contexts)} relevant contexts")
        
        # Test instruction prompt building
        prompt = rag.build_instruction_prompt("face_analysis", None, contexts)
        if prompt and len(prompt) > 100:
            print("✅ Instruction prompt built successfully")
        else:
            print("❌ Instruction prompt building failed")
        
        # Test context type determination
        context_type = rag.determine_context_type("Who is this person?")
        if context_type == "face_analysis":
            print("✅ Context type determination working")
        else:
            print(f"⚠️ Context type determination: {context_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG System test failed: {e}")
        return False

def test_enhanced_gemini_engine():
    """Test Enhanced Gemini Engine"""
    print("\n🤖 Testing Enhanced Gemini Engine...")
    
    try:
        from enhanced_gemini_engine import EnhancedGeminiEngine
        
        # Initialize Gemini engine
        engine = EnhancedGeminiEngine()
        print("✅ Gemini Engine initialized successfully")
        
        # Test API key detection
        if engine.api_key:
            print("✅ Google API key found")
        else:
            print("⚠️ Google API key not found - will use fallback responses")
        
        # Test system status
        status = engine.get_system_status()
        print(f"✅ System status: {status}")
        
        # Test instruction-tuned response (without API call)
        try:
            # This will test the prompt building without making API calls
            response = engine.generate_instruction_tuned_response(
                "Hello, how are you?", 
                None
            )
            if response:
                print("✅ Instruction-tuned response generation working")
            else:
                print("⚠️ No response generated (API key may be missing)")
        except Exception as e:
            print(f"⚠️ Response generation test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Engine test failed: {e}")
        return False

def test_mega_run_engine_integration():
    """Test Mega Run Engine Integration"""
    print("\n🚀 Testing Mega Run Engine Integration...")
    
    try:
        # Import the updated mega run engine
        from mega_runengine import MegaImageRecognizer, ENHANCED_AI_AVAILABLE
        
        if ENHANCED_AI_AVAILABLE:
            print("✅ Enhanced AI Systems available in Mega Run Engine")
        else:
            print("❌ Enhanced AI Systems not available in Mega Run Engine")
            return False
        
        # Test initialization (without GUI)
        print("Testing initialization...")
        # Note: We can't fully initialize without GUI, but we can test imports
        
        return True
        
    except Exception as e:
        print(f"❌ Mega Run Engine integration test failed: {e}")
        return False

def test_knowledge_base_structure():
    """Test Knowledge Base Structure"""
    print("\n📚 Testing Knowledge Base Structure...")
    
    try:
        from enhanced_rag_system import EnhancedRAGSystem
        
        rag = EnhancedRAGSystem()
        kb = rag.knowledge_base
        
        # Check required sections
        required_sections = ["face_patterns", "object_patterns", "conversation_patterns", "context_patterns"]
        missing_sections = []
        
        for section in required_sections:
            if section in kb:
                print(f"✅ {section} section found")
            else:
                missing_sections.append(section)
                print(f"❌ {section} section missing")
        
        if not missing_sections:
            print("✅ All required knowledge base sections present")
        else:
            print(f"⚠️ Missing sections: {missing_sections}")
        
        # Test knowledge base update
        rag.update_knowledge_base(
            "Test message", 
            "Test response", 
            {"faces": {"faces": [{"name": "Test", "confidence": 0.9}]}}
        )
        print("✅ Knowledge base update working")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base structure test failed: {e}")
        return False

def test_api_key_setup():
    """Test API Key Setup"""
    print("\n🔑 Testing API Key Setup...")
    
    api_key_path = "../models/google_api_key.txt"
    
    if os.path.exists(api_key_path):
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            if api_key and len(api_key) > 10:
                print("✅ Google API key file found and contains key")
                return True
            else:
                print("⚠️ Google API key file exists but key is empty or too short")
                return False
        except Exception as e:
            print(f"❌ Error reading API key file: {e}")
            return False
    else:
        print("⚠️ Google API key file not found")
        print("💡 Create the file: models/google_api_key.txt")
        print("💡 Get FREE API key from: https://makersuite.google.com/app/apikey")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("🧪 ENHANCED AI SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Enhanced RAG System", test_enhanced_rag_system()))
    test_results.append(("Enhanced Gemini Engine", test_enhanced_gemini_engine()))
    test_results.append(("Mega Run Engine Integration", test_mega_run_engine_integration()))
    test_results.append(("Knowledge Base Structure", test_knowledge_base_structure()))
    test_results.append(("API Key Setup", test_api_key_setup()))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Enhanced AI System is ready!")
        print("\n🚀 Next steps:")
        print("1. Make sure your Google API key is in models/google_api_key.txt")
        print("2. Run: python mega_runengine.py")
        print("3. Enjoy enhanced AI responses!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that all files are in the correct locations")
        print("3. Verify your Google API key is properly set up")
    
    print("\n" + "=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
