#!/usr/bin/env python3
"""
🚀 LIVE CAMERA LAUNCHER
Quick launcher for the Live Camera Face & Object Detection System
"""

import os
import sys
from live_camera_detector import LiveCameraDetector

def main():
    """Main launcher function"""
    print("🎥 LIVE CAMERA FACE & OBJECT DETECTOR")
    print("=" * 50)
    print("Features:")
    print("✅ Real-time face detection (MediaPipe + OpenCV)")
    print("✅ Real-time object detection (YOLO)")
    print("✅ Interactive labeling system")
    print("✅ Automatic data collection")
    print("✅ Live training capabilities")
    print("=" * 50)
    
    try:
        # Create and run the detector
        detector = LiveCameraDetector()
        detector.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Live camera system stopped by user")
    except Exception as e:
        print(f"❌ Error running live camera system: {e}")
        print("Make sure your camera is connected and not being used by another application")

if __name__ == "__main__":
    main()
