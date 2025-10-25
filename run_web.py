#!/usr/bin/env python3
"""
Web Application Launcher for Gas Fire Detection System
"""
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if model and data files exist"""
    required_files = [
        'gas_fire_model.pkl',
        'safehome_sensor_data.csv',
        'app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        
        if 'gas_fire_model.pkl' in missing_files:
            print("\n💡 To train the model, run: python main.py")
        
        return False
    
    return True

def main():
    print("🔥 Gas Fire Detection System - Web Interface")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start web application. Please resolve the issues above.")
        return
    
    print("✅ All requirements satisfied!")
    print("\n🚀 Starting web application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Web application stopped.")
    except Exception as e:
        print(f"\n❌ Error starting web application: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()