#!/usr/bin/env python3
"""
Installation script for HR Multi-Agent System
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("🚀 HR Multi-Agent System Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    success, output = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print(f"⚠️ Warning: Could not upgrade pip: {output}")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    success, output = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install dependencies: {output}")
        print("\n🔧 Try installing manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ Dependencies installed successfully")
    
    # Create sample CVs
    print("\n📝 Creating sample CV files...")
    success, output = run_command(f"{sys.executable} main.py --mode demo")
    if success:
        print("✅ Sample CV files created")
    else:
        print(f"⚠️ Could not create sample files: {output}")
    
    print("\n🎉 Installation completed successfully!")
    print("\n🚀 To start the system:")
    print("   Web Interface: python main.py --mode web")
    print("   CLI Demo:      python main.py --mode cli --cv-dir sample_cvs")
    print("\n🌐 Web interface will be available at: http://localhost:7860")

if __name__ == "__main__":
    main() 