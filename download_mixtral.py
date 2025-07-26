#!/usr/bin/env python3
"""
Download Mixtral 8x7B Model Script

This script downloads the Mixtral 8x7B model in GGUF format for local use.
"""

import os
import sys
import subprocess
from pathlib import Path

def download_mixtral():
    """Download Mixtral 8x7B model in GGUF format."""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model details
    model_name = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    model_path = models_dir / model_name
    
    # Download URL (using TheBloke's quantized version)
    download_url = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    
    print(f"🎯 Downloading Mixtral 8x7B model...")
    print(f"📁 Model will be saved to: {model_path}")
    print(f"🌐 Download URL: {download_url}")
    print(f"📦 File size: ~26GB (Q4_K_M quantization)")
    print()
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        print(f"📏 File size: {model_path.stat().st_size / (1024**3):.1f} GB")
        return str(model_path)
    
    # Download using curl
    try:
        print("⏳ Starting download... (this may take 30-60 minutes depending on your internet)")
        print("💡 Tip: You can stop and resume the download later")
        print()
        
        # Use curl with progress bar
        cmd = [
            "curl", "-L", download_url,
            "-o", str(model_path),
            "--progress-bar"
        ]
        
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0 and model_path.exists():
            print(f"\n✅ Download completed successfully!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📏 File size: {model_path.stat().st_size / (1024**3):.1f} GB")
            return str(model_path)
        else:
            print("❌ Download failed")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return None
    except KeyboardInterrupt:
        print("\n⏹️ Download interrupted by user")
        if model_path.exists():
            print(f"📁 Partial file exists at: {model_path}")
        return None

def check_system_requirements():
    """Check if system meets requirements for Mixtral."""
    print("🔍 Checking system requirements...")
    
    # Check available RAM (need at least 16GB for Q4 model)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 16:
            print("⚠️ Warning: Mixtral 8x7B requires at least 16GB RAM")
            print("💡 Consider using a smaller model or increasing RAM")
        else:
            print("✅ RAM requirements met")
            
    except ImportError:
        print("⚠️ Could not check RAM (psutil not installed)")
    
    # Check available disk space (need at least 30GB)
    try:
        disk_usage = os.statvfs(".")
        free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
        print(f"💿 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 30:
            print("⚠️ Warning: Need at least 30GB free disk space")
            print("💡 Consider freeing up space or using a smaller model")
        else:
            print("✅ Disk space requirements met")
            
    except Exception as e:
        print(f"⚠️ Could not check disk space: {e}")

def main():
    """Main function."""
    print("🚀 Mixtral 8x7B Model Downloader")
    print("=" * 50)
    
    # Check requirements
    check_system_requirements()
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the download? (y/N): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        return
    
    # Download model
    model_path = download_mixtral()
    
    if model_path:
        print("\n🎉 Setup complete!")
        print(f"📁 Model location: {model_path}")
        print("\n💡 Next steps:")
        print("1. Test the model: python3 -m src.cli.main process-status")
        print("2. Process a transcript: python3 -m src.cli.main process --transcript data/transcripts/your_transcript.json")
        print("3. Record and process: python3 -m src.cli.main pipeline --duration 300")
    else:
        print("\n❌ Setup failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 