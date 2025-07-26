#!/usr/bin/env python3
"""
Alternative Mixtral 8x7B Model Downloader
Downloads a different Mixtral model that's known to work well
"""

import os
import sys
import requests
import psutil
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements."""
    print("🔍 Checking system requirements...")
    
    # Check RAM
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"🧠 Total RAM: {ram_gb:.1f} GB")
        if ram_gb < 16:
            print("⚠️  Warning: Less than 16GB RAM detected")
        else:
            print("✅ RAM requirements met")
    except:
        print("⚠️  Could not check RAM")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"💿 Available disk space: {free_gb:.1f} GB")
        if free_gb < 30:
            print("❌ Insufficient disk space (need at least 30GB)")
            return False
        else:
            print("✅ Disk space requirements met")
    except:
        print("⚠️  Could not check disk space")
    
    return True

def download_model():
    """Download the alternative Mixtral model."""
    
    # Alternative model URL (different quantization)
    model_url = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
    
    # Local path
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
    
    print("🎯 Downloading alternative Mixtral 8x7B model...")
    print(f"📁 Model will be saved to: {model_path}")
    print(f"🌐 Download URL: {model_url}")
    print("📦 File size: ~32GB (Q5_K_M quantization - better quality)")
    
    # Check if already exists
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"✅ Model already exists at: {model_path}")
        print(f"📏 File size: {size_gb:.1f} GB")
        return str(model_path)
    
    # Download with progress
    try:
        print("⏳ Starting download (this may take 30-60 minutes)...")
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress update every 100MB
                    if downloaded % (100 * 1024 * 1024) == 0:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"📥 Downloaded: {mb_downloaded:.0f} MB")
        
        print("✅ Download completed successfully!")
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial file
        return None

def main():
    """Main function."""
    print("🚀 Alternative Mixtral 8x7B Model Downloader")
    print("=" * 50)
    
    # Check requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with the download? (y/N): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        return
    
    # Download model
    model_path = download_model()
    
    if model_path:
        print("\n🎉 Setup complete!")
        print(f"📁 Model location: {model_path}")
        print("\n💡 Next steps:")
        print("1. Test the model: python3 -m src.cli.main process-status")
        print("2. Process a transcript: python3 -m src.cli.main process --transcript data/transcripts/your_transcript.json")
        print("3. Record and process: python3 -m src.cli.main pipeline --duration 300")
    else:
        print("❌ Setup failed")

if __name__ == "__main__":
    main() 