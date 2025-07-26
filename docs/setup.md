# Setup Instructions

## Prerequisites

### 1. System Requirements
- **macOS** (required for BlackHole audio driver)
- **Python 3.8+**
- **Homebrew** (for installing BlackHole)

### 2. Audio Setup (macOS)
```bash
# Install BlackHole audio driver
brew install blackhole-2ch

# Verify installation
system_profiler SPAudioDataType | grep BlackHole
```

### 3. Model Downloads

#### LLaMA 2 7B Model
```bash
# Create models directory
mkdir -p models

# Download LLaMA 2 7B GGUF model (choose one):
# Option 1: From Hugging Face
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -O models/llama-2-7b.gguf

# Option 2: From local source
# Copy your LLaMA 2 7B GGUF file to models/llama-2-7b.gguf
```

#### Whisper Model
```bash
# Download whisper.cpp models
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
./models/download-ggml-model.sh base.en
cp models/ggml-base.en.bin ../models/whisper-base.gguf
cd ..
rm -rf whisper.cpp
```

## Installation

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd Notes_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp config/env.example config/.env

# Edit configuration
nano config/.env
# or
code config/.env
```

### 3. API Setup

#### Google Docs API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Docs API and Google Drive API
4. Create OAuth 2.0 credentials
5. Download credentials JSON file
6. Add credentials to `config/.env`

#### Notion API
1. Go to [Notion Developers](https://developers.notion.com/)
2. Create a new integration
3. Get the integration token
4. Add token to `config/.env`
5. Share your Notion workspace with the integration

### 4. Audio Device Setup
```bash
# Check available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set BlackHole as default input (optional)
# System Preferences > Sound > Input > BlackHole 2ch
```

## Testing Setup

### 1. Verify Audio Capture
```bash
# Test audio capture
python -c "
import sounddevice as sd
import numpy as np

# Record 5 seconds of audio
recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
sd.wait()
print('Audio capture test successful')
"
```

### 2. Test Transcription
```bash
# Test whisper.cpp (if installed)
cd whisper.cpp
./main -m models/ggml-base.en.bin -f test.wav
```

### 3. Test LLaMA
```bash
# Test LLaMA model loading
python -c "
from llama_cpp import Llama
llm = Llama(model_path='models/llama-2-7b.gguf')
print('LLaMA model loaded successfully')
"
```

## Troubleshooting

### Common Issues

#### 1. BlackHole Not Found
```bash
# Reinstall BlackHole
brew uninstall blackhole-2ch
brew install blackhole-2ch
# Restart system
```

#### 2. Audio Permission Issues
- Go to System Preferences > Security & Privacy > Privacy > Microphone
- Add your terminal/IDE to allowed applications

#### 3. Model Loading Errors
```bash
# Check model file exists
ls -la models/

# Verify model file integrity
file models/llama-2-7b.gguf
```

#### 4. Memory Issues
- LLaMA 2 7B requires ~4GB RAM
- Consider using smaller models for testing
- Close other applications to free memory

## Development Setup

### 1. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install black flake8 pytest
```

### 2. Pre-commit Hooks (Optional)
```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

### 3. IDE Setup
- **VS Code**: Install Python extension
- **PyCharm**: Configure virtual environment
- **Vim/Neovim**: Install Python language server

## Next Steps

After setup is complete:

1. **Test Audio Module**: `python -m tests.test_audio`
2. **Test Transcription**: `python -m tests.test_transcription`
3. **Test Summarization**: `python -m tests.test_summarization`
4. **Run Full Pipeline**: `python src/cli/main.py --help`

## Support

For issues:
1. Check the troubleshooting section
2. Review logs in `logs/app.log`
3. Create an issue with detailed error information
