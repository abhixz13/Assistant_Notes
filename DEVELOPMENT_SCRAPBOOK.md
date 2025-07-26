# AI Notes Assistant - Development Scrapbook

## 📋 Project Overview
**Goal:** Build a personal AI Tutor that captures YouTube audio, transcribes it, and generates structured notes for AI learning content.

**Architecture:** Modular system with independent components that can be built and tested separately.

---

## 🏗️ Module 1: Project Setup & Infrastructure

### What We Did
1. **Created complete project structure** with modular architecture
2. **Set up configuration management** (YAML + environment variables)
3. **Implemented logging system** with file rotation and debug modes
4. **Created CLI interface** with comprehensive commands
5. **Set up testing framework** with pytest
6. **Created documentation** and setup instructions

### Problems Faced & Solutions

#### Problem 1: Import Issues with Relative Imports
**What was the problem?**
- Python relative imports (`from ..utils.config`) failed with "attempted relative import beyond top-level package"
- This is a common issue when running scripts directly vs. as modules

**Technological Concepts:**
- **Python Module System:** How Python handles imports and package structure
- **Relative vs Absolute Imports:** When to use each type
- **sys.path manipulation:** Adding directories to Python's module search path

**Purpose of solving this problem:**
- Enable proper module imports across the project
- Allow scripts to run independently
- Maintain clean code structure

**Solution:**
```python
# Changed from relative imports:
from ..utils.config import config

# To absolute imports with path manipulation:
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config
```

#### Problem 2: Missing Dependencies
**What was the problem?**
- `whisper-cpp-python==0.0.1` version didn't exist
- `soundfile` package was missing for audio playback
- Various Python packages needed for audio processing

**Technological Concepts:**
- **Package Versioning:** Understanding semantic versioning and package availability
- **Virtual Environments:** Isolating project dependencies
- **pip requirements:** Managing Python package dependencies

**Purpose of solving this problem:**
- Ensure all required packages are available
- Avoid version conflicts
- Enable audio processing functionality

**Solution:**
```bash
# Created virtual environment
python3 -m venv venv
source venv/bin/activate

# Updated requirements.txt with correct versions
pip install soundfile openai-whisper llama-cpp-python
```

#### Problem 3: Homebrew Not Installed
**What was the problem?**
- `brew` command not found when trying to install BlackHole
- macOS package manager not available

**Technological Concepts:**
- **Package Managers:** Homebrew for macOS, apt for Ubuntu, etc.
- **System-level vs User-level installations**
- **macOS audio drivers:** How audio drivers work on macOS

**Purpose of solving this problem:**
- Install BlackHole audio driver for system audio capture
- Enable virtual audio routing

**Solution:**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install BlackHole
brew install blackhole-2ch
```

---

## 🎧 Module 2: Audio Module

### What We Did
1. **Created AudioCapture class** with BlackHole integration
2. **Implemented audio recording** with configurable parameters
3. **Added audio utilities** for device detection and file processing
4. **Created comprehensive tests** for all audio functionality
5. **Built audio playback** for testing recordings

### Problems Faced & Solutions

#### Problem 4: BlackHole Audio Routing Issues
**What was the problem?**
- BlackHole was installed but not receiving audio from applications
- Recordings were silent even with audio playing
- System audio wasn't being routed to BlackHole

**Technological Concepts:**
- **macOS Audio Architecture:** Core Audio, HAL (Hardware Abstraction Layer)
- **Virtual Audio Devices:** How BlackHole creates virtual audio endpoints
- **Audio Routing:** How macOS routes audio between applications and devices
- **Multi-Output Devices:** Combining multiple audio outputs

**Purpose of solving this problem:**
- Capture system audio (YouTube, music, etc.) for transcription
- Enable real-time audio processing
- Create virtual audio pipeline

**Solution:**
1. **Created Multi-Output Device** in Audio MIDI Setup:
   - Combined "Built-in Output" and "BlackHole 2ch"
   - Set as system audio output
   - Enabled simultaneous output to speakers and BlackHole

2. **Alternative Device Detection:**
   - Tested all available audio devices
   - Found working alternatives (Desk Digital Audio, iPhone Microphone)
   - Implemented fallback mechanism

#### Problem 5: Audio Device Detection
**What was the problem?**
- BlackHole device wasn't being detected properly
- Device indices were inconsistent
- Audio device enumeration was failing

**Technological Concepts:**
- **Audio Device APIs:** How applications discover and interact with audio hardware
- **Device Indices:** How audio devices are numbered by the system
- **Cross-platform Audio:** Differences between macOS, Windows, Linux audio systems

**Purpose of solving this problem:**
- Automatically detect available audio devices
- Handle different audio setups
- Provide reliable audio capture

**Solution:**
```python
def _find_blackhole_device(self) -> int:
    """Find BlackHole device index."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if 'blackhole' in str(device).lower():
            return i
    return 0  # Fallback to default device
```

#### Problem 6: Audio File Format Issues
**What was the problem?**
- WAV files weren't being saved correctly
- Audio data format mismatches
- Sample rate and bit depth issues

**Technological Concepts:**
- **Digital Audio Formats:** WAV, MP3, FLAC, etc.
- **Audio Encoding:** PCM, bit depth, sample rates
- **Audio Processing:** How digital audio is represented and manipulated

**Purpose of solving this problem:**
- Save audio recordings in compatible formats
- Ensure audio quality for transcription
- Enable proper audio playback

**Solution:**
```python
def _save_wav_file(self, audio_data: np.ndarray, output_path: str):
    """Save audio data to WAV file."""
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(self.sample_rate)
        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
```

---

## 🎤 Module 3: Transcription Module

### What We Did
1. **Implemented WhisperTranscriber class** for real-time and batch transcription
2. **Created transcription utilities** for file processing and output formatting
3. **Built CLI commands** for transcribe, transcribe-live, and transcribe-batch
4. **Added structured output** with timestamps, key points, and technical terms
5. **Implemented comprehensive testing** for all transcription functionality
6. **Fixed critical CLI bug** in duration parameter passing

### Problems Faced & Solutions

#### Problem 7: CLI Duration Parameter Bug
**What was the problem?**
- When using `--output` parameter, the duration wasn't being passed correctly
- `capture.record_to_file(output)` was passing output as duration parameter
- This caused recordings to be much shorter than requested (2-6 seconds instead of 30)

**Technological Concepts:**
- **Function Parameter Order:** How Python handles positional vs keyword arguments
- **CLI Parameter Mapping:** How Click maps CLI options to function parameters
- **Method Signatures:** Understanding parameter order in method definitions

**Purpose of solving this problem:**
- Ensure accurate recording durations
- Fix parameter passing in CLI commands
- Enable proper audio capture for transcription

**Solution:**
```python
# BUGGY CODE (before fix):
capture.record_to_file(output)  # output was passed as duration parameter!

# FIXED CODE (after fix):
capture.record_to_file(duration=duration, output_path=output)
```

#### Problem 8: Audio Device Selection for Transcription
**What was the problem?**
- BlackHole device was capturing silence (no audio routing)
- System needed to use alternative audio devices for actual content capture
- Device selection logic was hardcoded to BlackHole only

**Technological Concepts:**
- **Audio Device APIs:** How applications discover and select audio devices
- **Device Enumeration:** Listing all available audio devices
- **Fallback Mechanisms:** Providing alternatives when primary device fails

**Purpose of solving this problem:**
- Enable actual audio content capture
- Provide reliable audio device selection
- Support multiple audio input sources

**Solution:**
```python
def _find_blackhole_device(self) -> int:
    """Find audio device index by name or default to BlackHole."""
    devices = sd.query_devices()
    
    # First, try to find the specified device name
    target_device = self.blackhole_device.lower()
    for i, device in enumerate(devices):
        device_str = str(device).lower()
        if target_device in device_str:
            return i
    
    # Fallback to BlackHole if specified device not found
    for i, device in enumerate(devices):
        if 'blackhole' in str(device).lower():
            return i
    
    return 0  # Default device
```

#### Problem 9: Whisper Model Loading and Performance
**What was the problem?**
- Initial attempt with `whisper-cpp-python` had compilation issues
- Needed to switch to `openai-whisper` for easier installation
- Model loading was slow for first-time use

**Technological Concepts:**
- **Local LLM Deployment:** Running AI models locally vs cloud APIs
- **Model Caching:** How Whisper caches models for faster subsequent loads
- **Model Sizes:** Trade-offs between tiny, base, small, medium, large models

**Purpose of solving this problem:**
- Enable local transcription without API costs
- Provide fast transcription for real-time use
- Balance accuracy vs performance

**Solution:**
```python
# Switched from whisper-cpp-python to openai-whisper
pip install openai-whisper

# Model loading with caching
self.model = whisper.load_model(model_size)
```

### Transcription Module Features

#### Real-Time Transcription
- **Live Audio Capture:** Records system audio in real-time
- **Streaming Processing:** Processes audio in 5-second chunks
- **Live Status Updates:** Shows duration, segments, and queue size
- **Graceful Shutdown:** Ctrl+C stops recording and saves transcript

#### Batch Transcription
- **Directory Processing:** Transcribes all audio files in a directory
- **Model Reuse:** Loads model once for multiple files
- **Progress Tracking:** Shows progress across multiple files

#### Structured Output
- **Timestamps:** Precise time markers for each segment
- **Key Points Extraction:** Identifies important concepts
- **Technical Terms:** Detects AI/ML terminology
- **Formatted Notes:** Clean Markdown output

#### Performance Results
- **2-Minute Audio:** 28 segments, 1:59 duration captured
- **Transcription Speed:** ~6 seconds for 2-minute audio
- **Content Quality:** High-quality AI/ML lecture content
- **Accuracy:** Excellent speech-to-text conversion

#### Note Structure Evolution
- **Original Format:** Complex 8-section structure with detailed subsections
- **User Feedback:** Requested simpler, more focused format
- **New Format:** Simple 3-section structure with key topics and discussion points
- **Implementation:** Updated prompts and Markdown generation to match user preferences
- **Fallback Processing:** Added capability to generate notes without Mixtral model

#### Model Upgrade
- **Previous Model:** LLaMA 2 7B (planned)
- **New Model:** Mixtral 8x7B (12.7B parameters, MoE architecture)
- **Benefits:** Better performance, larger context window (8K vs 4K), improved reasoning
- **Configuration:** Updated settings.yaml and processor class
- **Download Script:** Created download_mixtral.py for easy model setup

### Technical Concepts Learned

#### OpenAI Whisper
- **Model Architecture:** Transformer-based speech recognition
- **Model Sizes:** tiny (39M), base (74M), small (244M), medium (769M), large (1550M)
- **Language Support:** Multi-language transcription
- **Local Deployment:** Running without internet connection

#### Audio Processing for Transcription
- **Chunk Processing:** Breaking long audio into manageable segments
- **Real-time Streaming:** Processing audio as it's captured
- **Threading:** Separate threads for audio capture and transcription
- **Queue Management:** Buffering audio chunks for processing

#### Structured Data Output
- **JSON Schema:** Structured transcript data
- **Markdown Formatting:** Human-readable notes
- **Metadata Extraction:** Duration, segments, timestamps
- **Content Analysis:** Key points and technical terms

---

## 🔧 Technical Concepts Learned

### Audio Processing
- **Sample Rate:** Number of audio samples per second (16kHz, 44.1kHz, 48kHz)
- **Bit Depth:** Number of bits per sample (16-bit, 24-bit)
- **Channels:** Mono (1) vs Stereo (2) audio
- **RMS (Root Mean Square):** Measure of audio signal strength

### macOS Audio System
- **Core Audio:** macOS audio framework
- **BlackHole:** Virtual audio driver for routing system audio
- **Audio MIDI Setup:** macOS utility for configuring audio devices
- **Multi-Output Devices:** Combining multiple audio outputs

### Python Audio Libraries
- **sounddevice:** Real-time audio I/O
- **soundfile:** Audio file reading/writing
- **librosa:** Audio analysis and processing
- **numpy:** Numerical operations on audio data

### Development Practices
- **Modular Architecture:** Building independent, testable components
- **Configuration Management:** Centralized settings with YAML and environment variables
- **Logging:** Comprehensive logging with different levels and outputs
- **Testing:** Unit tests, integration tests, and performance tests

---

## 📊 Current Status

### ✅ Completed Modules
1. **Project Infrastructure** - Complete
   - Configuration management
   - Logging system
   - CLI interface
   - Testing framework

2. **Audio Module** - Complete
   - Audio recording functionality
   - Device detection
   - File management
   - Playback testing

3. **Transcription Module** - Complete
   - OpenAI Whisper integration
   - Real-time and batch transcription
   - Structured output with timestamps
   - Key points and technical terms extraction

### 🔄 Next Steps
1. **Transcription Module** - ✅ Complete
   - OpenAI Whisper integration
   - Real-time and batch transcription
   - Structured output generation

2. **Processing Module** - ✅ Complete (Enhanced Version)
   - Mixtral 8x7B integration (configured, model download needed)
   - Simple note structure ✅
   - Key topics and discussion points ✅
   - Fallback processing without Mixtral ✅
   - Automatic note generation ✅

3. **Export Module**
   - Google Docs integration
   - Notion integration
   - Markdown formatting

---

## 🎯 Key Learnings

### Project Management
- **Incremental Development:** Build and test each module independently
- **Problem Documentation:** Record issues and solutions for future reference
- **User Testing:** Test with real scenarios (YouTube videos, audio playback)

### Technical Architecture
- **Modular Design:** Each component can be developed and tested separately
- **Configuration-Driven:** Easy to adjust settings without code changes
- **Error Handling:** Comprehensive logging and error recovery
- **Cross-Platform Considerations:** Audio systems differ across operating systems

### Development Workflow
- **Virtual Environments:** Essential for Python project isolation
- **Version Control:** Track changes and rollback when needed
- **Testing Strategy:** Unit tests, integration tests, and manual testing
- **Documentation:** Keep detailed notes of problems and solutions

---

## 🚀 Success Metrics

### Audio Module Success
- ✅ Audio recording works with multiple devices
- ✅ File saving and playback functional
- ✅ Device detection and configuration working
- ✅ Error handling and logging implemented
- ✅ Tests passing and comprehensive

### Transcription Module Success
- ✅ Real-time transcription working with live audio
- ✅ Batch transcription processing multiple files
- ✅ Structured output with timestamps and metadata
- ✅ Key points and technical terms extraction
- ✅ 2-minute audio processed in ~6 seconds
- ✅ High-quality AI/ML content transcription
- ✅ CLI commands for all transcription modes

### Project Infrastructure Success
- ✅ Modular architecture established
- ✅ Configuration management working
- ✅ CLI interface functional
- ✅ Development environment set up
- ✅ Documentation complete

---

*This scrapbook will be updated as we continue development through the remaining modules.* 