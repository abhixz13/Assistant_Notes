# AI Notes Assistant

A personal AI tutor that captures YouTube audio, transcribes it, and generates structured notes for AI learning content.

## 🎯 Purpose
Build a personal AI Tutor that listens to YouTube AI audio in real time and generates structured notes (key points, summaries, takeaways, AI concepts) that are easy to digest for learners.

## 🏗️ Project Structure

```
Notes_assistant/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   ├── __init__.py
│   ├── settings.yaml           # Configuration settings
│   └── .env.example           # Environment variables template
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py         # Audio capture using BlackHole
│   │   └── utils.py           # Audio utilities
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── whisper_local.py   # Local whisper.cpp integration
│   │   └── utils.py           # Transcription utilities
│   ├── summarization/
│   │   ├── __init__.py
│   │   ├── llama_local.py     # LLaMA 2 7B integration
│   │   └── utils.py           # Summarization utilities
│   ├── export/
│   │   ├── __init__.py
│   │   ├── google_docs.py     # Google Docs export
│   │   ├── notion.py          # Notion export
│   │   └── utils.py           # Export utilities
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── local_storage.py   # Local file storage
│   │   └── metadata.py        # Metadata management
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py            # CLI interface
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       └── logger.py          # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_transcription.py
│   ├── test_summarization.py
│   └── test_export.py
├── data/
│   ├── transcripts/           # Stored transcripts
│   ├── notes/                # Generated notes
│   └── metadata/             # Metadata files
├── models/                   # Local model files (LLaMA, whisper)
└── docs/                    # Documentation
    └── setup.md             # Setup instructions
```

## 🚀 Development Roadmap

### Phase 1: Core Infrastructure
1. **Project Setup** - Dependencies, config management, logging
2. **Audio Module** - BlackHole integration, system audio capture
3. **Transcription Module** - whisper.cpp integration
4. **Summarization Module** - LLaMA 2 7B integration
5. **Storage Module** - Local file management, metadata handling

### Phase 2: Export & Integration
6. **Export Module** - Google Docs and Notion integration
7. **CLI Interface** - Command-line interface for all modules
8. **Integration Testing** - End-to-end testing

### Phase 3: Enhancement
9. **GUI Interface** - Tkinter-based GUI
10. **Advanced Features** - Model switching, advanced config

## 🛠️ Setup Instructions

### Prerequisites
- macOS (for BlackHole audio driver)
- Python 3.8+
- BlackHole audio driver installed
- LLaMA 2 7B model files
- whisper.cpp compiled

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Notes_assistant

# Install dependencies
pip install -r requirements.txt

# Install BlackHole (macOS)
brew install blackhole-2ch

# Setup environment
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

## 📋 Module Dependencies

### Audio Module
- `pyaudio` - Audio capture
- `sounddevice` - Audio processing
- BlackHole driver (macOS)

### Transcription Module
- `whisper.cpp` - Local transcription
- `numpy` - Audio processing

### Summarization Module
- `llama-cpp-python` - LLaMA 2 integration
- `transformers` - Model utilities

### Export Module
- `google-api-python-client` - Google Docs API
- `notion-client` - Notion API
- `markdown` - Markdown processing

### Storage Module
- `sqlite3` - Local database
- `yaml` - Configuration files

## 🔧 Configuration

The system uses YAML configuration files for:
- Model settings (LLaMA, Whisper parameters)
- Export settings (Google Docs, Notion)
- Storage paths and organization
- Audio capture settings

## 🧪 Testing Strategy

Each module will be tested independently:
1. **Unit Tests** - Individual functions
2. **Integration Tests** - Module interactions
3. **End-to-End Tests** - Complete workflow

## 📝 Notes Structure

Generated notes will include:
- **Key Points** - Main takeaways
- **Tools Mentioned** - AI tools, frameworks, libraries
- **Concepts Explained** - AI concepts and definitions
- **Q&A Section** - Questions and answers from content
- **Metadata** - Tags, topics, YouTube URL, timestamp

## 🎯 Next Steps

1. Set up project structure and dependencies
2. Build and test Audio Module
3. Build and test Transcription Module
4. Build and test Summarization Module
5. Build and test Storage Module
6. Build and test Export Module
7. Integrate all modules in CLI
8. Add GUI interface

---

**Status**: 🚧 In Development - Phase 1 

##  **Fix: Use Simpler Whisper Alternative**

### **Step 1: Update requirements.txt to remove problematic packages**
Let me create a simpler requirements file that avoids the compilation issues:

```bash
<code_block_to_apply_changes_from>
```

### **Step 2: Install the simpler requirements**
```bash
pip install -r requirements_simple.txt
```

### **Step 3: Verify installation**
```bash
pip list | grep -E "(whisper|llama|click|yaml)"
```

**Expected:** You should see packages installed without compilation errors.

---

**Why this approach is better:**
- `openai-whisper` is the official Python package (no compilation needed)
- It downloads models automatically when first used
- Much simpler installation process
- Still works offline after initial download

**Go ahead and run these commands. Let me know if you get any errors!**

**Note:** The whisper model will be downloaded automatically when you first use it (about 1GB for the base model). 