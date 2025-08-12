Product Use Case
🎯 Purpose:
The idea here is to build a personal AI Tutor that has a strong understanding of AI concepts. The tutor can listen into YouTube AI audio in real time and generate structured notes (key points, summaries, takeaways, AI concepts that become easy to digest for the learner. The notes should be comprehensive and be created like a third person talking about AI- Product, technology, concepts, business, Only. This will help the learner to glean information and expedite learning. The notes can be created or pushed into Google Docs and/or Notion AI, based on user choice. THe notes should be very well structured and use of markdown is very much appreciated. 

👤 Target Users:
AI learners and researchers
Product managers, developers, students
Anyone who consumes educational or professional YouTube content


✅ 2. Key Functional & Non-Functional Requirements
🛠 Functional Requirements
Feature	Description
Audio Capture	Capture YouTube video audio playing on Mac (using BlackHole).
Real-Time Transcription	Use Whisper (OpenAI or local Whisper.cpp) to transcribe audio.
Summarization	Use GPT-4 (prototype) or local LLMs (e.g., Mistral, LLaMA) for summarization.
Note Structuring	Organize notes into headings like Key Points, Tools Mentioned, Q&A, etc.
Export to Google Docs	Push notes to Google Docs via API with formatting.
Push to Notion	Send notes to Notion workspace via SDK with metadata.
Model Switching	Toggle between OpenAI and local models for both transcription and summary.
Storage & Organization	Save transcript + notes in structured folder hierarchy.
GUI Interface	Lightweight GUI for ease of use, future support for Shortcuts or menu bar app.

⚙️ Non-Functional Requirements
NFR	Description
Budget Friendly	API costs only during prototyping, local execution preferred later.
Performance	Responsive processing (real-time/near real-time transcription).
Accuracy	Use high-fidelity STT and LLMs to ensure relevance of notes.
Privacy	Local processing preferred; secure use of APIs.
Extensibility	Easily plug in other models (e.g., Claude, RAG, diarization, etc.).
Portability	Primary focus on macOS; later support for iPad and iPhone via workarounds.

🧩 3. System Design
🎯 Architecture Overview
scss
Copy
Edit
YouTube Video (macOS)
   └──▶ 🎧 Audio Capture (BlackHole)
        └──▶ 🧠 Transcription (OpenAI Whisper API OR whisper.cpp)
             └──▶ ✍️ Summarization (OpenAI GPT-4 OR Local LLM like Mistral)
                  └──▶ 📄 Structuring Engine (Section-wise breakdown)
                       ├──▶ 📑 Export to Google Docs
                       └──▶ 🧾 Push to Notion
                            └──▶ 📁 Store locally (org. by topic, date, etc.)
🔌 Key Components
Component	Tool/Tech
Audio Capture	BlackHole (macOS virtual audio driver)
Transcription	whisper.cpp, OpenAI Whisper API
Summarization	OpenAI GPT-4, Mistral/LLaMA via llama.cpp
Exporting	google-api-python-client, notion-client
UI	Tkinter (simple GUI), Electron + Flask or SwiftUI (advanced)
Note Format	Markdown → .docx, then Google Docs
Config & Switch	dotenv, YAML or GUI toggle panel

🧱 4. Critical Development Phases
🚀 Phase 1: CLI MVP (Mac)
Set up BlackHole for audio capture

Build CLI tool for:

Record system audio

Transcribe (Whisper)

Summarize (GPT)

Export to Google Docs

🔄 Phase 2: Notion + Config Support
Add Notion SDK for pushing notes

Add model/config toggle via YAML or UI

Create file structure & metadata tagging

🖥️ Phase 3: GUI Prototype
Build Python Tkinter or Electron-based GUI

UI actions: Start, Stop, Export, Choose Model, View History

📲 Phase 4: iPad/iPhone Support
Investigate iOS audio forwarding to Mac

Build Apple Shortcut to trigger Mac processing remotely

iCloud/Dropbox sync of processed notes

📈 Phase 5: Optimization & Background Tasks
Queue-based background processing

Diarization (speaker separation)

Advanced note format templates

Video metadata extraction via YouTube API

⚙️ 5. Underlying Technology
Functionality	Stack/Tool
Audio Routing	BlackHole
Transcription	OpenAI Whisper, whisper.cpp
Summarization	OpenAI GPT-4 (via API), Mistral, Ollama
Google Docs Export	google-api-python-client, gspread, Markdown2Docs
Notion Export	notion-client, Notion API
GUI	Python Tkinter (simple), Electron/Flask/SwiftUI (flexible)
Storage	Local folder + SQLite or JSON metadata logs
Config	.env, YAML, or GUI toggle panel

📌 6. High-Level Execution Plan (continuation of System Design)
Week	Milestone	Description
Week 1	Project Setup	Setup Python environment, BlackHole, APIs
Week 2	Audio Capture + Transcription	CLI to record & transcribe audio using Whisper (both APIs and local)
Week 3	Summarization Engine	Integrate OpenAI GPT + OSS LLMs for summary
Week 4	Export System	Format + export to Google Docs and Notion
Week 5	Basic GUI	Add minimal GUI with Start, Stop, Export buttons
Week 6	Config Management	Add model toggles, folder selection, metadata tagging
Week 7	Notion Enrichment	Tagging, topic grouping, syncing
Week 8+	iPad/iPhone Extension	Optional Shortcuts + iCloud sync, or macOS remote trigger

Further clarifying points to determine the scope

Audio Input:
Is the expectation that the tool will record all system audio, or only audio from a specific app/browser tab?
- All system audio

Should the tool support both live (real-time) and pre-recorded YouTube videos?
- Both. Mostly real-time

Transcription:
Should the user be able to choose between OpenAI Whisper API and local whisper.cpp at runtime?
- Not at runtime. It will be pre-decided before the runtime. 

Is diarization (speaker separation) a must-have for the MVP, or only for later phases?
- No. Not needed for MVP

Summarization:
Should the summarization be section-wise (e.g., per topic/segment), or just a single summary for the whole video?
- Section wise will make the notes more structured. So yes section-wise

Is there a preferred local LLM (Mistral, LLaMA, etc.), or should the system be modular to support any?
- Preferably use open source. The system prototype should be modular as we will test out the performance based on different models. But lets start with one open source for MVP but have the code flexible enought to add another

Export/Integration:
For Google Docs/Notion export, should the tool support both at once, or is it user-selectable per session?
- Support both

Is there a preferred metadata structure for notes (e.g., tags, topics, timestamps)?
- Tags, topics and YouTube video URL link