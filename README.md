# Lotti — Live Voice Agent with Local LanceDB RAG (Windows 11)

**Lotti** is a proactive, real-time AI team member built on the **Mistral Realtime Voice API** (`mistralai[voice-agents]`) and a local **LanceDB** knowledge base (`C:\integriti\lancedb_data`). She listens to technical discussions in internal team/Slack calls, checks architectural constraints against internal Architecture Decision Records (ADRs), and provides concise voice feedback.

## Project Structure

- `config.py` — Strongly-typed configuration loader (`python-dotenv`) with Windows 11 defaults (`C:\integriti\...`).
- `prompt_loader.py` — Dynamic Markdown system prompt loader (`SystemPromptLoader`) with `st_mtime_ns` hot-reloading and working-memory composition.
- `rag_manager.py` — Local LanceDB connector (`RAGManager`), ADR seeding utility, technical keyword trigger detector, and async vector retrieval helper.
- `audio_utils.py` — Windows 11 `DefaultAudioInterface` setup, `PyAudio`/`PortAudio` device enumeration, and headless fallback handling.
- `main.py` — `LottiVoiceAgent` async event loop integrating `client.audio.sessions.init(...)`, `@function_tool` RAG tools, live transcript listeners, and graceful shutdown.
- `prompts/system_prompt.md` — Default modular system instructions template.
- `tests/test_lotti.py` — Automated test suite verifying prompt hot-reloading, LanceDB RAG retrieval, and voice session event handling.

## Windows 11 Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (64-bit recommended for Windows 11)
- **Virtual Audio Routing for Slack (Optional)**: Install [VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/) or Voicemeeter on Windows 11 if routing Lotti directly into a Slack Huddle / call.

### 2. Install Dependencies (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
*(Note: Modern `pyaudio>=0.2.14` ships precompiled binary wheels for Windows 11 x64 that bundle PortAudio automatically.)*

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your `MISTRAL_API_KEY`:
```powershell
Copy-Item .env.example .env
notepad .env
```

By default, Lotti uses:
- **System Prompt**: `C:\integriti\prompts\system_prompt.md` (created automatically on first run if missing)
- **LanceDB Storage**: `C:\integriti\lancedb_data` (created and seeded with sample ADRs on first run if empty)

### 4. Run a Local RAG Diagnostic Check
Verify that the Markdown prompt loader and LanceDB vector search work before joining a call:
```powershell
python main.py --test-rag "Can we use MongoDB or Redis for durable domain events?"
```

### 5. Start the Live Voice Agent
```powershell
python main.py
```
Edit `C:\integriti\prompts\system_prompt.md` at any time while Lotti is running—changes are detected and hot-reloaded automatically.
