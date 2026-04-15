# DARWIX AI - Voice Agent

A real-time AI voice agent for **DARWIX AI** with bilingual voice interaction, live transcription, streaming responses, and interruption handling.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js, TypeScript, Web Audio API |
| **Backend** | FastAPI, Python, asyncio |
| **STT** | Deepgram |
| **LLM** | Groq |
| **TTS** | Sarvam AI |
| **Transport** | WebSocket |

---

## Features

- Real-time voice capture and playback
- English and Hindi support
- Low-latency streaming response pipeline
- Rolling conversation memory with summarization
- Barge-in interruption support
- Text input fallback alongside voice

---

## Project Structure

```text
DARWIX/
|-- backend/
|   |-- main.py
|   |-- modules/
|   |   |-- conversation_engine.py
|   |   |-- speech_to_text.py
|   |   |-- text_to_speech.py
|   |   `-- memory_manager.py
|   |-- construction_updates.json
|   `-- requirements.txt
`-- frontend/
    |-- app/
    |   `-- page.tsx
    |-- components/
    |   `-- VoiceAgent.tsx
    `-- package.json
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/DARWIX.git
cd DARWIX
```

Backend:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

---

## Usage

1. Select a language.
2. Start a call with the mic button.
3. Speak naturally and DARWIX AI will respond.
4. Interrupt at any time while the assistant is speaking.
5. Use the text box if you prefer typing.
