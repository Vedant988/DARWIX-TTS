# DARWIX AI Voice Agent

DARWIX AI is a real-time bilingual voice assistant built with a Next.js frontend and a FastAPI backend. It streams microphone audio to the backend, transcribes speech, generates responses with an LLM, synthesizes audio, and plays the reply back in the browser.

## Stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, asyncio
- STT: Deepgram
- LLM: Groq
- TTS: Sarvam AI
- Transport: WebSocket

## Core Features

- Live speech-to-text
- Streaming assistant responses
- Bilingual voice flow for English and Hindi
- Barge-in interruption support
- Rolling conversation memory and summarization
- Text input fallback

## Structure

```text
DARWIX/
|-- backend/
|   |-- main.py
|   |-- construction_updates.json
|   |-- requirements.txt
|   `-- modules/
|       |-- conversation_engine.py
|       |-- memory_manager.py
|       |-- speech_to_text.py
|       `-- text_to_speech.py
|-- frontend/
|   |-- app/
|   |-- components/
|   `-- package.json
|-- read_docx.py
|-- read_pdf.py
`-- README1.md
```

## Run

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

Open `http://localhost:3000` and start a conversation with DARWIX AI.
