# The Empathy Engine - DARWIX TTS

The Empathy Engine is a FastAPI + Next.js project that turns plain text into emotionally modulated speech. Instead of sending monotone text directly to a TTS engine, the system first detects emotion, maps that emotion to voice parameters, reshapes the text for prosody, and then synthesizes clause-level audio that feels more expressive and human.

This project was built to satisfy the assignment goal of giving AI a more human voice by bridging text sentiment and playable audio output.

## Project Goal

The assignment asks for a service that:
- accepts text input
- detects emotion
- modulates at least two voice parameters
- maps emotion to voice behavior with clear logic
- generates a playable audio file

This project does all of that, and adds a few stretch-goal features such as:
- granular emotion detection beyond just positive/negative/neutral
- emotion intensity scaling
- web UI for instant testing
- lexical prosody shaping to simulate pauses, breath, hesitation, and emphasis
- clause-aware chunking for more natural rhythm

## Why We Chose `bulbul:v2`

We chose Sarvam `bulbul:v2` because it fits the assignment especially well.

### 1. It exposes the exact parameters the challenge requires

The assignment explicitly asks for vocal parameter modulation. `bulbul:v2` supports:
- `pitch`
- `pace`
- `loudness`

That gives us a direct and programmable way to satisfy the requirement of changing at least two vocal parameters based on detected emotion.

### 2. It supports Indian languages and realistic conversational use cases

The project is designed for real conversational text, including English and Indian-language workflows. `bulbul:v2` supports BCP-47 language codes such as:
- `en-IN`
- `hi-IN`
- `ta-IN`
- `te-IN`
- `bn-IN`
- and several more

That makes it a better match than a generic offline engine for the assignment context of customer interaction and trust-building speech.

### 3. It supports preprocessing, which improves pronunciation quality

Sarvam preprocessing is useful for:
- numbers
- dates
- abbreviations
- symbols
- mixed-language text

Because the assignment is about believable voice interaction, this matters a lot. Human-sounding delivery is not only emotional prosody; it also depends on correct pronunciation.

### 4. It gives enough control without forcing a heavyweight speech stack

`bulbul:v2` is simple to call from Python over HTTP while still giving us:
- speaker selection
- audio codec control
- sample rate control
- cached responses
- low-friction deployment in a web application

That let us spend more effort on the empathy layer itself instead of on infrastructure.

### 5. It pairs well with our prosody-engineering approach

Sarvam does not expose direct controls like `breathiness`, `tension`, or `energy_rise` as native API sliders. That sounds like a limitation at first, but it actually fits the assignment well because it forced us to build a real empathy layer on top:
- emotion detection
- emotion-to-voice mapping
- clause chunking
- punctuation shaping
- lexical prosody injection
- per-chunk pitch/pace/loudness control

So `bulbul:v2` became the synthesis engine, while our code became the emotional intelligence layer.

## High-Level Pipeline

```text
Text Input
  -> Emotion Analysis
  -> Emotion-to-Voice Mapping
  -> Advanced Voice Enhancement
  -> Intelligent Text Formatting
  -> Clause/Chunk Prosody Direction
  -> Sarvam bulbul:v2 Synthesis
  -> Audio Stitching
  -> Playable WAV Output
```

## Current Architecture

### Backend
- `FastAPI` for API and WebSocket handling
- `emotion_engine.py` for emotion analysis using Hugging Face inference
- `voice_mapper.py` for base mapping from emotion dimensions to `pitch`, `pace`, and `loudness`
- `advanced_voice_mapper.py` for advanced attributes like `breathiness`, `tension`, `brightness`, `pitch_variance`, and `energy_rise`
- `intelligent_text_formatter.py` for lexical prosody injection such as pauses, ellipses, breath markers, stutters, and emphatic stretching
- `prosody_director.py` for clause chunking and per-chunk shaping
- `text_to_speech.py` for Sarvam `bulbul:v2` requests
- `audio_stitcher.py` for combining chunk audio into one final WAV file

### Frontend
- `Next.js` app for entering text and testing the generated speech
- WebSocket connection to the backend for real-time pipeline updates
- UI for language and speaker selection

## Assignment Requirements Mapping

### 1. Text Input
Satisfied.
- The service accepts text through the frontend and backend WebSocket flow.
- Main path: `backend/main.py`

### 2. Emotion Detection
Satisfied.
- The backend analyzes text emotion before synthesis.
- It supports more than three emotions, not just positive/negative/neutral.
- Examples include `joy`, `anger`, `fear`, `sadness`, `confusion`, `curiosity`, and `neutral`.

### 3. Vocal Parameter Modulation
Satisfied.
- The system modulates:
  - `pace`
  - `pitch`
  - `loudness`
- These are applied both at a base voice level and again per chunk for finer control.

### 4. Emotion-to-Voice Mapping
Satisfied.
- There is explicit logic mapping detected emotion to voice behavior.
- `voice_mapper.py` maps emotional dimensions like valence, arousal, and dominance into base TTS parameters.
- `advanced_voice_mapper.py` extends this into higher-level expressive cues.

### 5. Audio Output
Satisfied.
- The final result is a playable `.wav` file.
- Chunk audio is synthesized and then stitched into one final audio file.

## Stretch Goals Implemented

### Granular Emotions
Implemented.
- The system handles nuanced emotions such as fear, sadness, confusion, curiosity, excitement, amusement, and more.

### Intensity Scaling
Implemented.
- Emotion confidence and clarity affect how strongly vocal parameters are modulated.
- Example: stronger frustration can produce more aggressive pace changes, punctuation shaping, and emphasis.

### Web Interface
Implemented.
- The project includes a frontend UI built with Next.js.

### SSML-Like Expressivity
Implemented as a workaround, not literal SSML.
- Sarvam `bulbul:v2` does not provide direct native controls for all human voice traits.
- To compensate, the formatter rewrites text itself to influence prosody, for example:
  - `I... *hhhhh* I don't know what to do.`
  - `d-don't`
  - `thhhreee times already`
- This gives us SSML-like expressive control through lexical shaping.

## Key Design Choices

### 1. Why not use only positive / negative / neutral?

That would satisfy the minimum requirement, but it would not sound convincing. Human speech differs not only by sentiment polarity, but by the type of emotion.

For example:
- `fear` should sound tense and hesitant
- `sadness` should sound slower and flatter
- `anger` should sound sharper and more clipped
- `curiosity` should sound lighter and more lifted

So we chose a more granular emotion model to improve realism.

### 2. Why clause-based chunking instead of tiny word groups?

Very short chunks often sound robotic because they break rhythm unnaturally. We moved to clause-oriented chunking so phrases follow punctuation and conjunction boundaries more closely.

This gives us:
- better pacing
- more natural pauses
- stronger phrase endings
- smoother final audio after stitching

### 3. Why lexical prosody injection?

Sarvam accepts `pitch`, `pace`, and `loudness`, but not direct sliders for:
- breathiness
- tension
- vocal fry
- hesitation
- trailing-off behavior

So we convert these advanced features into things the TTS engine can actually express:
- commas
- ellipses
- breath tags like `*hhhhh*`
- stutters like `d-don't`
- elongated emphasis like `thhhreee`
- detached ending words for a fading sentence ending

This was a deliberate design decision to move beyond flat robotic output.

### 4. Why keep `pitch`, `pace`, and `loudness` subtle?

Although Sarvam supports wide numeric ranges, human speech usually stays near neutral and varies in small amounts. Large values quickly sound synthetic.

So the system uses:
- relatively tight final clamps for `pitch`
- moderate `pace` shifts
- controlled `loudness` adjustments

This keeps emotional variation believable instead of exaggerated.

## How Emotion Maps to Voice

At a high level:
- positive / energetic emotions -> slightly higher pitch, a bit more pace, brighter delivery
- sad emotions -> slower pace, softer loudness, flatter contour
- angry / frustrated emotions -> tighter phrasing, clipped delivery, emphasis on key words
- anxious emotions -> hesitations, commas, breath cues, occasional stutter behavior
- neutral / calm emotions -> longer clauses, fewer breaks, steadier parameter values

Examples of mapping logic:
- high `tension` -> shorter chunks, more pauses, occasional stutter shaping
- high `breathiness` -> ellipses and breath markers like `*hhhhh*`
- high `pace_variance` -> more comma-based segmentation
- strong emphasis phrases -> selected words can be stretched, such as `thhhreee`
- sentence endings -> last word can be detached and softened to feel like a human trailing off

## Repository Structure

```text
DARWIX-TTS/
|-- backend/
|   |-- main.py
|   |-- requirements.txt
|   |-- modules/
|   |   |-- advanced_voice_mapper.py
|   |   |-- audio_stitcher.py
|   |   |-- emotion_engine.py
|   |   |-- intelligent_text_formatter.py
|   |   |-- memory_manager.py
|   |   |-- prosody_director.py
|   |   |-- text_to_speech.py
|   |   |-- voice_mapper.py
|   |   `-- word_prosody_engine.py
|-- frontend/
|   |-- app/
|   |-- components/
|   `-- package.json
|-- README.md
`-- README1.md
```

## Setup Instructions

## Prerequisites
- Python 3.11+
- Node.js 18+
- npm

## Environment Variables
Create `backend/.env` and set the following:

```env
HF_TOKEN=your_huggingface_token
SARVAM_API_KEY=your_sarvam_api_key
GROQ_API_KEY=your_groq_api_key
ALLOWED_ORIGINS=http://localhost:3000
DEFAULT_EMOTION_MODEL=hartmann
```

Handoff document for HR:
`https://docs.google.com/document/d/1aPJa3t_CUdeCqJM4cf1KCdy1bXGOCVeo0jtcLIqp-YQ/edit?usp=sharing`

Notes:
- `HF_TOKEN` is required for emotion inference.
- `SARVAM_API_KEY` is required for TTS generation.
- `GROQ_API_KEY` is optional in practice, but recommended for richer prosody direction.
- Without Groq, the app falls back to deterministic clause chunking.

## Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Backend default URL:
- `http://localhost:8000`

Useful backend endpoints:
- `GET /health`
- `GET /debug`
- `GET /emotion-models`
- `GET /outputs/{filename}`
- `WS /ws/voice`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL:
- `http://localhost:3000`

## How to Run the Application

1. Start the backend.
2. Start the frontend.
3. Open `http://localhost:3000`.
4. Enter text into the UI.
5. Choose language and speaker.
6. Submit the text.
7. Listen to the generated audio and review the emotional delivery.

## Example Flow

Input:
```text
I'm getting really frustrated because I have told them three times already, yet nothing has changed in the report!
```

Possible empathy-engine shaping:
```text
I'm g-getting really frustrated,
because I have told them thhhreee times already,
yet nothing has changed in the report!
```

Then the system:
- classifies the emotion
- adjusts pitch/pace/loudness
- shapes phrase boundaries
- synthesizes chunk audio with Sarvam
- stitches the result into a final WAV

## Limitations

- Sarvam `bulbul:v2` does not expose native controls for every human vocal trait.
- Some advanced expressive behaviors are simulated through text shaping rather than official TTS parameters.
- Final realism still depends on the underlying speaker voice quality.
- Groq-based chunk guidance improves results, but fallback chunking is still rule-based.

## Future Improvements

- stronger per-emotion tuning tables
- direct A/B comparison mode between neutral and emotional synthesis
- more controlled SSML-style abstraction layer
- evaluation metrics for perceived empathy / naturalness
- automated listening-test harness

## Summary

This project satisfies the core assignment requirements and extends them with a practical empathy layer. We chose Sarvam `bulbul:v2` because it gives us reliable, programmable TTS controls for `pitch`, `pace`, and `loudness`, while still letting us build a richer emotional system on top through chunking, lexical prosody injection, and intelligent voice mapping.
