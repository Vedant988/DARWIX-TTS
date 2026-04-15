import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from modules.emotion_engine import EmotionAnalysisError, emotion_engine
from modules.text_to_speech import TTSGenerationError, tts_engine
from modules.voice_mapper import voice_mapper

app = FastAPI(title="DARWIX AI Emotion Voice Pipeline")

raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",")] if raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Log initialization and check HF configuration."""
    hf_token = os.environ.get("HF_TOKEN")
    print("\n" + "="*60)
    print("DARWIX AI Backend - Inference API Mode")
    print("="*60)
    print(f"Mode: Using Hugging Face Inference API (no local model downloads)")
    print(f"HF_TOKEN: {'✓ Set' if hf_token else '✗ MISSING (REQUIRED!)'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    if not hf_token:
        print("❌ ERROR: HF_TOKEN is REQUIRED for Inference API mode")
        print("   Please set HF_TOKEN in .env file")
        print("   Get your token: https://huggingface.co/settings/tokens")
        print()
        raise RuntimeError("HF_TOKEN environment variable not set")


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or "sample"


def _build_assistant_summary(analysis, voice_profile, file_name: str) -> str:
    top_summary = ", ".join(f"{item.label}:{item.score:.2f}" for item in analysis.top_emotions[:3])
    return (
        f"Emotion: {analysis.primary_emotion} ({analysis.confidence:.2f}). "
        f"Voice -> speaker={voice_profile.speaker}, pace={voice_profile.pace:.2f}, "
        f"pitch={voice_profile.pitch:.2f}, loudness={voice_profile.loudness:.2f}. "
        f"Top scores: {top_summary}. File: {file_name}."
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": "text -> emotion -> voice mapping -> tts",
        "voice_input_enabled": False,
    }


@app.get("/debug")
async def debug_info():
    """Debug endpoint to check environment and dependencies"""
    try:
        from huggingface_hub import InferenceClient
        inference_available = True
        inference_version = "via huggingface_hub"
    except ImportError:
        inference_available = False
        inference_version = None
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_token_status = "✓ Set" if hf_token else "✗ MISSING"
    
    return {
        "debug_info": {
            "mode": "Hugging Face Inference API (Cloud-based, no local models)",
            "inference_client": {
                "available": inference_available,
                "version": inference_version,
            },
            "hf_token": hf_token_status,
            "hf_token_preview": f"{hf_token[:10]}..." if hf_token else None,
            "python_version": sys.version,
            "output_directory": str(OUTPUT_DIR),
        }
    }


@app.get("/emotion-models")
async def list_emotion_models():
    return {"models": emotion_engine.list_models()}


@app.get("/outputs/{filename}")
async def get_generated_output(filename: str):
    output_path = (OUTPUT_DIR / filename).resolve()
    if output_path.parent != OUTPUT_DIR.resolve() or not output_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(output_path)


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket, language: str = "en"):
    await websocket.accept()
    session_id = str(id(websocket))
    current_processing_task: asyncio.Task | None = None
    audio_disabled_notified = False

    async def cancel_current_task(send_interrupt: bool = True):
        nonlocal current_processing_task
        task = current_processing_task
        current_processing_task = None
        if task and not task.done():
            task.cancel()
            if send_interrupt:
                await websocket.send_text(json.dumps({"type": "interrupt"}))

    async def process_text_input(payload: dict):
        text = payload.get("text", "").strip()
        if not text:
            logger.warning(f"[{session_id}] Empty text received")
            return

        selected_model = payload.get("emotion_model") or payload.get("model")
        selected_language = payload.get("language", language)
        selected_speaker = payload.get("speaker")
        started_at = time.time()

        logger.info(f"[{session_id}] Starting emotion analysis: text_length={len(text)}, model={selected_model}, language={selected_language}, speaker={selected_speaker}")
        
        try:
            analysis = await asyncio.to_thread(emotion_engine.analyze, text, selected_model)
            logger.info(f"[{session_id}] Emotion analysis completed in {time.time() - started_at:.2f}s: primary_emotion={analysis.primary_emotion}")
        except Exception as e:
            logger.error(f"[{session_id}] Emotion analysis failed: {e}", exc_info=True)
            await websocket.send_text(json.dumps({"type": "error", "message": f"Emotion analysis failed: {e}"}))
            return

        logger.debug(f"[{session_id}] Mapping voice profile...")
        voice_profile = voice_mapper.map_to_voice(analysis, selected_language, selected_speaker)
        logger.debug(f"[{session_id}] Voice profile mapped: speaker={voice_profile.speaker}")

        file_stem = "_".join(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                _sanitize_filename_part(analysis.primary_emotion),
                _sanitize_filename_part(voice_profile.speaker),
            ]
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "analysis",
                    "text": text,
                    "analysis": analysis.to_public_dict(),
                    "voice_profile": voice_profile.to_public_dict(),
                }
            )
        )

        synthesis = await tts_engine.synthesize_text(
            text=text,
            language=selected_language,
            voice_profile=voice_profile,
            output_dir=OUTPUT_DIR,
            file_stem=file_stem,
        )

        await websocket.send_bytes(synthesis.audio_bytes)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "transcript",
                    "role": "assistant",
                    "text": _build_assistant_summary(analysis, voice_profile, synthesis.file_path.name),
                    "latency": round(time.time() - started_at, 2),
                    "file_url": f"/outputs/{synthesis.file_path.name}",
                }
            )
        )

    async def run_processing_task(payload: dict):
        nonlocal current_processing_task
        try:
            await process_text_input(payload)
        except asyncio.CancelledError:
            pass
        except (EmotionAnalysisError, TTSGenerationError, ValueError) as exc:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "transcript",
                        "role": "assistant",
                        "text": f"Processing failed: {exc}",
                    }
                )
            )
        finally:
            if current_processing_task is asyncio.current_task():
                current_processing_task = None

    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            payload_type = payload.get("type")
            logger.debug(f"[{session_id}] Received WebSocket message: type={payload_type}")

            if payload_type == "interrupt":
                logger.info(f"[{session_id}] Interrupt received")
                await cancel_current_task(send_interrupt=True)
                continue

            if payload_type in {"audio_start", "audio_chunk", "audio_stop"}:
                if payload_type == "audio_start" and not audio_disabled_notified:
                    audio_disabled_notified = True
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "transcript",
                                "role": "assistant",
                                "text": "Voice input is disabled on the backend now. Send text input instead.",
                            }
                        )
                    )
                continue

            if payload_type == "text":
                logger.info(f"[{session_id}] Text payload received, starting processing task")
                await cancel_current_task(send_interrupt=False)
                current_processing_task = asyncio.create_task(run_processing_task(payload))

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
    except Exception as exc:
        logger.error(f"[{session_id}] WebSocket Error: {exc}", exc_info=True)
    finally:
        await cancel_current_task(send_interrupt=False)
