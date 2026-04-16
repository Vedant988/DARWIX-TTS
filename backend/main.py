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
from modules.prosody_director import prosody_director
from modules.audio_stitcher import audio_stitcher
from modules.advanced_voice_mapper import advanced_voice_mapper
from modules.intelligent_text_formatter import (
    intelligent_text_formatter,
    sentiment_curve_generator,
)
from modules.word_prosody_engine import word_prosody_engine

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

        # === ADVANCED VOICE ENHANCEMENT ===
        logger.debug(f"[{session_id}] Enhancing voice with advanced micro-prosody...")
        advanced_voice = advanced_voice_mapper.enhance_voice_for_emotion(
            voice_profile,
            primary_emotion=analysis.primary_emotion,
            confidence=analysis.confidence,
            clarity=analysis.clarity,
        )
        micro_prosody_instructions = advanced_voice_mapper.create_micro_prosody_instructions(advanced_voice)
        logger.debug(f"[{session_id}] Advanced voice parameters: breathing_freq={micro_prosody_instructions.get('breathing_frequency'):.2f}, vibrato={micro_prosody_instructions.get('vibrato_rate_hz'):.1f}Hz")

        # === INTELLIGENT TEXT FORMATTING ===
        logger.debug(f"[{session_id}] Formatting text for emotion '{analysis.primary_emotion}'...")
        formatted_result = await asyncio.to_thread(
            intelligent_text_formatter.format_for_emotion,
            text,
            analysis.primary_emotion,
            analysis.confidence,
            advanced_voice,
        )
        formatted_text = formatted_result.formatted_text
        sentiment_curve_type = formatted_result.sentiment_curve.get("type", "flat")
        dynamic_loudness_mult = formatted_result.dynamic_loudness_multiplier
        logger.info(
            f"[{session_id}] Text formatted: curve={sentiment_curve_type}, "
            f"loudness_mult={dynamic_loudness_mult:.2f}, "
            f"lexical_hints={formatted_result.lexical_prosody_hints}"
        )

        await websocket.send_text(json.dumps({
            "type": "pipeline_status",
            "stage": "Optimizing Text for Emotion",
            "progress": 0,
        }))

        # === NEW PROSODY DIRECTOR PIPELINE ===
        await websocket.send_text(json.dumps({
            "type": "pipeline_status",
            "stage": "Generating Semantic Micro-Chunks",
            "progress": 0,
        }))

        voice_quality_context = {}
        try:
            # Build voice quality context for prosody director
            voice_quality_context = {
                "tensionLevel": "tense" if advanced_voice.tension > 0.6
                                else "relaxed" if advanced_voice.tension < 0.3
                                else "normal",
                "breathiness": "high" if advanced_voice.breathiness > 0.4
                               else "low" if advanced_voice.breathiness < 0.25
                               else "medium",
                "energyPattern": "rising" if advanced_voice.energy_rise > 0.25
                                 else "falling" if advanced_voice.energy_rise < -0.25
                                 else "flat",
                "tension": round(advanced_voice.tension, 4),
                "breathinessValue": round(advanced_voice.breathiness, 4),
                "brightness": round(advanced_voice.brightness, 4),
                "pitchVariance": round(advanced_voice.pitch_variance, 4),
                "paceVariance": round(advanced_voice.pace_variance, 4),
                "energyRise": round(advanced_voice.energy_rise, 4),
                "basePace": round(voice_profile.pace, 4),
                "basePitch": round(voice_profile.pitch, 4),
                "baseLoudness": round(voice_profile.loudness, 4),
            }
            
            # Use FORMATTED text for prosody direction, not original
            chunks = await prosody_director.direct_prosody(formatted_text, voice_quality_context)
            logger.info(f"[{session_id}] Prosody director produced {len(chunks)} chunks")

            # === APPLY SENTIMENT CURVE ===
            chunks = sentiment_curve_generator.apply_sentiment_curve(
                chunks,
                sentiment_curve_type,
                voice_profile.loudness,
                dynamic_loudness_mult,
            )
            logger.info(f"[{session_id}] Applied {sentiment_curve_type} sentiment curve to chunks")

            # Use advanced voice parameters to shape the actual engine knobs per clause.
            chunks = advanced_voice_mapper.apply_to_chunks(chunks, advanced_voice)
            logger.info(f"[{session_id}] Applied advanced chunk shaping to {len(chunks)} chunks")
            
            # === ANALYZE WORD-LEVEL PROSODY ===
            word_prosody_info = []
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.words:
                    metadata = word_prosody_engine.extract_word_prosody_metadata(chunk)
                    word_prosody_info.append({
                        "chunk_idx": chunk_idx,
                        "word_count": len(chunk.words),
                        "significant_pauses": metadata.significant_pauses,
                        "emotion_context": metadata.emotion_context,
                    })
                    logger.debug(
                        f"[{session_id}] Chunk {chunk_idx}: "
                        f"{len(chunk.words)} words with {metadata.significant_pauses} significant pauses"
                    )
            
            if word_prosody_info:
                logger.info(f"[{session_id}] Word-level prosody extracted from {len(word_prosody_info)} chunks")
                await websocket.send_text(json.dumps({
                    "type": "debug_info",
                    "message": f"Word-level prosody: {len(word_prosody_info)} chunks with word control",
                }))

        except Exception as e:
            logger.error(f"[{session_id}] Prosody direction failed, falling back to single chunk: {e}")
            chunks = prosody_director.fallback_clause_chunks(formatted_text, voice_quality_context)
            chunks = sentiment_curve_generator.apply_sentiment_curve(
                chunks,
                sentiment_curve_type,
                voice_profile.loudness,
                dynamic_loudness_mult,
            )
            chunks = advanced_voice_mapper.apply_to_chunks(chunks, advanced_voice)

        await websocket.send_text(json.dumps({
            "type": "pipeline_status",
            "stage": "Mapping Chunk-Level Prosody Parameters",
            "progress": 1,
            "chunk_count": len(chunks),
        }))

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

        # === CONCURRENT CHUNK SYNTHESIS ===
        await websocket.send_text(json.dumps({
            "type": "pipeline_status",
            "stage": "Synthesizing Concurrent Audio Streams",
            "progress": 2,
        }))

        async def synthesize_chunk(chunk, chunk_idx):
            """Synthesize a single chunk with its custom prosody parameters."""
            try:
                # Use the chunk-level prosody values directly, with the selected speaker preserved.
                chunk_voice_profile = type(voice_profile)(
                    speaker=voice_profile.speaker,
                    pitch=chunk.pitch,
                    pace=chunk.pace,
                    loudness=chunk.loudness,
                    intensity=voice_profile.intensity,
                    stability=voice_profile.stability,
                    dimensions=voice_profile.dimensions,
                    reason=f"{voice_profile.reason} [chunk {chunk_idx+1}: {chunk.emotion_context}]",
                    tts_model=voice_profile.tts_model,
                    output_audio_codec=voice_profile.output_audio_codec,
                    speech_sample_rate=voice_profile.speech_sample_rate,
                    enable_preprocessing=voice_profile.enable_preprocessing,
                )
                
                chunk_stem = f"{file_stem}_chunk_{chunk_idx:02d}"
                synthesis = await tts_engine.synthesize_text(
                    text=chunk.text,
                    language=selected_language,
                    voice_profile=chunk_voice_profile,
                    output_dir=OUTPUT_DIR,
                    file_stem=chunk_stem,
                )
                
                logger.info(f"[{session_id}] Synthesized chunk {chunk_idx+1}/{len(chunks)}")
                return synthesis
            except Exception as e:
                logger.error(f"[{session_id}] Chunk synthesis failed: {e}")
                raise

        try:
            # Synthesize all chunks concurrently
            syntheses = await asyncio.gather(
                *[synthesize_chunk(chunk, i) for i, chunk in enumerate(chunks)],
                return_exceptions=False
            )

            await websocket.send_text(json.dumps({
                "type": "pipeline_status",
                "stage": "Assembling Final Audio Vector",
                "progress": 3,
            }))

            # === AUDIO STITCHING ===
            chunk_files = [s.file_path for s in syntheses]
            pause_durations = [chunks[i].post_chunk_pause_ms for i in range(len(chunks) - 1)] + [0]

            final_path = OUTPUT_DIR / f"{file_stem}_final.wav"
            await asyncio.to_thread(
                audio_stitcher.stitch_chunks,
                chunk_files,
                pause_durations,
                final_path,
            )

            # Read final stitched audio
            final_audio_bytes = final_path.read_bytes()

            await websocket.send_bytes(final_audio_bytes)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "transcript",
                        "role": "assistant",
                        "text": _build_assistant_summary(analysis, voice_profile, final_path.name),
                        "latency": round(time.time() - started_at, 2),
                        "file_url": f"/outputs/{final_path.name}",
                        "chunks_processed": len(chunks),
                    }
                )
            )

        except Exception as e:
            logger.error(f"[{session_id}] Prosody pipeline failed: {e}", exc_info=True)
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Audio processing failed: {e}"
            }))

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
