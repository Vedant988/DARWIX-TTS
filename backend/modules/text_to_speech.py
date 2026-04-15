import base64
import os
from dataclasses import dataclass
from pathlib import Path


class TTSGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SynthesisResult:
    audio_bytes: bytes
    file_path: Path
    request_id: str | None
    codec: str


class TTSEngine:
    API_URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self):
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY")
        if not self.sarvam_api_key:
            print("WARNING: SARVAM_API_KEY is missing in .env")

    async def synthesize_text(self, text: str, language: str, voice_profile, output_dir: Path, file_stem: str) -> SynthesisResult:
        if not self.sarvam_api_key:
            raise TTSGenerationError("SARVAM_API_KEY is missing")
        if not text.strip():
            raise TTSGenerationError("Text input is empty")
        if len(text) > 1500:
            raise TTSGenerationError("Sarvam bulbul:v2 supports up to 1500 characters per request")

        try:
            import httpx
        except ImportError as exc:
            raise TTSGenerationError("httpx is required for TTS requests") from exc

        target_code = "hi-IN" if language == "hi" else "en-IN"
        payload = {
            "text": text,
            "target_language_code": target_code,
            "speaker": voice_profile.speaker,
            "pitch": voice_profile.pitch,
            "pace": voice_profile.pace,
            "loudness": voice_profile.loudness,
            "speech_sample_rate": voice_profile.speech_sample_rate,
            "enable_preprocessing": voice_profile.enable_preprocessing,
            "model": voice_profile.tts_model,
            "output_audio_codec": voice_profile.output_audio_codec,
            "enable_cached_responses": True,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "api-subscription-key": self.sarvam_api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise TTSGenerationError(f"Sarvam TTS request failed: {exc}") from exc

        audios = data.get("audios") or []
        if not audios:
            raise TTSGenerationError("Sarvam TTS returned no audio data")

        try:
            audio_bytes = base64.b64decode(audios[0])
        except Exception as exc:
            raise TTSGenerationError("Unable to decode Sarvam audio payload") from exc

        extension = voice_profile.output_audio_codec.lower()
        file_path = output_dir / f"{file_stem}.{extension}"
        file_path.write_bytes(audio_bytes)

        return SynthesisResult(
            audio_bytes=audio_bytes,
            file_path=file_path,
            request_id=data.get("request_id"),
            codec=voice_profile.output_audio_codec,
        )


tts_engine = TTSEngine()
