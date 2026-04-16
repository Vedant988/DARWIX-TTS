import base64
import os
import re
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
    GLOBAL_PACE_MULTIPLIER = 0.95
    LANGUAGE_CODE_MAP = {
        "en": "en-IN",
        "en-in": "en-IN",
        "hi": "hi-IN",
        "hi-in": "hi-IN",
        "bn": "bn-IN",
        "bn-in": "bn-IN",
        "gu": "gu-IN",
        "gu-in": "gu-IN",
        "kn": "kn-IN",
        "kn-in": "kn-IN",
        "ml": "ml-IN",
        "ml-in": "ml-IN",
        "mr": "mr-IN",
        "mr-in": "mr-IN",
        "od": "od-IN",
        "od-in": "od-IN",
        "pa": "pa-IN",
        "pa-in": "pa-IN",
        "ta": "ta-IN",
        "ta-in": "ta-IN",
        "te": "te-IN",
        "te-in": "te-IN",
    }

    def __init__(self):
        self.sarvam_api_key = os.environ.get("SARVAM_API_KEY")
        if not self.sarvam_api_key:
            print("WARNING: SARVAM_API_KEY is missing in .env")

    def _normalize_target_language_code(self, language: str) -> str:
        """Normalize language inputs to the BCP-47 tag Sarvam expects."""
        normalized = (language or "en").strip()
        lowered = normalized.lower()

        if lowered in self.LANGUAGE_CODE_MAP:
            return self.LANGUAGE_CODE_MAP[lowered]

        if re.fullmatch(r"[a-z]{2,3}-[A-Z]{2}", normalized):
            return normalized

        if re.fullmatch(r"[a-z]{2,3}-[a-z]{2}", lowered):
            language_part, region_part = lowered.split("-", 1)
            return f"{language_part}-{region_part.upper()}"

        return "hi-IN" if lowered.startswith("hi") else "en-IN"

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

        target_code = self._normalize_target_language_code(language)

        # Sarvam expects pitch, pace, and loudness in constrained ranges.
        def clamp(value: float, minimum: float, maximum: float) -> float:
            return max(minimum, min(maximum, value))

        # Bulbul v2 sounds most natural with small movements around neutral.
        pitch = clamp(voice_profile.pitch, -0.22, 0.22)
        pace = clamp(voice_profile.pace * self.GLOBAL_PACE_MULTIPLIER, 0.82, 1.08)
        loudness = clamp(voice_profile.loudness, 0.82, 1.22)

        payload = {
            "text": text,
            "target_language_code": target_code,
            "speaker": voice_profile.speaker,
            "pitch": pitch,
            "pace": pace,
            "loudness": loudness,
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
                response_text = response.text
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = response_text
            except NameError:
                body = exc.response.text if exc.response is not None else ""
            raise TTSGenerationError(
                f"Sarvam TTS request failed: {exc} | response: {body}"
            ) from exc
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
