from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalDimensions:
    valence: float
    arousal: float
    dominance: float


@dataclass(frozen=True)
class VoiceProfile:
    speaker: str
    pitch: float
    pace: float
    loudness: float
    intensity: float
    stability: float
    dimensions: EmotionalDimensions
    reason: str
    tts_model: str = "bulbul:v2"
    output_audio_codec: str = "wav"
    speech_sample_rate: int = 22050
    enable_preprocessing: bool = True

    def to_public_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "pitch": round(self.pitch, 4),
            "pace": round(self.pace, 4),
            "loudness": round(self.loudness, 4),
            "intensity": round(self.intensity, 4),
            "stability": round(self.stability, 4),
            "dimensions": {
                "valence": round(self.dimensions.valence, 4),
                "arousal": round(self.dimensions.arousal, 4),
                "dominance": round(self.dimensions.dominance, 4),
            },
            "tts_model": self.tts_model,
            "output_audio_codec": self.output_audio_codec,
            "speech_sample_rate": self.speech_sample_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "reason": self.reason,
        }


class EmotionToVoiceMapper:
    LABEL_DIMENSIONS = {
        "admiration": EmotionalDimensions(0.72, 0.42, 0.64),
        "amusement": EmotionalDimensions(0.76, 0.7, 0.54),
        "anger": EmotionalDimensions(-0.88, 0.92, 0.78),
        "annoyance": EmotionalDimensions(-0.56, 0.76, 0.64),
        "approval": EmotionalDimensions(0.42, 0.38, 0.58),
        "caring": EmotionalDimensions(0.67, 0.28, 0.48),
        "confusion": EmotionalDimensions(-0.1, 0.54, 0.26),
        "curiosity": EmotionalDimensions(0.24, 0.56, 0.38),
        "desire": EmotionalDimensions(0.46, 0.62, 0.56),
        "disappointment": EmotionalDimensions(-0.62, 0.42, 0.3),
        "disapproval": EmotionalDimensions(-0.48, 0.52, 0.6),
        "disgust": EmotionalDimensions(-0.74, 0.66, 0.58),
        "embarrassment": EmotionalDimensions(-0.42, 0.72, 0.22),
        "excitement": EmotionalDimensions(0.84, 0.95, 0.62),
        "fear": EmotionalDimensions(-0.92, 0.95, 0.12),
        "gratitude": EmotionalDimensions(0.78, 0.44, 0.5),
        "grief": EmotionalDimensions(-0.96, 0.2, 0.08),
        "joy": EmotionalDimensions(0.92, 0.74, 0.7),
        "love": EmotionalDimensions(0.88, 0.48, 0.62),
        "neutral": EmotionalDimensions(0.0, 0.34, 0.5),
        "nervousness": EmotionalDimensions(-0.58, 0.84, 0.16),
        "optimism": EmotionalDimensions(0.74, 0.5, 0.62),
        "pride": EmotionalDimensions(0.66, 0.62, 0.84),
        "realization": EmotionalDimensions(0.08, 0.48, 0.44),
        "relief": EmotionalDimensions(0.54, 0.26, 0.38),
        "remorse": EmotionalDimensions(-0.72, 0.4, 0.16),
        "sadness": EmotionalDimensions(-0.82, 0.22, 0.18),
        "surprise": EmotionalDimensions(0.18, 0.88, 0.46),
    }

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _lerp(self, value: float, neutral: float, pull: float) -> float:
        return value * (1.0 - pull) + neutral * pull

    def _score_lookup(self, analysis) -> dict[str, float]:
        return {item.label: item.score for item in analysis.normalized_emotions}

    def _weighted_dimensions(self, analysis) -> EmotionalDimensions:
        dimensions = EmotionalDimensions(0.0, 0.34, 0.5)
        weighted = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
        }
        total_weight = 0.0

        for item in analysis.normalized_emotions:
            dims = self.LABEL_DIMENSIONS.get(item.label, dimensions)
            weighted["valence"] += dims.valence * item.score
            weighted["arousal"] += dims.arousal * item.score
            weighted["dominance"] += dims.dominance * item.score
            total_weight += item.score

        if total_weight == 0:
            return dimensions

        return EmotionalDimensions(
            valence=weighted["valence"] / total_weight,
            arousal=weighted["arousal"] / total_weight,
            dominance=weighted["dominance"] / total_weight,
        )

    def _pick_speaker(self, dimensions: EmotionalDimensions, scores: dict[str, float]) -> str:
        negative_pressure = max(0.0, -dimensions.valence)
        sadness_cluster = scores.get("sadness", 0.0) + scores.get("grief", 0.0) + scores.get("remorse", 0.0)

        if dimensions.valence > 0.45 and dimensions.arousal > 0.62:
            return "arya"
        if negative_pressure > 0.45 and dimensions.dominance > 0.55:
            return "anushka"
        if sadness_cluster > 0.28 or (negative_pressure > 0.35 and dimensions.arousal < 0.45):
            return "vidya"
        if dimensions.valence > 0.15 and dimensions.arousal < 0.58:
            return "manisha"
        return "anushka"

    def map_to_voice(self, analysis, language: str = "en", override_speaker: str | None = None) -> VoiceProfile:
        dimensions = self._weighted_dimensions(analysis)
        scores = self._score_lookup(analysis)

        anger_like = scores.get("anger", 0.0) + scores.get("annoyance", 0.0) + scores.get("disapproval", 0.0)
        fear_like = scores.get("fear", 0.0) + scores.get("nervousness", 0.0)
        sadness_like = scores.get("sadness", 0.0) + scores.get("grief", 0.0) + scores.get("remorse", 0.0)
        surprise_like = scores.get("surprise", 0.0) + scores.get("realization", 0.0)
        neutral_like = scores.get("neutral", 0.0) + (scores.get("confusion", 0.0) * 0.45)

        clarity = analysis.clarity
        intensity = self._clamp(
            0.32 + (0.34 * abs(dimensions.valence)) + (0.28 * dimensions.arousal) + (0.24 * clarity),
            0.2,
            1.0,
        )
        stability = self._clamp(neutral_like + (0.25 * (1.0 - clarity)), 0.0, 0.75)

        # --- SARVAM-SPECIFIC PACE FIX ---
        # Drop the baseline anchor from 1.0 to 0.88 to compensate for Sarvam's high WPM
        baseline_pace = 0.90
        
        # Pivot around the new baseline. Neutral arousal is ~0.34, Neutral dominance is 0.5
        pace = baseline_pace + (0.15 * (dimensions.arousal - 0.34))
        pace += 0.05 * (dimensions.dominance - 0.5)
        
        # Emotion specific overrides (dampened slightly)
        pace -= 0.12 * sadness_like
        pace += 0.06 * surprise_like
        pace -= 0.05 * max(0.0, -dimensions.valence) * (1.0 - dimensions.dominance)

        pitch = (0.34 * dimensions.valence) + (0.28 * (dimensions.arousal - 0.5))
        pitch -= 0.42 * anger_like * max(0.0, dimensions.dominance)
        pitch += 0.38 * fear_like * (1.0 - dimensions.dominance)
        pitch -= 0.12 * sadness_like

        loudness = 0.94 + (1.0 * dimensions.arousal) + (0.26 * dimensions.dominance)
        loudness -= 0.56 * sadness_like
        loudness -= 0.14 * neutral_like

        # Apply stability interpolation using the new baseline
        pace = self._lerp(pace, baseline_pace, stability)
        pitch = self._lerp(pitch, 0.0, stability * 0.8)
        loudness = self._lerp(loudness, 1.0, stability * 0.7)

        # Shift the clamp down to match the new Sarvam baseline
        pace = self._clamp(pace, 0.75, 1.05)
        pitch = self._clamp(pitch, -0.75, 0.75)
        loudness = self._clamp(loudness, 0.3, 3.0)

        speaker = override_speaker or self._pick_speaker(dimensions, scores)
        top_labels = ", ".join(item.label for item in analysis.top_emotions[:3])
        reason = (
            f"Blended top emotions: {top_labels}. "
            f"Arousal={dimensions.arousal:.2f} drives pace, "
            f"valence={dimensions.valence:.2f} shapes brightness, "
            f"and dominance={dimensions.dominance:.2f} separates firm emotions like anger "
            f"from tense emotions like fear when setting pitch and loudness."
        )

        return VoiceProfile(
            speaker=speaker,
            pitch=pitch,
            pace=pace,
            loudness=loudness,
            intensity=intensity,
            stability=stability,
            dimensions=dimensions,
            reason=reason,
        )


voice_mapper = EmotionToVoiceMapper()
