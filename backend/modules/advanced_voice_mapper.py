"""
Advanced Voice Mapper: Implements human-like micro-prosody, natural variations,
and emotion-specific voice quality for more natural TTS delivery.
"""
import logging
import math
from dataclasses import dataclass, replace
from typing import NamedTuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancedVoiceParameters:
    """Extended voice parameters with micro-prosody and natural variations."""
    # Base parameters
    speaker: str
    pitch: float
    pace: float
    loudness: float
    
    # Micro-prosody parameters
    pitch_variance: float  # Natural pitch vibrato (0.0-1.0)
    pace_variance: float   # Natural speed variation (0.0-1.0)
    breathiness: float     # Voice breathiness (0.0-1.0, higher = more breath)
    
    # Emotion-specific voice quality
    tension: float         # Vocal tension (0.0=relaxed, 1.0=tense)
    brightness: float      # Spectral brightness (0.0=dark, 1.0=bright)
    
    # Energy curve
    energy_rise: float     # How much energy rises toward end of phrase (0.0-1.0)
    
    def to_sarvam_base(self) -> dict:
        """Convert to base Sarvam parameters."""
        return {
            "speaker": self.speaker,
            "pitch": self.pitch,
            "pace": self.pace,
            "loudness": self.loudness,
        }


class AdvancedVoiceMapper:
    """Maps emotions to nuanced, human-like voice characteristics."""
    
    # Emotion-specific voice quality profiles
    EMOTION_VOICE_PROFILES = {
        # Positive emotions: bright, energetic, natural breathiness
        "joy": {
            "pitch_variance": 0.65,      # More pitch variation
            "pace_variance": 0.55,       # Some natural speed variation
            "breathiness": 0.40,         # Warm, open
            "tension": 0.25,             # Relaxed
            "brightness": 0.75,          # Bright
            "energy_rise": 0.50,         # Energy builds
        },
        "excitement": {
            "pitch_variance": 0.75,
            "pace_variance": 0.70,
            "breathiness": 0.45,
            "tension": 0.40,
            "brightness": 0.85,
            "energy_rise": 0.65,
        },
        "amusement": {
            "pitch_variance": 0.68,
            "pace_variance": 0.60,
            "breathiness": 0.50,
            "tension": 0.20,
            "brightness": 0.80,
            "energy_rise": 0.55,
        },
        
        # Calm/neutral emotions: stable, natural
        "neutral": {
            "pitch_variance": 0.35,
            "pace_variance": 0.30,
            "breathiness": 0.25,
            "tension": 0.30,
            "brightness": 0.50,
            "energy_rise": 0.25,
        },
        "calm": {
            "pitch_variance": 0.30,
            "pace_variance": 0.25,
            "breathiness": 0.30,
            "tension": 0.15,
            "brightness": 0.45,
            "energy_rise": 0.15,
        },
        
        # Negative emotions: tense, darker, controlled
        "sadness": {
            "pitch_variance": 0.25,
            "pace_variance": 0.35,
            "breathiness": 0.15,
            "tension": 0.60,
            "brightness": 0.25,
            "energy_rise": -0.30,        # Energy falls
        },
        "grief": {
            "pitch_variance": 0.20,
            "pace_variance": 0.20,
            "breathiness": 0.10,
            "tension": 0.75,
            "brightness": 0.15,
            "energy_rise": -0.40,
        },
        "anger": {
            "pitch_variance": 0.80,      # Highly variable pitch for aggression
            "pace_variance": 0.85,       # Staccato, unpredictable pacing
            "breathiness": 0.15,         # Tight, harsh tone
            "tension": 0.95,             # Maximum vocal tension
            "brightness": 0.75,          # Bright, aggressive edge
            "energy_rise": 0.75,         # High energy throughout
        },
        "fear": {
            "pitch_variance": 0.70,
            "pace_variance": 0.75,
            "breathiness": 0.60,
            "tension": 0.85,
            "brightness": 0.55,
            "energy_rise": 0.45,
        },
        "anxiety": {
            "pitch_variance": 0.75,      # High pitch variation for nervousness
            "pace_variance": 0.80,       # Very variable pace for hesitation
            "breathiness": 0.65,         # Very breathy for anxiety
            "tension": 0.88,             # Very tense vocal cords
            "brightness": 0.35,          # Darker, constricted tone
            "energy_rise": 0.10,         # Energy controlled/suppressed
        },
        
        # Contemplative: thoughtful, measured
        "confusion": {
            "pitch_variance": 0.55,
            "pace_variance": 0.40,
            "breathiness": 0.30,
            "tension": 0.45,
            "brightness": 0.40,
            "energy_rise": 0.20,
        },
        "curiosity": {
            "pitch_variance": 0.60,
            "pace_variance": 0.50,
            "breathiness": 0.35,
            "tension": 0.35,
            "brightness": 0.60,
            "energy_rise": 0.40,
        },
    }

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _get_emotion_voice_profile(self, primary_emotion: str) -> dict:
        """Get voice quality profile for emotion, with safe fallback."""
        profile = self.EMOTION_VOICE_PROFILES.get(
            primary_emotion.lower(),
            self.EMOTION_VOICE_PROFILES["neutral"]
        )
        return profile.copy()

    def enhance_voice_for_emotion(
        self,
        base_voice_profile,
        primary_emotion: str,
        confidence: float,
        clarity: float,
    ) -> AdvancedVoiceParameters:
        """
        Enhance base voice profile with advanced emotion-aware parameters.
        
        Args:
            base_voice_profile: VoiceProfile from standard mapper
            primary_emotion: Primary emotion label
            confidence: Confidence in emotion detection (0.0-1.0)
            clarity: Clarity of speech from analysis (0.0-1.0)
        
        Returns:
            AdvancedVoiceParameters with all micro-prosody settings
        """
        emotion_profile = self._get_emotion_voice_profile(primary_emotion)
        
        # Scale all emotion parameters by confidence
        # If confidence is low, move toward neutral
        neutral_profile = self.EMOTION_VOICE_PROFILES["neutral"]
        for key in emotion_profile:
            emotion_profile[key] = (
                emotion_profile[key] * confidence +
                neutral_profile[key] * (1.0 - confidence)
            )

        # Brightness affected by clarity: clear speech = brighter
        brightness = emotion_profile["brightness"] + (0.15 * clarity)
        brightness = self._clamp(brightness, 0.0, 1.0)

        # Tension reduced slightly by clarity (confident/clear = less tense)
        tension = emotion_profile["tension"] * (1.0 - clarity * 0.2)
        tension = self._clamp(tension, 0.0, 1.0)

        return AdvancedVoiceParameters(
            speaker=base_voice_profile.speaker,
            pitch=base_voice_profile.pitch,
            pace=base_voice_profile.pace,
            loudness=base_voice_profile.loudness,
            pitch_variance=self._clamp(emotion_profile["pitch_variance"], 0.0, 1.0),
            pace_variance=self._clamp(emotion_profile["pace_variance"], 0.0, 1.0),
            breathiness=self._clamp(emotion_profile["breathiness"], 0.0, 1.0),
            tension=tension,
            brightness=brightness,
            energy_rise=self._clamp(emotion_profile["energy_rise"], -1.0, 1.0),
        )

    def create_micro_prosody_instructions(
        self,
        advanced_params: AdvancedVoiceParameters,
    ) -> dict:
        """
        Create text processing instructions for more human-like delivery.
        
        Returns a dict with instructions for:
        - Breath placement
        - Natural pausing patterns
        - Emphasis placement
        """
        instructions = {
            # Breathing: higher breathiness = more natural pauses for breathing
            "breathing_frequency": 0.3 + (advanced_params.breathiness * 0.4),
            
            # Pitch variation: add subtle micro-pitch movements
            "use_pitch_vibrato": advanced_params.pitch_variance > 0.4,
            "vibrato_rate_hz": 4.5 + (advanced_params.pitch_variance * 2.0),  # 4.5-6.5 Hz
            
            # Pause placement: tense/anxious = more frequent micro-pauses
            "use_micro_pauses": advanced_params.tension > 0.5,
            "micro_pause_frequency": 0.2 + (advanced_params.tension * 0.3),
            
            # Energy contour
            "energy_curve": "rising" if advanced_params.energy_rise > 0.3
                           else "falling" if advanced_params.energy_rise < -0.3
                           else "flat",
            
            # Spectral adjustments
            "frequency_range": "bright" if advanced_params.brightness > 0.6
                               else "warm" if advanced_params.brightness < 0.4
                               else "neutral",
            
            # Vocal quality
            "vocal_quality": "breathy" if advanced_params.breathiness > 0.4
                            else "tense" if advanced_params.tension > 0.6
                            else "natural",
        }
        return instructions

    def apply_to_chunks(self, chunks: list, advanced_params: AdvancedVoiceParameters) -> list:
        """Translate advanced voice parameters into subtle per-chunk engine controls."""
        if not chunks:
            return chunks

        shaped_chunks = []
        total = max(1, len(chunks) - 1)

        for index, chunk in enumerate(chunks):
            progress = index / total if len(chunks) > 1 else 0.0
            pitch_wave = math.sin(progress * math.pi)
            pace_wave = math.sin((progress + 0.15) * math.pi * 2)

            pitch = chunk.pitch
            pitch += (advanced_params.brightness - 0.5) * 0.14
            pitch += advanced_params.energy_rise * (progress - 0.25) * 0.08
            pitch += (pitch_wave - 0.5) * advanced_params.pitch_variance * 0.05

            pace = chunk.pace
            pace += (advanced_params.brightness - 0.5) * 0.05
            pace -= advanced_params.breathiness * 0.04
            pace += advanced_params.energy_rise * progress * 0.05
            pace += pace_wave * advanced_params.pace_variance * 0.03

            loudness = chunk.loudness
            loudness += advanced_params.energy_rise * progress * 0.05
            loudness -= advanced_params.breathiness * 0.03

            baseline_pause = chunk.post_chunk_pause_ms if chunk.post_chunk_pause_ms > 0 else 140
            if advanced_params.tension > 0.7:
                baseline_pause -= 35
            if advanced_params.breathiness > 0.58:
                baseline_pause += 45
            if advanced_params.energy_rise > 0.25:
                baseline_pause -= int(progress * 60)

            shaped_chunks.append(
                replace(
                    chunk,
                    pace=self._clamp(pace, 0.84, 1.12),
                    pitch=self._clamp(pitch, -0.22, 0.22),
                    loudness=self._clamp(loudness, 0.82, 1.22),
                    post_chunk_pause_ms=int(self._clamp(baseline_pause, 40, 340)),
                )
            )

        return shaped_chunks


advanced_voice_mapper = AdvancedVoiceMapper()
