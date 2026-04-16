"""
Prosody Director: Uses Groq/Llama 3 to break text into emotionally-aware chunks
with word-level prosody parameters (pace, pitch, loudness, pause per word).
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ProsodyWord(NamedTuple):
    """Represents a single word with individual prosody parameters."""
    word: str
    pace: float  # Word-level pace (0.7-1.3)
    loudness: float  # Word-level loudness (0.6-1.6)
    pitch: float  # Word-level pitch (-0.5 to 0.5)
    pause_after_ms: int  # Silence after word (0-400ms for micro-pauses)


@dataclass(frozen=True)
class ProsodyChunk:
    """Represents a chunk of text with word-level prosody granularity."""
    text: str
    emotion_context: str
    pace: float  # Average/default pace for chunk
    pitch: float  # Average/default pitch for chunk
    loudness: float  # Average/default loudness for chunk
    post_chunk_pause_ms: int
    words: list[ProsodyWord] = field(default_factory=list)  # Word-level prosody


class ProsodyDirector:
    """Directs prosody with word-level granularity using Groq."""

    CLAUSE_BREAK_RE = re.compile(r"(?<=[,;:!?])\s+|\n+")
    CONJUNCTION_BREAK_RE = re.compile(
        r"\b(?:and|but|or|so|because|although|though|however|yet|while|then|still|instead)\b",
        re.IGNORECASE,
    )
    INTENSIFIERS = {"such", "very", "absolutely", "really", "so", "too", "quite", "truly"}
    
    SYSTEM_PROMPT = """You are an expert audio director with deep knowledge of human prosody and emotional speech patterns.

Your task: Break the transcript into logical chunks. For EACH CHUNK, provide WORD-BY-WORD prosody parameters. This enables ultra-realistic emotional delivery where each word's loudness, pace, and pitch reflects the emotional intent.

CRITICAL: You MUST output word-level parameters for maximum realism. Words are the building blocks of natural speech emotion.

Intensifiers ("such", "very", "absolutely", "really", "so") are PROSODIC HOTSPOTS:
- Increase loudness (1.2-1.4) to emphasize
- Decrease pace (0.7-0.8) to stretch and emphasize
- Adjust pitch to match emotion (higher for joy, lower for sadness)

Emotional delivery rules:
- JOY/EXCITEMENT: Fast pace (1.1-1.3), high loudness (1.2-1.6), rising pitch, energetic word stresses
- ANXIETY/FEAR: Variable pace (0.8-1.1), micro-pauses after key words, slightly lower loudness (0.8-1.0), tense pitch
- ANGER: Short staccato words (0.8-1.0), high loudness (1.3-1.6), aggressive emphasis, frequent micro-pauses
- SADNESS: Slow pace (0.7-0.9), lower loudness (0.6-0.9), falling pitch, extended pauses

For each chunk, output:
- text: The exact phrase
- emotion_context: Current emotional state
- pace: Average chunk pace (0.7-1.3)
- pitch: Average chunk pitch (-0.5 to 0.5)
- loudness: Average chunk loudness (0.6-1.6)
- post_chunk_pause_ms: Pause after chunk (0-600ms)
- words: Array of word-level prosody. Each word object has:
  - word: The exact word
  - pace: Individual pace (0.7-1.3) - slow emphasizes, fast rushes
  - loudness: Individual loudness (0.6-1.6) - high emphasizes, low de-emphasizes
  - pitch: Individual pitch (-0.5 to 0.5) - high for questions/excitement, low for sadness
  - pause_after_ms: Micro-pause after word (0-200ms for natural breathing)

Word-level prosody examples:
- "such" in "such a wonderful day" → pace: 0.75 (stretch), loudness: 1.3 (emphasize), pause_after_ms: 50
- "not" in anxious context → pace: 0.85, loudness: 0.9, pause_after_ms: 100
- "absolutely" in angry context → pace: 0.8, loudness: 1.5, pause_after_ms: 150

Output ONLY valid JSON. No markdown."""

    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set - prosody director will be disabled")

    def _context_number(self, voice_quality_context: dict | None, key: str, default: float) -> float:
        """Read a float-like value from voice context."""
        if not voice_quality_context:
            return default
        try:
            return float(voice_quality_context.get(key, default))
        except (TypeError, ValueError):
            return default

    def _context_level(self, voice_quality_context: dict | None, key: str, default: str) -> str:
        """Read a string level from voice context."""
        if not voice_quality_context:
            return default
        return str(voice_quality_context.get(key, default)).lower()

    def _split_long_clause(self, clause: str, preferred_words: int = 11, hard_max_words: int = 16) -> list[str]:
        """Split a long clause at conjunctions before falling back to word-count boundaries."""
        words = clause.split()
        if len(words) <= hard_max_words:
            return [clause.strip()]

        phrases: list[str] = []
        current: list[str] = []

        for word in words:
            clean = re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()

            if current and len(current) >= preferred_words and self.CONJUNCTION_BREAK_RE.fullmatch(clean):
                phrases.append(" ".join(current).strip())
                current = [word]
                continue

            current.append(word)

            if len(current) >= hard_max_words:
                phrases.append(" ".join(current).strip())
                current = []

        if current:
            phrases.append(" ".join(current).strip())

        return [phrase for phrase in phrases if phrase]

    def _extract_clause_texts(self, text: str, voice_quality_context: dict | None = None) -> list[str]:
        """Split text into clause-sized phrases for more natural chunking."""
        tension = self._context_number(voice_quality_context, "tension", 0.3)
        breathiness = self._context_number(voice_quality_context, "breathinessValue", 0.25)

        preferred_words = 12
        if tension > 0.78:
            preferred_words = 8
        elif tension > 0.58:
            preferred_words = 10
        elif breathiness < 0.3:
            preferred_words = 13

        hard_max_words = max(preferred_words + 4, 12)

        raw_clauses = [
            part.strip()
            for part in self.CLAUSE_BREAK_RE.split(text)
            if part and part.strip()
        ]

        merged_clauses: list[str] = []
        pending_prefix = ""
        for clause in raw_clauses or [text]:
            if self.CONJUNCTION_BREAK_RE.fullmatch(clause):
                pending_prefix = f"{pending_prefix} {clause}".strip()
                continue

            if pending_prefix:
                clause = f"{pending_prefix} {clause}".strip()
                pending_prefix = ""

            merged_clauses.append(clause)

        if pending_prefix:
            if merged_clauses:
                merged_clauses[-1] = f"{merged_clauses[-1]} {pending_prefix}".strip()
            else:
                merged_clauses.append(pending_prefix)

        clauses: list[str] = []
        for clause in merged_clauses or [text]:
            clauses.extend(self._split_long_clause(clause, preferred_words, hard_max_words))

        return [clause for clause in clauses if clause]

    def _estimate_pause_ms(
        self,
        clause_text: str,
        index: int,
        total: int,
        voice_quality_context: dict | None = None,
    ) -> int:
        """Estimate a natural pause after a clause."""
        tension = self._context_number(voice_quality_context, "tension", 0.3)
        breathiness = self._context_number(voice_quality_context, "breathinessValue", 0.25)
        energy_rise = self._context_number(voice_quality_context, "energyRise", 0.0)
        progress = index / max(1, total - 1)

        pause_ms = 160
        if clause_text.endswith((",", ";", ":")):
            pause_ms += 40
        if clause_text.endswith(("?", "!", ".")):
            pause_ms += 70
        if tension > 0.7:
            pause_ms -= 45
        if breathiness > 0.58:
            pause_ms += 45
        if energy_rise > 0.25:
            pause_ms -= int(progress * 70)

        return max(40, min(340, pause_ms))

    def _build_fallback_words(
        self,
        clause_text: str,
        pace: float,
        pitch: float,
        loudness: float,
        voice_quality_context: dict | None = None,
    ) -> list[ProsodyWord]:
        """Create word-level defaults when Groq output is absent or sparse."""
        tension = self._context_number(voice_quality_context, "tension", 0.3)
        pitch_variance = self._context_number(voice_quality_context, "pitchVariance", 0.35)
        pace_variance = self._context_number(voice_quality_context, "paceVariance", 0.3)

        words: list[ProsodyWord] = []
        raw_words = clause_text.split()
        for idx, raw_word in enumerate(raw_words):
            clean = re.sub(r"^[^\w]+|[^\w]+$", "", raw_word).lower()
            pause_after_ms = 0
            if raw_word.endswith(","):
                pause_after_ms = 80
            elif raw_word.endswith((";", ":")):
                pause_after_ms = 110
            elif raw_word.endswith((".", "!", "?")):
                pause_after_ms = 140
            elif clean in {"and", "but", "or", "so", "because", "however"} and tension > 0.68:
                pause_after_ms = 60

            word_pace = pace
            word_pitch = pitch
            word_loudness = loudness

            if clean in self.INTENSIFIERS:
                word_pace = max(0.84, pace - 0.08)
                word_loudness = min(1.25, loudness + 0.12)

            if pace_variance > 0.55:
                word_pace += 0.03 if idx % 2 == 0 else -0.03
            if pitch_variance > 0.55:
                word_pitch += 0.04 if idx % 3 == 0 else -0.02

            words.append(
                ProsodyWord(
                    word=raw_word,
                    pace=max(0.84, min(1.12, word_pace)),
                    loudness=max(0.82, min(1.22, word_loudness)),
                    pitch=max(-0.22, min(0.22, word_pitch)),
                    pause_after_ms=max(0, min(160, pause_after_ms)),
                )
            )

        return words

    def _split_terminal_word(self, chunk: ProsodyChunk, voice_quality_context: dict | None = None) -> list[ProsodyChunk]:
        """Detach the last word into a softer terminal chunk so endings trail off more naturally."""
        words = list(chunk.words or [])
        if len(words) < 3:
            return [chunk]

        tokens = chunk.text.split()
        if len(tokens) < 3:
            return [chunk]

        last_word = words[-1]
        clean_last = re.sub(r"^[^\w]+|[^\w]+$", "", last_word.word)
        if len(clean_last) < 3:
            return [chunk]

        leading_words = words[:-1]
        trailing_words = [last_word]
        leading_text = " ".join(word.word for word in leading_words).strip()
        trailing_text = " ".join(word.word for word in trailing_words).strip()

        if not leading_text or not trailing_text:
            return [chunk]

        breathiness = self._context_number(voice_quality_context, "breathinessValue", 0.25)
        tension = self._context_number(voice_quality_context, "tension", 0.3)

        bridge_pause = 50 if tension > 0.65 else 70
        trailing_pace = max(0.82, min(1.02, chunk.pace - 0.08 - (breathiness * 0.03)))
        trailing_pitch = max(-0.28, min(0.18, chunk.pitch - 0.06))
        trailing_loudness = max(0.74, min(1.12, chunk.loudness - 0.14))

        lead_pace = sum(word.pace for word in leading_words) / len(leading_words)
        lead_pitch = sum(word.pitch for word in leading_words) / len(leading_words)
        lead_loudness = sum(word.loudness for word in leading_words) / len(leading_words)

        leading_chunk = ProsodyChunk(
            text=leading_text,
            emotion_context=chunk.emotion_context,
            pace=max(0.84, min(1.12, lead_pace)),
            pitch=max(-0.22, min(0.22, lead_pitch)),
            loudness=max(0.82, min(1.22, lead_loudness)),
            post_chunk_pause_ms=bridge_pause,
            words=leading_words,
        )
        terminal_chunk = ProsodyChunk(
            text=trailing_text,
            emotion_context=f"{chunk.emotion_context}_terminal",
            pace=trailing_pace,
            pitch=trailing_pitch,
            loudness=trailing_loudness,
            post_chunk_pause_ms=chunk.post_chunk_pause_ms,
            words=[
                ProsodyWord(
                    word=last_word.word,
                    pace=trailing_pace,
                    loudness=trailing_loudness,
                    pitch=trailing_pitch,
                    pause_after_ms=last_word.pause_after_ms,
                )
            ],
        )
        return [leading_chunk, terminal_chunk]

    def apply_terminal_decay(
        self,
        chunks: list[ProsodyChunk],
        voice_quality_context: dict | None = None,
    ) -> list[ProsodyChunk]:
        """Detach final words so phrase endings sound softer and more human."""
        if not chunks:
            return chunks

        decayed: list[ProsodyChunk] = []
        for index, chunk in enumerate(chunks):
            is_last_chunk = index == len(chunks) - 1
            has_terminal_punctuation = bool(re.search(r"(\.\.\.|[.!?])\s*$", chunk.text))
            should_detach = has_terminal_punctuation or is_last_chunk

            if should_detach:
                decayed.extend(self._split_terminal_word(chunk, voice_quality_context))
            else:
                decayed.append(chunk)

        return decayed

    def fallback_clause_chunks(
        self,
        text: str,
        voice_quality_context: dict | None = None,
    ) -> list[ProsodyChunk]:
        """Create deterministic clause-based chunks when model guidance is unavailable."""
        clauses = self._extract_clause_texts(text, voice_quality_context)
        total = max(1, len(clauses))
        base_pace = self._context_number(voice_quality_context, "basePace", 1.0)
        base_pitch = self._context_number(voice_quality_context, "basePitch", 0.0)
        base_loudness = self._context_number(voice_quality_context, "baseLoudness", 1.0)
        brightness = self._context_number(voice_quality_context, "brightness", 0.5)
        breathiness = self._context_number(voice_quality_context, "breathinessValue", 0.25)
        energy_rise = self._context_number(voice_quality_context, "energyRise", 0.0)
        tension_level = self._context_level(voice_quality_context, "tensionLevel", "normal")

        chunks: list[ProsodyChunk] = []
        for index, clause_text in enumerate(clauses):
            progress = index / max(1, total - 1)
            pace = base_pace + ((brightness - 0.5) * 0.05) - (breathiness * 0.03) + (energy_rise * progress * 0.04)
            pitch = base_pitch + ((brightness - 0.5) * 0.12) + (energy_rise * (progress - 0.2) * 0.05)
            loudness = base_loudness + (energy_rise * progress * 0.05) - (breathiness * 0.03)
            emotion_context = "tense" if tension_level == "tense" else "breathy" if breathiness > 0.58 else "steady"

            words = self._build_fallback_words(
                clause_text,
                pace,
                pitch,
                loudness,
                voice_quality_context,
            )
            chunks.append(
                ProsodyChunk(
                    text=clause_text,
                    emotion_context=emotion_context,
                    pace=max(0.84, min(1.12, pace)),
                    pitch=max(-0.22, min(0.22, pitch)),
                    loudness=max(0.82, min(1.22, loudness)),
                    post_chunk_pause_ms=self._estimate_pause_ms(
                        clause_text,
                        index,
                        total,
                        voice_quality_context,
                    ),
                    words=words,
                )
            )

        chunks = self.apply_terminal_decay(chunks, voice_quality_context)
        logger.info(f"Fallback clause chunker produced {len(chunks)} chunks")
        return chunks

    def rebalance_chunks(
        self,
        chunks: list[ProsodyChunk],
        voice_quality_context: dict | None = None,
    ) -> list[ProsodyChunk]:
        """Split overlong chunks at clause boundaries and backfill missing word metadata."""
        rebalanced: list[ProsodyChunk] = []
        for chunk in chunks:
            clause_texts = self._extract_clause_texts(chunk.text, voice_quality_context)
            should_split = len(clause_texts) > 1 and len(chunk.text.split()) > 14

            if should_split:
                per_clause_context = dict(voice_quality_context or {})
                per_clause_context.update(
                    {
                        "basePace": chunk.pace,
                        "basePitch": chunk.pitch,
                        "baseLoudness": chunk.loudness,
                    }
                )
                rebalanced.extend(self.fallback_clause_chunks(chunk.text, per_clause_context))
                continue

            words = chunk.words or self._build_fallback_words(
                chunk.text,
                chunk.pace,
                chunk.pitch,
                chunk.loudness,
                voice_quality_context,
            )
            rebalanced.append(
                ProsodyChunk(
                    text=chunk.text,
                    emotion_context=chunk.emotion_context,
                    pace=chunk.pace,
                    pitch=chunk.pitch,
                    loudness=chunk.loudness,
                    post_chunk_pause_ms=max(
                        40,
                        min(
                            340,
                            chunk.post_chunk_pause_ms
                            or self._estimate_pause_ms(chunk.text, 0, 1, voice_quality_context),
                        ),
                    ),
                    words=words,
                )
            )

        return rebalanced

    def _validate_word(self, word_dict: dict) -> ProsodyWord | None:
        """Validate and convert a raw word dict to ProsodyWord."""
        try:
            return ProsodyWord(
                word=word_dict.get("word", "").strip(),
                pace=max(0.7, min(1.3, float(word_dict.get("pace", 1.0)))),
                loudness=max(0.6, min(1.6, float(word_dict.get("loudness", 1.0)))),
                pitch=max(-0.5, min(0.5, float(word_dict.get("pitch", 0.0)))),
                pause_after_ms=max(0, min(200, int(word_dict.get("pause_after_ms", 0)))),
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid word format: {e}")
            return None

    def _validate_chunk(self, chunk: dict) -> ProsodyChunk | None:
        """Validate and convert a raw chunk dict to ProsodyChunk."""
        try:
            # Parse word-level prosody if provided
            words = []
            raw_words = chunk.get("words", [])
            if raw_words:
                for raw_word in raw_words:
                    word = self._validate_word(raw_word)
                    if word and word.word:
                        words.append(word)
            
            return ProsodyChunk(
                text=chunk.get("text", "").strip(),
                emotion_context=chunk.get("emotion_context", "neutral"),
                pace=max(0.7, min(1.3, float(chunk.get("pace", 1.0)))),
                pitch=max(-0.5, min(0.5, float(chunk.get("pitch", 0.0)))),
                loudness=max(0.6, min(1.6, float(chunk.get("loudness", 1.0)))),
                post_chunk_pause_ms=max(0, min(600, int(chunk.get("post_chunk_pause_ms", 0)))),
                words=words,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid chunk format: {e}")
            return None

    async def direct_prosody(self, text: str, voice_quality_context: dict | None = None) -> list[ProsodyChunk]:
        """Break text into prosody-directed chunks using Groq.
        
        Args:
            text: The text to analyze
            voice_quality_context: Optional dict with voice quality metadata:
                - tensionLevel: "relaxed"|"normal"|"tense"
                - breathiness: "low"|"medium"|"high"
                - energyPattern: "rising"|"falling"|"flat"
        """
        if not self.groq_api_key:
            logger.warning("Groq API key not configured - returning single chunk")
            return self.fallback_clause_chunks(text, voice_quality_context)

        try:
            from groq import Groq
        except ImportError as e:
            logger.error(f"Groq library not installed: {e}")
            return self.fallback_clause_chunks(text, voice_quality_context)

        try:
            client = Groq(api_key=self.groq_api_key)
            
            # Build user message with optional voice quality context
            user_message = f"{self.SYSTEM_PROMPT}\n\nText to direct:\n{text}"
            
            if voice_quality_context:
                tension = voice_quality_context.get("tensionLevel", "normal")
                breathiness = voice_quality_context.get("breathiness", "medium")
                energy = voice_quality_context.get("energyPattern", "flat")
                brightness = voice_quality_context.get("brightness", 0.5)
                pace_variance = voice_quality_context.get("paceVariance", 0.3)
                pitch_variance = voice_quality_context.get("pitchVariance", 0.35)
                user_message += (
                    f"\n\nVoice Quality Context:\n"
                    f"- Vocal Tension: {tension}\n"
                    f"- Breathiness: {breathiness}\n"
                    f"- Energy Pattern: {energy}\n"
                    f"- Brightness: {brightness}\n"
                    f"- Pace Variance: {pace_variance}\n"
                    f"- Pitch Variance: {pitch_variance}\n\n"
                    f"Prefer clause-sized chunks over ultra-short fragments unless the emotion is highly tense."
                )
            
            response = client.messages.create(
                model="llama-3.3-70b-versatile",  # Groq's fast open model
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
                temperature=0.7,
                max_tokens=2048,
            )

            # Extract response text
            response_text = response.content[0].text if response.content else ""
            
            # Parse JSON array from response
            # Try to find JSON array in the response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON array found in Groq response")
                return self.fallback_clause_chunks(text, voice_quality_context)
            
            json_str = response_text[json_start:json_end]
            raw_chunks = json.loads(json_str)

            # Validate and convert chunks
            chunks = []
            for raw_chunk in raw_chunks:
                chunk = self._validate_chunk(raw_chunk)
                if chunk and chunk.text:
                    chunks.append(chunk)

            if not chunks:
                logger.warning("No valid chunks produced from Groq")
                return self.fallback_clause_chunks(text, voice_quality_context)

            chunks = self.rebalance_chunks(chunks, voice_quality_context)
            chunks = self.apply_terminal_decay(chunks, voice_quality_context)
            logger.info(f"Prosody director produced {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Groq prosody direction failed: {e}", exc_info=True)
            return self.fallback_clause_chunks(text, voice_quality_context)


prosody_director = ProsodyDirector()
