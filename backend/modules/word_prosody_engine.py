"""
Word Prosody Engine: Applies word-level prosody from Groq to TTS synthesis.
Groups words intelligently for efficient synthesis while respecting prosody boundaries.
"""
import logging
import re
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger(__name__)


class SynthesisPhrase(NamedTuple):
    """A group of words synthesized together with average prosody."""
    text: str  # Joined words
    pace: float  # Average pace for phrase
    pitch: float  # Average pitch for phrase
    loudness: float  # Average loudness for phrase
    inter_word_pauses_ms: list[int]  # Pause after each word in phrase


@dataclass
class WordProsodyMetadata:
    """Metadata about word-level prosody for a chunk."""
    chunk_text: str
    words: list  # List of ProsodyWord objects
    emotion_context: str
    significant_pauses: int  # Count of pauses >= 100ms (indicates emotional moments)


class WordProsodyEngine:
    """Intelligently applies word-level prosody to synthesis."""

    CLAUSE_CONNECTORS = {
        "and", "but", "or", "so", "because", "although", "though",
        "however", "yet", "while", "then", "still", "instead",
    }

    @staticmethod
    def group_words_for_synthesis(chunk, max_words_per_phrase: int = 10) -> list[SynthesisPhrase]:
        """
        Group words from a chunk into synthesis phrases.
        
        Respects:
        - Pause boundaries: Create phrase breaks after words with significant pauses
        - Word count: Keep phrases at clause length instead of defaulting to ultra-short bursts
        - Emotion peaks: Keep emotionally intense words in smaller phrases
        
        Args:
            chunk: ProsodyChunk with words list
            max_words_per_phrase: Maximum words to synthesize together
            
        Returns:
            List of SynthesisPhrase objects
        """
        if not chunk.words:
            # No word-level data, use fallback
            return [SynthesisPhrase(
                text=chunk.text,
                pace=chunk.pace,
                pitch=chunk.pitch,
                loudness=chunk.loudness,
                inter_word_pauses_ms=[],
            )]

        phrases = []
        current_phrase_words = []
        current_pauses = []
        current_pace_sum = 0.0
        current_pitch_sum = 0.0
        current_loudness_sum = 0.0
        hard_max_words = max(max_words_per_phrase + 4, 12)

        for i, word in enumerate(chunk.words):
            current_phrase_words.append(word.word)
            current_pauses.append(word.pause_after_ms)
            current_pace_sum += word.pace
            current_pitch_sum += word.pitch
            current_loudness_sum += word.loudness
            clean_word = re.sub(r"^[^\w]+|[^\w]+$", "", word.word).lower()
            has_clause_punctuation = bool(re.search(r"[,;:!?]$", word.word))
            is_sentence_end = bool(re.search(r"[.!?]$", word.word))

            # Determine if we should break here
            should_break = False
            
            # Break at sentence/clause punctuation first.
            if is_sentence_end or has_clause_punctuation:
                should_break = True
            
            # Break after significant pauses - emotional or syntactic moments.
            elif word.pause_after_ms >= 120:
                should_break = True

            # Prefer to break on clause connectors once the phrase is long enough.
            elif len(current_phrase_words) >= max_words_per_phrase and clean_word in WordProsodyEngine.CLAUSE_CONNECTORS:
                should_break = True

            # If the phrase keeps growing, force a break.
            elif len(current_phrase_words) >= hard_max_words:
                should_break = True
            
            # Break at end of chunk
            elif i == len(chunk.words) - 1:
                should_break = True

            if should_break:
                # Create phrase with average prosody
                n = len(current_phrase_words)
                phrase_text = " ".join(current_phrase_words)
                
                phrases.append(SynthesisPhrase(
                    text=phrase_text,
                    pace=current_pace_sum / n,
                    pitch=current_pitch_sum / n,
                    loudness=current_loudness_sum / n,
                    inter_word_pauses_ms=current_pauses[:-1] if current_pauses else [],
                ))

                current_phrase_words = []
                current_pauses = []
                current_pace_sum = 0.0
                current_pitch_sum = 0.0
                current_loudness_sum = 0.0

        logger.debug(f"Grouped chunk into {len(phrases)} synthesis phrases")
        return phrases

    @staticmethod
    def extract_word_prosody_metadata(chunk) -> WordProsodyMetadata:
        """Extract metadata about word-level prosody for analysis."""
        significant_pauses = 0
        if chunk.words:
            significant_pauses = sum(1 for w in chunk.words if w.pause_after_ms >= 100)

        return WordProsodyMetadata(
            chunk_text=chunk.text,
            words=chunk.words,
            emotion_context=chunk.emotion_context,
            significant_pauses=significant_pauses,
        )

    @staticmethod
    def apply_word_prosody_emphasis(word_text: str, loudness: float, pace: float) -> str:
        """
        Apply word-level emphasis through text formatting.
        
        For highly emphasized words (loudness > 1.3):
        - Can add emphasis markers or repeat for stress
        
        For slow words (pace < 0.8):
        - Can extend with character repetition to hint at slowness
        
        Returns modified text that hints at prosody to TTS engine.
        """
        # For now, return original text
        # In future: could use markdown-style formatting if Sarvam supports it
        return word_text

    @staticmethod
    def calculate_inter_word_silence_ms(words_in_phrase: list, inter_word_pauses: list[int]) -> dict:
        """
        Calculate silence injection between words in a phrase.
        
        Returns dict mapping word index to pause_ms after that word.
        """
        pause_map = {}
        for i, pause_ms in enumerate(inter_word_pauses):
            if pause_ms > 0:
                pause_map[i] = pause_ms
        return pause_map


word_prosody_engine = WordProsodyEngine()
