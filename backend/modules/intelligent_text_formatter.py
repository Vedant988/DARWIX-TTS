"""
Intelligent Text Formatter: Reshapes input text with emotion-aware punctuation
and formatting for optimal TTS prosody. Also tracks overall sentiment curve.
"""
import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)


class FormattedTextResult(NamedTuple):
    """Result of text formatting."""
    formatted_text: str
    original_text: str
    sentiment_curve: dict  # "rising", "falling", "flat", "wave"
    dynamic_loudness_multiplier: float  # Global loudness factor (0.8-1.4)
    lexical_prosody_hints: dict  # Tracks punctuation/disfluency shaping applied


class IntelligentTextFormatter:
    """Reformats text for optimal TTS prosody based on emotion and sentiment."""

    CONJUNCTION_PATTERN = (
        r"(?:and|but|or|so|because|although|though|however|yet|while|then|still|instead)"
    )
    HESITATION_LEADERS = {"i", "we", "well", "so", "oh", "no"}
    STUTTER_EMOTIONS = {"fear", "anxiety", "nervousness", "confusion", "anger"}
    EMPHATIC_EMOTIONS = {"anger", "annoyance", "disapproval", "frustration"}
    EMPHATIC_COUNT_WORDS = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    }

    # Emotion to punctuation style mapping
    EMOTION_PUNCTUATION_STYLE = {
        # Joyful emotions: exclamation marks, ellipsis for wonder, short punchy phrases
        "joy": {
            "use_exclamation": True,
            "use_ellipsis": False,
            "use_question": False,
            "phrase_breaks": "short",  # Break into shorter phrases
            "comma_frequency": 0.4,
        },
        "excitement": {
            "use_exclamation": True,
            "use_ellipsis": False,
            "use_question": True,  # Rhetorical questions
            "phrase_breaks": "short",
            "comma_frequency": 0.3,
        },
        "amusement": {
            "use_exclamation": True,
            "use_ellipsis": True,  # For comic timing
            "use_question": True,
            "phrase_breaks": "varied",
            "comma_frequency": 0.5,
        },

        # Calm emotions: periods, minimal punctuation
        "calm": {
            "use_exclamation": False,
            "use_ellipsis": False,
            "use_question": False,
            "phrase_breaks": "long",
            "comma_frequency": 0.6,
        },
        "neutral": {
            "use_exclamation": False,
            "use_ellipsis": False,
            "use_question": False,
            "phrase_breaks": "medium",
            "comma_frequency": 0.4,
        },

        # Negative/tense emotions: ellipsis for pauses, fragmented
        "sadness": {
            "use_exclamation": False,
            "use_ellipsis": True,
            "use_question": True,
            "phrase_breaks": "fragmented",
            "comma_frequency": 0.7,
        },
        "fear": {
            "use_exclamation": False,
            "use_ellipsis": True,
            "use_question": False,
            "phrase_breaks": "fragmented",
            "comma_frequency": 0.8,
        },
        "anxiety": {
            "use_exclamation": False,
            "use_ellipsis": True,
            "use_question": False,
            "phrase_breaks": "fragmented",
            "comma_frequency": 0.75,
        },
        "anger": {
            "use_exclamation": True,
            "use_ellipsis": False,
            "use_question": False,
            "phrase_breaks": "short",
            "comma_frequency": 0.3,
        },

        # Contemplative
        "confusion": {
            "use_exclamation": False,
            "use_ellipsis": True,
            "use_question": True,
            "phrase_breaks": "medium",
            "comma_frequency": 0.6,
        },
    }

    def _get_style(self, emotion: str) -> dict:
        """Get punctuation style for emotion."""
        return self.EMOTION_PUNCTUATION_STYLE.get(
            emotion.lower(),
            self.EMOTION_PUNCTUATION_STYLE["neutral"]
        )

    def _normalize_text(self, text: str) -> str:
        """Collapse whitespace while preserving intentional line breaks."""
        text = re.sub(r"[ \t]+", " ", text.strip())
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _advanced_value(self, advanced_params, attr: str, default: float) -> float:
        """Safely read a numeric attribute from advanced params."""
        try:
            return float(getattr(advanced_params, attr))
        except (AttributeError, TypeError, ValueError):
            return default

    def _resolve_phrase_breaks(self, phrase_breaks: str, advanced_params) -> str:
        """Let advanced parameters steer phrase density."""
        if not advanced_params:
            return phrase_breaks

        tension = self._advanced_value(advanced_params, "tension", 0.3)
        breathiness = self._advanced_value(advanced_params, "breathiness", 0.25)
        pace_variance = self._advanced_value(advanced_params, "pace_variance", 0.3)

        if tension > 0.8 or pace_variance > 0.72:
            return "fragmented"
        if tension > 0.6:
            return "short"
        if breathiness < 0.3 and tension < 0.35:
            return "long"
        return phrase_breaks

    def _determine_sentiment_curve(self, emotion: str, confidence: float) -> str:
        """Determine the emotional arc of the text."""
        # High confidence positive → rising energy
        if emotion in ["joy", "excitement", "amusement"] and confidence > 0.6:
            return "rising"
        # Negative emotions → falling energy
        elif emotion in ["sadness", "grief", "fear", "anxiety"] and confidence > 0.5:
            return "falling"
        # Anger → wave (build then peak)
        elif emotion == "anger":
            return "wave"
        # Default: flat
        else:
            return "flat"

    def _compute_dynamic_loudness(self, emotion: str, confidence: float) -> float:
        """Compute overall loudness multiplier based on emotion intensity."""
        # Intense emotions (positive or negative) → louder
        if emotion in ["excitement", "anger", "fear"]:
            return 1.2 + (confidence * 0.2)
        # Positive calm → slightly louder
        elif emotion in ["joy", "amusement"]:
            return 1.1 + (confidence * 0.15)
        # Sad/calm emotions → quieter
        elif emotion in ["sadness", "grief", "anxiety"]:
            return 0.8 + (confidence * 0.1)
        # Neutral
        else:
            return 1.0

    def _inject_clause_commas(self, text: str, comma_frequency: float) -> tuple[str, int]:
        """Insert commas before conjunctions when delivery should feel more segmented."""
        if comma_frequency < 0.55:
            return text, 0

        replacements = 0

        def repl(match: re.Match) -> str:
            nonlocal replacements
            replacements += 1
            return f", {match.group(1)} "

        updated = re.sub(
            rf"(?<![,;:])\s+({self.CONJUNCTION_PATTERN})\s+",
            repl,
            text,
            flags=re.IGNORECASE,
        )

        if comma_frequency >= 0.82:
            updated = re.sub(
                r"(?<![,;:])\s+(because|since|when|while|if|unless|instead)\s+",
                repl,
                updated,
                flags=re.IGNORECASE,
            )
        return updated, replacements

    def _inject_breathiness(self, text: str, breathiness: float) -> tuple[str, dict]:
        """Use punctuation to hint at airy or hesitant delivery."""
        hints = {"ellipsis_count": 0, "hesitation_repeat": False, "breath_tag_count": 0}
        if breathiness <= 0.6:
            return text, hints

        updated = text
        leader_match = re.match(r'^([\'"(\[]?)([A-Za-z]+)\b', updated)
        if leader_match and leader_match.group(2).lower() in self.HESITATION_LEADERS:
            leader = leader_match.group(2)
            breath_token = "*hhhhh*" if breathiness > 0.72 else "*hh*"
            repeated = f"{leader_match.group(1)}{leader}... {breath_token} {leader}"
            updated = repeated + updated[leader_match.end():]
            hints["hesitation_repeat"] = True
            hints["ellipsis_count"] += 1
            hints["breath_tag_count"] += 1
        else:
            new_text, count = re.subn(r",\s+", "... ", updated, count=1)
            updated = new_text
            hints["ellipsis_count"] += count

        if breathiness > 0.72 and hints["breath_tag_count"] == 0:
            new_text, count = re.subn(
                r"(\.\.\.|\.)\s+",
                r"\1 *hhhhh* ",
                updated,
                count=1,
            )
            updated = new_text
            hints["breath_tag_count"] += count

        if breathiness > 0.78 and hints["ellipsis_count"] == 0:
            new_text, count = re.subn(r"\.\s+", "... ", updated, count=1)
            updated = new_text
            hints["ellipsis_count"] += count

        return updated, hints

    def _inject_tension_disfluency(
        self,
        text: str,
        emotion: str,
        confidence: float,
        tension: float,
    ) -> tuple[str, bool]:
        """Use a light stutter when the delivery should feel constricted or anxious."""
        if tension < 0.82 or confidence < 0.55 or emotion.lower() not in self.STUTTER_EMOTIONS:
            return text, False

        for match in re.finditer(r"\b([A-Za-z]{3,8})\b", text):
            word = match.group(1)
            start, end = match.span()
            if (start > 0 and text[start - 1] == "*") or (end < len(text) and text[end:end + 1] == "*"):
                continue
            if word.lower() in {"this", "that", "there", "their", "about", "would"}:
                continue
            stuttered = f"{word[:2]}-{word.lower()}" if tension > 0.9 and len(word) >= 4 else f"{word[0]}-{word.lower()}"
            return text[:start] + stuttered + text[end:], True

        return text, False

    def _elongate_word(self, word: str) -> str:
        """Stretch a word with repeated consonant/vowel hints for stronger emphasis."""
        lowered = word.lower()
        if lowered == "three":
            return "thhhreee"

        if len(word) < 4:
            return word

        head = word[:2]
        tail = word[2:]
        vowel_match = re.search(r"[aeiouy]", tail, flags=re.IGNORECASE)
        if not vowel_match:
            return f"{head}{tail[-1] * 2}"

        vowel_idx = 2 + vowel_match.start()
        return f"{word[:vowel_idx]}{word[vowel_idx] * 2}{word[vowel_idx + 1:]}"

    def _inject_emphatic_elongation(
        self,
        text: str,
        emotion: str,
        confidence: float,
        advanced_params,
    ) -> tuple[str, int]:
        """Stretch count words like 'three' when the line sounds frustrated/emphatic."""
        if confidence < 0.55:
            return text, 0

        tension = self._advanced_value(advanced_params, "tension", 0.3) if advanced_params else 0.3
        emotion_key = emotion.lower()
        is_emphatic = emotion_key in self.EMPHATIC_EMOTIONS or tension > 0.72
        if not is_emphatic:
            return text, 0

        pattern = re.compile(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b(?=\s+(?:times?|more|again|already)\b)",
            flags=re.IGNORECASE,
        )

        replacements = 0

        def repl(match: re.Match) -> str:
            nonlocal replacements
            word = match.group(1)
            if word.lower() not in self.EMPHATIC_COUNT_WORDS:
                return word
            replacements += 1
            stretched = self._elongate_word(word)
            if word.isupper():
                return stretched.upper()
            if word[0].isupper():
                return stretched.capitalize()
            return stretched

        updated = pattern.sub(repl, text, count=1)
        return updated, replacements

    def _inject_lexical_prosody(
        self,
        text: str,
        emotion: str,
        confidence: float,
        style: dict,
        advanced_params,
    ) -> tuple[str, dict]:
        """Convert advanced voice controls into punctuation and lexical hesitations."""
        hints = {
            "comma_injections": 0,
            "ellipsis_count": 0,
            "hesitation_repeat": False,
            "stuttered": False,
            "elongated_words": 0,
        }
        if not advanced_params:
            return text, hints

        tension = self._advanced_value(advanced_params, "tension", 0.3)
        breathiness = self._advanced_value(advanced_params, "breathiness", 0.25)
        pace_variance = self._advanced_value(advanced_params, "pace_variance", 0.3)

        updated, elongated_words = self._inject_emphatic_elongation(
            text,
            emotion,
            confidence,
            advanced_params,
        )
        hints["elongated_words"] = elongated_words

        comma_frequency = style["comma_frequency"]
        comma_frequency += max(0.0, pace_variance - 0.45) * 0.8
        comma_frequency += max(0.0, tension - 0.55) * 0.5
        comma_frequency -= max(0.0, 0.35 - breathiness) * 0.2

        updated, comma_count = self._inject_clause_commas(updated, comma_frequency)
        hints["comma_injections"] = comma_count

        updated, breathy_hints = self._inject_breathiness(updated, breathiness)
        hints.update(breathy_hints)

        updated, stuttered = self._inject_tension_disfluency(updated, emotion, confidence, tension)
        hints["stuttered"] = stuttered
        return updated, hints

    def _add_sentence_breaks(self, text: str, phrase_breaks: str) -> str:
        """Add line breaks for natural phrase pacing."""
        if phrase_breaks == "short":
            # Break after periods, but also add breaks before conjunctions
            text = re.sub(r'([.!?])\s+', r'\1\n', text)
            # Also break before "but", "and", "however", etc. occasionally
            text = re.sub(
                rf'\s+({self.CONJUNCTION_PATTERN})\s+',
                r'\n\1 ',
                text,
                flags=re.IGNORECASE,
            )
        elif phrase_breaks == "fragmented":
            # Break very frequently for anxious/fear delivery
            text = re.sub(r'([.!?])\s+', r'\1\n', text)
            text = re.sub(r'([,;:])\s+', r'\1\n', text)
            # Also break on conjunctions
            text = re.sub(
                rf'\s+({self.CONJUNCTION_PATTERN})\s+',
                r'\n\1 ',
                text,
                flags=re.IGNORECASE,
            )
        elif phrase_breaks == "long":
            # Minimal breaks, keep phrases flowing
            pass
        else:  # "medium" or "varied"
            # Standard sentence breaks
            text = re.sub(r'([.!?])\s+', r'\1\n', text)

        return text.strip()

    def _enhance_punctuation(self, text: str, style: dict) -> str:
        """Enhance text with emotion-aware punctuation."""
        # Clean up excessive punctuation without collapsing intentional ellipses.
        text = re.sub(r'(?<!\.)\.{4,}', '...', text)
        text = re.sub(r'([!?]){2,}', r'\1', text)
        text = re.sub(r'([!?])\s*([.!?]+)', r'\1', text)

        # Replace periods with exclamation marks if appropriate
        if style["use_exclamation"]:
            # Convert some periods to exclamation marks (especially at ends of sentences)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            enhanced = []
            for i, sent in enumerate(sentences):
                if sent.strip():
                    # Last sentence or strong sentiment → exclamation
                    if i == len(sentences) - 1 or (i > 0 and i % 2 == 0):
                        sent = re.sub(r'\.(\s*)$', r'!\1', sent)
                    enhanced.append(sent)
            text = ' '.join(enhanced)

        # Add ellipsis for dramatic/thoughtful pauses
        if style["use_ellipsis"]:
            # Find natural pause points (before conjunctions, after commas in some cases)
            text = re.sub(r',(\s+(?:but|however|although))', r'...\1', text, flags=re.IGNORECASE)

        # Add question marks for contemplative tone
        if style["use_question"]:
            # This is more subtle - add question marks to rhetorical statements
            # Example: "You know what..." → "You know what...?"
            text = re.sub(r'(\.\.\.)(\s+|$)', r'...?\2', text)

        return text

    def format_for_emotion(
        self,
        text: str,
        emotion: str,
        confidence: float = 0.7,
        advanced_params=None,
    ) -> FormattedTextResult:
        """
        Reformat text for optimal TTS prosody based on emotion.

        Args:
            text: Original input text
            emotion: Detected primary emotion
            confidence: Confidence in emotion detection (0.0-1.0)

        Returns:
            FormattedTextResult with formatted text and metadata
        """
        style = self._get_style(emotion)
        formatted = self._normalize_text(text)
        phrase_breaks = self._resolve_phrase_breaks(style["phrase_breaks"], advanced_params)

        # Step 1: Inject lexical prosody before chunk shaping.
        formatted, lexical_hints = self._inject_lexical_prosody(
            formatted,
            emotion,
            confidence,
            style,
            advanced_params,
        )

        # Step 2: Add intelligent sentence breaks
        formatted = self._add_sentence_breaks(formatted, phrase_breaks)

        # Step 3: Enhance punctuation
        formatted = self._enhance_punctuation(formatted, style)

        # Step 4: Determine sentiment curve
        sentiment_curve = self._determine_sentiment_curve(emotion, confidence)

        # Step 5: Compute dynamic loudness
        dynamic_loudness = self._compute_dynamic_loudness(emotion, confidence)

        logger.info(
            f"Text formatted for '{emotion}' (confidence={confidence:.2f}): "
            f"curve={sentiment_curve}, loudness_mult={dynamic_loudness:.2f}, "
            f"lexical_hints={lexical_hints}"
        )

        return FormattedTextResult(
            formatted_text=formatted,
            original_text=text,
            sentiment_curve={
                "type": sentiment_curve,
                "confidence": confidence,
            },
            dynamic_loudness_multiplier=dynamic_loudness,
            lexical_prosody_hints=lexical_hints,
        )


class SentimentCurveGenerator:
    """Generates dynamic loudness/pace curves based on sentiment arc."""

    @staticmethod
    def apply_sentiment_curve(
        chunks: list,
        curve_type: str,
        base_loudness: float = 1.0,
        dynamic_multiplier: float = 1.0,
    ) -> list:
        """
        Modulate chunk loudness/pace based on sentiment curve.

        Args:
            chunks: List of ProsodyChunk objects
            curve_type: "rising", "falling", "flat", or "wave"
            base_loudness: Base loudness from voice profile
            dynamic_multiplier: Global loudness multiplier from emotion

        Returns:
            List of chunks with modulated loudness
        """
        if not chunks or curve_type == "flat":
            return chunks

        n = len(chunks)
        modulated = []

        for i, chunk in enumerate(chunks):
            progress = i / max(1, n - 1)  # 0.0 to 1.0 through the chunks

            if curve_type == "rising":
                # Energy builds toward the end
                loudness_factor = 0.9 + (0.3 * progress)  # 0.9 to 1.2
            elif curve_type == "falling":
                # Energy fades
                loudness_factor = 1.1 - (0.3 * progress)  # 1.1 to 0.8
            elif curve_type == "wave":
                # Energy peaks in the middle
                wave_progress = abs(progress - 0.5) * 2  # 0.0 to 1.0 (0 at middle)
                loudness_factor = 1.0 + (0.2 * (1.0 - wave_progress))  # 0.8 to 1.2
            else:
                loudness_factor = 1.0

            # Apply modulation
            modulated_loudness = chunk.loudness * loudness_factor * dynamic_multiplier
            modulated_loudness = max(0.6, min(1.6, modulated_loudness))  # Clamp to valid range

            # Create modified chunk (preserve original, just update loudness)
            modulated_chunk = type(chunk)(
                text=chunk.text,
                emotion_context=chunk.emotion_context,
                pace=chunk.pace,
                pitch=chunk.pitch,
                loudness=modulated_loudness,
                post_chunk_pause_ms=chunk.post_chunk_pause_ms,
                words=list(getattr(chunk, "words", [])),
            )
            modulated.append(modulated_chunk)

        logger.debug(f"Applied {curve_type} sentiment curve to {len(chunks)} chunks")
        return modulated


intelligent_text_formatter = IntelligentTextFormatter()
sentiment_curve_generator = SentimentCurveGenerator()
