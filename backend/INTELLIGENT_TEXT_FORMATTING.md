# Intelligent Text Formatting & Dynamic Sentiment Curves

## Overview

The **Intelligent Text Formatter** reshapes input text with emotion-aware punctuation and formatting to optimize TTS prosody, ensuring the overall "feeling" of the sentence dynamically affects flow and loudness.

## Key Features

### 1. Emotion-Aware Punctuation Optimization

Different emotions trigger different formatting strategies:

#### Joyful Emotions (joy, excitement, amusement)
- **Exclamation marks**: Converts periods to `!` for energetic emphasis
- **Short phrases**: Breaks text into punchy, shorter phrases
- **Punctuation**: Minimal commas (0.3-0.4 frequency)
- **Example**:
  - Input: "Hello thank you for calling our service"
  - Output: "Hello! Thank you for calling our service."

#### Calm Emotions (calm, neutral)
- **Periods preserved**: Full stops for measured delivery
- **Long phrases**: Keeps sentences flowing naturally
- **Punctuation**: Standard comma usage (0.4-0.6 frequency)
- **Example**:
  - Input: "The package will arrive tomorrow at noon"
  - Output: "The package will arrive tomorrow at noon." (unchanged)

#### Negative/Tense Emotions (sadness, fear, anxiety)
- **Ellipsis (...)**: Adds dramatic pauses for reflection
- **Fragmented breaks**: Splits sentences at commas for hesitant delivery
- **Questions**: Adds rising intonation markers for uncertainty
- **Punctuation**: High comma frequency (0.7-0.8) for micro-pauses
- **Example**:
  - Input: "Something went wrong with the system"
  - Output: "Something went wrong... with the system." (contemplative pause)

#### Anger
- **Exclamation marks**: Converts to `!` for aggressive emphasis
- **Short bursts**: Fragments text for staccato delivery
- **No ellipsis**: Direct, no hesitation
- **Example**:
  - Input: "This is absolutely unacceptable"
  - Output: "This is absolutely unacceptable!" (emphatic)

### 2. Dynamic Sentiment Curves

The formatter analyzes the overall emotional content and applies an **emotional arc** to the entire synthesis:

#### Rising Curve (joy, excitement when high confidence)
- Loudness progression: 0.9 → 1.2 across chunks
- Energy builds toward the end of the sentence
- Perfect for: "This is going to be amazing!"

#### Falling Curve (sadness, grief, anxiety when high confidence)
- Loudness progression: 1.1 → 0.8 across chunks
- Energy fades gradually
- Perfect for: "I'm not sure if I can do this..."

#### Wave Curve (anger)
- Loudness peaks in middle: 0.8 → 1.2 → 0.8
- Energy spikes with intensity then settles
- Perfect for: "This is ABSOLUTELY unacceptable behavior!"

#### Flat Curve (neutral, low confidence)
- Consistent loudness throughout
- Maintains steady delivery

### 3. Dynamic Loudness Multiplier

Overall emotion intensity scales the global loudness:

| Emotion | Loudness Multiplier |
|---------|-------------------|
| Excitement | 1.2 - 1.4 |
| Anger | 1.2 - 1.4 |
| Fear | 1.2 - 1.4 |
| Joy | 1.1 - 1.25 |
| Amusement | 1.1 - 1.25 |
| Calm | 1.0 |
| Neutral | 1.0 |
| Sadness | 0.8 - 0.9 |
| Anxiety | 0.8 - 0.9 |
| Grief | 0.8 - 0.9 |

## Integration with Pipeline

### Processing Flow

```
User Text Input
    ↓
Emotion Analysis (Hugging Face)
    ↓
Advanced Voice Enhancement
    ↓
INTELLIGENT TEXT FORMATTING ← NEW
    • Emotion-aware punctuation
    • Sentiment curve determination
    • Dynamic loudness calculation
    ↓
FORMATTED TEXT + METADATA
    ↓
Prosody Director (Groq)
    • Receives optimized text
    • Understands punctuation for pauses
    ↓
SENTIMENT CURVE APPLICATION ← NEW
    • Modulates all chunk loudness
    • Applies emotional arc
    ↓
Concurrent Synthesis (Sarvam)
    ↓
Audio Stitching
    ↓
Final Output
```

### WebSocket Pipeline Status

The UI now displays 5 stages:
1. **Optimizing Text for Emotion** (text formatting)
2. **Generating Semantic Micro-Chunks** (Groq)
3. **Mapping Chunk-Level Prosody Parameters** (voice mapping)
4. **Synthesizing Concurrent Audio Streams** (Sarvam)
5. **Assembling Final Audio Vector** (stitching)

## Code Architecture

### intelligent_text_formatter.py

**Main Classes:**
- `IntelligentTextFormatter`: Reformats text based on emotion
- `SentimentCurveGenerator`: Applies emotional arcs to synthesized chunks

**Key Methods:**
- `format_for_emotion(text, emotion, confidence)` → `FormattedTextResult`
  - Returns: formatted text + sentiment curve + dynamic loudness
- `apply_sentiment_curve(chunks, curve_type, base_loudness, dynamic_multiplier)`
  - Returns: list of chunks with modulated loudness

**Result Structure:**
```python
FormattedTextResult(
    formatted_text: str,           # Emotion-optimized text
    original_text: str,            # Original unchanged text
    sentiment_curve: dict,         # {"type": "rising"|"falling"|"wave"|"flat", "confidence": 0.0-1.0}
    dynamic_loudness_multiplier: float  # 0.8-1.4
)
```

## Examples

### Example 1: Joyful Text
**Input:**
```
"I got the job offer today thank you so much for helping me prepare"
```
**After Formatting:**
```
"I got the job offer today! Thank you so much for helping me prepare!"
```
**Sentiment Curve:** Rising (energy builds)  
**Loudness:** 1.2x-1.4x (enthusiastic volume)

---

### Example 2: Anxious Text
**Input:**
```
"I am not sure if I can handle this responsibility properly"
```
**After Formatting:**
```
"I am not sure...
if I can handle,
this responsibility,
properly..."
```
**Sentiment Curve:** Falling (uncertain energy)  
**Loudness:** 0.8x-0.9x (quiet, hesitant)

---

### Example 3: Angry Text
**Input:**
```
"This is completely unacceptable and needs to change immediately"
```
**After Formatting:**
```
"This is completely unacceptable!
And needs to change,
immediately!"
```
**Sentiment Curve:** Wave (peaks mid-sentence)  
**Loudness:** 1.3x (loud, aggressive)

---

## Parameters Controlled

The formatter influences these Sarvam TTS parameters per chunk:

| Parameter | Range | Effect |
|-----------|-------|--------|
| **loudness** | 0.6-1.6 | Emotional intensity, energy curve |
| **pace** | 0.7-1.2 | Speech rate (unchanged in formatter) |
| **pitch** | -0.55-0.55 | Vocal tone (set by voice mapper) |

Punctuation added by formatter influences Sarvam's natural behavior:
- **!** → More energetic emphasis
- **...** → Dramatic pause
- **?** → Rising intonation
- **,** → Short micro-pause

## Usage in main.py

```python
# After emotion analysis
formatted_result = await asyncio.to_thread(
    intelligent_text_formatter.format_for_emotion,
    text,
    analysis.primary_emotion,
    analysis.confidence,
)

# Use formatted text for Groq (not original)
chunks = await prosody_director.direct_prosody(
    formatted_result.formatted_text,  # ← formatted, not original
    voice_quality_context
)

# Apply sentiment curve to all chunks
chunks = sentiment_curve_generator.apply_sentiment_curve(
    chunks,
    formatted_result.sentiment_curve["type"],
    voice_profile.loudness,
    formatted_result.dynamic_loudness_multiplier,
)
```

## Benefits

✅ **Natural Punctuation**: TTS engines handle punctuation natively (pauses, intonation)  
✅ **Dynamic Loudness**: Overall emotional flow affects all chunks, not just individual parameters  
✅ **Text Shaping**: Formatting creates better phonetic boundaries for Groq  
✅ **Emotional Arc**: Entire synthesis feels like a natural speech delivery  
✅ **Confidence-Scaled**: Subtle formatting for low-confidence, dramatic for high-confidence emotions  

## Future Enhancements

1. **Breathing Phrases**: Add strategic sentence breaks for vocal breathing (large texts)
2. **Pacing Acceleration**: Rising curves could include pace increase too
3. **Emphasis Markers**: Support markdown-style `*emphasis*` for Groq direction
4. **Stanza Breaks**: Detect natural topic shifts and add blank lines for prosodic boundaries
5. **Context-Aware Punctuation**: Remember sentence context to avoid over-punctuation
