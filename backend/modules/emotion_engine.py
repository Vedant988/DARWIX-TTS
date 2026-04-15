import os
import re
from dataclasses import dataclass


class EmotionAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmotionModelSpec:
    key: str
    hf_model_id: str
    display_name: str
    tradeoff: str
    label_count: int
    latency_hint_ms: str
    multi_label: bool
    threshold: float
    aliases: tuple[str, ...]

    def to_public_dict(self) -> dict:
        return {
            "key": self.key,
            "hf_model_id": self.hf_model_id,
            "display_name": self.display_name,
            "tradeoff": self.tradeoff,
            "label_count": self.label_count,
            "latency_hint_ms": self.latency_hint_ms,
            "multi_label": self.multi_label,
        }


@dataclass(frozen=True)
class EmotionScore:
    label: str
    score: float

    def to_public_dict(self) -> dict:
        return {"label": self.label, "score": round(self.score, 4)}


@dataclass(frozen=True)
class EmotionAnalysis:
    model_key: str
    model_id: str
    display_name: str
    multi_label: bool
    primary_emotion: str
    confidence: float
    top_emotions: list[EmotionScore]
    normalized_emotions: list[EmotionScore]
    clarity: float
    warnings: list[str]

    def to_public_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "multi_label": self.multi_label,
            "primary_emotion": self.primary_emotion,
            "confidence": round(self.confidence, 4),
            "clarity": round(self.clarity, 4),
            "top_emotions": [item.to_public_dict() for item in self.top_emotions],
            "normalized_emotions": [item.to_public_dict() for item in self.normalized_emotions],
            "warnings": self.warnings,
        }


class EmotionEngine:
    def __init__(self):
        self.model_specs = {
            "hartmann": EmotionModelSpec(
                key="hartmann",
                hf_model_id="j-hartmann/emotion-english-distilroberta-base",
                display_name="j-hartmann/emotion-english-distilroberta-base",
                tradeoff="Fastest premium-quality",
                label_count=7,
                latency_hint_ms="80-250",
                multi_label=False,
                threshold=0.05,
                aliases=("1", "j-hartmann", "distilroberta", "premium"),
            ),
            "go_emotions": EmotionModelSpec(
                key="go_emotions",
                hf_model_id="SamLowe/roberta-base-go_emotions",
                display_name="SamLowe/roberta-base-go_emotions",
                tradeoff="Best balance / richest label space",
                label_count=28,
                latency_hint_ms="150-400",
                multi_label=True,
                threshold=0.12,
                aliases=("2", "samlowe", "go-emotions", "roberta"),
            ),
            "bhadresh": EmotionModelSpec(
                key="bhadresh",
                hf_model_id="bhadresh-savani/distilbert-base-uncased-emotion",
                display_name="bhadresh-savani/distilbert-base-uncased-emotion",
                tradeoff="Very fast / compact",
                label_count=6,
                latency_hint_ms="60-200",
                multi_label=False,
                threshold=0.05,
                aliases=("3", "distilbert", "bhadresh-savani"),
            ),
        }
        self.alias_map = {}
        for spec in self.model_specs.values():
            self.alias_map[spec.key] = spec.key
            self.alias_map[spec.hf_model_id.lower()] = spec.key
            for alias in spec.aliases:
                self.alias_map[alias.lower()] = spec.key

        self.default_model_key = self._resolve_model_key(os.environ.get("DEFAULT_EMOTION_MODEL")) or "hartmann"
        
        # Initialize inference client
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EmotionAnalysisError("HF_TOKEN environment variable is required for emotion analysis via Inference API")
        
        try:
            from huggingface_hub import InferenceClient
            self.inference_client = InferenceClient(api_key=hf_token)
            print("[EMOTION_ENGINE] ✓ Initialized with Hugging Face Inference API (no local model downloads)")
        except ImportError:
            raise EmotionAnalysisError("huggingface_hub is required. Install: pip install huggingface_hub")

    def list_models(self) -> list[dict]:
        return [
            {
                **spec.to_public_dict(),
                "default": spec.key == self.default_model_key,
            }
            for spec in self.model_specs.values()
        ]

    def _resolve_model_key(self, requested_model: str | None) -> str | None:
        if not requested_model:
            return None
        return self.alias_map.get(requested_model.strip().lower())

    def _get_model_spec(self, requested_model: str | None) -> EmotionModelSpec:
        resolved_key = self._resolve_model_key(requested_model) or self.default_model_key
        if resolved_key not in self.model_specs:
            raise EmotionAnalysisError(f"Unsupported emotion model: {requested_model}")
        return self.model_specs[resolved_key]

    def _normalize_outputs(self, outputs, spec: EmotionModelSpec) -> tuple[list[EmotionScore], list[EmotionScore], float]:
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            outputs = outputs[0]

        scores = [EmotionScore(str(item["label"]).lower(), float(item["score"])) for item in outputs]
        scores.sort(key=lambda item: item.score, reverse=True)

        if spec.multi_label:
            selected = [item for item in scores if item.score >= spec.threshold][:8]
            if not selected:
                selected = scores[:5]
            total = sum(item.score for item in selected) or 1.0
            normalized = [EmotionScore(item.label, item.score / total) for item in selected]
        else:
            normalized = scores[:]
            selected = normalized[:5]

        clarity = selected[0].score - selected[1].score if len(selected) > 1 else selected[0].score
        clarity = max(0.0, min(1.0, clarity))
        return scores, normalized, clarity

    def analyze(self, text: str, requested_model: str | None = None) -> EmotionAnalysis:
        if not text or not text.strip():
            raise EmotionAnalysisError("No text provided for emotion analysis")

        spec = self._get_model_spec(requested_model)
        print(f"\n[ANALYZE] Starting emotion analysis via Inference API")
        print(f"[ANALYZE] Model: {spec.key} ({spec.hf_model_id})")
        print(f"[ANALYZE] Text: {text[:100]}...")
        
        try:
            print(f"[ANALYZE] Calling Inference API...")
            outputs = self.inference_client.text_classification(
                text.strip(),
                model=spec.hf_model_id,
            )
            print(f"[ANALYZE] API call successful")
            print(f"[ANALYZE] Raw outputs: {outputs}\n")
        except Exception as exc:
            print(f"[ANALYZE] API call FAILED: {exc}\n")
            raise EmotionAnalysisError(f"Emotion analysis via Inference API failed: {exc}") from exc

        raw_scores, normalized_scores, clarity = self._normalize_outputs(outputs, spec)
        warnings: list[str] = []

        non_ascii_chars = len(re.findall(r"[^\x00-\x7F]", text))
        if non_ascii_chars / max(len(text), 1) > 0.15:
            warnings.append("Selected emotion models are English-trained; non-English text may reduce accuracy.")

        if spec.multi_label:
            primary = max(normalized_scores, key=lambda item: item.score)
            confidence = primary.score
            top_emotions = normalized_scores[:5]
        else:
            primary = raw_scores[0]
            confidence = primary.score
            top_emotions = raw_scores[:5]

        return EmotionAnalysis(
            model_key=spec.key,
            model_id=spec.hf_model_id,
            display_name=spec.display_name,
            multi_label=spec.multi_label,
            primary_emotion=primary.label,
            confidence=confidence,
            top_emotions=top_emotions,
            normalized_emotions=normalized_scores[:8],
            clarity=clarity,
            warnings=warnings,
        )


emotion_engine = EmotionEngine()
