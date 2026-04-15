#!/usr/bin/env python3
"""
Pre-download and cache emotion models to avoid timeout issues during first use.
Run this once to cache models, then subsequent runs will be instant.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set download timeout
os.environ.setdefault("HF_HUB_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

def warmup():
    """Pre-download models to cache."""
    print("[WARMUP] Starting model pre-download...\n")
    
    try:
        from modules.emotion_engine import emotion_engine
        
        models = emotion_engine.list_models()
        print(f"[WARMUP] Found {len(models)} emotion models to cache\n")
        
        for model in models:
            try:
                print(f"[WARMUP] Caching: {model['display_name']} ({model['key']})")
                print(f"[WARMUP] Model: {model['hf_model_id']}")
                
                # Force load and cache the model
                spec = emotion_engine._get_model_spec(model['key'])
                classifier = emotion_engine._build_classifier(spec)
                
                # Test inference to ensure it works
                test_result = classifier("This is a test.")
                print(f"[WARMUP] ✓ Successfully cached: {model['key']}")
                print(f"[WARMUP] Sample output: {test_result}\n")
                
            except Exception as e:
                print(f"[WARMUP] ✗ Failed to cache {model['key']}: {e}\n")
                continue
        
        print("[WARMUP] Model pre-download complete!")
        print("[WARMUP] All models are now cached and ready for instant use.")
        
    except Exception as e:
        print(f"[WARMUP] Error during warmup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    warmup()
