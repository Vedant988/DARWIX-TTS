# Troubleshooting: Model Download Timeout

## Problem
When you click "Synthesize Emotion", the model download gets stuck at 0% and times out:
```
pytorch_model.bin:   0%|..| 0.00/329M [01:08<?, ?B/s]
```

## Root Causes
1. **No HF_TOKEN**: Downloads are rate-limited without authentication
2. **Slow Internet**: 329MB model takes time on slow connections
3. **Network Issues**: Intermittent connectivity during download
4. **First-Time Load**: Model needs to be downloaded and cached

## Solutions

### Quick Fix: Pre-Download Models
Run this once to cache all models before using the app:

```powershell
# Navigate to backend directory
cd C:\Users\vedan\Desktop\DARWIX\backend

# Activate venv
.\venv\Scripts\Activate.ps1

# Pre-download and cache models (first time only)
python warmup_models.py
```

This will:
- Download all emotion models to your local cache
- Test that they load correctly
- Take 5-10 minutes depending on internet speed
- After this, all subsequent runs are instant

### Setup HF_TOKEN (Recommended)
Add your Hugging Face token to `.env`:

1. Get token from: https://huggingface.co/settings/tokens
2. Add to `.env`:
   ```
   HF_TOKEN=hf_YOUR_TOKEN_HERE
   ```
3. Restart backend

### Increase Timeout (Already Done)
The backend now automatically sets:
```
HF_HUB_TIMEOUT=600 seconds (10 minutes)
```

### Use Local Model Cache
If you have the model cached already:
```powershell
# Check cache
$env:HF_HOME = "C:\Users\vedan\.cache\huggingface\hub"
ls $env:HF_HOME
```

### Alternative: Lighter Model
Edit `backend/modules/emotion_engine.py` and change default model:
```python
{
    "key": "bhadresh",  # Change to this for faster downloads
    "hf_model_id": "bhadresh-savani/distilbert-base-uncased-emotion",
    "default": True,  # Make it default
    ...
}
```

## Verification

Check if model is cached:
```powershell
$cache = "$env:USERPROFILE\.cache\huggingface\hub"
ls $cache
```

Check backend environment:
```
GET http://localhost:8000/debug
```

## Expected Timeline

**First Run (with HF_TOKEN):**
- Model download: 2-5 minutes
- Subsequent runs: <500ms

**First Run (without HF_TOKEN):**
- Model download: 5-10+ minutes (rate limited)
- Subsequent runs: <500ms

**After Pre-Warming:**
- All runs: <500ms (instant)
