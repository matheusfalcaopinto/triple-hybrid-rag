# Audio Assets

This directory contains pre-recorded audio clips in μ-law format for the voice agent.

## Format Specifications

All audio files must be in this exact format:
- **Codec:** μ-law (G.711)
- **Sample Rate:** 8000 Hz (8kHz)
- **Channels:** Mono
- **Frame Size:** 160 bytes (20ms per frame)
- **File Extension:** `.raw`, `.mulaw`, or `.ulaw`

## Directory Structure

```
audio_assets/
├── greetings.raw           # Welcome greeting
└── silence_fillings/       # Random silence fillers
    ├── filler1.raw
    ├── filler2.raw
    └── ...
```

### Current Files
- `greetings.raw` - Welcome greeting played at call start
- `silence_fillings/` - Random silence fillers picked during LLM processing

### Adding New Clips

Use the conversion script to create new clips:

```bash
# Convert all .wav files in audio_assets/
python scripts/convert_audio.py

# Convert single file
python scripts/convert_audio.py my_audio.wav audio_assets/my_clip.raw

# Convert and play to verify
python scripts/convert_audio.py my_audio.wav --play
```

## Using Clips in Code

Clips are loaded automatically by name from this directory:

```python
from providers.pre_recorded import get_clip_frames

# Get specific clip
frames = get_clip_frames("greetings", base_dir="audio_assets")

# Get random silence filling
frames = get_clip_frames("random_silence_filling", base_dir="audio_assets")
```

**Environment Configuration:**

```bash
# Use random silence fillings
SILENCE_FILL_AUDIO_CLIP=random_silence_filling
ENABLE_SILENCE_FILL=true

# Use specific clip
SILENCE_FILL_AUDIO_CLIP=my_custom_filler
```

## Clip Naming Convention

Use descriptive names:
- `greeting_<variant>.raw` - Welcome messages
- `silence_<duration>.raw` - Silence fills (e.g., `silence_500ms.raw`)
- `filler_<type>.raw` - Conversational fillers (e.g., `filler_hmm.raw`)
- `hold_<variant>.raw` - Hold music/messages
- `error_<type>.raw` - Error messages

## Creating Silence Fills

Silence fills are useful to prevent awkward pauses:

```bash
# Create 500ms of silence
ffmpeg -f lavfi -i anullsrc=r=8000:cl=mono -t 0.5 -f mulaw silence_500ms.raw

# Create 1 second of silence
ffmpeg -f lavfi -i anullsrc=r=8000:cl=mono -t 1.0 -f mulaw silence_1s.raw

# Or use the script
python scripts/convert_audio.py silence.wav audio_assets/silence_500ms.raw
```

## Tips for Recording

1. **Record in a quiet environment** - Background noise is amplified in phone calls
2. **Speak clearly and at moderate pace** - Phone audio has limited bandwidth
3. **Keep it short** - Aim for 1-3 seconds for most clips
4. **Test on phone first** - Listen to converted audio before deploying
5. **Avoid loud sounds** - μ-law has limited dynamic range

## Quality Guidelines

- **Sample clearly in high quality first** (44.1kHz/16-bit WAV)
- **Normalize audio levels** to avoid clipping
- **Apply noise reduction** if needed
- **Convert as final step** to preserve quality

## Example: Creating a Custom Greeting

```bash
# 1. Record your greeting in Audacity/other tool
# 2. Export as WAV (mono, any sample rate)
# 3. Convert to μ-law format
python scripts/convert_audio.py my_greeting.wav audio_assets/greeting_custom.raw

# 4. Update config to use it
# In .env:
GREETING_AUDIO_CLIP=greeting_custom
```

## Troubleshooting

**Clip sounds distorted:**
- Check input levels aren't too high
- Try normalizing to -3dB before converting

**Clip won't play:**
- Verify file size is divisible by 160
- Check file is in correct directory
- Ensure filename matches what code expects

**Silent or low volume:**
- Increase gain in source audio
- Normalize to -1dB peak before converting
