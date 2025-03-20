# DiffRhythm Reference Implementation

This repository provides a reference implementation for using DiffRhythm, a music generation model. It includes a simple interface to generate music from lyrics text.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/anurag12-webster/diffrhythm-ref.git
cd diffrhythm-ref
```

2. Install DiffRhythm:
```bash
cd DiffRhythm
pip install -e .
cd ..
```

## Usage

The repository provides a simple interface through `tasks.py` to generate music:

```python
from tasks import generate_music

# Example lyrics with timestamps
text = """
[00:00.00]这是第一行歌词
[00:03.00]这是第二行歌词
[00:06.00]这是第三行歌词
"""

# Generate music
generate_music(
    text=text,
    audio_style_path="path/to/style/audio.wav",  # Optional: reference audio for style
    text_style_prompt="happy and energetic",      # Optional: text description of style
    output_path="generated_music.wav"             # Output audio file path
)
```

### Parameters

- `text`: The lyrics text with timestamps
- `audio_style_path`: (Optional) Path to an audio file to use as style reference
- `text_style_prompt`: (Optional) Text description of the desired style
- `device`: (Optional) Device to run inference on ("cuda" or "cpu")
- `output_path`: (Optional) Path to save the generated audio

## Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- See `DiffRhythm/setup.py` for full list of dependencies

## License

This repository is licensed under the same terms as the original DiffRhythm repository. 