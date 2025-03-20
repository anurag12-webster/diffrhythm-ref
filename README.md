# DiffRhythm Reference Implementation

This repository provides a reference implementation for using DiffRhythm, a music generation model.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/anurag12-webster/diffrhythm-ref.git
cd diffrhythm-ref
```

2. Install DiffRhythm and its dependencies:
```bash
cd DiffRhythm
pip install -e .
```

## Usage

You can use DiffRhythm directly by importing from the app module:

```python
from diffrhythm.infer.infer import inference
from diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_audio_style_prompt,
    get_text_style_prompt,
    prepare_model,
    get_negative_style_prompt
)

# Example lyrics with timestamps
text = """
[00:00.00]这是第一行歌词
[00:03.00]这是第二行歌词
[00:06.00]这是第三行歌词
"""

# Initialize models
device = "cuda"  # or "cpu"
cfm, cfm_full, tokenizer, muq, vae = prepare_model(device)

# Get style prompts (optional)
text_style = get_text_style_prompt(muq, "happy and energetic")
negative_style = get_negative_style_prompt(device)

# Generate music
inference(
    cfm=cfm,
    cfm_full=cfm_full,
    tokenizer=tokenizer,
    muq=muq,
    vae=vae,
    text=text,
    text_style=text_style,
    negative_style=negative_style,
    device=device,
    output_path="generated_music.wav"
)
```

### Parameters

- `text`: The lyrics text with timestamps
- `audio_style_path`: (Optional) Path to an audio file to use as style reference
- `text_style_prompt`: (Optional) Text description of the desired style
- `device`: Device to run inference on ("cuda" or "cpu")
- `output_path`: Path to save the generated audio

## Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- Dependencies:
  - torch
  - numpy
  - librosa
  - mutagen
  - huggingface_hub
  - muq
  - wandb
  - accelerate
  - ema_pytorch
  - bitsandbytes
  - pypinyin
  - jieba
  - x-transformers
  - safetensors

## License

This repository is licensed under the same terms as the original DiffRhythm repository. 