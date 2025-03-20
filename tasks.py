import torch
from diffrhythm.infer.infer import inference
from diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_audio_style_prompt,
    get_text_style_prompt,
    prepare_model,
    get_negative_style_prompt
)

def generate_music(
    text: str,
    audio_style_path: str = None,
    text_style_prompt: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = "output.wav"
):
    """
    Generate music using DiffRhythm.
    
    Args:
        text: The lyrics text to generate music for
        audio_style_path: Path to an audio file to use as style reference
        text_style_prompt: Text description of the style
        device: Device to run inference on ("cuda" or "cpu")
        output_path: Path to save the generated audio
    """
    # Prepare models
    cfm, cfm_full, tokenizer, muq, vae = prepare_model(device)
    
    # Get style prompts
    if audio_style_path:
        audio_style, vocal_flag = get_audio_style_prompt(muq, audio_style_path)
    else:
        audio_style = None
        vocal_flag = False
        
    if text_style_prompt:
        text_style = get_text_style_prompt(muq, text_style_prompt)
    else:
        text_style = None
        
    # Get negative style prompt
    negative_style = get_negative_style_prompt(device)
    
    # Generate music
    inference(
        cfm=cfm,
        cfm_full=cfm_full,
        tokenizer=tokenizer,
        muq=muq,
        vae=vae,
        text=text,
        audio_style=audio_style,
        text_style=text_style,
        negative_style=negative_style,
        vocal_flag=vocal_flag,
        device=device,
        output_path=output_path
    )

if __name__ == "__main__":
    # Example usage
    text = """
    [00:00.00]这是第一行歌词
    [00:03.00]这是第二行歌词
    [00:06.00]这是第三行歌词
    """
    
    generate_music(
        text=text,
        audio_style_path="path/to/style/audio.wav",  # Optional
        text_style_prompt="happy and energetic",      # Optional
        output_path="generated_music.wav"
    ) 