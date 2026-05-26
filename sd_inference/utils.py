"""
Shared utilities for SD inference scripts.
Supports loading models from ModelScope (default) or HuggingFace.
"""
import os


def get_model_path(model_id, use_modelscope=True):
    """Download model from ModelScope or return HuggingFace model ID.

    Args:
        model_id: Model identifier (e.g., 'AI-ModelScope/stable-diffusion-v1-5')
        use_modelscope: If True, download from ModelScope; if False, use HuggingFace

    Returns:
        Local path (ModelScope) or model_id (HuggingFace)
    """
    if use_modelscope:
        from modelscope import snapshot_download
        print(f"Downloading from ModelScope: {model_id}")
        local_path = snapshot_download(model_id)
        print(f"Cached at: {local_path}")
        return local_path
    return model_id


# Default model IDs
SD_V15_MODELSCOPE = 'AI-ModelScope/stable-diffusion-v1-5'
SD_V15_HF = 'runwayml/stable-diffusion-v1-5'
SD_INPAINT_MODELSCOPE = 'AI-ModelScope/stable-diffusion-inpainting'
SD_INPAINT_HF = 'runwayml/stable-diffusion-inpainting'
CONTROLNET_CANNY_MODELSCOPE = 'AI-ModelScope/sd-controlnet-canny'
CONTROLNET_CANNY_HF = 'lllyasviel/sd-controlnet-canny'
