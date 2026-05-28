import os
import sys
import gc
import torch
import base64
import asyncio
from io import BytesIO
from typing import Optional, Callable
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sd_inference'))
from utils import get_model_path, SD_V15_MODELSCOPE


class ModelService:
    def __init__(self):
        self.pipe_txt2img: Optional[StableDiffusionPipeline] = None
        self.pipe_img2img: Optional[StableDiffusionImg2ImgPipeline] = None
        self.lora_path: Optional[str] = None
        self._lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.pipe_txt2img is not None

    def load(self, lora_path: str = "checkpoints/lora/final"):
        if self.pipe_txt2img is not None and self.lora_path == lora_path:
            return

        self.unload()

        model_path = get_model_path(SD_V15_MODELSCOPE)
        print(f"Loading SD model from: {model_path}")

        self.pipe_txt2img = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, safety_checker=None
        )

        if os.path.isdir(lora_path):
            print(f"Loading LoRA from: {lora_path}")
            self.pipe_txt2img.unet.load_adapter(lora_path)
            self.lora_path = lora_path

        self.pipe_txt2img = self.pipe_txt2img.to("cuda")
        self.pipe_txt2img.enable_attention_slicing()

        self.pipe_img2img = StableDiffusionImg2ImgPipeline(
            vae=self.pipe_txt2img.vae,
            text_encoder=self.pipe_txt2img.text_encoder,
            tokenizer=self.pipe_txt2img.tokenizer,
            unet=self.pipe_txt2img.unet,
            scheduler=self.pipe_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        print("Model loaded successfully!")

    def unload(self):
        self.pipe_txt2img = None
        self.pipe_img2img = None
        self.lora_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_generator(self, seed: Optional[int] = None):
        if seed is not None:
            return torch.Generator("cuda").manual_seed(seed)
        return None

    def _image_to_base64(self, image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        progress_callback: Optional[Callable] = None,
    ) -> list[str]:
        async with self._lock:
            if not self.is_loaded():
                self.load()

            def step_callback(pipe, step, timestep, callback_kwargs):
                if progress_callback:
                    asyncio.run_coroutine_threadsafe(
                        progress_callback(step, steps),
                        asyncio.get_event_loop(),
                    )
                return callback_kwargs

            generator = self._make_generator(seed)
            images = self.pipe_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                width=width,
                height=height,
                callback_on_step_end=step_callback if progress_callback else None,
            ).images

            return [self._image_to_base64(img) for img in images]

    async def img2img(
        self,
        prompt: str,
        image_base64: str,
        negative_prompt: str = "low quality, blurry",
        strength: float = 0.75,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        progress_callback: Optional[Callable] = None,
    ) -> list[str]:
        async with self._lock:
            if not self.is_loaded():
                self.load()

            from PIL import Image
            image_data = base64.b64decode(image_base64)
            init_image = Image.open(BytesIO(image_data)).convert("RGB")
            init_image = init_image.resize((512, 512))

            generator = self._make_generator(seed)
            images = self.pipe_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            ).images

            return [self._image_to_base64(img) for img in images]


model_service = ModelService()
