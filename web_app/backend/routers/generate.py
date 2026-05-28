import base64
import time
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional

from services.model_service import model_service
from services.user_service import save_history

router = APIRouter()


class Txt2ImgRequest(BaseModel):
    username: str
    prompt: str
    negative_prompt: str = "low quality, blurry"
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1
    width: int = 512
    height: int = 512


class Img2ImgRequest(BaseModel):
    username: str
    prompt: str
    image_base64: str
    negative_prompt: str = "low quality, blurry"
    strength: float = 0.75
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1


class BatchRequest(BaseModel):
    username: str
    prompts: list[str]
    negative_prompt: str = "low quality, blurry"
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    width: int = 512
    height: int = 512


@router.post("/txt2img")
async def txt2img(req: Txt2ImgRequest):
    start = time.time()
    try:
        images_b64 = await model_service.txt2img(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            num_images=req.num_images,
            width=req.width,
            height=req.height,
        )
    except Exception as e:
        raise HTTPException(500, f"生成失败: {str(e)}")

    elapsed = round(time.time() - start, 2)
    record = save_history(req.username, {
        "type": "txt2img",
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "params": {
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "width": req.width,
            "height": req.height,
        },
        "images": images_b64,
        "elapsed": elapsed,
    })

    return {"ok": True, "images": images_b64, "elapsed": elapsed, "id": record["id"]}


@router.post("/img2img")
async def img2img(req: Img2ImgRequest):
    start = time.time()
    try:
        images_b64 = await model_service.img2img(
            prompt=req.prompt,
            image_base64=req.image_base64,
            negative_prompt=req.negative_prompt,
            strength=req.strength,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            num_images=req.num_images,
        )
    except Exception as e:
        raise HTTPException(500, f"生成失败: {str(e)}")

    elapsed = round(time.time() - start, 2)
    record = save_history(req.username, {
        "type": "img2img",
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "params": {
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "strength": req.strength,
        },
        "init_image": req.image_base64,
        "images": images_b64,
        "elapsed": elapsed,
    })

    return {"ok": True, "images": images_b64, "elapsed": elapsed, "id": record["id"]}


@router.post("/batch")
async def batch_generate(req: BatchRequest):
    start = time.time()
    results = []
    for i, prompt in enumerate(req.prompts):
        try:
            images_b64 = await model_service.txt2img(
                prompt=prompt,
                negative_prompt=req.negative_prompt,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                num_images=1,
                width=req.width,
                height=req.height,
            )
            results.append({"ok": True, "images": images_b64, "prompt": prompt})
        except Exception as e:
            results.append({"ok": False, "error": str(e), "prompt": prompt})

        save_history(req.username, {
            "type": "batch",
            "prompt": prompt,
            "negative_prompt": req.negative_prompt,
            "params": {
                "steps": req.steps,
                "guidance_scale": req.guidance_scale,
                "seed": req.seed,
                "width": req.width,
                "height": req.height,
            },
            "images": results[-1].get("images", []),
            "elapsed": 0,
        })

    elapsed = round(time.time() - start, 2)
    return {"ok": True, "results": results, "elapsed": elapsed}


@router.websocket("/ws/{username}")
async def ws_generate(websocket: WebSocket, username: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "txt2img":
                await websocket.send_json({"type": "generating"})
                images = await model_service.txt2img(
                    prompt=data["prompt"],
                    negative_prompt=data.get("negative_prompt", "low quality, blurry"),
                    steps=data.get("steps", 30),
                    guidance_scale=data.get("guidance_scale", 7.5),
                    seed=data.get("seed"),
                    num_images=data.get("num_images", 1),
                    width=data.get("width", 512),
                    height=data.get("height", 512),
                )
                await websocket.send_json({"type": "done", "images": images})

    except WebSocketDisconnect:
        pass
