import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sd_inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sd_finetune'))

from routers import auth, generate, history
from services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    model_service.unload()


app = FastAPI(title="SD LoRA Web", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/history", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="data/history"), name="outputs")

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(generate.router, prefix="/api/generate", tags=["generate"])
app.include_router(history.router, prefix="/api/history", tags=["history"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": model_service.is_loaded()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
