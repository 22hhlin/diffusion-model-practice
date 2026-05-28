from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.user_service import register, login

router = APIRouter()


class AuthRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register_user(req: AuthRequest):
    if len(req.username) < 2 or len(req.password) < 4:
        raise HTTPException(400, "用户名至少2位，密码至少4位")
    try:
        user = register(req.username, req.password)
        return {"ok": True, "user": user}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/login")
async def login_user(req: AuthRequest):
    user = login(req.username, req.password)
    if not user:
        raise HTTPException(401, "用户名或密码错误")
    return {"ok": True, "user": user}
