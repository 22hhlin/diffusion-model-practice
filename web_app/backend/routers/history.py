from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.user_service import get_history, delete_history_item

router = APIRouter()


class DeleteRequest(BaseModel):
    username: str
    item_id: str


@router.get("/{username}")
async def list_history(username: str, limit: int = 50, offset: int = 0):
    items = get_history(username, limit, offset)
    return {"ok": True, "items": items, "total": len(items)}


@router.delete("/")
async def delete_item(req: DeleteRequest):
    ok = delete_history_item(req.username, req.item_id)
    if not ok:
        raise HTTPException(404, "记录不存在")
    return {"ok": True}
