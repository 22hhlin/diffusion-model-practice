import os
import json
import time
import uuid
import hashlib
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
HISTORY_DIR = os.path.join(DATA_DIR, 'history')


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _load_users() -> dict:
    _ensure_dirs()
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    _ensure_dirs()
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


def register(username: str, password: str) -> dict:
    users = _load_users()
    if username in users:
        raise ValueError("用户名已存在")

    salt = uuid.uuid4().hex[:16]
    user = {
        "username": username,
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "created_at": time.time(),
    }
    users[username] = user
    _save_users(users)

    user_dir = os.path.join(HISTORY_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    return {"username": username, "created_at": user["created_at"]}


def login(username: str, password: str) -> Optional[dict]:
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if _hash_password(password, user["salt"]) != user["password_hash"]:
        return None
    return {"username": username, "created_at": user["created_at"]}


def save_history(username: str, record: dict) -> dict:
    _ensure_dirs()
    user_dir = os.path.join(HISTORY_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    record["id"] = uuid.uuid4().hex[:12]
    record["timestamp"] = time.time()

    history_file = os.path.join(user_dir, "history.json")
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)

    history.insert(0, record)
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return record


def get_history(username: str, limit: int = 50, offset: int = 0) -> list:
    history_file = os.path.join(HISTORY_DIR, username, "history.json")
    if not os.path.exists(history_file):
        return []
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history[offset:offset + limit]


def delete_history_item(username: str, item_id: str) -> bool:
    history_file = os.path.join(HISTORY_DIR, username, "history.json")
    if not os.path.exists(history_file):
        return False
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    new_history = [h for h in history if h.get("id") != item_id]
    if len(new_history) == len(history):
        return False
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(new_history, f, ensure_ascii=False, indent=2)
    return True
