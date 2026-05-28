#!/bin/bash
# SD LoRA Web 一键启动脚本

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== SD LoRA Web 启动 ==="

# 启动后端
echo "启动后端 (FastAPI)..."
cd "$SCRIPT_DIR/backend"
pip install -q fastapi uvicorn[standard] websockets python-multipart 2>/dev/null
uvicorn main:app --host 0.0.0.0 --port 7860 &
BACKEND_PID=$!

# 启动前端
echo "启动前端 (Vite)..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi
npx vite --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "=== 启动完成 ==="
echo "前端: http://localhost:5173"
echo "后端: http://localhost:7860"
echo "API文档: http://localhost:7860/docs"
echo ""
echo "DSW用户请使用代理地址访问"
echo "按 Ctrl+C 停止"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
