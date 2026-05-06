#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Starting LLM Performance Evaluator Web UI ==="

cd "$PROJECT_ROOT"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "[1/2] Starting backend API on port 5000..."
cd "$PROJECT_ROOT/web2_api"
flask run --port 5000 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

sleep 2

echo "[2/2] Starting frontend on port 5173..."
cd "$PROJECT_ROOT/web2"
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "  Backend API:  http://localhost:5000"
echo "  Frontend:     http://localhost:5173"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop both servers..."
echo ""

wait