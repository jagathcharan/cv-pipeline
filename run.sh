#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1) Prepare dataset locally (optional)
"$ROOT_DIR/download.sh" || true

# 2) Build Docker image
IMAGE="traffic-detection-analysis"
docker build -t "$IMAGE" "$ROOT_DIR"

# 3) Run Streamlit app
docker run --rm -p 8501:8501 -v "$ROOT_DIR/data:/app/data" "$IMAGE"
