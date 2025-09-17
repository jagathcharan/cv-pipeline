#!/bin/bash
set -euo pipefail

# Optional dataset download (idempotent)
./download.sh || true

OUTPUTS_DIR="/app/data/outputs"
BEST_WEIGHTS="$OUTPUTS_DIR/yolo_train/weights/best.pt"
METRICS_JSON="$OUTPUTS_DIR/yolo_eval/metrics.json"
MARKER_DIR="/app/data/.markers"
IMAGE_BUILD_ID_FILE="/app/.build-id"
VOLUME_BUILD_ID_FILE="$MARKER_DIR/build-id"

mkdir -p "$MARKER_DIR"

# Detect image rebuilds to force a fresh pipeline once per image build
IMAGE_BUILD_ID="$(cat "$IMAGE_BUILD_ID_FILE" 2>/dev/null || echo unknown)"
VOLUME_BUILD_ID="$(cat "$VOLUME_BUILD_ID_FILE" 2>/dev/null || echo none)"

if [ "$IMAGE_BUILD_ID" != "$VOLUME_BUILD_ID" ]; then
  echo "[startup] Detected new image build ($IMAGE_BUILD_ID != $VOLUME_BUILD_ID). Clearing previous outputs and running pipeline..."
  rm -rf "$OUTPUTS_DIR" || true
fi

# Run the pipeline once if outputs are missing
if [ ! -f "$BEST_WEIGHTS" ] || [ ! -f "$METRICS_JSON" ]; then
  echo "[startup] Running one-time pipeline to produce training and evaluation outputs..."
  python src/pipeline.py \
    --data_yaml /app/data/traffic-detection-project/data.yaml \
    --project /app/data/outputs \
    --train_name yolo_train \
    --eval_name yolo_eval \
    --imgsz 416 || true
  # Record build id after successful pipeline attempt (even if partial) to avoid loops
  echo "$IMAGE_BUILD_ID" > "$VOLUME_BUILD_ID_FILE" || true
else
  echo "[startup] Found existing outputs. Skipping pipeline."
fi

exec streamlit run src/dashboard.py --server.port=8501 --server.address=0.0.0.0


