#!/bin/bash
set -euo pipefail

# Dataset preparation script with offline fallback
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/data/traffic-detection-project"
ZIP_PATH="$DATA_DIR/traffic-detection-project.zip"

echo "==> Preparing dataset..."
if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/data.yaml" ]; then
  echo "Dataset already present at: $DATA_DIR"
  exit 0
fi

mkdir -p "$DATA_DIR"

if [ -f "$ZIP_PATH" ]; then
  echo "Found archive at $ZIP_PATH. Extracting..."
  unzip -o "$ZIP_PATH" -d "$DATA_DIR" >/dev/null
  echo "Extraction complete."
  exit 0
fi

echo "No local archive found."
if [ -n "${DATA_ZIP_URL:-}" ]; then
  echo "Attempting download from DATA_ZIP_URL..."
  curl -L -o "$ZIP_PATH" "$DATA_ZIP_URL" || true
fi

# Fallback: attempt Kaggle dataset direct API URL via curl (may require credentials)
if [ ! -f "$ZIP_PATH" ]; then
  echo "Attempting download from Kaggle via curl..."
  echo "curl -L -o \"$ZIP_PATH\" https://www.kaggle.com/api/v1/datasets/download/yusufberksardoan/traffic-detection-project"
  curl -L -o "$ZIP_PATH" "https://www.kaggle.com/api/v1/datasets/download/yusufberksardoan/traffic-detection-project" || true
fi

if [ -f "$ZIP_PATH" ]; then
  echo "Unzipping downloaded archive..."
  unzip -o "$ZIP_PATH" -d "$DATA_DIR" >/dev/null
  echo "Download and extraction complete."
  exit 0
fi

echo "Dataset not found."
echo "Provide the archive via one of the following options:"
echo "  1) Manually place: $ZIP_PATH and re-run"
echo "  2) Set DATA_ZIP_URL to a direct .zip URL and re-run (e.g., export DATA_ZIP_URL=...)"
echo "  3) Bind-mount your dataset directory to /app/data when running Docker"
exit 1
