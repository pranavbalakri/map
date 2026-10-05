#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# --- Args ---
ALPHA_FLAG="--alpha"

usage() {
    echo "Usage: $0 <images_folder> [--no-alpha] [--output <file>]"
    echo "  images_folder   Path to folder containing DJI images"
    echo "  --no-alpha      Disable alpha blending for overlaps (default: on)"
    echo "  --output <file> Output map path (default: output.jpg)"
    exit 1
}

if [[ $# -lt 1 ]]; then usage; fi

FOLDER_ARG="$1"
shift

OUTPUT="output.jpg"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-alpha) ALPHA_FLAG="--no-alpha"; shift ;;
        --output)   OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

FOLDER="$(cd "$FOLDER_ARG" && pwd)"

# Derive CSV path: <folder>.csv next to the folder
FOLDER_NAME="$(basename "$FOLDER")"
CSV_PATH="${FOLDER}.csv"

# --- Python interpreter ---
if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: No Python interpreter found. Run 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt' first."
    exit 1
fi

echo "Using Python: $PYTHON"

# --- Step 1: Generate CSV if missing ---
if [[ -f "$CSV_PATH" ]]; then
    echo "CSV already exists at $CSV_PATH — skipping extraction."
else
    echo "Extracting DJI metadata from $FOLDER..."
    "$PYTHON" "$SCRIPT_DIR/dji_dataextraction.py" "$FOLDER" "$CSV_PATH"
    if [[ ! -f "$CSV_PATH" ]]; then
        echo "Error: CSV generation failed."
        exit 1
    fi
    echo "Metadata written to $CSV_PATH."
fi

# --- Step 2: Run stitching pipeline ---
echo "Running stitching pipeline..."
cd "$SCRIPT_DIR"
"$PYTHON" "$SCRIPT_DIR/with_stitch_main.py" "$FOLDER" --output "$OUTPUT" $ALPHA_FLAG
echo "Map saved to $OUTPUT"
