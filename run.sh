#!/usr/bin/env bash
# Run the Go2 object-tracking simulation.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/.venv312/bin/python3.12" "$SCRIPT_DIR/main.py"
