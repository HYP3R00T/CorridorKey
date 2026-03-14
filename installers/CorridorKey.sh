#!/usr/bin/env bash
# CorridorKey launcher - Linux fallback
#
# If the installer could not create a Desktop launcher automatically,
# copy this file to your Desktop and make it executable:
#   chmod +x CorridorKey.sh
#
# Then drag a clips folder onto it, or run:
#   ./CorridorKey.sh /path/to/clips

if [[ -z "${1:-}" ]]; then
    echo "[ERROR] No folder provided."
    echo ""
    echo "USAGE: Drag and drop a clips folder onto this file."
    echo "       Or run: ./CorridorKey.sh /path/to/clips"
    echo ""
    read -rp "Press Enter to exit..."
    exit 1
fi

TARGET="${1%/}"

if [[ ! -d "$TARGET" ]]; then
    echo "[ERROR] Not a directory: $TARGET"
    read -rp "Press Enter to exit..."
    exit 1
fi

if ! command -v corridorkey &>/dev/null; then
    echo "[ERROR] corridorkey is not installed."
    echo "        Run install.sh first."
    read -rp "Press Enter to exit..."
    exit 1
fi

echo "Starting CorridorKey..."
echo "Target: $TARGET"
echo ""

corridorkey wizard "$TARGET"

read -rp "Press Enter to close..."
