#!/usr/bin/env bash
# Extract GraSP tarballs in-place.
#
# The paper's pipeline (and ours) uses 1 fps everywhere. The 30 fps case
# archives can be very large and are NOT needed. They are skipped by default.
# Pass --with-30fps to opt in.
#
# Usage:
#   bash tools/extract_grasp.sh [GRASP_ROOT] [--with-30fps]
#
# Default GRASP_ROOT: ./GraSP (relative to CWD).

set -euo pipefail

GRASP_ROOT="GraSP"
DO_30FPS=0
for arg in "$@"; do
    case "$arg" in
        --with-30fps) DO_30FPS=1 ;;
        *) GRASP_ROOT="$arg" ;;
    esac
done

echo "[grasp] root       : $GRASP_ROOT"
echo "[grasp] with 30fps : $DO_30FPS"

# ---- 1 fps: frames and annotations
ONEFPS="$GRASP_ROOT/GraSP_1fps"
if [ -d "$ONEFPS" ]; then
    echo "[grasp] extracting 1 fps tarballs (this may take a while)"
    for f in "$ONEFPS/annotations.tar.gz" "$ONEFPS/frames.tar.gz"; do
        if [ -f "$f" ]; then
            # crude idempotency: if the corresponding directory already exists and is non-empty, skip
            base_name="$(basename "$f" .tar.gz)"
            target_dir="$ONEFPS/$base_name"
            if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null || true)" ]; then
                echo "  [skip] $base_name already extracted at $target_dir"
                continue
            fi
            echo "  tar xzf $f  -->  $ONEFPS"
            tar xzf "$f" -C "$ONEFPS"
        fi
    done
else
    echo "[grasp] skipping 1 fps (dir not found)"
fi

# ---- 30 fps: opt-in only
if [ "$DO_30FPS" -eq 1 ]; then
    THIRTYFPS="$GRASP_ROOT/GraSP_30fps/frames"
    if [ -d "$THIRTYFPS" ]; then
        echo "[grasp] extracting 30 fps case tarballs into $THIRTYFPS/<CASE>/"
        for f in "$THIRTYFPS"/*.tar.gz; do
            [ -e "$f" ] || continue
            case_name="$(basename "$f" .tar.gz)"
            target="$THIRTYFPS/$case_name"
            if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null || true)" ]; then
                echo "  [skip] $case_name already extracted"
                continue
            fi
            echo "  tar xzf $f  -->  $target"
            mkdir -p "$target"
            tar xzf "$f" -C "$target"
        done
    else
        echo "[grasp] skipping 30 fps (dir not found)"
    fi
else
    echo "[grasp] 30 fps extraction skipped (use --with-30fps to opt in; not needed for our pipeline)"
fi

echo "[grasp] done. Suggested next step:"
echo "        python tools/peek_labels.py --root ."
