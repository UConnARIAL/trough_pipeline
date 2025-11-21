#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG (edit these paths as needed) ---
PY=python
SCRIPT="ocean_filter_make_new_masks.py"

WATER_SHP="../water_mask/water_3413_clip.gpkg"
MASKED_ROOT="/scratch2/projects/PDG_shared/AlaskaTundraMosaicMasks/"            # where 43_19, 44_19, ... live
OUT_ROOT="./ocean_masked"   # we will create OUT_ROOT/<tile_id>/
THRESHOLD="0.99"
BUFFER_M="500"
SIMPLIFY_M="2"
MAX_WORKERS="1"
LOG_EVERY="1"

# --- USAGE ---
# 1) Pass tile IDs as args:
#    ./run_ocean_filter_by_tile.sh 43_19 44_19
# 2) Or pass a file with one tile_id per line:
#    ./run_ocean_filter_by_tile.sh --tiles-file tile_ids.txt

# --- parse args ---
TILES=()
if [[ "${1-}" == "--tiles-file" ]]; then
  [[ $# -ge 2 ]] || { echo "Need a file after --tiles-file"; exit 1; }
  mapfile -t TILES < "$2"
else
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <tile_id> [<tile_id> ...]   OR   $0 --tiles-file tile_ids.txt"
    exit 1
  fi
  TILES=("$@")
fi

# --- sanity checks ---
[[ -f "$SCRIPT" ]] || { echo "Cannot find $SCRIPT"; exit 1; }
[[ -f "$WATER_SHP" ]] || { echo "Cannot find water vector: $WATER_SHP"; exit 1; }

mkdir -p "$OUT_ROOT"

# Let globs expand to empty rather than literal if no matches
shopt -s nullglob

for TILE in "${TILES[@]}"; do
  TILE="$(echo "$TILE" | xargs)"   # trim
  [[ -n "$TILE" ]] || continue

  # Check there are inputs
  IN_PATTERN=${MASKED_ROOT}/${TILE}/*.tif
  FOUND=( $IN_PATTERN )
  if [[ ${#FOUND[@]} -eq 0 ]]; then
    echo "[SKIP] No TIFFs found for ${TILE} under ${MASKED_ROOT}/${TILE}/"
    continue
  fi

  OUT_DIR="${OUT_ROOT}/${TILE}"
  mkdir -p "$OUT_DIR"

  OUT_MASKS_DIR="$OUT_DIR"                   # put new masks directly under tile folder
  OUT_CSV="$OUT_DIR/ocean_filter.csv"
  OUT_DROP="$OUT_DIR/drop_tiles.txt"
  OUT_BOUNDARY="$OUT_DIR/boundary_tiles.txt"
  LOG="$OUT_DIR/run.log"

  echo "==> ${TILE}: writing outputs to ${OUT_DIR}"
  # Run and tee both stdout+stderr to the log (no process substitution)
  set -x
  $PY "$SCRIPT" \
    --water-shp "$WATER_SHP" \
    --tifs "${MASKED_ROOT}/${TILE}/*.tif" \
    --ocean-threshold "$THRESHOLD" \
    --buffer-m "$BUFFER_M" \
    --out-masks-dir "$OUT_MASKS_DIR" \
    --out-csv "$OUT_CSV" \
    --out-drop "$OUT_DROP" \
    --out-boundary "$OUT_BOUNDARY" \
    --max-workers "$MAX_WORKERS" \
    --simplify-m "$SIMPLIFY_M" \
    --log-every "$LOG_EVERY" \
    |& tee "$LOG"
  set +x
done

echo "All done. See per-tile outputs under: $OUT_ROOT/<tile_id>/"

