#!/bin/bash
# Move a big folder from home to scratch and leave a symlink at the old path (user rule 2026-10-30: big files belong on scratch).
#   move_to_scratch.sh <path under /home/d250009> [<path> ...]
# The target mirrors the path under /scratch/d250009/pi05-handover (projects/pi05-handover/runs/X -> runs/X); anything else goes to
# /scratch/d250009/home-moved/<relative path>. rsync removes each source file after it is copied, the symlink is made only when
# the copy finished, so an interrupted move can be rerun. Never run it on a folder that a running job reads or writes.
set -uo pipefail
for src in "$@"; do
  src=${src%/}
  [[ -L "$src" ]] && { echo "$src is already a symlink to $(readlink "$src")"; continue; }
  [[ -d "$src" ]] || { echo "$src: not a directory" >&2; continue; }
  case "$src" in
    /home/d250009/projects/pi05-handover/*) dst=/scratch/d250009/pi05-handover/${src#/home/d250009/projects/pi05-handover/} ;;
    /home/d250009/*) dst=/scratch/d250009/home-moved/${src#/home/d250009/} ;;
    *) echo "$src: not under /home/d250009" >&2; continue ;;
  esac
  mkdir -p "$dst" && echo "$(date -Is) moving $src -> $dst" || continue
  if rsync -a --remove-source-files "$src/" "$dst/"; then
    find "$src" -depth -type d -empty -delete
    if [[ -e "$src" ]]; then echo "$(date -Is) $src still has files (not replaced by a symlink)"; continue; fi
    ln -s "$dst" "$src" && echo "$(date -Is) done $src -> $dst"
  else
    echo "$(date -Is) rsync failed for $src" >&2
  fi
done
