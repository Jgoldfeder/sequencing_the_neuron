#!/bin/bash
# Snapshot _fast__*.pt dumps from the three running mergedbest512 runs before
# run.py deletes them after the staged solve. Copies land in recon/rescued/.
RECON=/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon
DEST=$RECON/rescued
mkdir -p "$DEST"
TAGS="784x128x64x10 784x128x64x32x10 784x128x80x40x32x10"

while true; do
  all_done=1
  for tag in $TAGS; do
    f="$RECON/_fast__${tag}__s0.pt"
    d="$DEST/_fast__${tag}__s0.pt"
    [ -f "$d" ] && continue
    all_done=0
    if [ -f "$f" ]; then
      s1=$(stat -c%s "$f" 2>/dev/null) || continue
      sleep 3
      s2=$(stat -c%s "$f" 2>/dev/null) || continue
      if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then
        cp "$f" "$d.tmp" && mv "$d.tmp" "$d" && echo "[rescued] $d ($s1 bytes) $(date)"
      fi
    fi
  done
  [ "$all_done" = 1 ] && { echo "[watcher] all dumps rescued, exiting"; exit 0; }
  # stop if no mergedbest512 runs remain (nothing left to rescue)
  if ! pgrep -f "run.py --variant mergedbest512" > /dev/null; then
    echo "[watcher] no runs left, exiting"; exit 0
  fi
  sleep 3
done
