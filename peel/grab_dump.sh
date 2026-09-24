#!/bin/bash
# Copy the iter-35 _fast dump the instant it appears, before the staged solve
# deletes it. Poll fast (1s) since the dump lives only during the solve.
SRC=/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/recon/_fast__3072x256x100__s0.pt
DST=/tmp/claude-1001/-home-judah/480010f4-8e92-4a50-8319-7f2f544ea513/scratchpad/grabbed_3072x256x100.pt
for i in $(seq 1 900); do
  if [ -f "$DST" ]; then echo "[grab] already have $DST"; exit 0; fi
  if [ -f "$SRC" ]; then
    s1=$(stat -c%s "$SRC" 2>/dev/null) || { sleep 1; continue; }
    sleep 1
    s2=$(stat -c%s "$SRC" 2>/dev/null) || continue
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 1000000 ]; then
      cp "$SRC" "$DST.tmp" && mv "$DST.tmp" "$DST" && echo "[grab] captured $DST ($s1 bytes)" && exit 0
    fi
  fi
  sleep 1
done
echo "[grab] timed out without seeing dump"; exit 1
