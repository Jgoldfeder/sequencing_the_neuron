#!/bin/bash
# copies every new version of every 10K run checkpoint to ckpt_hist/10K_<tag>_itNNN.pt (runs until killed)
P=/home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code/peel
while true; do
  for f in $P/recon_gpt__gaussian__p8__s0__10K_*.pt; do
    n=$(basename $f .pt); n=${n#recon_gpt__gaussian__p8__s0__10K_}
    it=$(python -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['iter'])" 2>/dev/null) || continue
    out=$P/ckpt_hist/10K_${n}_it$(printf %03d $it).pt
    [ -f "$out" ] || cp "$f" "$out"
  done
  sleep 60
done
