#!/bin/bash
# 5-layer, REST-ERROR-ZERO control: 8 members = exact teacher; the 4 units (L2 121; L3 3,43,117) forced as targets, re-drawn, then the SAME targeted-sampler protocol (output disagreement + activation of the unsolved units, sticky pins)
cd /home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code
PY=/opt/miniconda3/bin/python
SNAP=recon/_pop__teacherexact__3072x200x200x200x100__s0.pt
COMMON="--variant mergedbest512 --arch 3072,200,200,200,100 --outer 10 --q 48000 --window 30 --device cuda --teacher-seed 0 --seed 0 --layer-consensus --align-inverse --log-every 1 --pop-save-every 1 --resume-pop $SNAP --target-frac 1.0"
POP_SAVE_ALL=1 CUDA_VISIBLE_DEVICES=2 $PY run.py $COMMON --target-mode pin --target-reinit --target-units "2:121;3:3,43,117" --target-act-w 1.0 --tag target_exact5 > results/logs/target_exact__5layer_s0.log 2>&1
echo EXACT5DONE
