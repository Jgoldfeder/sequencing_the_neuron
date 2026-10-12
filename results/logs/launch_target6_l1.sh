#!/bin/bash
# v3 l1: lam=5 PER UNIT (exact-penalty rule), certified-only gauge, fixed reference
cd /home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code
PY=/opt/miniconda3/bin/python
SNAP=recon/_pop__mergedbest512_seed1__3072x200x200x200x200x100__s1_it020.pt
COMMON="--variant mergedbest512 --arch 3072,200,200,200,200,100 --outer 10 --q 48000 --window 30 --device cuda --teacher-seed 0 --seed 1 --layer-consensus --align-inverse --log-every 1 --pop-save-every 1 --resume-pop $SNAP --target-frac 1.0"
POP_SAVE_ALL=1 CUDA_VISIBLE_DEVICES=2 $PY run.py $COMMON --target-mode l1 --target-lam 5 --tag target_l1 > results/logs/target_l1__6layer_s1_it20.log 2>&1
echo L16DONE
