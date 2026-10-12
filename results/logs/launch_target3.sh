#!/bin/bash
cd /home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code
PY=/opt/miniconda3/bin/python
SNAP=recon/_pop__mergedbest512_seed1__3072x200x200x200x200x100__s1_it020.pt
COMMON="--variant mergedbest512 --arch 3072,200,200,200,200,100 --outer 10 --q 48000 --window 30 --device cuda --teacher-seed 0 --seed 1 --layer-consensus --align-inverse --log-every 1 --pop-save-every 1 --resume-pop $SNAP"
POP_SAVE_ALL=1 CUDA_VISIBLE_DEVICES=1 $PY run.py $COMMON --target-mode pin --tag target_pin > results/logs/target_pin__6layer_s1_it20.log 2>&1
echo TARGET3DONE
