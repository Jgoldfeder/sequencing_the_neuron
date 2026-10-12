#!/bin/bash
# Independent-committee reruns: SAME black box (teacher seed 0, cached teacher files), population seed 1.
# Results/recon names carry __s1 (seed) and the tag, so nothing from the seed-0 runs is overwritten.
cd /home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code
PY=/opt/miniconda3/bin/python
COMMON="--variant mergedbest512 --outer 30 --q 48000 --window 30 --device cuda --teacher-seed 0 --seed 1 --layer-consensus --align-inverse --log-every 1 --pop-save-every 1 --tag seed1"
POP_SAVE_ALL=1 CUDA_VISIBLE_DEVICES=1 $PY run.py $COMMON --arch 3072,200,200,200,100     > results/logs/mergedbest512_seed1__3072x200x200x200x100__s1.log 2>&1 &
POP_SAVE_ALL=1 CUDA_VISIBLE_DEVICES=2 $PY run.py $COMMON --arch 3072,200,200,200,200,100 > results/logs/mergedbest512_seed1__3072x200x200x200x200x100__s1.log 2>&1 &
wait
echo SEED1DONE
