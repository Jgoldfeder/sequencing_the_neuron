#!/bin/bash
# Clean end-to-end tanh demonstrations at every depth, via solve_tanh.py. Two workers share one task list
# with lock files: worker "gpu0" runs on GPU 0 in priority order; worker "opp" claims GPU 2 or 1 only when
# that GPU has had no other process for two consecutive checks (Judah's LeNet runs own them now).
# Usage: demo_schedule.sh gpu0 | opp
cd /home/judah/icml/philippe/SequencingTheNeurome_Revision_extracted/code
P=/opt/miniconda3/bin/python; LK=results/logs/demo
S="--refine-cg 2000 --refine-iters 8 --polish-cg 2000 --polish-iters 6 --final-cg 2000 --final-iters 4"
declare -A ARCH=( [5L]=3072,200,200,200,100 [6L]=3072,200,200,200,200,100 [8L]=3072,200,200,200,200,200,200,100
                  [9L]=3072,200,200,200,200,200,200,200,100 [10L]=3072,200,200,200,200,200,200,200,200,100
                  [11L]=3072,200,200,200,200,200,200,200,200,200,100 )
declare -A OPTS=( [5L]="--q 48000 --window 30" [6L]="--q 96000 --window 30"
                  [8L]="--q 300000 --window 30 --disk rings/demo_8L --disk-ram-gb 50"
                  [9L]="--q 800000 --window 20 --epochs 5 --disk rings/demo_9L --disk-ram-gb 50 --qg-block-gb 1.61 --qg-chunk-gb 6.5"
                  [10L]="--q 1200000 --window 20 --epochs 5 --disk rings/demo_10L --disk-ram-gb 50 --qg-block-gb 1.61 --qg-chunk-gb 6.5"
                  [11L]="--q 1800000 --window 20 --epochs 5 --disk rings/demo_11L --disk-ram-gb 50 --qg-block-gb 1.61 --qg-chunk-gb 6.5" )
declare -A NEEDGB=( [5L]=5 [6L]=5 [8L]=130 [9L]=240 [10L]=330 [11L]=480 )   # disk space needed (ring + flush)
run_task() {  # $1 task $2 gpu
  t=$1; g=$2
  if ! mkdir "$LK/$t.lock" 2>/dev/null; then return 1; fi
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt "${NEEDGB[$t]}" ]; then echo "$(date) skip $t: need ${NEEDGB[$t]} GB disk, have $free_gb" >> $LK/schedule.log; rmdir "$LK/$t.lock"; return 1; fi
  echo "$(date) START $t on gpu $g" >> $LK/schedule.log
  $P solve_tanh.py --arch ${ARCH[$t]} --tag demo_$t --gpus $g ${OPTS[$t]} $S > results/logs/solve_tanh__demo_$t.log 2>&1
  echo "$(date) END $t on gpu $g (exit $?)" >> $LK/schedule.log; touch "$LK/$t.done"
  return 0
}
gpu_busy() {  # other processes on gpu index $1 ?
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$1 '$1==i{print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$uuid"
}
if [ "$1" = "gpu0" ]; then
  for t in 5L 11L 8L 6L 10L 9L; do [ -e "$LK/$t.done" ] && continue; run_task $t 0; done
else
  while true; do
    for g in 2 1; do
      if ! gpu_busy $g; then sleep 300; if ! gpu_busy $g; then
        for t in 6L 8L 10L 9L; do [ -e "$LK/$t.done" ] && continue; run_task $t $g && break; done
      fi; fi
    done
    alldone=1; for t in 6L 8L 10L 9L; do [ -e "$LK/$t.done" ] || alldone=0; done; [ $alldone = 1 ] && exit 0
    sleep 300
  done
fi
