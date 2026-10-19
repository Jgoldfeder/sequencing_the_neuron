#!/usr/bin/env bash
# One-shot setup + LeNet no-peel committee run on a fresh Linux box with an NVIDIA GPU.
#
#   bash setup_new_machine.sh                # clone, install, tune the sampler flags, launch the run
#   GPU=1 Q=200000 bash setup_new_machine.sh # pick the GPU / queries per iteration
#   SKIP_TUNE=1 bash setup_new_machine.sh    # use the default --qg-chunk-gb 6 --qg-block-gb 2 (24 GB cards)
#
# Needs: git with SSH access to github.com/Jgoldfeder/sequencing_the_neuron, python >= 3.10, an NVIDIA driver.
# The LeNet teacher (teachers/teacher_cnn_1x28x28__..._leaky_relu.pt) is tracked in the repo so every machine
# attacks the SAME network; MNIST is downloaded on first use by torchvision.
set -euo pipefail
GPU="${GPU:-0}"; Q="${Q:-200000}"; TAG="${TAG:-lenet_nopeel_q200k_$(hostname -s)}"
BRANCH="${BRANCH:-october2026}"; DIR="${DIR:-$HOME/sequencing_the_neuron}"

if [ ! -d "$DIR/.git" ]; then
    git clone -b "$BRANCH" git@github.com:Jgoldfeder/sequencing_the_neuron.git "$DIR"
fi
cd "$DIR"; git checkout -q "$BRANCH"; git pull -q

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
# torch 2.7 / torchvision 0.22 were the versions the October runs used (CUDA 12.6 wheels)
pip install -q "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu126
pip install -q "numpy>=2" "scipy>=1.13" threadpoolctl
mkdir -p recon pool_logs teachers
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

FLAGS="--qg-chunk-gb 6 --qg-block-gb 2"
if [ -z "${SKIP_TUNE:-}" ]; then
    echo "== qg_tune: sampler block/chunk sweep for this GPU (a few minutes)"
    CUDA_VISIBLE_DEVICES="$GPU" python qg_tune.py --conv lenet --reps 1 | tee pool_logs/qg_tune_lenet.log
    REC=$(grep "^RECOMMENDED:" pool_logs/qg_tune_lenet.log | sed 's/^RECOMMENDED: //')
    if [ -n "$REC" ]; then FLAGS="$REC"; fi
fi
echo "== sampler flags: $FLAGS"

echo "== launching $TAG on GPU $GPU (log: pool_logs/$TAG.log)"
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python run.py \
  --variant mergedbest --conv lenet --cnn-act leaky_relu --device cuda \
  --outer 40 --window 20 --q "$Q" $FLAGS --disk-seg-gb 2 \
  --cluster-eps 0.1 --cluster-eps-l1 0.02 --masked-consensus --layer-consensus \
  --balance 1 --balance-refresh 5 \
  --log-every 1 --pop-save-every 5 --tag "$TAG" > "pool_logs/$TAG.log" 2>&1 &
echo "pid $!"; echo "follow with:  tail -f $DIR/pool_logs/$TAG.log"
