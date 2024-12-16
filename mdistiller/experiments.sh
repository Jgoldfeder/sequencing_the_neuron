# Run baseline
python3 tools/train.py --cfg configs/cifar100/CDD_new_baseline.yaml > CDD_baselines.out 2>&1 &

# Run 10 epochs trial
python3 tools/train.py --cfg configs/cifar100/CDD_10_epochs.yaml > CDD_10.out 2>&1 &

# Run 50 epochs trial
python3 tools/train.py --cfg configs/cifar100/CDD_50_epochs.yaml > CDD_50.out 2>&1 &

