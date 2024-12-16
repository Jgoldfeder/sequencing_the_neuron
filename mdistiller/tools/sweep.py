import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import wandb

cudnn.benchmark = True

import sys
sys.path.append('..')

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from mdistiller.tools import main_train

wandb.login()

def main():
    wandb.init(project="CDD-sweep")
    cfg.CD.LR = wandb.config.lr
    cfg.CD.EPOCHS = wandb.config.epochs
    cfg.CD.PROB = wandb.config.prob
    main_train(cfg, False, None)

# Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "best_acc"},
    "parameters": {
        "lr": {"max": 0.1, "min": 0.001},
        "epochs": {"max": 3, "min":1},
        "prob": {"max": 1.0, "min":0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 100,
        "eta": 50,
    },
}

# Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="CDD-sweep")
cfg.merge_from_file("configs/cifar100/CDD_sweep.yaml")
wandb.agent(sweep_id, function=main, count=200)
