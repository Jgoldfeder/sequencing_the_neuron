import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('../reconstruction')
sys.path.append('./reconstruction')
from util import Population

from standardize_align_new import Standardizer, SingleTransformerEncoderStandardizer

from models import base_CNN, two_CNN, var_CNN, base_RNN, var_RNN, base_TransformerEncoder
from test_utils import *

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
np.set_printoptions(precision=3, suppress=True, linewidth=200, formatter={'float_kind': lambda x: f"{x:.3f}"})

# blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_20_CNNx1-3-3-2x3-3-3-2_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_bothalign_bothredist/black_box.pt"
# final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_20_CNNx1-3-3-2x3-3-3-2_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_bothalign_bothredist/population_iteration_54.pt"
# layer_string = "CNNx1-3-3-2x3-3-3-2"
# best_model_index = 2 #needs to be manually inspected from log file

# blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_23_CNNx1-10-3-1x10-5-3-2x5-3-3-2_outer_iterations_80_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_bothalign_bothredist/black_box.pt"
# final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_23_CNNx1-10-3-1x10-5-3-2x5-3-3-2_outer_iterations_80_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_bothalign_bothredist/population_iteration_53.pt"
# layer_string = "CNNx1-10-3-1x10-5-3-2x5-3-3-2"
# best_model_index = 8 #needs to be manually inspected from log file

blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_24_CNNx1-40-3-1x40-20-3-1x20-10-3-2x10-3-3-2_outer_iterations_50_num_samples_40000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_newalign/black_box.pt"
final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_24_CNNx1-40-3-1x40-20-3-1x20-10-3-2x10-3-3-2_outer_iterations_50_num_samples_40000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_newalign/population_iteration_49.pt"
layer_string = "CNNx1-40-3-1x40-20-3-1x20-10-3-2x10-3-3-2"
best_model_index = 1 #needs to be manually inspected from log file


input_shape = (1, 28, 28)
model_type = 'cnn'
activation_f = nn.LeakyReLU()

layer_dim = layer_string.split('x')
layer_configs = []
for layer in layer_dim[1:]:
	layer = layer.split('-')
	layer_configs.append({'in_channels': int(layer[0]), 'out_channels': int(layer[1]), 'kernel_size': int(layer[2]), 'stride': int(layer[3])})

blackbox_dict = torch.load(blackbox_dict_path)
final_population_dict = torch.load(final_population_dict_path)

blackbox = var_CNN(input_shape, layer_configs, activation_f)
blackbox.load_state_dict(blackbox_dict)
subs = [var_CNN(input_shape, layer_configs, activation_f) for i in range(10)]
final_population = Population(subs=subs)
final_population.load_state_dict(final_population_dict)
final_population.best = best_model_index
best_model = final_population.subs[best_model_index]

param_count = sum(p.numel() for p in best_model.parameters())
print("Best model param count: ", param_count)
print()

final_population.evaluate(blackbox, model_type='cnn')
print()

