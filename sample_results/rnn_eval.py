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
from eval_utils import *

torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)
np.set_printoptions(precision=3, suppress=True, linewidth=200, formatter={'float_kind': lambda x: f"{x:.3f}"})

# #hidden layer 28
# blackbox_dict_path = "rnn/seed_31_RNNx28_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_var_RNN_28/black_box.pt"
# final_population_dict_path = "rnn/seed_31_RNNx28_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_var_RNN_28/population_iteration_54.pt"
# hidden_size = 28
# best_model_index = 0 #needs to be manually inspected from log file

#hidden layre 64
# blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx64_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_do-we-solve/black_box.pt"
# final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx64_outer_iterations_55_num_samples_10000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_do-we-solve/population_iteration_54.pt"
# hidden_size = 64
# best_model_index = 0 #needs to be manually inspected from log file

# #hidden layer 128
# blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx128_outer_iterations_55_num_samples_40000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_128_evenmoresamples/black_box.pt"
# final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx128_outer_iterations_55_num_samples_40000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee_aligner_128_evenmoresamples/population_iteration_54.pt"
# hidden_size = 128
# best_model_index = 0 #needs to be manually inspected from log file

#hidden layer 128 more epochs
blackbox_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx128_80ksamples-moreepochs_outer_iterations_80_num_samples_80000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee/black_box.pt"
final_population_dict_path = "/home/elvin/nn_sequencing/sequencing_the_neuron/models/seed_40_RNNx128_80ksamples-moreepochs_outer_iterations_80_num_samples_80000_num_epochs_5_dataset_mnist_optim_adam_activation_relu_sampling_method_committee/population_iteration_69.pt"
hidden_size = 128
best_model_index = 0 #needs to be manually inspected from log file

num_std = 1

blackbox_dict = torch.load(blackbox_dict_path)
final_population_dict = torch.load(final_population_dict_path)

blackbox = var_RNN(28, [hidden_size]) #input_size, layer_configs
blackbox.load_state_dict(blackbox_dict)

subs = [var_RNN(28, [hidden_size]) for i in range(10)]
final_population = Population(subs=subs)
final_population.load_state_dict(final_population_dict)
final_population.best = best_model_index
best_model = final_population.subs[best_model_index]

param_count = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
print("Best model param count: ", param_count)

final_population.evaluate(blackbox, model_type='rnn')

# print_diff(best_model, blackbox)
print("-----------")

best_og = copy.deepcopy(best_model)
blackbox_og = copy.deepcopy(blackbox)

# test_model = var_RNN(28, [hidden_size])
# std = Standardizer(test_model)

# print_indices_unaligned(best_model, blackbox)

# for i in range(num_std):
#     best_model_std = Standardizer(best_model)
#     best_model = best_model_std.reload()
# for i in range(num_std):
#     blackbox_std = Standardizer(blackbox)
#     blackbox = blackbox_std.reload()
best_model_std = Standardizer(best_model)
blackbox_std = Standardizer(blackbox)

# print("*************************")
# print("*************************")

best_model_std.rnn_align(blackbox_std)
best_model = best_model_std.reload()
blackbox = blackbox_std.reload()

mean_weight_value = torch.cat([p.data.view(-1).abs() for p in best_model.parameters()]).mean()
min_weight_value = torch.cat([p.data.view(-1).abs() for p in best_model.parameters()]).min()
min_nonzero_weight_value = torch.cat([p.data.view(-1).abs() for p in best_model.parameters() if p.data.view(-1).abs().nonzero(as_tuple=True)[0].numel() > 0]).min()
max_weight_value = torch.cat([p.data.view(-1).abs() for p in best_model.parameters()]).max()
print("Mean weight magnitude: ", mean_weight_value.item())
# print("Min weight value: ", min_weight_value.item())
print("Min nonzero weight magnitude: ", min_nonzero_weight_value.item())
print("Max weight magnitude: ", max_weight_value.item())
print("-----------")

# re_std_model = Standardizer(best_model)
# re_std_blackbox = Standardizer(blackbox)
# re_std_model.reload()
# re_std_blackbox.reload()
# print("-----------")
# print_diff(best_model, blackbox)
print_flattened_loss(best_model, blackbox)
print()
print("abs loss:")
print_abs_loss(best_model, blackbox)
print()
print_loss(best_model, blackbox)
# print()
# print_indices_unaligned(best_model, blackbox)


for i in range(1, 20):
    input = torch.randn(1, i, 28)
    # input = torch.randn(784)
    if torch.allclose(best_og(input), best_model(input), atol=1e-5) and torch.allclose(blackbox_og(input), blackbox(input), atol=1e-5):
        continue
    else:
        print("ERROR, OUTPUTS ARE NOT SAME: ", i)
        # if not torch.allclose(model_og(input), model(input)):
        #     print("model_og and model are not same")
        #     print(model_og(input))
        #     print(model(input))
        # if not torch.allclose(target_og(input), target(input)):
        #     print("target_og and target are not same")
        #     print(target_og(input))
        #     print(target(input))
print("all same")

# M = torch.randn(2, 3)
# norms = torch.norm(M, p=2, dim=1)
# print(M)
# print(norms)
# sorted_indices = torch.argsort(norms)
# sorted_M = M[sorted_indices]
# print(M)