import torch
import numpy as np
import standardize
import order
import bisect
import copy
import max_ae 
import math
import meanse_meanae
import torch.nn as nn
from itertools import permutations


#print("generalized align")

def get_all_permutations_for_kernel_indices(num_kernels): #assumes only 3 kernels, ideally recursion for this to generalize. 
   #all_permuations_kernel_indices = [[0,1,2], [1,0,2], [2,0,1], [1,2,0], [0,2,1],[2,1,0]]
   all_permuations_kernel_indices = list(permutations(range(0, num_kernels)))
   #print("all_permuations_kernel_indices", all_permuations_kernel_indices)
   return all_permuations_kernel_indices

def heuristic_ordering_kernels_cnn(original_cnn_layer, model_to_align_cnn_layer): 
    weights_original = original_cnn_layer.weight
    weights_align_net = model_to_align_cnn_layer.weight

    align_index = 0
    num_kernels = len(model_to_align_cnn_layer.weight)
    min_err_ordering = [-1]*num_kernels # for each kernel ->  min mean abs error ordering. The elements represent the new ordering. 
    index_taken = set()
    for weight_align in weights_align_net:
        min_mean_abs_err = 1000
        min_mapping_op_index = -1 
        og_index = 0
        for weight_og in weights_original:
            num_els = weight_og.numel()
            weight_align_flat = weight_align.flatten()
            weight_og_flat = weight_og.flatten()
            sum_abs_error = torch.nn.functional.l1_loss(weight_align_flat, weight_og_flat, reduction="sum")
            mean_abs_error = sum_abs_error/num_els
            if min_mean_abs_err > mean_abs_error and og_index not in index_taken:
                min_mean_abs_err = mean_abs_error 
                min_mapping_op_index = og_index

            og_index+=1
        
        min_err_ordering[align_index] = min_mapping_op_index
        index_taken.add(min_mapping_op_index)
        align_index +=1
    
    return min_err_ordering
# order kernels according to this permutation. 

def order_kernels_cnn(num_kernels, permutation, network2_layer): 
    if type(network2_layer).__name__ != 'Conv2d': 
        #print(type(network2_layer))
        return network2_layer
    
    network_2_weight_copies = []
    network_2_bias_copies = []

    for i in range(0,num_kernels): 
         weights = network2_layer.weight[i]
         bias =  network2_layer.bias[i]

         network_2_weight_copies.append(weights.clone())
         network_2_bias_copies.append(bias.clone())

    with torch.no_grad():
        for i in range(0, num_kernels): 
            network2_layer.weight[i] = network_2_weight_copies[permutation[i]]
            network2_layer.bias[i] = network_2_bias_copies[permutation[i]]
    
    return network2_layer

def order_fnn_weights(num_kernels, permutation, network2_layer):
    if type(network2_layer).__name__ != 'Linear': 
        #print(type(network2_layer))
        return network2_layer

    number_input_neurons = int(network2_layer.weight.shape[1])
   # print(number_input_neurons)

    fnn_start_idx = 0
    fnn_end_idx = int(number_input_neurons/num_kernels) 

    fnn_weights_copes = []
    # only weights because you can't apply to this biases because those are specific to the outputs. 
    for i in range(0, num_kernels): 
        weights =  network2_layer.weight[:, fnn_start_idx:fnn_end_idx]
        fnn_weights_copes.append(weights.clone())
        fnn_start_idx+= int(number_input_neurons/num_kernels)
        fnn_end_idx+= int(number_input_neurons/num_kernels)
    
    fnn_start_idx = 0
    fnn_end_idx = int(number_input_neurons/num_kernels) 
    # match with inter-kernel alignment 
    with torch.no_grad():
        if permutation != None:
            for i in range(0, num_kernels): 
                network2_layer.weight[:, fnn_start_idx:fnn_end_idx]= fnn_weights_copes[permutation[i]]
                fnn_start_idx+= int(number_input_neurons/num_kernels)
                fnn_end_idx+= int(number_input_neurons/num_kernels)
    
    return network2_layer

def cnn_align_to_perm(num_kernels, model_to_align: torch.nn.Module, perm): 

    align_layers = standardize.get_layers(model_to_align)
    cnn_layer_model_to_align = align_layers[0]
    initial_magnitude_cnn = torch.round(torch.mean(torch.abs(cnn_layer_model_to_align.weight)), decimals=4).item()

    #print("before alignment cnn layer", cnn_layer_model_to_align.weight)

    # inter kernel alignment 
    #print('inter-kernel alignment')
    cnn_layer = order_kernels_cnn(num_kernels, perm,cnn_layer_model_to_align)

    after_magnitude_cnn = torch.round(torch.mean(torch.abs(cnn_layer.weight)), decimals=4).item()

    assert initial_magnitude_cnn == after_magnitude_cnn, f"initial mag: {initial_magnitude_cnn}, after mag: {after_magnitude_cnn}"

    align_layers[0] = cnn_layer

    fnn_layer_model_to_align = align_layers[1]
    initial_magnitude_fnn =  torch.round(torch.mean(torch.abs(fnn_layer_model_to_align.weight)), decimals=4).item()

    fnn_layer = order_fnn_weights(num_kernels, perm, fnn_layer_model_to_align)

    after_magnitude_fnn = torch.round(torch.mean(torch.abs(fnn_layer.weight)), decimals=4).item()

    assert initial_magnitude_fnn == after_magnitude_fnn, f"initial mag: {initial_magnitude_fnn}, after mag: {after_magnitude_fnn}"
    # print("after ordering:", cnn_layer.weight)
    align_layers[1] = fnn_layer

    return model_to_align


def standardize_scale_cnn(model: torch.nn.Module, tanh: bool =None): 
    #print("original scaling")
    layers = standardize.get_layers(model)
    cnn_layer = layers[0]
    fnn_layer = layers[1]
    num_kernels = len(cnn_layer.weight)
    num_input_channels = cnn_layer.weight[0].size()[0] # the 1st dimension of any kernal will give this. 
    number_fnn_input_neurons =  int(fnn_layer.weight.shape[1]) 
    num_fnn_output_neurons = int(fnn_layer.weight.shape[0])

    weights_biases = (fnn_layer.weight, fnn_layer.bias.reshape(-1, 1))
    fnn_layer_weights_biases = torch.hstack(weights_biases)

    fnn_end_idx = int(number_fnn_input_neurons/num_kernels)
    fnn_start_idx = 0
    # normalize each kernel (divide the cnn_weights_biases with the kernel_scales )
    with torch.no_grad(): 
        each_kernel_for_fnn = int(number_fnn_input_neurons/num_kernels)

        for i in range(0, num_kernels): 
            # reassign the kernels to normalized weights and biases. 
            cnn_layer_weights_biases = torch.cat((cnn_layer.weight[i].flatten(), cnn_layer.bias[i].view(1)))
            each_kernel_num_els = cnn_layer_weights_biases.shape[0] # num of elements in one individual kernel  
            num_weights_in_kernel = each_kernel_num_els - 1
            each_kernel_for_fnn = int(number_fnn_input_neurons/num_kernels)

            # normalize cnn kernels 
            cnn_layer_weights_biases = cnn_layer_weights_biases.clone().expand(each_kernel_for_fnn, each_kernel_num_els) # want to make sure that the reference isn't modified later. 
            kernel_scales =  torch.norm(cnn_layer_weights_biases, dim=1, p=2)
            cnn_layer_weights_biases = cnn_layer_weights_biases/kernel_scales.reshape(-1,1)

            num_weights_in_kernel_per_input_channel = num_weights_in_kernel/num_input_channels
            squared_dim_kernel = int(math.sqrt(num_weights_in_kernel_per_input_channel))
            cnn_layer.weight[i] = cnn_layer_weights_biases[0, 0:num_weights_in_kernel].reshape(num_input_channels, squared_dim_kernel,squared_dim_kernel)  # all rows are the same so take any one except bias
            cnn_layer.bias[i] = cnn_layer_weights_biases[0,num_weights_in_kernel] # # all rows are the same val here. you're making it look like fnn by expanding it like that. use only the bias

            # only need to apply kernel scales to weights because those are ones affected from kernel. 
            #print("fnn_start_idx", fnn_start_idx)
            #print("fnn_end_idx", fnn_end_idx)
            fnn_layer_weights_biases[:, fnn_start_idx:fnn_end_idx] = fnn_layer_weights_biases[:, fnn_start_idx:fnn_end_idx] * kernel_scales

            # normalize 
            appended_fnn_weights_biases = torch.cat((fnn_layer_weights_biases[:, fnn_start_idx:fnn_end_idx],fnn_layer_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)# want to be careful with reference so that later kernel associated fnn layer don't modify prev.
            fnn_layer_norm = torch.norm(appended_fnn_weights_biases.clone() ,dim=1, p=2)
            
            #compute the avg scale to spread across
            avg_out_scale_mul = (sum(fnn_layer_norm)/len(fnn_layer_norm))**0.5

            # multiply these avg out scales across the CNN 
            cnn_layer.weight[i] =   cnn_layer.weight[i]*avg_out_scale_mul # all 196 rows are the same so take any one except bias
            cnn_layer.bias[i] =   cnn_layer.bias[i]*avg_out_scale_mul
    
             # divide this for FNN 
            fnn_layer.weight[:, fnn_start_idx:fnn_end_idx] =  fnn_layer_weights_biases[:, fnn_start_idx:fnn_end_idx]/avg_out_scale_mul

            fnn_start_idx+= int(number_fnn_input_neurons/num_kernels)
            fnn_end_idx+= int(number_fnn_input_neurons/num_kernels)

def other_version_standardize_scale(model): 
    layers = standardize.get_layers(model)
    cnn_layer = layers[0]
    fnn_layer = layers[1]

    kernel_norm = torch.norm(cnn_layer.weight)
    with torch.no_grad(): 
        cnn_layer.weight = cnn_layer.weight/kernel_norm


def get_mae(original, reconstructed): 
    original_layers = standardize.get_layers(original)
    reconstruced_layers = standardize.get_layers(reconstructed)

    total_size = sum(
            weights.numel() for weights in original.state_dict().values()
        )
    
    sum_cnn_weights = torch.nn.functional.l1_loss(original_layers[0].weight.flatten(), reconstruced_layers[0].weight.flatten(), reduction='sum')

    sum_cnn_bias  = torch.nn.functional.l1_loss(original_layers[0].bias.flatten(), reconstruced_layers[0].bias.flatten(), reduction='sum')

    sum_fnn_weights = torch.nn.functional.l1_loss(original_layers[1].weight.flatten(), reconstruced_layers[1].weight.flatten(), reduction='sum')

    sum_fnn_bias = torch.nn.functional.l1_loss(original_layers[1].bias.flatten(), reconstruced_layers[1].bias.flatten(), reduction='sum')

    overall_error = (sum_cnn_weights + sum_cnn_bias + sum_fnn_weights + sum_fnn_bias)/total_size

    return ([sum_cnn_weights/torch.numel(original_layers[0].weight.flatten()), 
    sum_cnn_bias/torch.numel(original_layers[0].bias.flatten()), sum_fnn_weights/torch.numel(original_layers[1].weight.flatten()), 
    sum_fnn_bias/torch.numel(original_layers[1].bias.flatten())], overall_error)

def bruteforce_cnn_evaluate(model: torch.nn.Module, model_to_evaluate: torch.nn.Module, tanh: bool = None):   
    #print("bruteforce cnn eval")
    standardize_scale_cnn(model, tanh=None)
    standardize_scale_cnn(model_to_evaluate, tanh=None)
    layers = standardize.get_layers(model)
    cnn_layer = layers[0]
    num_kernels = len(cnn_layer.weight)

    perms = get_all_permutations_for_kernel_indices(num_kernels)
    min_max_abs_error = math.inf
    perm_model_w_lowest_max_error = None
    for perm in perms:
        #print("perm", perm)
        #print("current min_max_abs_error", min_max_abs_error)
        model_copy = copy.deepcopy(model_to_evaluate)
        aligned_model_copy = cnn_align_to_perm(num_kernels, model_copy, perm)
        # get max error of all the permuted models. use the one with lowest max error for all evaluation. 
        mae = max_ae.calculate_distance_mae(model,aligned_model_copy)
        if mae < min_max_abs_error: 
            min_max_abs_error = mae
            perm_model_w_lowest_max_error = copy.deepcopy(aligned_model_copy)
    
    model_to_evaluate = perm_model_w_lowest_max_error
    low_max_error_model_layers =   standardize.get_layers(perm_model_w_lowest_max_error)  
    #print('cnn of lowest max error, ',low_max_error_model_layers[0].weight)
    #print('fnn of lowest max error, ',low_max_error_model_layers[1].weight)

    #print("avg abs magnitude cnn_layer_weights", torch.mean(torch.abs(low_max_error_model_layers[0].weight.flatten())))
    #print("avg abs magnitude cnn_layer_biases", torch.mean(torch.abs(low_max_error_model_layers[0].bias.flatten())))
   # print("avg abs magnitute fnn_layer_weights", torch.mean(torch.abs(low_max_error_model_layers[1].weight.flatten())))
   # print("avg abs magnitudefnn_layer_biases", torch.mean(torch.abs(low_max_error_model_layers[1].bias.flatten())))

    # now evaluate for all of them. 
    #mean_se, layers_mean_se = meanse_meanae.calculate_distance_mse_or_mae('mse', model, perm_model_w_lowest_max_error)
    layers_mean_ae, mean_ae = meanse_meanae.calculate_distance_mse_or_mae('mae', model, perm_model_w_lowest_max_error)
    max_overall_error =  max_ae.calculate_distance_mae(model,perm_model_w_lowest_max_error)

    #print("get mae", get_mae(model, model_to_evaluate))

    return (layers_mean_ae, mean_ae, max_overall_error)


def cnn_evaluate(model: torch.nn.Module, model_to_evaluate: torch.nn.Module, tanh: bool = None):
    print('heuristic evaluate')
    standardize_scale_cnn(model, tanh=None)
    standardize_scale_cnn(model_to_evaluate, tanh=None)

    cnn_layer_original = standardize.get_layers(model)[0]
    cnn_layer_align = standardize.get_layers(model_to_evaluate)[0]

    num_kernels = len(cnn_layer_align.weight)
    kernel_ordering = heuristic_ordering_kernels_cnn(cnn_layer_original, cnn_layer_align)
    print(f"best heuristic ordering: {kernel_ordering}")
    aligned_model_copy = cnn_align_to_perm(num_kernels, model_to_evaluate, kernel_ordering)

    #mean_se, layers_mean_se = meanse_meanae.calculate_distance_mse_or_mae('mse', model, aligned_model_copy)
    mean_ae, layers_mean_ae = meanse_meanae.calculate_distance_mse_or_mae('mae', model, aligned_model_copy)
    max_overall_error =  max_ae.calculate_distance_mae(model,aligned_model_copy)

    print("get mae", get_mae(model, model_to_evaluate))

    return (mean_ae, layers_mean_ae, max_overall_error)