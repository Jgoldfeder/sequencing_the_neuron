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

print("generalized align")

def get_all_permutations_for_kernel_indices(): #assumes only 3 kernels, ideally recursion for this to generalize. 
   all_permuations_kernel_indices = [[0,1,2], [1,0,2], [2,0,1], [1,2,0], [0,2,1],[2,1,0]]

   return all_permuations_kernel_indices

def get_mean_abs_between_kernel(kernel_O, kernel_R):
    pass

def heuristic_ordering_kernels_cnn(original_cnn_layer, model_to_align_cnn_layer): 
    weights_original = original_cnn_layer.weight
    weights_align_net = model_to_align_cnn_layer.weight

    align_index = 0

    min_err_ordering = [-1]*weights_original.shape[0] # for each kernel ->  min mean abs error ordering. The elements represent the new ordering. 
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

def order_kernels_cnn(permutation, network2_layer): 
    if type(network2_layer).__name__ != 'Conv2d': 
        #print(type(network2_layer))
        return network2_layer

    weights_0 = network2_layer.weight[0].clone()
    weights_1 = network2_layer.weight[1].clone()
    weights_2 = network2_layer.weight[2].clone()

    bias_0 =  network2_layer.bias[0].clone()
    bias_1 = network2_layer.bias[1].clone()
    bias_2 = network2_layer.bias[2].clone()
    with torch.no_grad():
        network2_weight_copies = [weights_0, weights_1, weights_2]
        network2_layer.weight[0] = network2_weight_copies[permutation[0]]
        network2_layer.weight[1] = network2_weight_copies[permutation[1]]
        network2_layer.weight[2] = network2_weight_copies[permutation[2]]

        # bias 
        network2_bias_copies = [bias_0, bias_1, bias_2]
        network2_layer.bias[0] = network2_bias_copies[permutation[0]]
        network2_layer.bias[1] = network2_bias_copies[permutation[1]]
        network2_layer.bias[2] = network2_bias_copies[permutation[2]]
    
    return network2_layer
def order_fnn_weights(permutation, network2_layer):
    if type(network2_layer).__name__ != 'Linear': 
        #print(type(network2_layer))
        return network2_layer

    number_input_neurons = int(network2_layer.weight.shape[1])
   # print(number_input_neurons)
    index_1 = int(number_input_neurons/3) # 3 because that's the number of kernels
    index_2 = int(2*number_input_neurons/3)
    weights_0 = network2_layer.weight[:, 0:index_1].clone()
    weights_1 = network2_layer.weight[:, index_1:index_2].clone()
    weights_2 = network2_layer.weight[:, index_2:number_input_neurons].clone()

    # match with inter-kernel alignment 
    with torch.no_grad():
        if permutation != None:
            network2_weight_copies = [weights_0, weights_1, weights_2]
        # print("weights 0 shape")
        # print(weights_0.shape)
            network2_layer.weight[:, 0:index_1]= network2_weight_copies[permutation[0]]
            network2_layer.weight[:, index_1:index_2] =  network2_weight_copies[permutation[1]]
            network2_layer.weight[:, index_2:number_input_neurons] = network2_weight_copies[permutation[2]]
    return network2_layer

def cnn_align_to_perm(model_to_align: torch.nn.Module, perm): 

    align_layers = standardize.get_layers(model_to_align)
    cnn_layer_model_to_align = align_layers[0]
    initial_magnitude_cnn = torch.round(torch.mean(torch.abs(cnn_layer_model_to_align.weight)), decimals=4).item()

    #print("before alignment cnn layer", cnn_layer_model_to_align.weight)

    # inter kernel alignment 
    #print('inter-kernel alignment')
    cnn_layer = order_kernels_cnn(perm,cnn_layer_model_to_align)

    after_magnitude_cnn = torch.round(torch.mean(torch.abs(cnn_layer.weight)), decimals=4).item()

    assert initial_magnitude_cnn == after_magnitude_cnn, f"initial mag: {initial_magnitude_cnn}, after mag: {after_magnitude_cnn}"

    align_layers[0] = cnn_layer

    fnn_layer_model_to_align = align_layers[1]
    initial_magnitude_fnn =  torch.round(torch.mean(torch.abs(fnn_layer_model_to_align.weight)), decimals=4).item()

    fnn_layer = order_fnn_weights(perm, fnn_layer_model_to_align)

    after_magnitude_fnn = torch.round(torch.mean(torch.abs(fnn_layer.weight)), decimals=4).item()

    assert initial_magnitude_fnn == after_magnitude_fnn, f"initial mag: {initial_magnitude_fnn}, after mag: {after_magnitude_fnn}"
    # print("after ordering:", cnn_layer.weight)
    align_layers[1] = fnn_layer
    # print("assign model ordering:", cnn_layer.weight)
    

    align_after_layers = standardize.get_layers(model_to_align)
    #print("after alignment cnn layer", align_after_layers[0].weight)


    return model_to_align


def standardize_scale_cnn(model: torch.nn.Module, tanh: bool =None): 
    print("original scaling")
    layers = standardize.get_layers(model)
    cnn_layer = layers[0]
    fnn_layer = layers[1]
    num_kernels = len(cnn_layer.weight)
    num_input_channels = cnn_layer.weight[0].size()[0] # the 1st dimension of any kernal will give this. 
    number_fnn_input_neurons =  int(fnn_layer.weight.shape[1]) 
    num_fnn_output_neurons = int(fnn_layer.weight.shape[0])

    # cnn layer normalize and then multiply 
    # concat weights and biases
    cnn_layer_weights_biases_1 = torch.cat((cnn_layer.weight[0].flatten(), cnn_layer.bias[0].view(1)))
    cnn_layer_weights_biases_2 = torch.cat((cnn_layer.weight[1].flatten(), cnn_layer.bias[1].view(1)))
    cnn_layer_weights_biases_3 = torch.cat((cnn_layer.weight[2].flatten(), cnn_layer.bias[2].view(1)))
    each_kernel_num_els = cnn_layer_weights_biases_1.shape[0]

    with torch.no_grad(): 
        each_kernel_for_fnn = int(number_fnn_input_neurons/num_kernels)
        cnn_layer_weights_biases_1 = cnn_layer_weights_biases_1.expand(each_kernel_for_fnn, each_kernel_num_els)
        kernel_1_scales =   torch.norm(cnn_layer_weights_biases_1, dim=1, p=2)  
        cnn_layer_weights_biases_2 = cnn_layer_weights_biases_2.expand(each_kernel_for_fnn, each_kernel_num_els)
        kernel_2_scales =  torch.norm(cnn_layer_weights_biases_2, dim=1, p=2)  
        cnn_layer_weights_biases_3 = cnn_layer_weights_biases_3.expand(each_kernel_for_fnn, each_kernel_num_els)
        kernel_3_scales =  torch.norm(cnn_layer_weights_biases_3, dim=1, p=2)  

        # divide the cnn_weights_biases with the kernel_scales 
        cnn_layer_weights_biases_1 = cnn_layer_weights_biases_1/kernel_1_scales.reshape(-1,1)
        cnn_layer_weights_biases_2 = cnn_layer_weights_biases_2/kernel_2_scales.reshape(-1,1)
        cnn_layer_weights_biases_3 = cnn_layer_weights_biases_3/kernel_3_scales.reshape(-1,1)

        # reassign the kernels to normalized weights and biases. 
        num_weights_in_kernel = each_kernel_num_els - 1
        num_weights_in_kernel_per_input_channel = num_weights_in_kernel/num_input_channels
        squared_dim_kernel = int(math.sqrt(num_weights_in_kernel_per_input_channel))
        cnn_layer.weight[0] = cnn_layer_weights_biases_1[0, 0:num_weights_in_kernel].reshape(num_input_channels, squared_dim_kernel,squared_dim_kernel)  # all rows are the same so take any one except bias

        cnn_layer.weight[1] =  cnn_layer_weights_biases_2[0, 0:num_weights_in_kernel].reshape(num_input_channels, squared_dim_kernel,squared_dim_kernel) # want to only use the weights and not the biases
        cnn_layer.weight[2] =  cnn_layer_weights_biases_3[0, 0:num_weights_in_kernel].reshape(num_input_channels, squared_dim_kernel,squared_dim_kernel) #  want to only use the weights and not the biases

        cnn_layer.bias[0] = cnn_layer_weights_biases_1[0,num_weights_in_kernel] # # all rows are the same val here. you're making it look like fnn by expanding it like that. use only the bias
        cnn_layer.bias[1] = cnn_layer_weights_biases_2[0,num_weights_in_kernel]
        cnn_layer.bias[2] = cnn_layer_weights_biases_3[0,num_weights_in_kernel] 

        weights_biases = (fnn_layer.weight, fnn_layer.bias.reshape(-1, 1))
        fnn_layer_weights_biases = torch.hstack(weights_biases)

        # only need to apply kernel scales to weights because those are ones affected from kernel. 
        index_1 = int(number_fnn_input_neurons/3) # 3 because that's the number of kernels
        index_2 = int(2*number_fnn_input_neurons/3)
        
        fnn_layer_weights_biases[:, 0:index_1] = fnn_layer_weights_biases[:, 0:index_1] * kernel_1_scales
        fnn_layer_weights_biases[:, index_1:index_2] = fnn_layer_weights_biases[:, index_1:index_2] * kernel_2_scales
        fnn_layer_weights_biases[:, index_2:number_fnn_input_neurons] = fnn_layer_weights_biases[:, index_2:number_fnn_input_neurons] * kernel_3_scales

        # norms of fnn weights and biases 

        appended_fnn_weights_biases_1 = torch.cat((fnn_layer_weights_biases[:, 0:index_1],fnn_layer_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
        fnn_layer_norm_1 = torch.norm(appended_fnn_weights_biases_1 ,dim=1, p=2)
        appended_fnn_weights_biases_2 = torch.cat((fnn_layer_weights_biases[:, index_1:index_2],fnn_layer_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
        fnn_layer_norm_2 = torch.norm(appended_fnn_weights_biases_2, dim=1, p=2)
        appended_fnn_weights_biases_3 = torch.cat((fnn_layer_weights_biases[:,  index_2:number_fnn_input_neurons],fnn_layer_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
        fnn_layer_norm_3 = torch.norm(appended_fnn_weights_biases_3, dim=1, p=2)
        
        
        #compute the avg scale to spread across
        
        avg_out_scale_mul_1 = (sum(fnn_layer_norm_1)/len(fnn_layer_norm_1))**0.5
        avg_out_scale_mul_2 = (sum(fnn_layer_norm_2)/len(fnn_layer_norm_2)) **0.5
        avg_out_scale_mul_3 = (sum(fnn_layer_norm_3)/len(fnn_layer_norm_3)) ** 0.5


        # multiply these avg out scales across the CNN 
        cnn_layer.weight[0] =   cnn_layer.weight[0]*avg_out_scale_mul_1 # all 196 rows are the same so take any one except bias
        cnn_layer.bias[0] =   cnn_layer.bias[0]*avg_out_scale_mul_1
        
        cnn_layer.weight[1] =   cnn_layer.weight[1]*avg_out_scale_mul_2 # want to only use the weights and not the biases
        cnn_layer.bias[1] =   cnn_layer.bias[1]*avg_out_scale_mul_2

        cnn_layer.weight[2] =  cnn_layer.weight[2]*avg_out_scale_mul_3 #  want to only use the weights and not the biases
        cnn_layer.bias[2] =   cnn_layer.bias[2]*avg_out_scale_mul_3

        # divide this for FNN 
        fnn_layer.weight[:, 0:index_1] =  fnn_layer_weights_biases[:, 0:index_1]/avg_out_scale_mul_1
        fnn_layer.weight[:, index_1:index_2] =  fnn_layer_weights_biases[:, index_1:index_2]/avg_out_scale_mul_2
        fnn_layer.weight[:, index_2:number_fnn_input_neurons] = fnn_layer_weights_biases[:, index_2:number_fnn_input_neurons]/avg_out_scale_mul_3

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
    print("bruteforce cnn eval")
    standardize_scale_cnn(model, tanh=None)
    standardize_scale_cnn(model_to_evaluate, tanh=None)

    perms = get_all_permutations_for_kernel_indices()
    min_max_abs_error = math.inf
    perm_model_w_lowest_max_error = None
    for perm in perms:
        #print("perm", perm)
        #print("current min_max_abs_error", min_max_abs_error)
        model_copy = copy.deepcopy(model_to_evaluate)
        aligned_model_copy = cnn_align_to_perm(model_copy, perm)
        # get max error of all the permuted models. use the one with lowest max error for all evaluation. 
        mae = max_ae.calculate_distance_mae(model,aligned_model_copy)
        if mae < min_max_abs_error: 
            min_max_abs_error = mae
            perm_model_w_lowest_max_error = copy.deepcopy(aligned_model_copy)
    
    model_to_evaluate = perm_model_w_lowest_max_error
    low_max_error_model_layers =   standardize.get_layers(perm_model_w_lowest_max_error)  
    #print('cnn of lowest max error, ',low_max_error_model_layers[0].weight)
    #print('fnn of lowest max error, ',low_max_error_model_layers[1].weight)

    print("avg abs magnitude cnn_layer_weights", torch.mean(torch.abs(low_max_error_model_layers[0].weight.flatten())))
    print("avg abs magnitude cnn_layer_biases", torch.mean(torch.abs(low_max_error_model_layers[0].bias.flatten())))
    print("avg abs magnitute fnn_layer_weights", torch.mean(torch.abs(low_max_error_model_layers[1].weight.flatten())))
    print("avg abs magnitudefnn_layer_biases", torch.mean(torch.abs(low_max_error_model_layers[1].bias.flatten())))

    # now evaluate for all of them. 
    #mean_se, layers_mean_se = meanse_meanae.calculate_distance_mse_or_mae('mse', model, perm_model_w_lowest_max_error)
    layers_mean_ae, mean_ae = meanse_meanae.calculate_distance_mse_or_mae('mae', model, perm_model_w_lowest_max_error)
    max_overall_error =  max_ae.calculate_distance_mae(model,perm_model_w_lowest_max_error)

    print("get mae", get_mae(model, model_to_evaluate))

    return (layers_mean_ae, mean_ae, max_overall_error)


def cnn_evaluate(model: torch.nn.Module, model_to_evaluate: torch.nn.Module, tanh: bool = None):
    print('heuristic evaluate')
    standardize_scale_cnn(model, tanh=None)
    standardize_scale_cnn(model_to_evaluate, tanh=None)

    cnn_layer_original = standardize.get_layers(model)[0]
    cnn_layer_align = standardize.get_layers(model_to_evaluate)[0]

    kernel_ordering = heuristic_ordering_kernels_cnn(cnn_layer_original, cnn_layer_align)
    print(f"best heuristic ordering: {kernel_ordering}")
    aligned_model_copy = cnn_align_to_perm(model_to_evaluate, kernel_ordering)

    #mean_se, layers_mean_se = meanse_meanae.calculate_distance_mse_or_mae('mse', model, aligned_model_copy)
    mean_ae, layers_mean_ae = meanse_meanae.calculate_distance_mse_or_mae('mae', model, aligned_model_copy)
    max_overall_error =  max_ae.calculate_distance_mae(model,aligned_model_copy)

    print("get mae", get_mae(model, model_to_evaluate))

    return (mean_ae, layers_mean_ae, max_overall_error)