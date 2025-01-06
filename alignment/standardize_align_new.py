from __future__ import annotations

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch.nn import Module
import torch.nn.functional as F
import torch.fx as fx
import sys


activation_module_types = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.ELU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.Hardtanh,
    nn.ReLU6,
    nn.PReLU,
    nn.Softplus,
    nn.Softsign,
    nn.Softmax,
    nn.Softmin,
    nn.Softshrink,
    nn.Softmax2d,
    nn.LogSoftmax,
    nn.Hardshrink,
    nn.Hardswish,
    nn.Hardsigmoid,
    nn.Threshold,
    nn.SiLU,
    nn.Mish
)
activation_function_targets = (
    F.relu,
    F.leaky_relu,
    F.tanh,
    F.sigmoid,
    F.elu,
    F.selu,
    F.celu,
    F.gelu,
    F.hardtanh,
    F.relu6,
    F.prelu,
    F.softplus,
    F.softsign,
    F.softmax,
    F.softmin,
    F.softshrink,
    F.log_softmax,
    F.hardshrink,
    F.hardswish,
    F.hardsigmoid,
    F.threshold,
    F.silu,
    F.mish
)

class Layer:
    layername: str
    layertype: str
    activation: str

    def __init__(self, layername, layertype, weights, bias, activation):
        self.layername = layername
        self.layertype = layertype
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.next = None

    def nextlayer(self, nextLayer):
        self.next = nextLayer

class Standardizer:
    model: nn.Module
    layers: list

    def __init__(self, model: Module, old_redist=False):
        self.model = model
        self.old_redist = old_redist
        self.layers = []
        self.get_layers()
        for i in range(len(self.layers) - 1):
            self.layers[i].nextlayer(self.layers[i + 1])
        self.canonical_forms()

    def get_layers(self):
        symbolic_traced = fx.symbolic_trace(self.model)
        module_dict = dict(symbolic_traced.named_modules())

        for node in symbolic_traced.graph.nodes:
            if node.op == 'call_module':
                module = module_dict[node.target]
                module_type = type(module).__name__
                layername = node.target

                # Check if this module is an activation function
                if isinstance(module, activation_module_types):
                    # Skip activation function modules for now
                    continue

                # Initialize activation function to None
                activation_function = None

                # Iterate over the users of the node
                for user_node in node.users:
                    if user_node.op == 'call_module':
                        user_module = module_dict[user_node.target]
                        if isinstance(user_module, activation_module_types):
                            activation_function = type(user_module).__name__.lower()
                            break  # Found activation function module
                    elif user_node.op == 'call_function':
                        if user_node.target in activation_function_targets:
                            activation_function = user_node.target.__name__.lower()
                            break  # Found activation function function
                
                if isinstance(module, nn.RNN):
                    weight_ih = module.weight_ih_l0.detach()
                    weight_hh = module.weight_hh_l0.detach()
                    bias_ih = module.bias_ih_l0.detach()
                    bias_hh = module.bias_hh_l0.detach()
                    activation_function = 'tanh'
                    self.layers.append(Layer(layername, module_type, [weight_ih, weight_hh], [bias_ih, bias_hh], activation_function))
                else:
                    if hasattr(module, 'weight') and module.weight is not None:
                        weight = module.weight.detach()
                    else:
                        weight = None
                    if hasattr(module, 'bias') and module.bias is not None:
                        bias = module.bias.detach()
                    else:
                        bias = None
                    if weight is not None:
                        self.layers.append(Layer(layername, module_type, weight, bias, activation_function))

    def canonical_forms(self):
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            if layer.activation in ('relu', 'leakyrelu', 'leaky_relu', 'prelu', 'relu6'): #(nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6)
                if layer.next is not None:
                    if layer.layertype == 'RNN':
                        # weights4norm = (layer.weights[0], layer.weights[1], layer.bias[0].reshape(-1, 1), layer.bias[1].reshape(-1, 1))
                        # weights4norm = torch.hstack(weights4norm)
                        # L2norms = weights4norm.norm(dim=1, p=2, keepdim=True)
                        # L2norms = torch.where(L2norms == 0, torch.tensor(1e-8), L2norms)
                        
                        # layer.weights[0] /= L2norms
                        # layer.weights[1] /= L2norms
                        # layer.bias[0] /= L2norms.squeeze()
                        # layer.bias[1] /= L2norms.squeeze()

                        # if layer.next.layertype == "RNN":
                        #     layer.next.weights[0] *= L2norms.transpose(0, 1)
                        #     layer.next.weights[1] *= L2norms.transpose(0, 1)
                        # else:
                        #     layer.next.weights *= L2norms.transpose(0, 1)
                        continue

                    else:
                        if layer.weights.dim() == 2:
                            #weight of a linear layer
                            weights4norm = (layer.weights, layer.bias.reshape(-1, 1))
                            weights4norm = torch.hstack(weights4norm)
                            L2norms = weights4norm.norm(dim=1, p=2, keepdim=True)
                            L2norms = torch.where(L2norms == 0, torch.tensor(1e-8), L2norms)
                        elif layer.weights.dim() == 4:
                            #weight of a convolutional layer
                            weights4norm = (layer.weights.view(layer.weights.shape[0], -1), layer.bias.reshape(-1, 1))
                            weights4norm = torch.hstack(weights4norm)
                            L2norms = weights4norm.norm(dim=1, p=2, keepdim=True)
                            L2norms = torch.where(L2norms == 0, torch.tensor(1e-8), L2norms).view(-1, 1, 1, 1)
                        
                        #make weight matrix columns have unit norm
                        layer.weights /= L2norms
                        layer.bias /= L2norms.squeeze()

                        #check that both layers are of the same type
                        if layer.next.layertype == layer.layertype:
                            layer.next.weights *= L2norms.transpose(0, 1)
                        elif 'Conv' in layer.layertype and layer.next.layertype == 'Linear':
                            #is convoluational layer followed by linear layer
                            #multiply index-sectioned flattened layer by each kernel's norm
                            k = layer.weights.shape[0]
                            n = layer.next.weights.shape[1]
                            if n % k != 0:
                                raise ValueError('Number of output channels of convolutional layer must be a multiple of number of input channels')
                            section_size = n // k
                            for i in range(k):
                                layer.next.weights[:, i*section_size:(i+1)*section_size] *= L2norms[i, 0]

            elif layer.activation == 'tanh': #(nn.Tanh)
                if layer.next is not None:
                    if layer.layertype == 'RNN':

                        layer.bias[0] = layer.bias[0] + layer.bias[1]
                        layer.bias[1].zero_()
                        weights4sign = (layer.weights[0], layer.weights[1], layer.bias[0].reshape(-1, 1))

                        #weights4sign = (layer.weights[0], layer.weights[1], layer.bias[0].reshape(-1, 1), layer.bias[1].reshape(-1, 1))
                        weights4sign = torch.hstack(weights4sign)
                        sums = weights4sign.sum(dim=1, keepdim=True)
                        signs = torch.sign(sums)
                        signs[signs==0] = 1

                        layer.weights[0] *= signs
                        layer.weights[1] *= signs
                        layer.weights[1] *= signs.transpose(0, 1)
                        layer.bias[0] *= signs.squeeze()
                        #layer.bias[1] *= signs.squeeze()

                        if layer.next.layertype == "RNN":
                            layer.next.weights[0] *= signs.transpose(0, 1)
                        else:
                            layer.next.weights *= signs.transpose(0, 1)
                        # continue
                    else:
                        if layer.weights.dim() == 2:
                            weights4sign = (layer.weights, layer.bias.reshape(-1, 1))
                            weights4sign = torch.hstack(weights4sign)
                            sums = weights4sign.sum(dim=1, keepdim=True)
                            signs = torch.sign(sums)
                            signs[signs==0] = 1

                        elif layer.weights.dim() == 4:
                            weights4sign = (layer.weights.view(layer.weights.shape[0], -1), layer.bias.reshape(-1, 1))
                            weights4sign = torch.hstack(weights4sign)
                            sums = weights4sign.sum(dim=1, keepdim=True)
                            signs = torch.sign(sums)
                            signs[signs==0] = 1
                            signs = signs.view(-1, 1, 1, 1)
                        
                        layer.weights *= signs
                        layer.bias *= signs.squeeze()

                        if layer.next.layertype == layer.layertype:
                            #make weight matrix columns have positive sum
                            layer.next.weights *= signs.transpose(0, 1)
                        elif 'Conv' in layer.layertype and layer.next.layertype == 'Linear':
                            #is convoluational layer followed by linear layer
                            #flatten the weight matrix of convolutional layer and set to linear layer
                            k = layer.weights.shape[0]
                            n = layer.next.weights.shape[1]
                            if n % k != 0:
                                raise ValueError('Number of output channels of convolutional layer must be a multiple of number of input channels')
                            section_size = n // k
                            for i in range(k):
                                layer.next.weights[:, i*section_size:(i+1)*section_size] *= signs[i, 0]

        # 2) optimize mae by distributing last layer scale factor over all layers
        if not self.old_redist:
        
            out_scale = torch.hstack((self.layers[-1].weights, self.layers[-1].bias.reshape(-1,1))).norm(dim=1, p=2)
            out_scale_total = sum(out_scale) / len(out_scale)
            avg_scale = out_scale_total ** (1 / len(self.layers))
            for i in range(len(self.layers)-1):
                layer = self.layers[i]
                if layer.activation in ('relu', 'leakyrelu', 'leaky_relu', 'prelu', 'relu6'): #(nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6)
                    if layer.next is not None:
                        if layer.layertype == "RNN":
                            print('redist')
                            layer.weights[0] *= avg_scale
                            #skip layer.weights[1] (hidden weights) because multiplication then division
                            layer.bias[0] *= avg_scale
                            layer.bias[1] *= avg_scale
                            if layer.next.layertype == "RNN":
                                layer.next.weights[0] /= avg_scale
                            else:
                                layer.next.weights /= avg_scale
                        else:
                            layer.weights *= avg_scale
                            layer.bias *= avg_scale
                            layer.next.weights /= avg_scale

        else:
            
            number_fnn_input_neurons = self.layers[-1].weights.shape[1]
            num_fnn_output_neurons = self.layers[-1].weights.shape[0]
            index_1 = int(number_fnn_input_neurons/3)
            index_2 = int(2*number_fnn_input_neurons/3)
            fnn_weights_biases = torch.hstack((self.layers[-1].weights, self.layers[-1].bias.reshape(-1,1)))

            # print(fnn_weights_biases)

            appended_fnn_weights_biases_1 = torch.cat((fnn_weights_biases[:, 0:index_1],fnn_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
            fnn_layer_norm_1 = torch.norm(appended_fnn_weights_biases_1 ,dim=1, p=2)
            appended_fnn_weights_biases_2 = torch.cat((fnn_weights_biases[:, index_1:index_2],fnn_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
            fnn_layer_norm_2 = torch.norm(appended_fnn_weights_biases_2, dim=1, p=2)
            appended_fnn_weights_biases_3 = torch.cat((fnn_weights_biases[:,  index_2:number_fnn_input_neurons],fnn_weights_biases[:, number_fnn_input_neurons].view(num_fnn_output_neurons,1)), dim=1)
            fnn_layer_norm_3 = torch.norm(appended_fnn_weights_biases_3, dim=1, p=2)

            avg_out_scale_mul_1 = (sum(fnn_layer_norm_1)/len(fnn_layer_norm_1))**0.5
            avg_out_scale_mul_2 = (sum(fnn_layer_norm_2)/len(fnn_layer_norm_2)) **0.5
            avg_out_scale_mul_3 = (sum(fnn_layer_norm_3)/len(fnn_layer_norm_3)) ** 0.5

            cnn_layer = self.layers[0]
            cnn_layer.weights[0] =   cnn_layer.weights[0]*avg_out_scale_mul_1 # all 196 rows are the same so take any one except bias
            cnn_layer.bias[0] =   cnn_layer.bias[0]*avg_out_scale_mul_1
            
            cnn_layer.weights[1] =   cnn_layer.weights[1]*avg_out_scale_mul_2 # want to only use the weights and not the biases
            cnn_layer.bias[1] =   cnn_layer.bias[1]*avg_out_scale_mul_2

            cnn_layer.weights[2] =  cnn_layer.weights[2]*avg_out_scale_mul_3 #  want to only use the weights and not the biases
            cnn_layer.bias[2] =   cnn_layer.bias[2]*avg_out_scale_mul_3

            fnn_layer = self.layers[-1]
            fnn_layer.weights[:, 0:index_1] =  fnn_weights_biases[:, 0:index_1]/avg_out_scale_mul_1
            fnn_layer.weights[:, index_1:index_2] =  fnn_weights_biases[:, index_1:index_2]/avg_out_scale_mul_2
            fnn_layer.weights[:, index_2:number_fnn_input_neurons] = fnn_weights_biases[:, index_2:number_fnn_input_neurons]/avg_out_scale_mul_3


    def align(self, std_target): #takes the standardizer object of the target network as argument

        for i in range(len(self.layers) - 1):
            layer = self.layers[i]

            if layer.layertype == 'RNN':
                self_weights = torch.hstack(layer.weights).clone()
                target_weights = torch.hstack(std_target.layers[i].weights).clone()
            else:
                if layer.weights.dim() == 4:
                    #flatten the weight matrix of convolutional layer
                    self_weights = layer.weights.clone().view(layer.weights.shape[0], -1)
                    target_weights = std_target.layers[i].weights.clone().view(std_target.layers[i].weights.shape[0], -1)
                else:
                    self_weights = layer.weights
                    target_weights = std_target.layers[i].weights
            distances = torch.cdist(target_weights, self_weights, p=1).cpu().numpy()
            indices = torch.from_numpy(linear_sum_assignment(distances)[1])
            #permute outgoing weights
            if layer.layertype == 'RNN':
                layer.weights[0] = layer.weights[0][indices]
                layer.weights[1] = layer.weights[1][indices] 
                layer.weights[1] = layer.weights[1][:, indices]#since square, "next" recurrent weights also get permuted?
                layer.bias[0] = layer.bias[0][indices]
                layer.bias[1] = layer.bias[1][indices]

                if layer.next.layertype == 'RNN':
                    layer.next.weights[0] = layer.weights[0][:, indices] #only permute input hidden layer
                else:
                    layer.next.weights = layer.next.weights[:, indices]

            else:
                layer.weights = layer.weights[indices]
                if layer.bias is not None:
                    layer.bias = layer.bias[indices]
                #permute incoming weights of next layer
                if layer.weights.dim() != layer.next.weights.dim():
                    #last convolutional layer not matchin weight dimensions
                    k = layer.weights.shape[0]
                    m, n = layer.next.weights.shape
                    if n % k != 0:
                        raise ValueError('Number of output channels of convolutional layer must be a multiple of number of input channels')
                    section_size = n // k
                    reshaped = layer.next.weights.view(m, k, section_size)
                    reshaped = reshaped[:, indices, :]
                    layer.next.weights = reshaped.view(m, n)
                else:  
                    layer.next.weights = layer.next.weights[:, indices]

    def check_dimensions(original_layer, custom_layer):
        # Check if the weight dimensions match
        if original_layer.weight.shape != custom_layer.weights.shape:
            return False
        
        # Check if the bias dimensions match (if applicable)
        if original_layer.bias is not None and custom_layer.bias is not None:
            if original_layer.bias.shape != custom_layer.bias.shape:
                return False
        elif (original_layer.bias is not None and custom_layer.bias is None) or (original_layer.bias is None and custom_layer.bias is not None):
            return False
        return True
    
    def reload(self):
        model_params = dict(self.model.named_parameters())
        for layer in self.layers:
            with torch.no_grad():
                if layer.layertype == 'RNN':
                    model_params[f"{layer.layername}.weight_ih_l0"].copy_(layer.weights[0])
                    model_params[f"{layer.layername}.weight_hh_l0"].copy_(layer.weights[1])
                    model_params[f"{layer.layername}.bias_ih_l0"].copy_(layer.bias[0])
                    model_params[f"{layer.layername}.bias_hh_l0"].copy_(layer.bias[1])
                else:
                    if layer.weights is not None:
                        model_params[f"{layer.layername}.weight"].copy_(layer.weights)
                    if layer.bias is not None:
                        model_params[f"{layer.layername}.bias"].copy_(layer.bias)
        return self.model