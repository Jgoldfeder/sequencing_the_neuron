from __future__ import annotations

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch.nn import Module
import torch.nn.functional as F
import torch.fx as fx


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
    weights: torch.Tensor
    bias: torch.Tensor
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

    def __init__(self, model: Module):
        self.model = model
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
                            layer.next.weights[:, i*section_size:(i+1)*section_size] /= L2norms[i, 0]

            elif layer.activation == 'tanh': #(nn.Tanh)
                if layer.next is not None:
                    if layer.weights.dim() == 2:
                        dims = 1
                    elif layer.weights.dim() == 4:
                        dims = (1, 2, 3)

                    if layer.next.layertype == layer.layertype:
                        #make weight matrix columns have positive sum
                        sums = layer.weights.sum(dim=dims, keepdim=True)
                        signs = torch.sign(sums)
                        signs[signs == 0] = 1
                        layer.weights *= signs
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
        
        out_scale = torch.hstack((self.layers[-1].weights, self.layers[-1].bias.reshape(-1,1))).norm(dim=1, p=2)
        out_scale_total = sum(out_scale) / len(out_scale)
        avg_scale = out_scale_total ** (1 / len(self.layers))
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            if layer.activation in ('relu', 'leakyrelu', 'leaky_relu', 'prelu', 'relu6'): #(nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6)
                if layer.next is not None:
                    layer.weights *= avg_scale
                    layer.bias *= avg_scale
                    layer.next.weights /= avg_scale
            

    def align(self, std_target): #takes the standardizer object of the target network as argument

        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
           
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
    
    def scale_layers(layer, nextlayer, norms, dims):
        layer.weights /= norms
        layer.bias /= norms.squeeze()

        if layer.layertype == nextlayer.layertype:
            nextlayer.weights *= norms.transpose(0, 1)
        elif 'Conv' in layer.layertype and nextlayer.layertype == 'Linear':
            #is convoluational layer followed by linear layer
            #flatten the weight matrix of convolutional layer and set to linear layer
            k = layer.weights.shape[0]
            n = layer.next.weights.shape[1]
            if n % k != 0:
                raise ValueError('Number of output channels of convolutional layer must be a multiple of number of input channels')
            section_size = n // k
            for i in range(k):
                layer.next.weights[:, i*section_size:(i+1)*section_size] /= L2norms[i, 0]

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
                if layer.weights is not None:
                    model_params[f"{layer.layername}.weight"].copy_(layer.weights)
                if layer.bias is not None:
                    model_params[f"{layer.layername}.bias"].copy_(layer.bias)
        return self.model