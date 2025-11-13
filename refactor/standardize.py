from __future__ import annotations

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch.nn import Module
import torch.nn.functional as F
import torch.fx as fx
import sys
import numpy as np

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

activation_module_types = (
	nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.Hardtanh, nn.ReLU6, nn.PReLU,
	nn.Softplus, nn.Softsign, nn.Softmax, nn.Softmin, nn.Softshrink, nn.Softmax2d, nn.LogSoftmax, nn.Hardshrink,
	nn.Hardswish, nn.Hardsigmoid, nn.Threshold, nn.SiLU, nn.Mish
)
activation_function_targets = (
	F.relu, F.leaky_relu, F.tanh, F.sigmoid, F.elu, F.selu, F.celu, F.gelu, F.hardtanh, F.relu6, F.prelu, F.softplus,
	F.softsign, F.softmax, F.softmin, F.softshrink, F.log_softmax, F.hardshrink, F.hardswish, F.hardsigmoid, F.threshold,
	F.silu, F.mish
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
        def is_conv(L): return L.weights.dim() == 4
        def flat_wb(L):
            # Flatten weights per output channel and append bias as a column
            Wf = L.weights.view(L.weights.shape[0], -1) if is_conv(L) else L.weights
            b  = L.bias.reshape(-1, 1)
            return torch.hstack((Wf, b))

        def row_l2(Wb):
            # L2 across each row (out-channel or neuron)
            n = Wb.norm(dim=1, p=2, keepdim=True)
            return torch.where(n == 0, torch.tensor(1e-8, device=Wb.device, dtype=Wb.dtype), n)

        def row_sign(Wb):
            s = torch.sign(Wb.sum(dim=1, keepdim=True))
            s[s == 0] = 1
            return s

        def as_weight_factor(L, v):
            # v is [out,1]; return shape for broadcasting onto weights
            return v.view(-1, 1, 1, 1) if is_conv(L) else v

        def squeeze_bias_factor(v):
            # bias expects [out], not [out,1] or [out,1,1,1]
            return v.squeeze()

        def scale_next(next_layer, factor, prev_layer):
            # factor is [out,1]; push to next layer's input dimension
            if next_layer.layertype == prev_layer.layertype:
                next_layer.weights *= factor.transpose(0, 1)
                return
            if 'Conv' in prev_layer.layertype and next_layer.layertype == 'Linear':
                k = prev_layer.weights.shape[0]  # out-channels of conv
                n = next_layer.weights.shape[1]  # fan-in of linear
                if n % k != 0:
                    raise ValueError('Number of output channels of convolutional layer must be a multiple of number of input channels')
                sec = n // k
                # factor: [k,1] → scale each input block
                f = factor.squeeze(1)
                for i in range(k):
                    next_layer.weights[:, i*sec:(i+1)*sec] *= f[i]

        def normalize_relu_block(L):
            Wb = flat_wb(L)
            f = row_l2(Wb)                 # [out,1]
            L.weights /= as_weight_factor(L, f)
            L.bias   /= squeeze_bias_factor(f)
            if L.next is not None:
                scale_next(L.next, f, L)

        def align_tanh_block(L):
            if L.layertype == 'RNN':
                # Merge biases into bias[0], zero bias[1]
                L.bias[0] = L.bias[0] + L.bias[1]
                L.bias[1].zero_()
                # Build rows for sign decision: [W_ih, W_hh, W_hh^T, b]
                Wb = torch.hstack((
                    L.weights[0],               # W_ih
                    L.weights[1],               # W_hh
                    L.weights[1].t(),           # W_hh^T (symmetry cue)
                    L.bias[0].reshape(-1, 1)    # bias
                ))
                s = row_sign(Wb)               # [out,1]
                # Flip rows/cols consistently
                L.weights[0] *= s
                L.weights[1] *= s
                L.weights[1] *= s.transpose(0, 1)
                L.bias[0]   *= squeeze_bias_factor(s)
                if L.next is not None:
                    if L.next.layertype == 'RNN':
                        L.next.weights[0] *= s.transpose(0, 1)
                    else:
                        L.next.weights *= s.transpose(0, 1)
                # (Optional) recompute/verify sign after flip if you need
                return

            # Non-RNN tanh: sign-align rows (neurons or out-channels)
            Wb = flat_wb(L)
            s = row_sign(Wb)                   # [out,1]
            L.weights *= as_weight_factor(L, s)
            L.bias    *= squeeze_bias_factor(s)
            if L.next is not None:
                scale_next(L.next, s, L)

        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            nxt   = layer.next
            if nxt is None:
                continue

            # ReLU family → normalize rows (unit L2)
            if layer.activation in {'relu', 'leakyrelu', 'leaky_relu', 'prelu', 'relu6'}:
                if layer.layertype != 'RNN':
                    normalize_relu_block(layer)
                continue

            # tanh → sign canonicalization
            if layer.activation == 'tanh':
                align_tanh_block(layer)
                continue
    
    def align(self, std_target): #takes the standardizer object of the target network as argument

        for i in range(len(self.layers) - 1):
            layer = self.layers[i]

            if layer.layertype == 'RNN':
                # #rnn layer
                # self_weights = torch.hstack(layer.weights).clone()
                # target_weights = torch.hstack(std_target.layers[i].weights).clone()
                raise RuntimeError('Use rnn_align for RNN models')
            else:
                if layer.weights.dim() == 4:
                    #convolutional layer
                    #flatten the weight matrix of convolutional layer
                    self_weights = layer.weights.clone().view(layer.weights.shape[0], -1)
                    target_weights = std_target.layers[i].weights.clone().view(std_target.layers[i].weights.shape[0], -1)
                else:
                    #fnn layer
                    self_weights = layer.weights
                    target_weights = std_target.layers[i].weights
            distances = torch.cdist(torch.abs(target_weights), torch.abs(self_weights), p=1).cpu().numpy()
            indices = torch.from_numpy(linear_sum_assignment(distances)[1])
            #permute outgoing weights
            if layer.layertype == 'RNN':
                # layer.weights[0] = layer.weights[0][indices]
                # layer.weights[1] = layer.weights[1][indices] 
                # layer.weights[1] = layer.weights[1][:, indices]#since square, "next" recurrent weights also get permuted?
                # layer.bias[0] = layer.bias[0][indices]
                # layer.bias[1] = layer.bias[1][indices]

                # if layer.next.layertype == 'RNN':
                #     layer.next.weights[0] = layer.weights[0][:, indices] #only permute input hidden layer
                # else:
                #     layer.next.weights = layer.next.weights[:, indices]
                raise RuntimeError('Use rnn_align for RNN models')

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
    
    def sort_permute(self, layer):
        weights4norm = (layer.weights[0], layer.weights[1], layer.weights[1].t(), layer.bias[0].reshape(-1, 1))
        weights4norm = torch.hstack(weights4norm)
        indices = torch.argsort(torch.norm(weights4norm, dim=1, p=2))
        layer.weights[0] = layer.weights[0][indices]
        layer.weights[1] = layer.weights[1][indices] 
        layer.weights[1] = layer.weights[1][:, indices]#since square, "next" recurrent weights also get permuted?
        layer.bias[0] = layer.bias[0][indices]
        layer.bias[1] = layer.bias[1][indices]

        if layer.next.layertype == 'RNN':
            layer.next.weights[0] = layer.weights[0][:, indices] #only permute input hidden layer
        else:
            layer.next.weights = layer.next.weights[:, indices]
    
    def rnn_align(self, std_target):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            target_layer = std_target.layers[i]
            if layer.layertype == 'RNN':
                #permutations, sort by norm
                before = layer.weights[1].clone()
                self.sort_permute(layer)
                self.sort_permute(target_layer)

                #polarity
                self_weights4sign = (layer.weights[0], layer.weights[1], layer.weights[1].t(), layer.bias[0].reshape(-1, 1))
                self_weights4sign = torch.hstack(self_weights4sign)
                target_weights4sign = (target_layer.weights[0], target_layer.weights[1], target_layer.weights[1].t(), target_layer.bias[0].reshape(-1, 1))
                target_weights4sign = torch.hstack(target_weights4sign)
                
                loss_orig = torch.abs(self_weights4sign - target_weights4sign)
                loss_orig[loss_orig > 0.001] = 1
                loss_orig = loss_orig.sum(dim=1)

                loss_flip = torch.abs(-self_weights4sign - target_weights4sign)
                loss_flip[loss_flip > 0.001] = 1
                loss_flip = loss_flip.sum(dim=1)

                # Choose polarity that minimizes loss
                signs = (loss_flip > loss_orig).float().unsqueeze(1)  # shape (n, 1)
                signs[signs == 0] = -1  # Set flip indices to -1

                layer.weights[0] *= signs
                layer.weights[1] *= signs
                layer.weights[1] *= signs.transpose(0, 1)
                layer.bias[0] *= signs.squeeze()

                if layer.next.layertype == "RNN":
                    layer.next.weights[0] *= signs.transpose(0, 1)
                else:
                    layer.next.weights *= signs.transpose(0, 1)

            else:
                print("ERROR")

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