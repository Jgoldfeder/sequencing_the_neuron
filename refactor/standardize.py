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
            if 'Conv' in prev_layer.layertype and 'Conv' in next_layer.layertype:
                # factor: [C_prev, 1] -> [1, C_prev, 1, 1]
                f = factor.squeeze(1).view(1, -1, 1, 1)
                next_layer.weights *= f
                return
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
    

class SingleTransformerEncoderStandardizer:
    #all learnable parameters of a single transformer encoder
    layer: nn.TransformerEncoderLayer
    dim: int

    def __init__(self, layer):
        if not isinstance (layer, nn.TransformerEncoderLayer):
            raise TypeError(f"Expected nn.TransformerEncoderLayer, but got {type(layer).__name__}")
        
        self.layer = layer
        self.dim = layer.self_attn.out_proj.weight.shape[1]
        self.get_params()
        self.canonical_forms()

    def get_params(self):
        layer = self.layer
        mha = self.layer.self_attn
        dim = self.dim

        self.W_q = mha.in_proj_weight.detach().clone()[:dim, :]  # Query weight
        self.W_k = mha.in_proj_weight.detach().clone()[dim:2*dim, :]  # Key weight
        self.W_v = mha.in_proj_weight.detach().clone()[2*dim:, :]  # Value weight

        self.b_q = mha.in_proj_bias.detach().clone()[:dim]      # Query bias
        self.b_k = mha.in_proj_bias.detach().clone()[dim:2*dim]  # Key bias
        self.b_v = mha.in_proj_bias.detach().clone()[2*dim:]    # Value bias

        self.W_O = mha.out_proj.weight.detach().clone()
        self.b_O = mha.out_proj.bias.detach().clone()

        self.gam1 = layer.norm1.weight.detach().clone()
        self.bet1 = layer.norm1.bias.detach().clone()

        self.W1 = layer.linear1.weight.detach().clone()
        self.b1 = layer.linear1.bias.detach().clone()
        self.W2 = layer.linear2.weight.detach().clone()
        self.b2 = layer.linear2.bias.detach().clone()

        self.gam2 = layer.norm2.weight.detach().clone()
        self.bet2 = layer.norm2.bias.detach().clone()

        self.layers = {'W_q': self.W_q, 'b_q': self.b_q, #query
                       'W_k': self.W_k, 'b_k': self.b_k, #key
                       'W_v': self.W_v, 'b_v': self.b_v, #value
                       'W_O': self.W_O, 'b_O': self.b_O, #output
                       'gam1': self.gam1, 'bet1': self.bet1, #layernorm 1
                       'W1': self.W1, 'b1': self.b1, #fnn 1
                       'W2': self.W2, 'b2': self.b2, #fnn 2
                       'gam2': self.gam2, 'bet2': self.bet2} #layernorm 2

    def balance_linear_layers(self, balancelayers):
        balancelayers[0][0] *= 7
        # print("weight: ", balancelayers[0][0])
        # print("bias: ", balancelayers[0][1])
        # print("weight l1: ", balancelayers[1][0])
        for i in range(len(balancelayers) - 1):
            weight, bias = balancelayers[i]
            if weight.dim() == 1: #since gamma has dim 1
                weight = weight.reshape(-1, 1)
            L2norms = torch.hstack((weight, bias.reshape(-1, 1))).norm(dim=1, p=2, keepdim=True)
            L2norms = torch.where(L2norms == 0, torch.tensor(1e-8), L2norms)
            weight /= L2norms
            bias /= L2norms.squeeze()
            balancelayers[i+1][0] *= L2norms.transpose(0, 1)

        print('')

    def balance_two_matrices(self, W1, b1, W2, b2):
        norm1 = torch.hstack((W1, b1.reshape(-1, 1))).norm(p=2)
        norm2 = torch.hstack((W2, b2.reshape(-1, 1))).norm(p=2)
        balance_factor = torch.sqrt(norm1 / norm2)
        W1 *= balance_factor
        b1 *= balance_factor
        W2 /= balance_factor
        return W1, b1, W2

    def canonical_forms(self):
        #balance Key and Query weights

        self.W_q, self.b_q, self.W_k = self.balance_two_matrices(self.W_q, self.b_q, self.W_k, self.b_k)

        #negative -> switch order after softmax?

        #balance Value and output projection matrices
        self.W_v, self.b_v, self.W_O = self.balance_two_matrices(self.W_v, self.b_v, self.W_O, self.b_O)

        #balance LayerNorm, linear1, and linear2 weights
        balancelayers = [[self.W1, self.b1], [self.W2, self.b2]]
        # self.balance_linear_layers(balancelayers)
        self.W1, self.b1, self.W2 = self.balance_two_matrices(self.W1, self.b1, self.W2, self.b2)

    def permute(self, std_target, wname, bname, next_wname):
        self_weights = self.layers[wname].clone()
        self_biases = self.layers[bname].clone()
        target_weights = std_target.layers[wname].clone()
        target_biases = std_target.layers[bname].clone()

        self_weightstack = torch.hstack((self_weights, self_biases.reshape(-1, 1)))
        target_weightstack = torch.hstack((target_weights, target_biases.reshape(-1, 1)))

        distances = torch.cdist(target_weightstack, self_weightstack, p=1).cpu().numpy()
        indices = torch.from_numpy(linear_sum_assignment(distances)[1])

        #permute weights
        self.layers[wname].copy_(self.layers[wname][indices])
        self.layers[bname].copy_(self.layers[bname][indices])
        self.layers[next_wname].copy_(self.layers[next_wname][:, indices])

    def test_perms(self):
        layername = "W1"
        biasname = "b1"
        nextlayername = "W2"
        print(self.layers[layername].shape[0], file=sys.stderr)
        dim = self.layers[layername].shape[0]
        perm = np.random.permutation(dim)
        self.layers[layername].copy_(self.layers[layername][perm])
        self.layers[biasname].copy_(self.layers[biasname][perm])
        self.layers[nextlayername].copy_(self.layers[nextlayername][:, perm])
        
    def align(self, std_target):
        #permute key and query
        selfstack = torch.hstack((self.W_q, self.b_q.reshape(-1, 1), self.W_k, self.b_k.reshape(-1, 1)))
        targetstack = torch.hstack((std_target.W_q, std_target.b_q.reshape(-1, 1), std_target.W_k, std_target.b_k.reshape(-1, 1)))
        distances = torch.cdist(targetstack, selfstack, p=1).cpu().numpy()
        indices = torch.from_numpy(linear_sum_assignment(distances)[1])

        self.W_q = self.W_q[indices]
        self.b_q = self.b_q[indices]
        self.W_k = self.W_k[indices]
        self.b_k = self.b_k[indices]

        #permute value
        self.permute(std_target, "W_v", "b_v", "W_O")
        #permute linear
        self.permute(std_target, "W1", "b1", "W2")

        # for i in range(1, len(self.layers)/2): #start at value matrix
        #     selfvals = self.layers.values()
        #     targetvals = std_target.layers.values()

        #     #every even layer is the weight, and odd layer is bias
        #     self_weights = selfvals[i*2].clone()
        #     self_biases = selfvals[i*2+1].clone()
        #     target_weights = targetvals[i*2].clone()
        #     target_biases = targetvals[i*2+1].clone()

        #     if self_weights.dim() == 1: #gamma has dim 1
        #         self_weights = self_weights.reshape(-1, 1)
        #         target_weights = target_weights.reshape(-1, 1)
            
        #     self_weightstack = torch.hstack((self_weights, self_biases.reshape(-1, 1)))
        #     target_weightstack = torch.hstack((target_weights, target_biases.reshape(-1, 1)))
            
        #     distances = torch.cdist(target_weightstack, self_weightstack, p=1).cpu().numpy()
        #     indices = torch.from_numpy(linear_sum_assignment(distances)[1])

        #     #permute weights
        #     selfvals[i*2] = selfvals[i*2][indices]
        #     selfvals[i*2+1] = selfvals[i*2+1][indices]
        #     selfvals[(i+1)*2] = selfvals[(i+1)*2][:, indices]


    def reload(self):
        layer = self.layer
        in_proj_weight = torch.cat([self.W_q, self.W_k, self.W_v], dim=0)
        in_proj_bias = torch.cat([self.b_q, self.b_k, self.b_v], dim=0)

        with torch.no_grad():
            layer.self_attn.in_proj_weight.copy_(in_proj_weight)
            layer.self_attn.in_proj_bias.copy_(in_proj_bias)
            layer.self_attn.out_proj.weight.copy_(self.W_O)
            layer.self_attn.out_proj.bias.copy_(self.b_O)

            layer.norm1.weight.copy_(self.gam1)
            layer.norm1.bias.copy_(self.bet1)

            layer.linear1.weight.copy_(self.W1)
            layer.linear1.bias.copy_(self.b1)
            layer.linear2.weight.copy_(self.W2)
            layer.linear2.bias.copy_(self.b2)

            layer.norm2.weight.copy_(self.gam2)
            layer.norm2.bias.copy_(self.bet2)

        return self.layer