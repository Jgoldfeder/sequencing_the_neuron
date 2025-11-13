import copy
import sys
import torch
from standardize import Standardizer

def evaluate_reconstruction(original, reconstruction, model_type='fnn'):
    #standardize both original and reconstruction, align networks, then calculate metrics
    reconstruction = copy.deepcopy(reconstruction)
    original = copy.deepcopy(original)
    original = original.cuda()
    reconstruction = reconstruction.cuda()

    #standardize network
    print("model_type: ", model_type, file=sys.stderr)
    if model_type == 'trans':
        raise NotImplementedError("Transformer evaluation not implemented yet")
        std_reconstruction = SingleTransformerEncoderStandardizer(reconstruction.encoder_layer)
        std_target = SingleTransformerEncoderStandardizer(original.encoder_layer)
    else:
        std_reconstruction = Standardizer(reconstruction)
        std_target = Standardizer(original)
    #align networks
    if model_type == 'rnn':
        std_reconstruction.rnn_align(std_target)
    else:
        std_reconstruction.align(std_target)
    reconstruction = std_reconstruction.reload()
    original = std_target.reload()

    if model_type == 'trans':
        re_layerdict = std_reconstruction.layers
        og_layerdict = std_target.layers
    else:
        re_layerdict = reconstruction.state_dict()
        og_layerdict = original.state_dict()

    #calculate metrics
    total_size = sum(
        weights.numel() for weights in og_layerdict.values()
    )
    total_se = 0
    total_ae = 0
    total_ape = 0
    max_ae = float('-inf')
    max_pe = float('-inf')
    layerwise_metrics = []

    for og_weight, re_weight, layername in zip(og_layerdict.values(), re_layerdict.values(), og_layerdict.keys()):
        #squared error, absolute error, max absolute errors, average percent errors for total
        total_se += torch.nn.functional.mse_loss(og_weight, re_weight, reduction="sum").item()
        total_ae += torch.nn.functional.l1_loss(og_weight, re_weight, reduction="sum").item()
        layermax_ae = torch.nn.functional.l1_loss(og_weight, re_weight, reduction="none").max().item()
        max_ae = max(layermax_ae, max_ae)
        pe = (torch.abs((og_weight - re_weight) / og_weight) * 100)
        pe = torch.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0)
        total_ape += pe.sum().item()
        max_pe = max(pe.max().item(), max_pe)

        #layerwise metrics
        layer_se = torch.nn.functional.mse_loss(og_weight, re_weight, reduction="mean").item()
        layer_ae = torch.nn.functional.l1_loss(og_weight, re_weight, reduction="mean").item()
        layer_ape = pe.mean().item()
        layermax_pe = pe.max().item()
        layerwise_metrics.append((layer_se, layer_ae, layermax_ae, layer_ape, layermax_pe, layername))
        
    mse = total_se / total_size
    mae = total_ae / total_size
    mape = total_ape / total_size
    #mape of entire 

    #can separate out biases and see if the biases are worse than the weights?

    return mse, mae, max_ae, mape, max_pe, layerwise_metrics