import torch
import copy
from recon_evals import e_mae, e_layers_mae, e_max_ae, e_mse
from standardize_align_new import Standardizer

def network_accuracy(network, test_loader):
    # Set the network to evaluation mode
    network.eval()
    
    # Move the network to CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 784)

            outputs = network(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def evaluate_reconstruction(original, reconstruction,return_blackbox=False,tanh=False,return_nets=False, old_redist=False):
    #standardize both original and reconstruction, align networks, then calculate metrics
    reconstruction = copy.deepcopy(reconstruction)
    original = copy.deepcopy(original)
    original = original.cuda()
    reconstruction = reconstruction.cuda()

    #standardize network
    std_reconstruction = Standardizer(reconstruction, old_redist)
    std_target = Standardizer(original, old_redist)
    #align networks
    std_reconstruction.align(std_target)
    reconstruction = std_reconstruction.reload()
    original = std_target.reload()

    #calculate metrics
    total_size = sum(
        weights.numel() for weights in original.state_dict().values()
    )
    total_se = 0
    total_ae = 0
    total_ape = 0
    max_ae = float('-inf')
    layerwise_metrics = []
    for og_weight, re_weight, layername in zip(original.state_dict().values(), reconstruction.state_dict().values(), original.state_dict().keys()):
        #squared error, absolute error, max absolute errors, average percent errors for total
        total_se += torch.nn.functional.mse_loss(og_weight, re_weight, reduction="sum").item()
        total_ae += torch.nn.functional.l1_loss(og_weight, re_weight, reduction="sum").item()
        layermax_ae = torch.nn.functional.l1_loss(og_weight, re_weight, reduction="none").max().item()
        max_ae = max(layermax_ae, max_ae)
        total_ape += (torch.abs((og_weight - re_weight) / og_weight) * 100).sum().item()

        #layerwise metrics
        layer_se = torch.nn.functional.mse_loss(og_weight, re_weight, reduction="mean").item()
        layer_ae = torch.nn.functional.l1_loss(og_weight, re_weight, reduction="mean").item()
        layer_ape = (torch.abs((og_weight - re_weight) / og_weight) * 100).mean().item()
        layerwise_metrics.append((layer_se, layer_ae, layermax_ae, layer_ape, layername))
        
    mse = total_se / total_size
    mae = total_ae / total_size
    mape = total_ape / total_size
    #mape of entire 

    #can separate out biases and see if the biases are worse than the weights?

    return mse, mae, max_ae, mape, layerwise_metrics

def evaluate_reconstruction_old(original, reconstruction,return_blackbox=False,tanh=False,return_nets=False):
    reconstruction = copy.deepcopy(reconstruction)
    original = copy.deepcopy(original)
    original = original.cuda()
    reconstruction = reconstruction.cuda()

    # align
    evaluator = e_mae(reconstruction, None,use_align=True,tanh=tanh)
    evaluator.original = original # blackbox
    metric = evaluator.get_evaluation()

    metrics = []
    for Evaluator in [e_mae,e_layers_mae,e_max_ae,e_mse]:
        evaluator = Evaluator(reconstruction, None,use_align=True,tanh=tanh)
        evaluator.original = original # blackbox
        metric = evaluator.calculate_distance()
        metrics.append(metric)
    
    if return_blackbox:
        return original
    if return_nets:
        return original,reconstruction        
    return metrics

def eval_layers_mae(original, reconstruction):
    #return a tuple, (mean absolute error per layer, mean absolute value of weight per layer, mape per layer)
    return