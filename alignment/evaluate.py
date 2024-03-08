import torch
import copy
from recon_evals import e_mae, e_layers_mae, e_max_ae, e_mse

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



def evaluate_reconstruction(original, reconstruction,return_blackbox=False,tanh=False):
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
    return metrics
