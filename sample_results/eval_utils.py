import torch

def print_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'Layer: {name} - Weights')
            print(param)
        elif 'bias' in name:
            print(f'Layer: {name} - Biases')
            print(param)

def print_pairs_firstelem(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        param1.clone()[param1<=0] = 0
        param2.clone()[param2<=0] = 0
        param1.clone()[param1>0] = 1
        param2.clone()[param2>0] = 1
        if 'weight' in name1:
            print(f'Layer: {name1} - Reconstructed Weights')
            print(param1.detach().numpy())
            print(f'Layer: {name2} - Blackbox Weights')
            print(param2.detach().numpy())
        elif 'bias' in name1:
            print(f'Layer: {name1} - Reconstructed Biases')
            print(param1.detach().numpy())
            print(f'Layer: {name2} - Blackbox Biases')
            print(param2.detach().numpy())
            
def print_sorted_abs_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'Layer: {name} - Weights')
            sorted = torch.abs(param)
            sorted, _ = torch.sort(sorted)
            print(sorted)
        elif 'bias' in name:
            print(f'Layer: {name} - Biases')
            sorted = torch.abs(param)
            sorted, _ = torch.sort(sorted)
            print(sorted)
            
def print_diff(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            print(f'Layer: {name1} - Weights')
        elif 'bias' in name1:
            print(f'Layer: {name1} - Biases')
        diff = torch.abs(param1 - param2)
        diff[diff < 1e-4] = 0
        diff[diff > 1e-4] = 1
        print(diff.detach().numpy())
            
def print_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            print(f'Layer: {name1} - Weights')
            print(torch.mean(torch.abs(param1 - param2)).item(), torch.max(torch.abs(param1 - param2)).item())
        elif 'bias' in name1:
            print(f'Layer: {name1} - Biases')
            print(torch.mean(torch.abs(param1 - param2)).item(), torch.max(torch.abs(param1 - param2)).item())
            
def print_sorted_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            if 'layers.0.weight_hh_l0' in name1:
                param1indeces = torch.argsort(torch.norm(param1, dim=1))
                param1_sorted = param1[param1indeces].T
                # param1_sorted = param1_sorted[torch.argsort(torch.norm(param1_sorted, dim=1))].T
                param1_sorted = param1_sorted[param1indeces].T

                param2indeces = torch.argsort(torch.norm(param2, dim=1))
                param2_sorted = param2[param2indeces].T
                # param2_sorted = param2[torch.argsort(torch.norm(param2, dim=1))].T
                param2_sorted = param2_sorted[param2indeces].T
                # param2_sorted = param2_sorted[torch.argsort(torch.norm(param2_sorted, dim=1))].T

            elif 'layers.0.weight_ih_l0' in name1:
                param1indeces = torch.argsort(torch.norm(param1, dim=1))
                param1_sorted = param1[param1indeces]

                param2indeces = torch.argsort(torch.norm(param2, dim=1))
                param2_sorted = param2[param2indeces]

            else:
                continue
            sorted_loss = torch.abs(param1_sorted - param2_sorted)
            print(f'Layer: {name1} - Weights')
            print(torch.mean(sorted_loss).item(), torch.max(sorted_loss).item())
        elif 'bias' in name1:
            continue
            print(f'Layer: {name1} - Biases')
            print(torch.mean(abs_sorted_loss).item(), torch.max(abs_sorted_loss).item())

def print_abs_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        abs_loss = torch.abs(param1.abs() - param2.abs())
        if 'weight' in name1:
            print(f'Layer: {name1} - Weights')
            print(torch.mean(abs_loss).item(), torch.max(abs_loss).item())
        elif 'bias' in name1:
            print(f'Layer: {name1} - Biases')
            print(torch.mean(abs_loss).item(), torch.max(abs_loss).item())    

def print_neg_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        abs_loss = torch.abs(param1*(-1) - param2)
        if 'weight' in name1:
            print(f'Layer: {name1} - Weights')
            print(torch.mean(abs_loss).item(), torch.max(abs_loss).item())
        elif 'bias' in name1:
            print(f'Layer: {name1} - Biases')
            print(torch.mean(abs_loss).item(), torch.max(abs_loss).item())  

def print_abs_sorted_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            if 'layers.0.weight_hh_l0' in name1:
                param1indeces = torch.argsort(torch.norm(param1, dim=1))
                param1_sorted = param1[param1indeces].T
                # param1_sorted = param1_sorted[torch.argsort(torch.norm(param1_sorted, dim=1))].T
                param1_sorted = param1_sorted[param1indeces].T

                param2indeces = torch.argsort(torch.norm(param2, dim=1))
                param2_sorted = param2[param2indeces].T
                # param2_sorted = param2[torch.argsort(torch.norm(param2, dim=1))].T
                param2_sorted = param2_sorted[param2indeces].T
                # param2_sorted = param2_sorted[torch.argsort(torch.norm(param2_sorted, dim=1))].T

                abs_sorted_loss = torch.abs(param1_sorted.abs() - param2_sorted.abs())
            elif 'layers.0.weight_ih_l0' in name1:
                param1indeces = torch.argsort(torch.norm(param1, dim=1))
                param1_sorted = param1[param1indeces]

                param2indeces = torch.argsort(torch.norm(param2, dim=1))
                param2_sorted = param2[param2indeces]

                abs_sorted_loss = torch.abs(param1_sorted.abs() - param2_sorted.abs())
            else:
                continue
            print(f'Layer: {name1} - Weights')
            print(torch.mean(abs_sorted_loss).item(), torch.max(abs_sorted_loss).item())
        elif 'bias' in name1:
            continue
            print(f'Layer: {name1} - Biases')
            print(torch.mean(abs_sorted_loss).item(), torch.max(abs_sorted_loss).item())

def print_flattened_loss(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            param1_flattened = param1.clone().flatten()
            param2_flattened = param2.clone().flatten()
            abs_sorted_loss = torch.abs(param1_flattened.abs().sort()[0] - param2_flattened.abs().sort()[0])
            print(f'Layer: {name1} - Weights')
            print(torch.mean(abs_sorted_loss).item(), torch.max(abs_sorted_loss).item())
        elif 'bias' in name1:
            continue
            print(f'Layer: {name1} - Biases')
            print(torch.mean(abs_sorted_loss).item(), torch.max(abs_sorted_loss).item())

def print_select_flipped_loss(model, blackbox, indices):
    signs = indices.clone()
    signs[signs > 0] = -1
    signs[signs <= 0] = 1

def print_indices_unaligned(model, blackbox):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), blackbox.named_parameters()):
        if 'weight' in name1:
            losses = torch.abs(param1 - param2).mean(dim=1)
            indices = torch.where(losses > 0.005)[0]
            print(f'Layer: {name1} - Weights')
            print(indices)
        elif 'bias' in name1:
            losses = torch.abs(param1 - param2)
            indices = torch.where(losses > 0.005)[0]
            print(f'Layer: {name1} - Biases')
            print(indices)