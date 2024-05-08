from solver import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import standardize
from solver import CNN_5_5
from align_cnn import get_all_permutations_for_kernel_indices
from align_cnn import order_kernels_cnn
from align_cnn import cnn_evaluate
from align_cnn import bruteforce_cnn_evaluate
from align_cnn import cnn_align_to_perm
from align_cnn import heuristic_ordering_kernels_cnn
from align_cnn import standardize_scale_cnn
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(30)
np.random.seed(30)
random.seed(30)

torch.backends.cudnn.deterministic=True

# Tests: 

# All of these have initialize network. 
# Also, do with training. Cause just initializing will be muchh more well behaved. 
# just with initialization rescaling in alignment won't kick in. 

### Easy:  Assign the same network to two variables. Realign shouldn't do anything and both will be the same.## 

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


def evaluate_accuracy(network, save_outputs=False):
        # Set the network to evaluation mode
    network.eval()
    
    # Move the network to CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    correct = 0
    total = 0

    save_preds_network  = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = network(images)
            if save_outputs:
                save_preds_network.append(outputs)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, save_preds_network

def train_network(net): 
    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the neural network
    for epoch in range(25):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")
        # Calculate and print accuracy
        accuracy, _ = evaluate_accuracy(net)
        print(f"Accuracy on MNIST: {accuracy:.4f}")

def are_two_models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
     if p1.data.ne(p2.data).sum() > 0:
        return False

network_name = "CNN" # right now only generalizing for kernel sizes. # CNN_5_5
def get_network(): 
    if network_name == "CNN": 
        print("In align test CNN_2_2")
        return CNN()
    
    if network_name == "CNN_5_5": 
        print("In align test CNN_5_5")
        return CNN_5_5()
# scaling test
def scaling_test(): 
    model = get_network()
    standardize_scale_cnn(model)

def heuristic_inter_kernels_test(): 
    model = get_network()
    model_layers = standardize.get_layers(model)
    model_copy = copy.deepcopy(model)
    model_copy_layers = standardize.get_layers(model_copy)

    cnn_layer_copy = model_copy_layers[0]
    cnn_layer = model_layers[0]

    perm = heuristic_ordering_kernels_cnn(cnn_layer, cnn_layer_copy)
    print(perm)
    assert perm == [0, 1, 2]

    # reorder and then call heuristic again 
    reorder = [2,1,0]

    order_kernels_cnn(reorder,cnn_layer_copy)

    perm = heuristic_ordering_kernels_cnn(cnn_layer, cnn_layer_copy)
    print(perm)
    assert perm == [2, 1, 0]


def inter_kernels_test(): 
    perms = get_all_permutations_for_kernel_indices()
    torch.cuda.empty_cache()
    model = get_network()
    #model.to(device)
    perms = [[2,0,1]]
    for perm in perms:
        model_copy = copy.deepcopy(model)
        model_layers = standardize.get_layers(model_copy)
        cnn_layer = model_layers[0]
        print(model_layers)
        print("before ordering:", cnn_layer.weight)
        order_kernels_cnn(perm,cnn_layer)
        print("after ordering:", cnn_layer.weight)
        model_layers[0] = cnn_layer
        print("assign model ordering:",  model_layers[0].weight)

def test_cnn_align(): 
    model = get_network()
    model_layers = standardize.get_layers(model)
    model_copy = copy.deepcopy(model)
    model_copy_layers = standardize.get_layers(model_copy)

    # should be equal 
    cnn_layer_copy = model_copy_layers[0]
    fnn_layer_copy = model_copy_layers[1]
    cnn_layer = model_layers[0]
    fnn_layer = model_layers[1]
    print("before alignment:", cnn_layer_copy.weight)
    print("before alignment: ",fnn_layer_copy.weight)
    cnn_align_to_perm(model, model_copy, [0, 1, 2])
    print("after alignment:", cnn_layer_copy.weight)
    print("after alignment: ",fnn_layer_copy.weight)
    print(fnn_layer.weight.shape)
    print(torch.equal(cnn_layer.weight, cnn_layer_copy.weight))
    print(torch.equal(fnn_layer.weight, fnn_layer_copy.weight))

    # won't be equal 
    print("before alignment:", cnn_layer_copy.weight)
    print("before alignment: ",fnn_layer_copy.weight)
    cnn_align_to_perm(model, model_copy, [2, 1, 0])
    print("after alignment:", cnn_layer_copy.weight)
    print("after alignment: ",fnn_layer_copy.weight)
    print(torch.equal(cnn_layer.weight, cnn_layer_copy.weight))
    print(torch.equal(fnn_layer.weight, fnn_layer_copy.weight))

def easy_test(): # test cnn_evaluate
    model = get_network()
    model.to(device)
    train_network(model)
    model_copy = copy.deepcopy(model)
    errors = cnn_evaluate(model, model_copy)

    print(are_two_models_equal(model, model_copy))
    print(errors) # should be zero error 

### Easy 2:  initialize two networks. Align network 2 to network 1. 
# It'll be something absurd since it's not learning from network 1 outputs (no training). 
# Then take that and realign back to network 2. You should get 0 errors (mean, max error between weights). 

def easy_2_test(): 
    network_1 = get_network()
    original_network_2 = get_network()
    network_1.to(device)
    original_network_2.to(device)
    train_network(network_1)

    train_network(original_network_2)

    align_network2_to_1 =  copy.deepcopy(original_network_2)
    print(cnn_evaluate(network_1, align_network2_to_1))
    # make sure that network2_aligned_to_network_1 is actually shuffled. 
    # it should have high max error per layer when you compare network2 and network2_aligned_to_network_1
    align_network2_to_1_back_to_2 = copy.deepcopy(align_network2_to_1)
    print(cnn_evaluate(original_network_2, align_network2_to_1_back_to_2)) # => this should give 0 error - considering numerical instability there might be a slight error. 


# Judag reccomended: 

# Another test: 
# Failures: 
# moving around weights that actually breaks the network. 
# relu scaling imperfect. 
# Take a lot of random slamples run them through a network save the ouputs. 

# then deal with the alignment on that network. 

# output on these samples of this networks should be the same. 

# do this a bunch of times. 

# fails -> \delta 10^-8 (first try this). scaling/swapping. floating point precision stuff might affect this. 

# isomorphims are legal. 
def easy_3_test(): 
    model = get_network()
    model.to(device)
    train_network(model)

    # TODO: save outputs of test_dataloader, shuffle false so order will remain same when you iterate. 
    acc, saved_preds_net = evaluate_accuracy(model, save_outputs=True)
        

    model_2 = copy.deepcopy(model)
    print(cnn_evaluate(model, model_2)) # should give low error. 

    acc_2, saved_preds_net_2 = evaluate_accuracy(model_2, save_outputs=True) # same accuracy after evaluate

    print(f'acc:{acc}')
    print(f'acc_2: {acc_2}')

    for i in range(0, len(saved_preds_net)):
        print("equal outputs")
        print(saved_preds_net[i])
        print(saved_preds_net_2[i])
        #print(torch.eq(saved_preds_net[i], saved_preds_net_2[i])) 
    
    

def add_gaussian_noise(model, mean=0, std=0.1):
    for param in model.parameters():
        if param.requires_grad:
            noise = torch.normal(mean=mean, std=std, size=param.data.size()).to(device)
            param.data.add_(noise)

def add_uniform_noise(model):
    # using range (-0.1, 0.1) range
    for param in model.parameters():
        if param.requires_grad:
            print(param.data)
            noise = (0.2 * (torch.rand(size=param.data.size()) - 0.5)).to(device) 
            param.data.add_(noise)
            print(noise)
            print("after noise")
            print(param.data)

def random_noise_test():
    network_1 = get_network()
    network_1.to(device)
    train_network(network_1)

    network_2 =  copy.deepcopy(network_1)
    add_uniform_noise(network_2)

    print(cnn_evaluate(network_1, network_2))


def all_intg_tests(): 
    print('easy test')
    easy_test()

    print('easy 2 test')
    easy_2_test()

    print('easy 3 test')
    easy_3_test()
    
    print("random_noise - uniform test")
    print(random_noise_test())

# Try 1 kernel
# do intra-kernel alignment, inter-kernel alignment, scaling, fully-connected. 

# unit tests 

# inter_kernels_test()

# test_cnn_align()
#scaling_test()

# integration tests 

#print('easy test')
#easy_test()

print('easy 2 test')
easy_2_test()


#heuristic_inter_kernels_test()


#print('easy 3 test')
#easy_3_test()


#print("random_noise - uniform test")
#random_noise_test()

#print("all tests")
#all_intg_tests()