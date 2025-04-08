import copy
import torch
import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import util
import gc

from models import base_CNN, two_CNN, var_CNN, base_RNN, var_RNN, base_TransformerEncoder

# Get seed, network 2nd layer dimension, outer iter, num_samples 

seed = int(sys.argv[1])
layer_dim =  str(sys.argv[2])
outer_iterations = int(sys.argv[3])
num_samples = int(sys.argv[4])
num_epochs  = int(sys.argv[5])
dataset  = str(sys.argv[6])
optim_ = str(sys.argv[7]) # optimizer for black box network
activation  = str(sys.argv[8])
aligner = str(sys.argv[9])
input_shape = str(sys.argv[10])
input_shape = tuple(int(x) for x in input_shape.split('x')) if input_shape[0].isdigit() else None
if input_shape is not None:
    if len(input_shape) == 1:
        input_shape = input_shape[0] #input size of RNN or transformer
    else:
        assert len(input_shape) == 3 #checking just for cnns
sampling_method = 'committee'
#sampling_method  = str(sys.argv[9])
sampling_options = ['committee','rand_gauss','rand_uni','dataset','expanded_dataset','fully_expanded_dataset',"easy","hard"]
if sampling_method not in sampling_options:
    raise ValueError("invalid sampling argument")
strong_start=False
single_strong_start = False
save_samples=True

input_dim=784
if dataset in ['cifar10','cifar100']:
    input_dim = 1024*3
if dataset in ['places365']:
    input_dim = 256*256*3
strong_start_str=""
if strong_start:
    strong_start_str="strong_start_"
if single_strong_start:
    strong_start_str="single_strong_start_"    
name = strong_start_str+"seed_"+str(seed)+"_"+layer_dim+"_outer_iterations_"+str(outer_iterations)+"_num_samples_"+str(num_samples)+"_num_epochs_"+str(num_epochs)+"_dataset_"+dataset+"_optim_"+optim_ + "_activation_"+activation + "_sampling_method_"+sampling_method+"_aligner_"+aligner

import os
if not os.path.exists("./results/"):
    os.makedirs("./results/")

models_path = "./models/"+name+"/"

if not os.path.exists(models_path):
    os.makedirs(models_path)


sys.stdout = open("./results/"+name, "w")
print ("Log file for:"+name)

layer_dim = layer_dim.split("x")
if layer_dim[0].isdigit():
    layer_dim = [int(x) for x in layer_dim]
    model_type = 'fnn'
elif layer_dim[0].lower() == 'cnn':
    model_type = 'cnn'
    layer_configs = []
    for layer in layer_dim[1:]:
        layer = layer.split('-')
        layer_configs.append({'in_channels': int(layer[0]), 'out_channels': int(layer[1]), 'kernel_size': int(layer[2]), 'stride': int(layer[3])})
elif layer_dim[0].lower() == 'rnn':
    model_type = 'rnn'
    layer_configs = [int(x) for x in layer_dim[1:]]
elif layer_dim[0].lower() == 'trans':
    model_type = 'trans'
    layer_configs = [int(x) for x in layer_dim[1:]]
else:
    raise ValueError('cannot parse layers')

print(f"seed: {seed}")
print(f"layer_dim: {layer_dim}")
print(f"outer_iterations: {outer_iterations}")
print(f"given_num_samples: {num_samples}")
print(f"num_epochs: {num_epochs}")
print(f"dataset: {dataset}")
print(f"optim: {optim_}")
print(f"activation: {activation}")


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

print("device",device)
torch.manual_seed(seed)

if activation not in ["tanh","relu",'nonleakyrelu','nonleakyreluapproximation']:
    raise ValueError("unknown activation")

if activation == "tanh":
    tanh = True
    activation_f = nn.Tanh()
else:
    tanh = False
    if activation == "nonleakyrelu":
        activation_f = nn.ReLU()
    elif activation == "nonleakyreluapproximation":
        activation_f = nn.LeakyReLU(negative_slope=0.0001)
    elif activation =="relu":
        activation_f = nn.LeakyReLU()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.activation = activation_f

        layers = []
        for i in range(len(layer_dim)-1):
            in_dim = layer_dim[i]
            out_dim = layer_dim[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers =nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

if model_type == 'rnn':
    print('RNN model')
    net = var_RNN(input_shape, layer_configs)
elif model_type == 'cnn':
    print('CNN model')
    # if layer_dim ==2:
    #     net = two_CNN()
    # else:
    #     net = base_CNN()
    net = var_CNN(input_shape, layer_configs, activation_f)
elif model_type == 'trans':
    print('Transformer model')
    net = base_TransformerEncoder(input_shape, layer_configs)
else:
    net = Net()

torch.save(net.state_dict(), models_path+"original_params_black_box.pt")
net.to(device)
util.train_blackbox(net,num_epochs,dataset,optim_, model_type=model_type)
print(net)
print("weight mean magnitude per layer")

if model_type != 'trans':
    for l in net.layers:
        if isinstance(l, nn.RNN):
            print("input weights:", l.weight_ih_l0.abs().mean())
            print("hidden weights:", l.weight_hh_l0.abs().mean())
        else:
            print("weights:", l.weight.abs().mean())

if model_type == 'rnn':
    og_net = var_RNN(input_shape, layer_configs)
elif model_type == 'trans':
    og_net = base_TransformerEncoder(input_shape, layer_configs)
elif model_type == 'cnn':
    # if layer_dim ==2:
    #     og_net = two_CNN()
    # else:
    #     og_net = base_CNN()
    og_net = var_CNN(input_shape, layer_configs, activation_f)
else:
    og_net=Net()
og_net.to(device)
og_net.load_state_dict(torch.load(models_path+"original_params_black_box.pt"))
# print("distance weights moved during training, mean and max")
# for i in range(len(net.layers)):
#     dists=[]
#     for j in range(net.layers[i].weight.shape[0]):
#         for k in range(net.layers[i].weight.shape[1]):
#             dist = abs(net.layers[i].weight[j][k]-og_net.layers[i].weight[j][k])
#             dists.append(dist)
#     dists=torch.tensor(dists)
#     print(dists.mean(),dists.max())
    
# save net
torch.save(net.state_dict(), models_path+"black_box.pt")

pop_size = 10
subs = []
for j in range(pop_size):
    if model_type == 'rnn':
        subs.append(var_RNN(input_shape, layer_configs))
    elif model_type == 'trans':
        subs.append(base_TransformerEncoder(input_shape, layer_configs))
    elif model_type == 'cnn':
    # if layer_dim ==2:
    #     subs.append(two_CNN())
    # else:
    #     subs.append(base_CNN())
        subs.append(var_CNN(input_shape, layer_configs, activation_f))
    else:
        subs.append(Net())      
    if strong_start:
        subs[-1].load_state_dict(torch.load(models_path+"original_params_black_box.pt"))
        with torch.no_grad():
            for i in range(len(net.layers)):
                for j in range(net.layers[i].weight.shape[0]):
                    for k in range(net.layers[i].weight.shape[1]):
                        subs[-1].layers[i].weight[j][k] += torch.rand(1).item()/100
if single_strong_start:
    subs[0].load_state_dict(torch.load(models_path+"original_params_black_box.pt"))
population = util.Population(subs)
population.cuda(device)
net = net.cuda(device)
criterion = nn.L1Loss()

lr = 0.001

population.set_optimizer(optim.Adam(population.parameters(), lr=lr))

if sampling_method =="dataset":
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset(dataset)
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            print()
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)

    
if sampling_method =="expanded_dataset":
    # only for MNIST
    if dataset!="mnist":
        raise ValueError("expanded_dataset only supported for mnist")

    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset(dataset)
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
            
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset("emnist")
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)

if sampling_method =="fully_expanded_dataset":
    # only for MNIST
    if dataset!="mnist":
        raise ValueError("fully expanded_dataset only supported for mnist")

    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset(dataset)
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
            
    # trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset("emnist")
    # with torch.no_grad():
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data
    #         inputs = inputs.view(-1, input_dim)
    #         new_inputs, labels = inputs.to(device), labels.to(device)            
    #         new_outputs = net(new_inputs.cuda(device)).cpu().detach()
    #         population.add_data(new_inputs, new_outputs,window=50000)
    #     for i, data in enumerate(test_loader, 0):
    #         inputs, labels = data
    #         inputs = inputs.view(-1, input_dim)
    #         new_inputs, labels = inputs.to(device), labels.to(device)            
    #         new_outputs = net(new_inputs.cuda(device)).cpu().detach()
    #         population.add_data(new_inputs, new_outputs,window=50000)


    
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset("qmnist")
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset("fmnist")
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset("kmnist")
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if model_type == 'fnn':
                inputs = inputs.view(-1, input_dim)
            new_inputs, labels = inputs.to(device), labels.to(device)            
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=50000)

with torch.enable_grad():
    for outer_iter in range(outer_iterations):
        sys.stdout.flush()

        restore = False

        if outer_iter > 25:
            lr = lr* 0.8
            population.set_optimizer(optim.Adam(population.parameters(), lr=lr))
        
        
        print("ITERATION: ",outer_iter, len(population.inputs))
        print("model type: ", model_type, file=sys.stderr)
        if sampling_method =="committee":
            samples_to_generate = num_samples
            seq_len = None
            if model_type == 'rnn' or model_type == 'trans':
                for l in range(1, 4):
                    samples_per_seq_len = samples_to_generate // 3 #hardcoded # of sequence lengths (1, 2, 3), maybe change to parameter?
                    seq_len = l
                    while samples_per_seq_len > 0:
                        new_inputs = util.get_adv(population.subs,lr=0.01,num_samples=min(samples_per_seq_len,6002),epochs=2000,schedule = [500,1000,1500],reverse=False,range_=1.000,input_dim=input_dim, model_type=model_type, sequence_length=seq_len) 
                        samples_per_seq_len -= 6002
                        new_outputs = net(new_inputs.cuda(device)).cpu().detach()
                        population.add_rnn_dataset(new_inputs, new_outputs, seq_len, window=500)
                        if save_samples:
                            torch.save(new_inputs,models_path +"/data_iteration_"+str(outer_iter)+".pt")
            else:
                while samples_to_generate > 0:
                    new_inputs = util.get_adv(population.subs,lr=0.01,num_samples=min(samples_to_generate,6002),epochs=2000,schedule = [500,1000,1500],reverse=False,range_=1.000,input_dim=input_dim, model_type=model_type, sequence_length=seq_len) 
                    samples_to_generate -= 6002
                    new_outputs = net(new_inputs.cuda(device)).cpu().detach()
                    population.add_data(new_inputs, new_outputs,window=500)
                    if save_samples:
                        torch.save(new_inputs,models_path +"/data_iteration_"+str(outer_iter)+".pt")
            
            gc.collect()
        if sampling_method =="rand_gauss":
            new_inputs=util.get_random_gauss(num_samples,input_dim=input_dim)
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=500)
        if sampling_method =="rand_uni":
            new_inputs=util.get_random_uniform(num_samples,input_dim=input_dim)
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=500)
        if sampling_method =="hard" or sampling_method =="easy":
            reverse=True
            if sampling_method =="easy":
                reverse=False
            if outer_iter>3:
                new_inputs=util.get_hard(num_samples,population,input_dim=input_dim,reverse=reverse)
            else:
                new_inputs=util.get_random_gauss(num_samples,input_dim=input_dim)
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=500)
            
#sampling_options = ['committee','rand_gauss','rand_uni','dataset','expanded_dataset',"easy","hard"]

        
        for i in range(10):
           population.train_one_epoch(batch_size=128, epoch_num=i,restore=False) 
           sys.stdout.flush()
        population.save(models_path +"/population_iteration_"+str(outer_iter)+".pt")
        population.evaluate(net,tanh=tanh)
        

for i in range(10):
    print(population.subs[i].loss)
sys.stdout.flush()

for i in range(10):
    print(util.evaluate(population.subs[i],net,tanh=tanh))
