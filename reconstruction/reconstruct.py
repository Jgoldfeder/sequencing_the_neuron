import copy
import torch
import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import util
import gc
import standardize

# Get seed, network 2nd layer dimension, outer iter, num_samples 

seed = int(sys.argv[1])
model_type = str(sys.argv[2])
layer_dim =  str(sys.argv[3]) # for cnn this is kernel dim. 
outer_iterations = int(sys.argv[4])
num_samples = int(sys.argv[5])
num_epochs  = int(sys.argv[6])
dataset  = str(sys.argv[7])
optim_ = str(sys.argv[8]) # optimizer for black box network
activation  = str(sys.argv[9])

input_dim=784
if dataset in ['cifar10','cifar100']:
    input_dim = 1024*3
if dataset in ['places365']:
    input_dim = 256*256*3
    
name = "seed_"+str(seed)+"_"+model_type + "_" + layer_dim+"_outer_iterations_"+str(outer_iterations)+"_num_samples_"+str(num_samples)+"_num_epochs_"+str(num_epochs)+"_dataset_"+dataset+"_optim_"+optim_ + "_activation_"+activation

import os
if not os.path.exists("./results/"):
    os.makedirs("./results/")

models_path = "./models/"+name+"/"

if not os.path.exists(models_path):
    os.makedirs(models_path)


sys.stdout = open("./results/"+name, "w")
print ("Log file for:"+name)

layer_dim = layer_dim.split("x")
layer_dim = [int(x) for x in layer_dim]


print(f"seed: {seed}")
if model_type == 'fnn': 
    print(f"layer_dim: {layer_dim}")
if model_type == 'cnn': 
    print(f"kernel dim: {layer_dim}")
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

        layers = []
        for i in range(len(layer_dim)-1):
            in_dim = layer_dim[i]
            out_dim = layer_dim[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers =nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = activation_f(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# TODO: use the formula on pytorch to get the dimension of fnn layer (depends only on cnn layer) instead of hardcoding. 
kernel_dim= layer_dim # to make it clear what layer_dim means to CNNs. 

def get_fnn_dimensions_from_kernel(conv_stride, conv_padding, pool_stride, pool_padding): 
    stride = conv_stride
    padding = conv_padding
    dilation = 1 
    kernel_size = kernel_dim[0]
    h_in = None 
    w_in = None 
    if dataset == 'mnist': 
        h_in = 28
        w_in = 28
    if dataset == 'cifar100': 
        h_in = 32
        w_in = 32

    # using formula from here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html (assuming squared kernels.)
    conv_h_out =  int((h_in + 2*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    conv_w_out =  int((w_in + 2*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    # using formula from here: 
    stride = pool_stride
    padding = pool_padding
    dilation = 1
    kernel_size = 2 
    h_in = conv_h_out 
    w_in = conv_w_out

    pool_h_out = int((h_in + 2*padding - kernel_size)/stride) + 1 
    pool_w_out = int((w_in + 2*padding - kernel_size)/stride) + 1 

    return pool_h_out, pool_w_out
class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__()
        out_channels = kernel_dim[3]
        self.conv_stride = 1
        self.conv_padding = 1
        self.pool_stride = 2
        self.pool_padding = 0
        self.conv1 = nn.Conv2d(in_channels=kernel_dim[2],out_channels=out_channels,kernel_size=kernel_dim[0],stride=1, padding=1) # square kernels 
        pool_w_out, pool_h_out = get_fnn_dimensions_from_kernel(self.conv_stride, self.conv_padding, self.pool_stride, self.pool_padding)
        number_outputs = 10 
        if dataset == 'cifar100': 
            number_outputs = 100
        self.fc1 = nn.Linear(out_channels * pool_w_out * pool_h_out, number_outputs)
    

    def forward(self, x):
        if dataset == 'mnist': 
            x = x.view(-1, 1, 28, 28)
        if dataset == 'cifar100': 
            x = x.view(-1, 3, 32, 32) 
        # Convolutional layers
        x = self.conv1(x)
        activation = nn.LeakyReLU()
        x =  activation(x)
        pool = nn.AvgPool2d(kernel_size=2, stride=self.pool_stride, padding=self.pool_padding)
        x = pool(x)
        # Flatten the tensor before passing it through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


def get_network(): 
    if model_type == 'cnn': 
        return CNN()

    if model_type == 'fnn': 
        return Net()
    
    return None

net = get_network()
net.to(device)

util.train_blackbox(net,num_epochs,dataset,optim_)
print(net)

net_layers = standardize.get_layers(net) 

for l in net_layers:
    print(l.weight.abs().mean())
# save net
torch.save(net.state_dict(), models_path+"black_box.pt")

pop_size = 10
subs = []
for j in range(pop_size):
  subs.append(get_network())
population = util.Population(subs)
population.cuda(device)
net = net.cuda(device)
criterion = nn.L1Loss()

lr = 0.001

population.set_optimizer(optim.Adam(population.parameters(), lr=lr))

with torch.enable_grad():
    for outer_iter in range(outer_iterations):
        sys.stdout.flush()

        restore = False

        if outer_iter > 25:
            lr = lr* 0.8
            population.set_optimizer(optim.Adam(population.parameters(), lr=lr))
        
        
        print("ITERATION: ",outer_iter, len(population.inputs))    
        samples_to_generate = num_samples
        while samples_to_generate > 0:
            
            new_inputs = util.get_adv(population.subs,lr=0.01,num_samples=min(samples_to_generate,100002),epochs=2000,schedule = [500,1000,1500],reverse=False,range_=1.000,input_dim=input_dim) 
            samples_to_generate -= 100002
            new_outputs = net(new_inputs.cuda(device)).cpu().detach()
            population.add_data(new_inputs, new_outputs,window=500)
        
        gc.collect()
        for i in range(10):
           population.train_one_epoch(batch_size=128, epoch_num=i,restore=False) 
           sys.stdout.flush()
        population.save(models_path +"/population_iteration_"+str(outer_iter)+".pt")
        if outer_iter in [15,30,54]:
            population.evaluate(net,tanh=tanh)
        

for i in range(10):
    print(population.subs[i].loss)
sys.stdout.flush()

for i in range(10):
    print(util.evaluate(population.subs[i],net,tanh=tanh))
