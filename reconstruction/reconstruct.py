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


# Get seed, network 2nd layer dimension, outer iter, num_samples 

seed = int(sys.argv[1])
layer_dim =  str(sys.argv[2])
outer_iterations = int(sys.argv[3])
num_samples = int(sys.argv[4])
num_epochs  = int(sys.argv[5])
dataset  = str(sys.argv[6])
optim_ = str(sys.argv[7]) # optimizer for black box network
activation  = str(sys.argv[8])

input_dim=784
if dataset in ['cifar10','cifar100']:
    input_dim = 1024*3
    
name = "seed_"+str(seed)+"_"+layer_dim+"_outer_iterations_"+str(outer_iterations)+"_num_samples_"+str(num_samples)+"_num_epochs_"+str(num_epochs)+"_dataset_"+dataset+"_optim_"+optim_ + "_activation_"+activation

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
    else if activation == "nonleakyreluapproximation":
        nn.LeakyReLU(negative_slope=0.0001)
    else if activation =="relu":
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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(3 * 14 * 14, 10) # using only one layer for now. 
       # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        # Convolutional layers
        x = self.conv1(x)
        activation = nn.LeakyReLU()
        x =  activation(x)
        pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        x = pool(x)
        # Flatten the tensor before passing it through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


net = Net()
net.to(device)


util.train_blackbox(net,num_epochs,dataset,optim_)
print(net)

# save net
torch.save(net.state_dict(), models_path+"black_box.pt")

pop_size = 10
subs = []
for j in range(pop_size):
  subs.append(Net())
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
