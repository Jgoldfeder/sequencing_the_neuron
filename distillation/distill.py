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
sys.path.append('./reconstruction')
sys.path.append('../reconstruction')
import util
import gc
import argparse
from models import imagenet_model_dict

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Script for configuring training parameters")

# Add arguments
parser.add_argument('--seed', type=int, help='Random seed for initialization')
parser.add_argument('--outer_iter', type=int, help='Number of outer iterations')
parser.add_argument('--samples', type=int, help='Number of samples to generate')
parser.add_argument('--epochs', type=int, help='Number of epochs for training')
parser.add_argument('--pop', type=int, help='Student population size')
parser.add_argument('--dataset', type=str, help='Name of the dataset to use', default="imagenet")
parser.add_argument('--optim_', type=str, help='Optimizer for black box network', default="adam")
parser.add_argument('--sampling_method', type=str, help='Sampling method to use', default="committee")
parser.add_argument('--multigpu', type=bool, help='Student population size', default=False)

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
seed = args.seed
outer_iterations = args.outer_iter
num_samples = args.samples
num_epochs = args.epochs
dataset = args.dataset
optim_ = args.optim_
sampling_method = args.sampling_method
pop_size = args.pop

# Will add dataset options where we train on both CD generated samples and real training data.
sampling_options = ['committee']
if sampling_method not in sampling_options:
    raise ValueError("invalid sampling argument")

save_samples=True
  
name = "seed_"+str(seed)+"_outer_iterations_"+str(outer_iterations)+"_num_samples_"+str(num_samples)+"_num_epochs_"+str(num_epochs)+"_dataset_"+dataset+"_optim_"+optim_ + "sampling_method_"+sampling_method

import os
if not os.path.exists("./results/"):
    os.makedirs("./results/")

models_path = "./models/"+name+"/"

if not os.path.exists(models_path):
    os.makedirs(models_path)

sys.stdout = open("./results/"+name, "w")
print ("Log file for:"+name)

input_dim=784
if dataset in ['cifar10','cifar100']:
    input_dim = 1024*3
if dataset in ['imagenet']:
    input_dim = 224*224*3


print(f"seed: {seed}")
print(f"outer_iterations: {outer_iterations}")
print(f"given_num_samples: {num_samples}")
print(f"num_epochs: {num_epochs}")
print(f"dataset: {dataset}")
print(f"optim: {optim_}")
print(f"sampling method:{sampling_method}")

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

print("device",device)
torch.manual_seed(seed)

net = imagenet_model_dict["ResNet50"](pretrained=True)
net = net.to(device)

subs = []
for j in range(pop_size):
  if multigpu:
    subs.append(nn.DataParallel(imagenet_model_dict["ResNet18"]())) 
  else: 
    subs.append(imagenet_model_dict["ResNet18"]())

population = util.Population(subs)
population.cuda(device)
criterion = nn.L1Loss()

lr = 0.001

population.set_optimizer(optim.Adam(population.parameters(), lr=lr))

if sampling_method =="dataset":
    trainloader,test_loader,input_dim,test_dataset,trainset = util.get_dataset(dataset)
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
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
        if sampling_method =="committee":
            samples_to_generate = num_samples
            while samples_to_generate > 0:
                
                new_inputs = util.get_adv(population.subs,lr=0.01,num_samples=min(samples_to_generate,100002),epochs=100,schedule = [500,1000,1500],reverse=False,range_=1.000,input_dim=input_dim) 
                samples_to_generate -= 100002
                new_outputs = net(new_inputs.cuda(device)).cpu().detach()
                population.add_data(new_inputs, new_outputs,window=500)
                if save_samples:
                    torch.save(new_inputs,models_path +"/data_iteration_"+str(outer_iter)+".pt")
            
            gc.collect()
        
        print("Training Model")
        for i in range(num_epochs):
           population.train_one_epoch(batch_size=128, epoch_num=i,restore=False) 
           sys.stdout.flush()
        population.save(models_path +"/population_iteration_"+str(outer_iter)+".pt")

        # Needs a different evaluation method
        # if outer_iter in [15,30,54]:
        #     population.evaluate(net,tanh=False)
        

for i in range(10):
    print(population.subs[i].loss)
sys.stdout.flush()

# Needs a different evaluation method
# for i in range(10):
#     print(util.evaluate(population.subs[i],net,tanh=False))
