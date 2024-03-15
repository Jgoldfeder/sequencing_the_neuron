
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.append('./alignment')
sys.path.append('../alignment')

import evaluate as evaluate_
from recon_evals import e_mae,e_layers_mae,e_max_ae,e_mse
import align_cnn 
device = 0



# note: pytorch currently has a bug where this won't work
# https://discuss.pytorch.org/t/tensors-of-the-same-index-must-be-on-the-same-device-and-the-same-dtype-except-step-tensors-that-can-be-cpu-and-float32-notwithstanding/190335
#torch.set_default_dtype(torch.float64)

def evaluate(original, reconstruction,return_blackbox=False,tanh=False,cnn=False):
    if cnn:
        return align_cnn.bruteforce_cnn_evaluate(original,reconstruction,tanh)
    else:
        return evaluate_.evaluate_reconstruction(original, reconstruction,return_blackbox=return_blackbox,tanh=tanh)



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_mean_weight_delta(net1,net2):
    mae = []
    max = []
    for i in range(len(net1.layers)):
        w1 = net1.layers[i].weight.detach().clone()
        w2 = net2.layers[i].weight.detach().clone()
        mae.append(torch.nn.L1Loss()(w1,w2))
        max.append(torch.nn.L1Loss(reduction='none')(w1,w2).flatten().max())
    return mae,max





class Population(nn.Module):
    def __init__(self,subs):
        super(Population, self).__init__()
        self.subs = nn.ModuleList(subs)
        self.inputs = []
        self.outputs = []
        self.best = None
        self.pop_size = len(subs)

    def save(self,PATH):
        torch.save(self.state_dict(), PATH)

    def load(self,PATH):
        self.load_state_dict(torch.load(PATH))
        
    def add_data(self,inputs,outputs,window = None):
        self.inputs.append(inputs)
        self.outputs.append(outputs)

        if window is None or len(self.inputs) <=window:
            self.ds = SampleDataset(torch.cat(self.inputs),torch.cat(self.outputs))      
        else:
            self.ds = SampleDataset(torch.cat(self.inputs[-window:]),torch.cat(self.outputs[-window:]))      
            
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def train_one_epoch(self,batch_size = 128,epoch_num=0,restore=False,bottom_half=False):
        if bottom_half:
            original = self.subs
            self.subs = nn.ModuleList(sorted(self.subs, key=lambda x: x.loss,reverse=True))
            self.subs = nn.ModuleList(self.subs[:len(self.subs)//2])
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=True)
        best = None
        pop_size = len(self.subs)
        optimizer = self.optimizer
        criterion = nn.L1Loss()

        running_losses = np.array([0.0]*pop_size)
        #ratios=[]
        for i in dl:
            x,y = i
            x=x.cuda(device)
            y=y.cuda(device)
            optimizer.zero_grad()
            y_hats = self(x)
    
            loss = [criterion(y_hats[i], y) for i in range(pop_size)]
            
            (sum(loss)*200).backward()

            #ratios.append(self.subs[0].fc1.weight.grad.abs().mean()/self.subs[0].fc2.weight.grad.abs().mean())
            if restore:
                self.restore_grad()
            optimizer.step()   
            running_losses += torch.tensor(loss).detach().numpy()

        losses = list(running_losses/len(dl))
        
        print(f"Epoch {epoch_num+1}, Min Loss: {min(losses)}, Max Loss: {max(losses)},Mean Loss: {np.array(losses).mean()}") 
        self.best = losses.index(min(losses))
        for i in range(len(losses)):
            self.subs[i].loss = losses[i]

        if bottom_half:
            self.subs = original
        #print(torch.tensor(ratios).mean())
    
    def evaluate(self,net,tanh=False):
        print(evaluate(net,self.subs[self.best],tanh=tanh))
        
    def forward(self, x):
        outs = []
        for s in self.subs:
            outs.append(s(x))
        return outs

    def align(self,blackbox=None,anchor=0, verbose = False,tanh=False):
        # aligns population in place, may mess up momentum in optimizer
        
        subs = self.subs
        if blackbox is None:
            blackbox =  subs[anchor]
        else:
            blackbox = copy.deepcopy(blackbox)
            blackbox = evaluate(blackbox,self.subs[self.best], return_blackbox= True,tanh=tanh)
            
        cannonical = subs[anchor]
    
    
        # align cannonical to blackbox
        original = blackbox.cuda()
        reconstruction = cannonical.cuda()
    
        evaluator = e_mae(reconstruction, None,tanh=tanh)
        evaluator.original = original # blackbox
        metric = evaluator.get_evaluation()
    
        metrics = []
        for Evaluator in [e_mae,e_layers_mae,e_max_ae,e_mse]:
            evaluator = Evaluator(reconstruction, None,tanh=tanh)
            evaluator.original = original # blackbox
            metric = evaluator.calculate_distance()
            metrics.append(metric)
        if verbose:
            print("blackbox:")
            print(metrics)
            print("inter population:")
        
        
        
        for s in subs:
            # align
        
            original = cannonical.cuda()
            reconstruction = s.cuda()
    
            # align
            evaluator = e_mae(reconstruction, None,tanh=tanh)
            evaluator.original = original # blackbox
            metric = evaluator.get_evaluation()
        
            metrics = []
            for Evaluator in [e_mae,e_layers_mae,e_max_ae,e_mse]:
                evaluator = Evaluator(reconstruction, None,tanh=tanh)
                evaluator.original = original # blackbox
                metric = evaluator.calculate_distance()
                metrics.append(metric)
            if verbose:
                print(metrics)

    
    def analyze(self,net=None,eps=0.01,delta=0.01,off=5):
        if net is None:
            net = self.subs[0]
        else:
            net = evaluate(net,self.subs[self.best], return_blackbox= True)

        aligned = self.subs

        restore_vals = {}
        for l in range(len(aligned[0].layers)):
            restore_vals["layer_" + str(l)+".weight"] = {}
            uncorrected_dists = []
            corrected_dists = []
            uncorrected_dists_below_eps = []
            corrected_dists_below_eps = []
            below_eps = 0
            below_delta = 0
            total = 0
            layer_below_eps = torch.zeros((aligned[0].layers[l].weight.shape[0],aligned[0].layers[l].weight.shape[1]))
            layer_below_delta = torch.zeros((aligned[0].layers[l].weight.shape[0],aligned[0].layers[l].weight.shape[1]))

            for i in range(aligned[0].layers[l].weight.shape[0]):
                for j in range(aligned[0].layers[l].weight.shape[1]):
                    vals = []
                    for a in aligned:
                        vals.append(a.layers[l].weight[i][j].item())
                    vals.sort()  
                    vals = torch.tensor(vals)

                    uncorrected_dist = abs(net.layers[l].weight[i][j].item() - vals.mean())
                    if off>0:
                        corrected_dist = abs(net.layers[l].weight[i][j].item() - np.array(vals[off:-off]).mean())
                        corrected_range = abs(vals[off]-vals[-off])
                        corrected_val = np.array(vals[off:-off]).mean()
                    else:
                        corrected_dist = uncorrected_dist
                        corrected_range = abs(vals[0]-vals[-1])
                        corrected_val = vals.mean()
 
                    if corrected_range < eps:
                        uncorrected_dists_below_eps.append(uncorrected_dist) 
                        corrected_dists_below_eps.append(corrected_dist) 
                        below_eps+=1
                        layer_below_eps[i][j] = 1
                        if corrected_dist < delta:
                            below_delta+=1
                            layer_below_delta[i][j] = 1

                        restore_vals["layer_" + str(l)+".weight"][(i,j)] = corrected_val
                    total+=1
                    uncorrected_dists.append(uncorrected_dist)
                    corrected_dists.append(corrected_dist)
            print("layer " + str(l) + ":")
            print("weight stats")
            print("max uncorrected error:",max(uncorrected_dists))
            print("max corrected error:",max(corrected_dists))
            print("max uncorrected error below eps:",max(uncorrected_dists_below_eps+[0]))
            print("max corrected error below eps:",max(corrected_dists_below_eps+[0]))

            print("mean uncorrected error:",torch.tensor(uncorrected_dists).mean())
            print("mean corrected error:",torch.tensor(corrected_dists).mean())
            print("mean uncorrected error below eps:",torch.tensor(uncorrected_dists_below_eps).mean())
            print("mean corrected error below eps:",torch.tensor(corrected_dists_below_eps).mean())

            
            print("P(range < e)=",below_eps/total,below_eps,"/",total)
            if below_eps>0:
                print("P(error < d | range < e)=",below_delta/below_eps,below_delta,"/",below_eps)

            dim0 = layer_below_eps.shape[0]
            dim1 = layer_below_eps.shape[1]
            
            rows_below_eps = layer_below_eps.sum(dim=0)
            rows_below_delta  = layer_below_delta.sum(dim=0)
            solved_rows = 0
            below_eps = 0.0001
            below_delta = 0
            for i in range(len(rows_below_eps)):
                if rows_below_eps[i]==dim0:
                    # all in row were below eps
                    solved_rows +=1
                    below_eps += dim0
                    below_delta += rows_below_delta[i]
            print("solved rows:", solved_rows,below_delta/ below_eps,dim1)
            
            cols_below_eps = layer_below_eps.sum(dim=1)
            cols_below_delta  = layer_below_delta.sum(dim=1)
            solved_cols = 0
            below_eps =  0.0001
            below_delta = 0

            for i in range(len(cols_below_eps)):
                if cols_below_eps[i]==dim1:
                    # all in row were below eps
                    solved_cols +=1
                    below_eps += dim1
                    below_delta += cols_below_delta[i]
            print("solved cols:", solved_cols, below_delta/ below_eps,dim0)
                    
            
            restore_vals["layer_" + str(l)+".bias"] = {}
            uncorrected_dists = []
            corrected_dists = []
            uncorrected_dists_below_eps = []
            corrected_dists_below_eps = []
            below_eps = 0
            below_delta = 0
            total = 0
            for i in range(aligned[0].layers[l].bias.shape[0]):
                vals = []
                for a in aligned:
                    vals.append(a.layers[l].bias[i].item())
                vals.sort()   
                vals = torch.tensor(vals)
                uncorrected_dist = abs(net.layers[l].bias[i].item() - vals.mean())
                if off>0:
                    corrected_dist = abs(net.layers[l].bias[i].item() - np.array(vals[off:-off]).mean())
                    corrected_range = abs(vals[off]-vals[-off])
                    corrected_val = np.array(vals[off:-off]).mean()
                else:
                    corrected_dist = uncorrected_dist
                    corrected_range = abs(vals[0]-vals[-1])
                    corrected_val = vals.mean()

                if corrected_range < eps:
                    uncorrected_dists_below_eps.append(uncorrected_dist) 
                    corrected_dists_below_eps.append(corrected_dist) 
                    below_eps+=1
                    if corrected_dist < delta:
                        below_delta+=1
                    restore_vals["layer_" + str(l)+".bias"][i] = corrected_val
                total+=1
                uncorrected_dists.append(uncorrected_dist)
                corrected_dists.append(corrected_dist)
            print("bias stats")
            print("max uncorrected error:",max(uncorrected_dists))
            print("max corrected error:",max(corrected_dists))
            print("max uncorrected error below eps:",max(uncorrected_dists_below_eps+[0]))
            print("max corrected error below eps:",max(corrected_dists_below_eps+[0]))

            print("mean uncorrected error:",torch.tensor(uncorrected_dists).mean())
            print("mean corrected error:",torch.tensor(corrected_dists).mean())
            print("mean uncorrected error below eps:",torch.tensor(uncorrected_dists_below_eps).mean())
            print("mean corrected error below eps:",torch.tensor(corrected_dists_below_eps).mean())
            
            print("P(range < e)=",below_eps/total,below_eps,"/",total)
            if below_eps>0:
                print("P(error < d | range < e)=",below_delta/below_eps,below_delta,"/",below_eps)
        self.restore_vals = restore_vals

        grad_masks = {}
        for l in range(len(aligned[0].layers)):
            grad_masks["layer_" + str(l)+".weight"] = torch.ones(aligned[0].layers[l].weight.shape).cuda()
            grad_masks["layer_" + str(l)+".bias"] = torch.ones(aligned[0].layers[l].bias.shape).cuda()
        
        
            for idx,val in restore_vals["layer_" + str(l)+".weight"].items():
                grad_masks["layer_" + str(l)+".weight"][idx[0]][idx[1]] = 0
            for idx,val in restore_vals["layer_" + str(l)+".bias"].items():
                grad_masks["layer_" + str(l)+".bias"][idx] = 0
        self.grad_masks = grad_masks
        return restore_vals,grad_masks

    def restore_grad(self,):
        grad_masks = self.grad_masks
        for net in  self.subs:
            for l in range(len(net.layers)):            
                net.layers[l].weight.grad *= grad_masks["layer_" + str(l)+".weight"]
                net.layers[l].bias.grad *= grad_masks["layer_" + str(l)+".bias"]

    def _restore(self,net,restore_vals):
        for l in range(len(net.layers)):            
            for idx,val in restore_vals["layer_" + str(l)+".weight"].items():
                with torch.no_grad():
                    net.layers[l].weight[idx[0]][idx[1]] = val
            for idx,val in restore_vals["layer_" + str(l)+".bias"].items():
                with torch.no_grad():
                    net.layers[l].bias[idx] = val
    
    def restore(self):
        for s in self.subs:
            self._restore(s,self.restore_vals)

    

# Define a custom dataset
from torch.utils.data import Dataset, DataLoader
class SampleDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        output_sample = self.outputs[idx]
        return input_sample, output_sample

def init_uniform(net,scale=20):
    def init_weights(m):
        #if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight,-1*scale,1*scale)

    net.apply(init_weights) 

def get_adv(sub_list,lr=0.01,epochs=100,num_samples=1000,schedule = [],reverse=False,range_=1,device=device,input_dim=784):
    adv = nn.Embedding(num_samples,input_dim)
    adv.cuda(device)
    range_ = range_
    init_uniform(adv,range_)#50
    print(adv.weight.detach().abs().cpu().mean())
    optimizer = torch.optim.Adam(adv.parameters(), lr=lr)
    error=0
    softmax = torch.nn.Softmax()
    for epoch in range(epochs):
        if epoch in schedule:
            lr = lr/10
            optimizer = torch.optim.Adam(adv.parameters(), lr=lr)
        outs = []
        for idx,s in enumerate(sub_list):
            s.cuda(device)
            #out = softmax(s(adv.weight)) 
            out = torch.nn.functional.normalize(s(adv.weight), p=1.0, dim=-1)
            #out = s(adv.weight)
            
            outs.append(out)
            s.zero_grad()
        outs = torch.stack(outs)
        outs = torch.transpose(outs,1,0).contiguous()
        dists = torch.cdist(outs,outs)
        #dists = -cosine_cdist(outs,outs)
        if error == 0:
            if reverse:
                print("init. error:", (dists.flatten().mean()))
            else:
                print("init. error:", -(dists.flatten().mean()))
                
        error = -(dists.flatten().mean())
        if reverse:
            error = -error
        #print(error)
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("final error:",error)
    print("stats:", adv.weight.detach().abs().cpu().mean(),adv.weight.detach().cpu().mean())
    return adv.weight.detach().cpu()




def train_blackbox(net,num_epochs=25,dataset="mnist",optim_="adam"):    
    if not dataset in ['mnist','fmnist','kmnist','cifar10','cifar100','places365']:
        raise ValueError("Unknown Dataset")
    if not optim_ in ["adam","rmsprop","sgd","adagrad"]:
        raise ValueError("Unknown Optimizer")
    
    input_dim = 784
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    if dataset == "fmnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    if dataset == "kmnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    if dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        input_dim = 3072
    if dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        input_dim = 3072

    if dataset=='places365':
        
        big_transform = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
                    ])
        image_dir = "./data/places365"
        trainset = torchvision.datasets.Places365(root=image_dir, split='train-standard', small=True, transform=big_transform,download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

        test_dataset = torchvision.datasets.Places365(root=image_dir, split='val', small=True, transform=big_transform,download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        input_dim = 256*256*3
       
    
    def evaluate_accuracy(network):
        # Set the network to evaluation mode
        network.eval()
        
        # Move the network to CUDA if available
        network.to(device)
    
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                images = images.view(-1, input_dim)
    
                outputs = network(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        return accuracy
    
    # Initialize the neural network, loss function, and optimizer

    criterion = nn.CrossEntropyLoss()
    if optim_ == "adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    if optim_ == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=0.01)
    if optim_ == "adagrad":
        optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    if optim_ == "rmsprop":
        optimizer = optim.Adagrad(net.parameters(), lr=0.01)
    
    # Train the neural network
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, input_dim)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")
        # Calculate and print accuracy
        accuracy = evaluate_accuracy(net)
        print(f"Accuracy on MNIST: {accuracy:.4f}")
