import math
import sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from align_evaluate import evaluate_reconstruction
device = 0

def train_blackbox(net,num_epochs=25,dataset="mnist",optim_="adam", model_type='fnn'):    
	if not dataset in ['mnist','fmnist','kmnist','cifar10','cifar100','places365', 'tinyimagenet']:
		raise ValueError("Unknown Dataset")
	if not optim_ in ["adam","rmsprop","sgd","adagrad","adadelta","rprop"]:
		raise ValueError("Unknown Optimizer")
	
	# Prepare datasets
	if dataset=='places365':
		big_transform = transforms.Compose([
			transforms.Resize(256),
			#transforms.CenterCrop(224),
			transforms.ToTensor(), # convert the images to a PyTorch tensor
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
					])
		image_dir = "./data/places365"
		download = True
		from pathlib import Path
		if Path(image_dir).is_dir():
			download = False
		trainset = torchvision.datasets.Places365(root=image_dir, split='train-standard', small=True, transform=big_transform,download=download)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

		test_dataset = torchvision.datasets.Places365(root=image_dir, split='val', small=True, transform=big_transform,download=download)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
		input_dim = 256*256*3
	else:
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
		if dataset=='mnist':
			trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
			test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
			input_dim = 28*28
		elif dataset == "cifar10":
			trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
			test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
			input_dim = 32*32*3
		elif dataset == "cifar100":
			trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
			test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
			input_dim = 32*32*3
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
		
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
				
				if model_type == 'fnn':
					images = images.view(-1, input_dim)
				elif model_type == 'rnn' or model_type == 'transformer':
					images = images.squeeze(1)
	
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
	if optim_ == "adadelta":
		optimizer = optim.Adadelta(net.parameters(), lr=0.01)
	if optim_ == "rprop":
		optimizer = optim.Rprop(net.parameters(), lr=0.01)
		
	# Train the neural network
	for epoch in range(num_epochs):
		net.train()
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			if model_type == 'fnn':
				inputs = inputs.view(-1, input_dim)
			elif model_type == 'rnn' or model_type == 'transformer':
				inputs = inputs.squeeze(1)
				sequence_length = 4
				sequence_length = min(sequence_length, inputs.size(1))
				inputs = inputs[:, :4, :]

			inputs, labels = inputs.to(device), labels.to(device)
	   
			optimizer.zero_grad()
			
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
	
			running_loss += loss.item()
		print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")
		# Calculate and print accuracy
		net.eval()
		accuracy = evaluate_accuracy(net)
		print(f"Accuracy on dataset: {accuracy:.4f}")
		net.train()

def get_adv(sub_list,lr=0.01,epochs=100,num_samples=1000,schedule = [],reverse=False,range_=1,device=device,input_dim=784, model_type='fnn', sequence_length=None):
	# Generate adversarial inputs that maximize disagreement among the sub-models
	if model_type == 'rnn' or model_type == 'transformer':
		input_dim = int(input_dim*sequence_length/(math.sqrt(input_dim)))
	
	adv = nn.Embedding(num_samples,input_dim)
	adv.cuda(device)
	range_ = range_
	adv.apply(lambda x: nn.init.uniform_(x.weight, -range_, range_))
	print(adv.weight.detach().abs().cpu().mean())
	optimizer = torch.optim.Adam(adv.parameters(), lr=lr)
	error=0
	# softmax = torch.nn.Softmax()
	for epoch in range(epochs):
		if epoch in schedule:
			lr = lr/10
			optimizer = torch.optim.Adam(adv.parameters(), lr=lr)
		outs = []
		for idx,s in enumerate(sub_list):
			s.cuda(device)
			#out = softmax(s(adv.weight)) 
			# Reshape adv.weight according to model type
			if model_type == 'cnn':
				weight = adv.weight.view(num_samples, 1, int(math.sqrt(input_dim)), int(math.sqrt(input_dim)))
			elif model_type == 'rnn' or model_type == 'transformer':
				batch_size = num_samples
				weight = adv.weight.view(batch_size, sequence_length, input_dim//sequence_length)
			else:
				weight = adv.weight
			out = torch.nn.functional.normalize(s(weight), p=1.0, dim=-1)
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
	print("stats:", weight.detach().abs().cpu().mean(),adv.weight.detach().cpu().mean())
	return weight.detach().cpu()

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

class Population(nn.Module):
	def __init__(self,subs):
		super(Population, self).__init__()
		self.subs = nn.ModuleList(subs)
		self.inputs = []
		self.outputs = []
		self.best = None
		self.pop_size = len(subs)
		self.ds = None

		self.inputs_dict = defaultdict(list)
		self.outputs_dict = defaultdict(list)
		self.datasets = {}
	
	def set_optimizer(self, optimizer):
		self.optimizer = optimizer
	
	def add_data(self,inputs,outputs,window = None):
		self.inputs.append(inputs)
		self.outputs.append(outputs)

		if window is None or len(self.inputs) <=window:
			self.datasets[0] = SampleDataset(torch.cat(self.inputs),torch.cat(self.outputs))      
		else:
			self.datasets[0] = SampleDataset(torch.cat(self.inputs[-window:]),torch.cat(self.outputs[-window:]))

	def add_seq_data(self, inputs, outputs, seq_len, window = None):
		self.inputs_dict[seq_len].append(inputs)
		self.outputs_dict[seq_len].append(outputs)

		if window is None or len(self.inputs_dict[seq_len]) <= window:
			self.datasets[seq_len] = SampleDataset(torch.cat(self.inputs_dict[seq_len]), torch.cat(self.outputs_dict[seq_len]))
		else:
			self.datasets[seq_len] = SampleDataset(torch.cat(self.inputs_dict[seq_len][-window:]), torch.cat(self.outputs_dict[seq_len][-window:]))

	def save(self,PATH):
		torch.save(self.state_dict(), PATH)

	def load(self,PATH):
		self.load_state_dict(torch.load(PATH))

	def train_one_epoch(self,batch_size = 128,epoch_num=0,restore=False,bottom_half=False):
		#check datasets. If only one dataset is used, populate the datasets dict with one item. 
		if self.ds is not None and len(self.datasets) == 0:
			self.datasets[0] = self.ds
		elif self.ds is None and len(self.datasets) == 0:
			raise Exception("no datasets")
		
		if bottom_half:
			original = self.subs
			self.subs = nn.ModuleList(sorted(self.subs, key=lambda x: x.loss,reverse=True))
			self.subs = nn.ModuleList(self.subs[:len(self.subs)//2])
		best = None
		pop_size = len(self.subs)
		optimizer = self.optimizer
		criterion = nn.L1Loss()

		running_losses = np.array([0.0]*pop_size)
		#initialize dataloaders for each sequence length dataset
		loaders = {length: iter(DataLoader(ds, batch_size=batch_size, shuffle=True))
				   for length, ds in self.datasets.items()}
		dataset_size = sum(len(dl) for dl in loaders.values())
		active_datasets = set(loaders.keys()) #set to keep track of each dataset that still has items
		while active_datasets:
			for seq_length in list(active_datasets):
				try:
					x,y = next(loaders[seq_length]) # if this fails, means that that dataset is exhausted
				except:
					active_datasets.remove(seq_length) #remove exhausted dataset
					continue
				x=x.cuda(device)
				y=y.cuda(device)
				optimizer.zero_grad()
				y_hats = self(x)
		
				loss = [criterion(y_hats[i], y) for i in range(pop_size)]
				
				(sum(loss)*200).backward()

				if restore:
					self.restore_grad()
				optimizer.step()   
				running_losses += torch.tensor(loss).detach().numpy()

		losses = list(running_losses/dataset_size)
		
		print(f"Epoch {epoch_num+1}, Min Loss: {min(losses)}, Max Loss: {max(losses)},Mean Loss: {np.array(losses).mean()}") 
		self.best = losses.index(min(losses))
		for i in range(len(losses)):
			self.subs[i].loss = losses[i]

		if bottom_half:
			self.subs = original

	def forward(self, x):
		outs = []
		for s in self.subs:
			outs.append(s(x))
		return outs
	
	def evaluate(self,net, model_type='fnn'):
		# Prints alignment metrics between the lowest-loss sub-model and the blackbox
		print("-"*50)
		mse, mae, max_ae, mape, max_pe, layerwise_metrics = evaluate_reconstruction(net,self.subs[self.best], model_type=model_type)
		print("total mse:", f"{mse:.3e}")
		print("total mae:", f"{mae:.3e}")
		print("total max_ae:", f"{max_ae:.3e}")
		print("total mape:", f"{mape:.3e}%")
		print("total max_pe:", f"{max_pe:.3e}%")
		for l_mse, l_mae, l_max_ae, l_mape, l_max_pe, layername in layerwise_metrics:
			print(f"{layername} - mse: {l_mse:.3e}, mae: {l_mae:.3e}, max_ae: {l_max_ae:.3e}, mape: {l_mape:.3e}%, max_pe: {l_max_pe:.3e}%")
		print("-"*50)