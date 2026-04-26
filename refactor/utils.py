import math
import sys
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from align_evaluate import evaluate_reconstruction

# Default device (legacy compatibility)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_gpu_ids(num_gpus=1):
    """Get list of GPU IDs to use."""
    if not torch.cuda.is_available():
        return []
    total_gpus = torch.cuda.device_count()
    return list(range(min(num_gpus, total_gpus)))


def probe_gpu_memory(model_sample, input_dim, gpu_id=0, model_type='fnn',
                     sequence_length=None, num_students=10, safety_factor=0.7):
    """
    Estimate maximum samples that can fit on GPU for get_adv() using analytical calculation.

    The main memory bottleneck is cdist which is O(n² * num_students).

    Args:
        model_sample: A sample model (used to estimate output dim)
        input_dim: Input dimension for samples
        gpu_id: Which GPU to check (assumes all GPUs are identical)
        model_type: Type of model ('fnn', 'cnn', 'rnn', 'transformer')
        sequence_length: Sequence length for RNN/transformer
        num_students: Number of students in population
        safety_factor: Fraction of estimated max to use (default 0.7)

    Returns:
        Maximum recommended samples per batch
    """
    # Get available GPU memory
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    reserved_mem = torch.cuda.memory_reserved(gpu_id)
    available_mem = total_mem - reserved_mem

    # Estimate output dimension from model
    output_dim = model_sample.layers[-1].out_features if hasattr(model_sample.layers[-1], 'out_features') else 10

    # Memory for cdist is the bottleneck: n² * num_students * output_dim * 4 bytes
    # Solve: n² * num_students * output_dim * 4 <= available_mem * safety_factor
    # n <= sqrt(available_mem * safety_factor / (num_students * output_dim * 4))

    max_samples = int(math.sqrt(available_mem * safety_factor / (num_students * output_dim * 4)))

    # Also account for embedding memory: n * input_dim * 4 bytes
    # But this is typically much smaller than cdist for reasonable n

    # Clamp to reasonable range
    max_samples = max(1000, min(max_samples, 50000))

    print(f"GPU {gpu_id}: {total_mem/1e9:.1f}GB total, estimated max samples = {max_samples}", file=sys.stderr)
    return max_samples

def train_blackbox(net,num_epochs=25,dataset="mnist",optim_="adam", model_type='fnn', seqlens=[28]):    
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
	elif dataset == 'tinyimagenet':
		from tinyimagenet import TinyImageNet
		from pathlib import Path
		tinyimg_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		data_dir = Path("./data/tinyimagenet/")
		trainset = TinyImageNet(data_dir, split='train', transform=tinyimg_transform)
		test_dataset = TinyImageNet(data_dir, split='val', transform=tinyimg_transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
		input_dim = 64*64*3
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
	if (model_type == 'rnn' or model_type == 'transformer') and seqlens is None:
		raise ValueError("sequence_length must be provided for rnn and transformer models")

	for epoch in range(num_epochs):
		net.train()
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			if model_type == 'fnn':
				inputs = inputs.view(-1, input_dim)
			elif model_type == 'rnn' or model_type == 'transformer':
				sequence_length = seqlens[i % len(seqlens)] # cycle through provided sequence lengths using modulo
				inputs = inputs.squeeze(1)
				sequence_length = min(sequence_length, inputs.size(1))
				inputs = inputs[:, :sequence_length, :]

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
	#input_dim should be flattened input size for all models

	# Normalize device to torch.device
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	elif isinstance(device, int):
		device = torch.device(f"cuda:{device}")

	if model_type == 'rnn' or model_type == 'transformer':
		#only generate truncated number of sequences if sequence_length is specified
		input_dim = int(input_dim*sequence_length/(math.sqrt(input_dim)))

	adv = nn.Embedding(num_samples,input_dim)
	adv.to(device)
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
			s.to(device)
			#out = softmax(s(adv.weight)) 
			# Reshape adv.weight according to model type
			if model_type == 'cnn':
				weight = adv.weight.view(num_samples, 1, int(math.sqrt(input_dim)), int(math.sqrt(input_dim)))
			elif model_type == 'rnn' or model_type == 'transformer':
				#input dim here has been changed to truncated then flattened size
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


def _get_adv_worker(rank, gpu_id, sub_list_state_dicts, model_config, num_samples,
                    lr, epochs, schedule, reverse, range_, input_dim, model_type,
                    sequence_length, results_dict):
    """
    Worker function for parallel get_adv execution.

    Runs in a separate process with its own CUDA context.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        # Reconstruct models from state dicts
        from models import var_FNN, var_CNN, var_RNN, base_TransformerEncoder

        sub_list = []
        for state_dict in sub_list_state_dicts:
            if model_config['type'] == 'fnn':
                model = var_FNN(model_config['activation'], model_config['layers'])
            elif model_config['type'] == 'cnn':
                model = var_CNN(model_config['input_shape'], model_config['layers'], model_config['activation'])
            elif model_config['type'] == 'rnn':
                model = var_RNN(model_config['input_shape'], model_config['layers'])
            elif model_config['type'] == 'transformer':
                model = base_TransformerEncoder(model_config['input_shape'], model_config['layers'])
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            sub_list.append(model)

        # Run get_adv on this GPU
        result = get_adv(
            sub_list, lr=lr, epochs=epochs, num_samples=num_samples,
            schedule=schedule, reverse=reverse, range_=range_,
            device=device, input_dim=input_dim, model_type=model_type,
            sequence_length=sequence_length
        )

        results_dict[rank] = result.cpu()

    except Exception as e:
        print(f"Worker {rank} on GPU {gpu_id} failed: {e}", file=sys.stderr)
        results_dict[rank] = None


def get_adv_parallel(sub_list, gpu_ids, samples_per_gpu, total_samples, model_config,
                     lr=0.01, epochs=100, schedule=[], reverse=False, range_=1,
                     input_dim=784, model_type='fnn', sequence_length=None):
    """
    Generate adversarial samples in parallel across multiple GPUs.

    Args:
        sub_list: List of student models (on any device)
        gpu_ids: List of GPU IDs to use
        samples_per_gpu: Maximum samples each GPU can handle
        total_samples: Total samples to generate
        model_config: Dict with model reconstruction info
        ... other get_adv params

    Returns:
        Concatenated tensor of all generated samples
    """
    num_gpus = len(gpu_ids)

    if num_gpus == 1:
        # Fall back to sequential batched approach
        all_results = []
        remaining = total_samples
        device = torch.device(f"cuda:{gpu_ids[0]}")
        while remaining > 0:
            batch_size = min(samples_per_gpu, remaining)
            result = get_adv(
                sub_list, lr=lr, epochs=epochs, num_samples=batch_size,
                schedule=schedule, reverse=reverse, range_=range_,
                device=device, input_dim=input_dim, model_type=model_type,
                sequence_length=sequence_length
            )
            all_results.append(result)
            remaining -= batch_size
        return torch.cat(all_results, dim=0)

    # Multi-GPU: Calculate samples distribution
    samples_distribution = []
    remaining = total_samples
    while remaining > 0:
        for gpu_id in gpu_ids:
            if remaining <= 0:
                break
            samples_this_gpu = min(samples_per_gpu, remaining)
            samples_distribution.append((gpu_id, samples_this_gpu))
            remaining -= samples_this_gpu

    # Prepare model state dicts for transfer (can't pickle CUDA tensors)
    sub_list_state_dicts = [s.cpu().state_dict() for s in sub_list]

    # Use multiprocessing Manager for shared dict
    manager = mp.Manager()
    results_dict = manager.dict()

    # Spawn processes
    processes = []
    for rank, (gpu_id, num_samples) in enumerate(samples_distribution):
        p = mp.Process(
            target=_get_adv_worker,
            args=(rank, gpu_id, sub_list_state_dicts, model_config, num_samples,
                  lr, epochs, schedule, reverse, range_, input_dim, model_type,
                  sequence_length, results_dict)
        )
        p.start()
        processes.append(p)
        print(f"Started worker {rank} on GPU {gpu_id} for {num_samples} samples", file=sys.stderr)

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect and concatenate results in order
    ordered_results = []
    for i in range(len(samples_distribution)):
        if results_dict.get(i) is not None:
            ordered_results.append(results_dict[i])
        else:
            raise RuntimeError(f"Worker {i} failed to produce results")

    # Move models back to original devices
    for s in sub_list:
        s.cuda()

    return torch.cat(ordered_results, dim=0)


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
	def __init__(self, subs, gpu_ids=None):
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

		# Multi-GPU support
		if gpu_ids is None:
			self.gpu_ids = [0]
		else:
			self.gpu_ids = gpu_ids
		self.device_assignments = {}
		self._assign_students_to_gpus()

	def _assign_students_to_gpus(self):
		"""Distribute students across available GPUs."""
		num_gpus = len(self.gpu_ids)
		print(f"Distributing {len(self.subs)} students across {num_gpus} GPUs: {self.gpu_ids}", file=sys.stderr)
		for i, student in enumerate(self.subs):
			gpu_idx = i % num_gpus
			gpu_id = self.gpu_ids[gpu_idx]
			self.device_assignments[i] = torch.device(f"cuda:{gpu_id}")
			student.to(self.device_assignments[i])
			print(f"  Student {i} -> GPU {gpu_id}", file=sys.stderr)

	def cuda(self, device=None):
		"""Override cuda to respect multi-GPU assignments or use single device."""
		if device is not None and len(self.gpu_ids) == 1:
			# Single GPU mode - move all to specified device
			for s in self.subs:
				s.to(device)
		# Multi-GPU mode already handled by _assign_students_to_gpus
		return self
	
	def set_optimizer(self, optimizer):
		# Legacy single optimizer (unused in multi-GPU mode)
		self.optimizer = optimizer

	def set_optimizers(self, optimizer_class, lr):
		"""Create one optimizer per student for multi-GPU training."""
		self.optimizers = []
		for s in self.subs:
			self.optimizers.append(optimizer_class(s.parameters(), lr=lr))
	
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
				# Multi-GPU: each student trains on its own GPU with its own CUDA stream
				# Create streams for each GPU if not already created
				if not hasattr(self, 'streams'):
					self.streams = {}
					for gpu_id in self.gpu_ids:
						self.streams[gpu_id] = torch.cuda.Stream(device=gpu_id)

				losses_list = []

				# Launch all students in parallel using streams
				for i, s in enumerate(self.subs):
					dev = next(s.parameters()).device
					gpu_id = dev.index
					stream = self.streams[gpu_id]

					with torch.cuda.stream(stream):
						x_dev = x.to(dev, non_blocking=True)
						y_dev = y.to(dev, non_blocking=True)
						self.optimizers[i].zero_grad()
						y_hat = s(x_dev)
						loss = criterion(y_hat, y_dev) * 200
						loss.backward()
						self.optimizers[i].step()
						losses_list.append(loss)

				# Sync all streams
				for gpu_id in self.gpu_ids:
					self.streams[gpu_id].synchronize()

				running_losses += np.array([l.detach().cpu().item() for l in losses_list])

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
		mse, mae, max_ae, mmpe, max_mpe, layerwise_metrics = evaluate_reconstruction(net,self.subs[self.best], model_type=model_type)
		print("total mse:", f"{mse:.3e}")
		print("total mae:", f"{mae:.3e}")
		print("total max_ae:", f"{max_ae:.3e}")
		print("total mean_mag_pe:", f"{mmpe:.3e}%")
		print("total max_mag_pe:", f"{max_mpe:.3e}%")
		for l_mse, l_mae, l_max_ae, l_mmpe, l_max_mpe, layername in layerwise_metrics:
			print(f"{layername} - mse: {l_mse:.3e}, mae: {l_mae:.3e}, max_ae: {l_max_ae:.3e}, mean_mag_pe: {l_mmpe:.3e}%, max_mag_pe: {l_max_mpe:.3e}%")
		print("-"*50)


def get_input_dim_and_shape(dataset, model_type):
	'''
	Returns flattened input dimension and input shape (either int or tuple) based on dataset and model type.
	input_dim is always flattened input size, used in get_adv.
	input_shape is used as parameter to model constructors.
	Returns:
	- input_dim: int
	- input_shape: int or tuple
	'''
	if dataset in ['mnist','fmnist','kmnist']:
		input_shape = (1, 28, 28)
	elif dataset in ['cifar10','cifar100']:
		input_shape = (3, 32, 32)
	elif dataset == 'tinyimagenet':
		input_shape = (3, 64, 64)
	elif dataset == 'places365':
		input_shape = (3, 256, 256)
	else:
		raise NotImplementedError("Dataset not supported for input shape inference")
	
	input_dim = math.prod(input_shape) #total number of dimensions for flattened input, used in get_adv

	#input_shape is used as parameter to model constructors
	if model_type == 'fnn':
		#input_shape is not used in fnn constructor, but we keep it for consistency
		return input_dim, input_shape
	elif model_type == 'cnn':
		return input_dim, input_shape
	elif model_type == 'rnn' or model_type == 'transformer':
		return input_dim, input_shape[-1] # for rnn and transformer, input shape is (batch_size, seq_len, input_size), so we return input_size
