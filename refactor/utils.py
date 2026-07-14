import math
import os
import sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.multiprocessing as mp
from align_evaluate import evaluate_reconstruction
device = 0

def prepare_tinyimagenet(root="./data"):
	"""Download + extract TinyImageNet-200 (64x64, 200 classes) if not already present,
	and reorganize the val split into per-class folders so torchvision's ImageFolder can
	read it. Self-contained (uses torchvision) — no third-party 'tinyimagenet' package.
	Returns (train_dir, val_dir).
	"""
	import shutil
	from torchvision.datasets.utils import download_and_extract_archive
	base = os.path.join(root, "tiny-imagenet-200")
	if not os.path.isdir(base):
		url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
		print(f"TinyImageNet not found; downloading (~240MB) to {root} ...", file=sys.__stdout__)
		download_and_extract_archive(url, download_root=root)
	train_dir = os.path.join(base, "train")
	val_dir = os.path.join(base, "val")
	# val ships as val/images/*.JPEG + val_annotations.txt; reshape into val/<wnid>/*.JPEG
	val_images = os.path.join(val_dir, "images")
	if os.path.isdir(val_images):
		with open(os.path.join(val_dir, "val_annotations.txt")) as f:
			for line in f:
				parts = line.split("\t")
				fname, wnid = parts[0], parts[1]
				cls_dir = os.path.join(val_dir, wnid)
				os.makedirs(cls_dir, exist_ok=True)
				shutil.move(os.path.join(val_images, fname), os.path.join(cls_dir, fname))
		shutil.rmtree(val_images)  # now empty
	return train_dir, val_dir


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
		elif dataset == "tinyimagenet":
			transform_tiny = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			train_dir, val_dir = prepare_tinyimagenet('./data')  # auto-downloads if missing
			# ImageFolder's default loader converts to RGB, so grayscale images become 3-channel
			trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_tiny)
			test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_tiny)
			input_dim = 64*64*3
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

def blackbox_cache_key(model_type, layers, activation, dataset, num_epochs, seed, optimizer="adam", seqlens=None):
	"""Identity of a trained black-box: everything that determines its weights.

	Deliberately excludes num_samples / outer_iterations / population_size — those affect
	the reconstruction, not the black-box, so a scaling sweep over K reuses one black-box.
	`layers` may be a list of ints (fnn/rnn) or of 'in,out,k,s' strings (cnn).
	"""
	layers_str = "-".join(str(l) for l in layers)
	key = f"type-{model_type}_layers-{layers_str}_act-{activation}_dataset-{dataset}_epochs-{num_epochs}_opt-{optimizer}_seed-{seed}"
	if model_type in ("rnn", "transformer") and seqlens is not None:
		key += "_seqlens-" + "-".join(str(s) for s in seqlens)
	return key.replace(",", "x").replace(" ", "")  # filename-safe (cnn specs contain commas)


def load_or_train_blackbox(model, cache_key, num_epochs, dataset, model_type, seqlens, cache_dir="./blackbox_cache"):
	"""Load a cached black-box matching cache_key, or train it and cache it.

	Returns True on cache hit, False on miss. The weights are loaded into `model` in place
	(on whatever device it already lives on).
	"""
	cache_path = os.path.join(cache_dir, cache_key + ".pt")
	if os.path.exists(cache_path):
		model.load_state_dict(torch.load(cache_path, map_location="cpu"))
		msg = f"[blackbox-cache] HIT: loaded {cache_path} (skipped training)"
		print(msg)
		print(msg, file=sys.__stdout__)
		return True
	train_blackbox(model, num_epochs=num_epochs, dataset=dataset, model_type=model_type, seqlens=seqlens)
	os.makedirs(cache_dir, exist_ok=True)
	torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, cache_path)
	msg = f"[blackbox-cache] MISS: trained and cached to {cache_path}"
	print(msg)
	print(msg, file=sys.__stdout__)
	return False


def get_adv(sub_list, lr=0.01, epochs=100, num_samples=1000, schedule=[], reverse=False, range_=1, device=device, input_dim=784, model_type='fnn', sequence_length=None, gpu_ids=None, gpu_model_copies=None):
	"""
	Generate adversarial inputs that maximize disagreement among sub-models.

	Args:
		gpu_ids: List of GPU ids to distribute SAMPLES across. Each GPU owns its
		         chunk of samples and optimizes independently (no cross-GPU communication).
		gpu_model_copies: Dict mapping gpu_id -> list of model copies on that GPU.
		                  If None and gpu_ids is provided, models will be copied (slow!).
		                  Pass pre-copied models for best performance.
	"""
	if model_type == 'rnn' or model_type == 'transformer':
		input_dim = int(input_dim * sequence_length / (math.sqrt(input_dim)))

	use_multi_gpu = gpu_ids is not None and len(gpu_ids) > 1

	if use_multi_gpu:
		import copy
		import threading

		num_gpus = len(gpu_ids)
		chunk_size = (num_samples + num_gpus - 1) // num_gpus
		results = [None] * num_gpus

		# Use pre-copied models if provided, otherwise copy (slow)
		if gpu_model_copies is None:
			print("Warning: copying models to GPUs (slow). Pass gpu_model_copies for better performance.")
			gpu_model_copies = {}
			for gpu_id in gpu_ids:
				gpu_model_copies[gpu_id] = [copy.deepcopy(s).cuda(gpu_id) for s in sub_list]

		# Pre-create embeddings and optimizers OUTSIDE threads to avoid GIL contention
		gpu_data = {}
		for gpu_idx in range(num_gpus):
			gpu_id = gpu_ids[gpu_idx]
			start_idx = gpu_idx * chunk_size
			end_idx = min(start_idx + chunk_size, num_samples)
			if start_idx >= num_samples:
				continue
			n_samples = end_idx - start_idx

			torch.cuda.set_device(gpu_id)
			adv = nn.Embedding(n_samples, input_dim).cuda(gpu_id)
			nn.init.uniform_(adv.weight, -range_, range_)
			optimizer = torch.optim.Adam(adv.parameters(), lr=lr)
			gpu_data[gpu_idx] = (adv, optimizer, n_samples)

		# Synchronize all GPUs before starting threads
		torch.cuda.synchronize()

		def train_on_gpu(gpu_idx):
			if gpu_idx not in gpu_data:
				return
			gpu_id = gpu_ids[gpu_idx]
			torch.cuda.set_device(gpu_id)

			adv, optimizer, n_samples = gpu_data[gpu_idx]
			models = gpu_model_copies[gpu_id]

			local_lr = lr
			for epoch in range(epochs):
				if epoch in schedule:
					local_lr = local_lr / 10
					optimizer = torch.optim.Adam(adv.parameters(), lr=local_lr)

				# Reshape weight according to model type
				if model_type == 'cnn':
					weight = adv.weight.view(n_samples, 1, int(math.sqrt(input_dim)), int(math.sqrt(input_dim)))
				elif model_type == 'rnn' or model_type == 'transformer':
					weight = adv.weight.view(n_samples, sequence_length, input_dim // sequence_length)
				else:
					weight = adv.weight

				# Forward through all models
				outs = [torch.nn.functional.normalize(m(weight), p=1.0, dim=-1) for m in models]
				outs = torch.stack(outs).transpose(1, 0).contiguous()
				dists = torch.cdist(outs, outs)

				error = -dists.flatten().mean()
				if reverse:
					error = -error
				error.backward()
				optimizer.step()
				optimizer.zero_grad()

			# Return final weights
			if model_type == 'cnn':
				results[gpu_idx] = adv.weight.view(n_samples, 1, int(math.sqrt(input_dim)), int(math.sqrt(input_dim))).detach().cpu()
			elif model_type == 'rnn' or model_type == 'transformer':
				results[gpu_idx] = adv.weight.view(n_samples, sequence_length, input_dim // sequence_length).detach().cpu()
			else:
				results[gpu_idx] = adv.weight.detach().cpu()

		# Launch all GPUs in parallel
		threads = [threading.Thread(target=train_on_gpu, args=(i,)) for i in range(num_gpus)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()

		# Concatenate results
		weight = torch.cat([r for r in results if r is not None], dim=0)
		return weight

	else:
		# Single GPU path (original code)
		adv = nn.Embedding(num_samples, input_dim)
		adv.cuda(device)
		adv.apply(lambda x: nn.init.uniform_(x.weight, -range_, range_))
		print(adv.weight.detach().abs().cpu().mean())
		optimizer = torch.optim.Adam(adv.parameters(), lr=lr)

		for s in sub_list:
			s.to(device)

		error = 0
		for epoch in range(epochs):
			if epoch in schedule:
				lr = lr / 10
				optimizer = torch.optim.Adam(adv.parameters(), lr=lr)

			if model_type == 'cnn':
				weight = adv.weight.view(num_samples, 1, int(math.sqrt(input_dim)), int(math.sqrt(input_dim)))
			elif model_type == 'rnn' or model_type == 'transformer':
				weight = adv.weight.view(num_samples, sequence_length, input_dim // sequence_length)
			else:
				weight = adv.weight

			outs = []
			for s in sub_list:
				out = torch.nn.functional.normalize(s(weight), p=1.0, dim=-1)
				outs.append(out)
				s.zero_grad()

			outs = torch.stack(outs)
			outs = torch.transpose(outs, 1, 0).contiguous()
			dists = torch.cdist(outs, outs)

			if error == 0:
				if reverse:
					print("init. error:", (dists.flatten().mean()))
				else:
					print("init. error:", -(dists.flatten().mean()))

			error = -(dists.flatten().mean())
			if reverse:
				error = -error
			error.backward()
			optimizer.step()
			optimizer.zero_grad()

		print("final error:", error)
		print("stats:", weight.detach().abs().cpu().mean(), adv.weight.detach().cpu().mean())
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

		# Disk spill: keep at most `ram_chunks` chunks in RAM; when `window` exceeds that,
		# write chunks to disk under `spill_dir` and stream them at training time. Entries in
		# self.inputs/outputs are then either tensors (RAM) or file-path strings (disk).
		self.ram_chunks = 0        # 0 => no limit => never spill (RAM-only)
		self.spill_dir = None
		self._cid = 0              # unique id for spill filenames

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_spill(self, ram_chunks, spill_dir):
		"""Enable disk spill: at most ram_chunks chunks kept in RAM, the rest on disk."""
		self.ram_chunks = ram_chunks
		self.spill_dir = spill_dir

	def _spilling(self, window):
		return bool(self.ram_chunks) and self.spill_dir is not None and window is not None and window > self.ram_chunks

	def _store_chunk(self, inputs, outputs, lst_in, lst_out, spilling):
		"""Append a chunk to (lst_in, lst_out) as an in-RAM tensor, or spill it to disk and
		append the file paths instead."""
		if spilling:
			os.makedirs(self.spill_dir, exist_ok=True)
			px = os.path.join(self.spill_dir, f"cx_{self._cid}.pt")
			py = os.path.join(self.spill_dir, f"cy_{self._cid}.pt")
			self._cid += 1
			torch.save(inputs, px)
			torch.save(outputs, py)
			lst_in.append(px)
			lst_out.append(py)
		else:
			lst_in.append(inputs)
			lst_out.append(outputs)

	@staticmethod
	def _trim(lst_in, lst_out, window):
		"""Keep only the last `window` chunks; delete disk files for the dropped ones."""
		if window is not None and len(lst_in) > window:
			for rx, ry in zip(lst_in[:-window], lst_out[:-window]):
				for r in (rx, ry):
					if isinstance(r, str) and os.path.exists(r):
						os.remove(r)
			del lst_in[:-window]
			del lst_out[:-window]

	def add_data(self,inputs,outputs,window = None):
		spilling = self._spilling(window)
		self._store_chunk(inputs, outputs, self.inputs, self.outputs, spilling)
		self._trim(self.inputs, self.outputs, window)

		# Legacy single-GPU train_one_epoch needs an in-RAM dataset; only build it when
		# nothing is spilled (all tensors). When spilling, the parallel trainer reads the
		# chunk refs (tensors/paths) directly, so no cat and no full in-RAM dataset.
		if all(torch.is_tensor(r) for r in self.inputs):
			self.datasets[0] = ConcatDataset([SampleDataset(x, y) for x, y in zip(self.inputs, self.outputs)])
		else:
			self.datasets.pop(0, None)

	def add_seq_data(self, inputs, outputs, seq_len, window = None):
		self.inputs_dict[seq_len].append(inputs)
		self.outputs_dict[seq_len].append(outputs)

		if window is not None and len(self.inputs_dict[seq_len]) > window:
			self.inputs_dict[seq_len] = self.inputs_dict[seq_len][-window:]
			self.outputs_dict[seq_len] = self.outputs_dict[seq_len][-window:]
		self.datasets[seq_len] = ConcatDataset([SampleDataset(x, y) for x, y in zip(self.inputs_dict[seq_len], self.outputs_dict[seq_len])])

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
