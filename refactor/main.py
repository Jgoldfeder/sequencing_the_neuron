import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import gc
from models import var_FNN, var_CNN, var_RNN, base_TransformerEncoder
from align_evaluate import evaluate_reconstruction
import utils

if __name__ == "__main__":
	# Parse arguments for run
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type', '-m', type=str, choices=['fnn', 'cnn', 'rnn', 'transformer'], required=True, 
					 help='Type of model')
	# parser.add_argument('--input_shape', '-is', required=True, help="for fnn/rnn: single int. For cnn: 'C,H,W'")
	parser.add_argument('--layers', '-l', required=True, nargs='+',
					 help="for fnn/rnn: list of ints. For cnn: list of 'in_channels,out_channels,kernel_size,stride'")
	parser.add_argument('--activation', '-a', type=str, choices=['relu', 'tanh', 'nonleakyrelu', 'nonleakyreluapproximation'],
					 default='relu', help='Activation function to use')
	parser.add_argument('--outer_iterations', '-oi', type=int, default=55, help='Number of outer iterations')
	parser.add_argument('--num_samples', '-ns', type=int, default=10000, help='Number of samples to generate')
	parser.add_argument('--num_epochs', '-ne', type=int, default=25, help='Number of training epochs to train the black box')
	parser.add_argument('--seq_len', '-sl', type=int, nargs='+', help='Sequence lengths for RNN/Transformer inputs')
	parser.add_argument('--dataset', '-d', type=str, default='mnist',choices=['mnist', 'cifar10', 'cifar100', 'places365', 'tinyimagenet'],
					 help='Dataset to use for training and evaluation')
	parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
	parser.add_argument('--cheat', action='store_true', help='If set, "cheat" by using gradients from blackbox and population of 1')
	parser.add_argument('--comment', '-c', type=str, default='', help='Additional comment for the run')
	args = parser.parse_args()

	# check that seq_len is provided for rnn and transformer
	if args.model_type in ['rnn', 'transformer'] and args.seq_len is None:
		raise ValueError("For RNN and Transformer models, --seq_len must be specified as a list of integers.")

	# set up logging and output
	name = f"{'-'.join(args.layers)}_outer-iterations-{args.outer_iterations}_samples-{args.num_samples}_epochs-{args.num_epochs}_dataset-{args.dataset}_activation-{args.activation}_seed-{args.seed}_{args.comment}"

	log_dir = "./results/"+args.model_type+"/"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	if args.cheat:
		print("cheating!!", file=sys.stderr)
		name = "cheat_"+name

	models_path = "./models/"+name+"/"+args.model_type+"/"

	if not os.path.exists(models_path):
		os.makedirs(models_path)

	sys.stdout = open(log_dir+name, "w")
	print("Log file for:"+name)
	for arg_name, arg_value in vars(args).items():
		print(f"{arg_name}: {arg_value}")

	# validate arguments, initialize variables
	device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}", file=sys.__stdout__)
	torch.manual_seed(args.seed)
	
	# input_dim = int(args.input_shape) if args.input_shape.isdigit() else np.prod([int(x) for x in args.input_shape.split(',')])
	# input_shape = int(args.input_shape) if args.input_shape.isdigit() else tuple(int(x) for x in args.input_shape.split(','))
	input_dim, input_shape = utils.get_input_dim_and_shape(args.dataset, args.model_type)
	if args.model_type == 'cnn':
		layers = []
		for l in args.layers:
			layerconfigs = {}
			p = l.split(',')
			if len(p) != 4:
				raise ValueError("Each CNN layer must be specified as in_channels,out_channels,kernel_size,stride")
			layerconfigs['in_channels'] = int(p[0])
			layerconfigs['out_channels'] =  int(p[1])
			layerconfigs['kernel_size'] =  int(p[2])
			layerconfigs['stride'] =  int(p[3])
			layers.append(layerconfigs)
	else:
		layers = [int(x) for x in args.layers]

	# print(layers, file=sys.__stdout__)
	# print(type(layers[0][0]), file=sys.__stdout__)
	if args.activation == "tanh":
		activation_f = nn.Tanh()
	elif args.activation == "nonleakyrelu":
		activation_f = nn.ReLU()
	elif args.activation == "nonleakyreluapproximation":
		activation_f = nn.LeakyReLU(negative_slope=0.0001)
	elif args.activation =="relu":
		activation_f = nn.LeakyReLU()
	else:
		raise ValueError("Unsupported activation function")
	
	# initialize model
	if args.model_type == 'fnn':
		model = var_FNN(activation_f, layers)
	elif args.model_type == 'cnn':
		model = var_CNN(input_shape, layers, activation_f)
	elif args.model_type == 'rnn':
		model = var_RNN(input_shape, layers)
	elif args.model_type == 'transformer':
		model = base_TransformerEncoder(input_shape, layers)

	torch.save(model.state_dict(), models_path+"original_params_black_box.pt)")
	model.to(device)

	# train black-box model
	print("Training black-box model", file=sys.__stdout__)
	#utils.train_blackbox(model, num_epochs=args.num_epochs, dataset=args.dataset, model_type=args.model_type, seqlens=args.seq_len)
	#using seq len of 28 instead of sampling seq lens
	utils.train_blackbox(model, num_epochs=args.num_epochs, dataset=args.dataset, model_type=args.model_type, seqlens=[28])
	print(model)
	print("weight mean magnitude per layer")
	if args.model_type != 'transformer':
		for l in model.layers:
			if isinstance(l, nn.RNN):
				print("input weights:", l.weight_ih_l0.abs().mean())
				print("hidden weights:", l.weight_hh_l0.abs().mean())
			else:
				print("weights:", l.weight.abs().mean())

	# load original model, save black box
	if args.model_type == 'fnn':
		og_model = var_FNN(activation_f, layers)
	elif args.model_type == 'cnn':
		og_model = var_CNN(input_shape, layers, activation_f)
	elif args.model_type == 'rnn':
		og_model = var_RNN(input_shape, layers)
	elif args.model_type == 'transformer':
		og_model = base_TransformerEncoder(input_shape, layers)

	og_model.to(device)
	og_model.load_state_dict(torch.load(models_path+"original_params_black_box.pt)"))
	torch.save(model.state_dict(), models_path+"black_box.pt")

	# train children population
	print("Training student population", file=sys.__stdout__)
	if args.cheat:
		pop_size = 1
	else:
		pop_size = 10
	subs = []
	for i in range(pop_size):
		if args.model_type == 'fnn':
			subs.append(var_FNN(activation_f, layers))
		elif args.model_type == 'cnn':
			subs.append(var_CNN(input_shape, layers, activation_f))
		elif args.model_type == 'rnn':
			subs.append(var_RNN(input_shape, layers))
		elif args.model_type == 'transformer':
			subs.append(base_TransformerEncoder(input_shape, layers))
	population = utils.Population(subs)
	population.cuda(device)
	model = model.cuda(device)
	
	criterion = nn.L1Loss()
	lr = 0.001
	population.set_optimizer(optim.Adam(population.parameters(), lr=lr))

	with torch.enable_grad():
		for outer_iter in range(args.outer_iterations):
			print(f"Outer iteration {outer_iter}", file=sys.__stdout__)
			sys.stdout.flush()
			restore = False
			if outer_iter > 25:
				lr = lr * 0.8
				population.set_optimizer(optim.Adam(population.parameters(), lr=lr))

			print("ITERATION: ",outer_iter, len(population.inputs))

			#use committee sampling to generate samples
			if args.cheat:
				subslist = population.subs + [model]
			else:
				subslist = population.subs
			samples_to_generate = args.num_samples
			if args.model_type == 'rnn' or args.model_type == 'transformer':
				samples_per_seq_len = samples_to_generate // len(args.seq_len) # equal number of samples per sequence length
				for slen in args.seq_len:
					to_generate = samples_per_seq_len
					while to_generate > 0:
						new_inputs = utils.get_adv(subslist, num_samples=min(to_generate, 10002), epochs=2000, schedule=[500, 1000, 1500], range_=1.000, input_dim=input_dim, model_type=args.model_type, sequence_length=slen)
						to_generate -= 10002
						new_outputs = model(new_inputs.cuda(device)).cpu().detach()
						population.add_seq_data(new_inputs, new_outputs, slen, window=500)
						#save samples
						torch.save(new_inputs,models_path +"/data_iteration_final.pt")
			else:
				while samples_to_generate > 0:
						new_inputs = utils.get_adv(subslist, num_samples=min(samples_to_generate, 10002), epochs=2000, schedule=[500, 1000, 1500], range_=1.000, input_dim=input_dim, model_type=args.model_type, sequence_length=None)
						samples_to_generate -= 10002
						new_outputs = model(new_inputs.cuda(device)).cpu().detach()
						population.add_data(new_inputs, new_outputs, window=500)
						#save samples
						torch.save(new_inputs,models_path +"/data_iteration_final.pt")
			gc.collect()

			for i in range(10):
				population.train_one_epoch(batch_size=128, epoch_num=i, restore=False)
				sys.stdout.flush()
			population.save(models_path +"/population_iteration_final.pt")
			population.evaluate(model, model_type=args.model_type)

	for i in range(pop_size):
		print(f"Student {i} loss:", population.subs[i].loss)
	sys.stdout.flush()
	for i in range(pop_size):
		print(f"Evaluating student {i}:")
		print(evaluate_reconstruction(model, population.subs[i], model_type=args.model_type))
