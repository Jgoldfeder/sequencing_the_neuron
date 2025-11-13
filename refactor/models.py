import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class var_FNN(nn.Module):
    def __init__(self, activation_f, layer_dim):
        super(var_FNN, self).__init__()

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


class base_CNN(nn.Module):
    def __init__(self):
        super(base_CNN, self).__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=1)
        fc1 = nn.Linear(2523, 10) # using only one layer for now. 
        self.layers = nn.ModuleList([conv1, fc1])
       # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.leaky_relu(self.layers[0](x))
        x = torch.flatten(x, 1)
        x = self.layers[-1](x)
        return x

class two_CNN(nn.Module):
    def __init__(self):
        super(two_CNN, self).__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)
        conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        fc1 = nn.Linear(147, 10)
        self.layers = nn.ModuleList([conv1, conv2, fc1])

    def forward(self, x):
        x = F.leaky_relu(self.layers[0](x))
        x = F.leaky_relu(self.layers[1](x))
        x = torch.flatten(x, 1)
        x = self.layers[-1](x)
        return x
    
class var_CNN(nn.Module):
    def __init__(self, input_shape, layer_configs, activation_f, pooling=None):
        super(var_CNN, self).__init__()

        self.activation = activation_f
        self.pooling = pooling #(pooling_kernel_size, pooling_stride_size), None if no pooling
        self.layers = nn.ModuleList()

        current_shape = input_shape #input shape is tuple (#channels, height, width)
        for layer in layer_configs:
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
            #find shape after convolution
            current_shape = self.calculate_output_shape(current_shape, layer['out_channels'], layer['kernel_size'], layer['stride'], 1)
            #find shape after pooling
            if pooling is not None:
                current_shape = self.calculate_output_shape(current_shape, current_shape[0], pooling[0], pooling[1], 0)
        
        final_channels, final_height, final_width = current_shape
        self.layers.append(nn.Linear(final_channels*final_height*final_width, 10))

    def calculate_output_shape(self, input_shape, out_channels, kernel_size, stride, padding):
        _, height, width = input_shape
        height_out = (height - kernel_size + 2 * padding) // stride + 1
        width_out = (width - kernel_size + 2 * padding) // stride + 1
        return out_channels, height_out, width_out
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            conv = self.layers[i]
            x = self.activation(conv(x))
            if self.pooling is not None:
                x = F.max_pool2d(x, kernel_size=self.pooling[0], stride=self.pooling[1])

        x = torch.flatten(x, 1)
        x = self.layers[-1](x)
        return x

class var_RNN(nn.Module):
    def __init__(self, input_size, layer_configs, batch_first=True):
        super(var_RNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_configs = layer_configs
        current_input_size = input_size

        for hidden_size in layer_configs:
            self.layers.append(nn.RNN(input_size=current_input_size, hidden_size=hidden_size, num_layers=1, batch_first=batch_first))
            current_input_size = hidden_size
        
        #append linear layer
        self.layers.append(nn.Linear(current_input_size, 10))

    def forward(self, x):
        batch_size = x.size(0)

        for i in range(len(self.layers)-1):
            h0 = x.new_zeros(1, batch_size, self.layer_configs[i])
            layer = self.layers[i]
            x, _ = layer(x, h0)

        x = self.layers[-1](x[:, -1, :])
        return x
    
class base_RNN(nn.Module):
    def __init__(self):
        super(base_RNN, self).__init__()
        rnn = nn.RNN(28, 3, 1, batch_first=True)
        fc = nn.Linear(3, 10)
        self.layers = nn.ModuleList([rnn,fc])
    def forward(self, x):
        h0 = x.new_zeros(1, x.size(0), 3)
        out, _ = self.layers[0](x, h0)
        out = self.layers[-1](out[:, -1, :])
        return out

class base_TransformerEncoder(nn.Module):
    def __init__(self, d_model, layer_configs):
        super(base_TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = 1,
            dim_feedforward = layer_configs[0],
            dropout = 0,
            activation = 'relu',
            batch_first = True
        )
        self.linear = nn.Linear(d_model, 10)
    def forward(self, x):
        x = self.encoder_layer(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x