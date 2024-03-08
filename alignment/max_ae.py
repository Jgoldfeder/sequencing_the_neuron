import torch 


def calculate_distance_mae(original, reconstruction):
    max_distance = max(
        _calculate_layer_max_distance(original_layer, reconstruction_layer)
        for original_layer, reconstruction_layer in _iterate_compared_layers(original, reconstruction)
    )
    return max_distance

def _calculate_layer_max_distance(original_layer_params, reconstruction_layer_params):
    distances = torch.nn.functional.l1_loss(
        original_layer_params,  reconstruction_layer_params, reduction="none"
    )
    
    import numpy as np 
    distance = distances.max()
    #print("max distance in layer", distance)
    #print("original_layer_params", original_layer_params)
    index_min = np.argmin(distances.cpu())
    #print("index_min", index_min)

    return distance.item()

def _iterate_compared_layers(original, reconstruction):
   # print("_iterate_compared_layers mae")
   # print(original.state_dict().values())
   # print(reconstruction.state_dict().values())
    yield from zip(
        original.state_dict().values(),
        reconstruction.state_dict().values(),
    )