import torch 
def calculate_distance_mse_or_mae(metric_for_eval, original, reconstruction):
        total_size = sum(
            weights.numel() for weights in original.state_dict().values()
        )

        num_weights = [weights.numel() for weights in original.state_dict().values()]

        layer_distances = [_calculate_layer_distance_mse_or_mae(metric_for_eval, original_param, reconstruction_param)
            for original_param, reconstruction_param in _iterate_compared_layers(original, reconstruction)]
        
        layer_err = [l/n for l,n in zip(layer_distances, num_weights)]

        overall_err = sum(layer_distances)/total_size

        return layer_err, overall_err

def _calculate_layer_distance_mse_or_mae(metric_for_eval, original_layer_params, reconstruction_layer_params): 
    if metric_for_eval == "mse":
        distance = torch.nn.functional.mse_loss(
            original_layer_params, reconstruction_layer_params, reduction="sum"
        )
    
    if metric_for_eval == "mae": 
         distance = torch.nn.functional.l1_loss(
            torch.flatten(original_layer_params), torch.flatten(reconstruction_layer_params), reduction="sum"
        )
    return distance.item()

def _iterate_compared_layers(original, reconstruction):
    yield from zip(
        original.state_dict().values(),
        reconstruction.state_dict().values(),
    )