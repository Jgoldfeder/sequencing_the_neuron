import standardize
import torch
import mse
import mae
import base
import align
from dataclasses import dataclass
from enum import Enum

class Activation(Enum):
    relu = "relu"
    leaky_relu = "leaky_relu"
    tanh = "tanh"

class e_mae(mse.Evaluator):
    @classmethod
    def calculate_weights_distance(cls, original_weights, reconstructed_weights):
        distance = torch.nn.functional.l1_loss(
            original_weights, reconstructed_weights, reduction="sum"
        )
        return distance.item()
    

class e_layers_mae(mae.Evaluator):
    def iterate_compared_layers(self, device=None):
        original_layers = standardize.get_layers(self.original)
        reconstruction_layers = standardize.get_layers(self.reconstruction)
        get_weights = standardize.order.get_layer_weights
        for original, reconstruction in zip(original_layers, reconstruction_layers):
            original_weights = get_weights(original, device)
            reconstruction_weights = get_weights(reconstruction, device)
            yield original_weights, reconstruction_weights

    def calculate_distance(self):
        return tuple(
            self.calculate_weights_distance(original, reconstructed)
            for original, reconstructed in self.iterate_compared_layers()
        )

    @classmethod
    def calculate_weights_distance(cls, original_weights, reconstructed_weights):
        distance = torch.nn.functional.l1_loss(original_weights, reconstructed_weights)
        return distance.item()

    @classmethod
    def format_evaluation(cls, value, precision=3) -> str:
        if value:
            values = (
                super(e_layers_mae, cls).format_evaluation(layer_value)
                for layer_value in value
            )
            formatted_value = ", ".join(values)
        else:
            formatted_value = "/"
        return formatted_value


class e_max_ae(mae.Evaluator):
    def calculate_distance(self):
        max_distance = max(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )
        return max_distance

    @classmethod
    def calculate_weights_distance(cls, original_weights, reconstructed_weights):
        distances = torch.nn.functional.l1_loss(
            original_weights, reconstructed_weights, reduction="none"
        )
        distance = distances.max()
        return distance.item()
    

@dataclass
class e_mse(base.Evaluator):
    use_align: bool = True
    tanh: bool = None

    def __post_init__(self):
        if self.tanh is None:
            self.tanh = "leaky_relu" == Activation.tanh
        super().__post_init__()

    def evaluate(self):
        return self.calculate_distance() if self.standardize_networks() else None

    def standardize_networks(self):
        standardized = self.same_architecture()
        if standardized:
            if self.use_align:
                align.align(
                    self.original, self.reconstruction, tanh=self.tanh
                )
            else:
                for model in (self.original, self.reconstruction):
                    standardize.standardize(model, tanh=self.tanh)
        return standardized

    def same_architecture(self):
        return all(
            original.shape == reconstruction.shape
            for original, reconstruction in self.iterate_compared_layers()
        )

    def calculate_distance(self):
        total_size = sum(
            weights.numel() for weights in self.original.state_dict().values()
        )
        total_distance = sum(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )
        distance = total_distance / total_size
        return distance

    @classmethod
    def calculate_weights_distance(cls, original, reconstruction):
        distance = torch.nn.functional.mse_loss(
            original, reconstruction, reduction="sum"
        )
        return distance.item()

    def iterate_compared_layers(self):
        yield from zip(
            self.original.state_dict().values(),
            self.reconstruction.state_dict().values(),
        )
