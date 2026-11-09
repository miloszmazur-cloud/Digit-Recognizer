import math
import random
import numpy as np
import h5py

LEARNING_RATE = 0.08
Z_CLAMP = 500.0


def sigmoid(z: float) -> float:
    z = max(-Z_CLAMP, min(Z_CLAMP, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


class Node:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = np.array(weights, dtype=np.float64)
        self.bias = float(bias)

        self.last_inputs = np.array([], dtype=np.float64)
        self.last_output = 0.0

    def predict(self, x: np.ndarray) -> float:
        self.last_inputs = x

        # z = w * x + b
        z = float(np.dot(self.weights, x)) + self.bias

        # unification of result (0, 1)
        self.last_output = sigmoid(z)
        return self.last_output

    def train(self, error: float) -> np.ndarray:
        # Derivative of Sigmoid = slope of activation curve
        derivative = self.last_output * (1.0 - self.last_output)

        # How much should node's z update
        delta = error * derivative

        # Error calculation for previous layer (use weights before update)
        errors_for_layer = delta * self.weights

        # Update weights and bias
        self.bias -= LEARNING_RATE * delta
        self.weights -= LEARNING_RATE * delta * self.last_inputs

        return errors_for_layer


class Layer:
    def __init__(self, nodes_number: int, inputs_number: int, weights: list | None = None,
                 biases: list | None = None):
        self.nodes = []

        if weights is not None and biases is not None:
            self.inputs_number = len(weights[0])

            for i in range(nodes_number):
                node = Node(weights[i], biases[i])
                self.nodes.append(node)
        else:
            self.inputs_number = inputs_number
            limit = 1.0 / math.sqrt(self.inputs_number)

            for _ in range(nodes_number):
                # Random initialization to avoid Symmetry Problem
                initialize_weights = np.random.uniform(-limit, limit, self.inputs_number)
                initialize_bias = random.uniform(-limit, limit)

                node = Node(initialize_weights, initialize_bias)
                self.nodes.append(node)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([node.predict(x) for node in self.nodes], dtype=np.float64)

    def train(self, errors: np.ndarray) -> np.ndarray:
        errors_for_layer = np.zeros(self.inputs_number, dtype=np.float64)

        for i, node in enumerate(self.nodes):
            errors_for_layer += node.train(float(errors[i]))

        return errors_for_layer


class AI:
    def __init__(self):
        self.layers = [
            Layer(nodes_number=128, inputs_number=784),  # First Hidden Layer
            Layer(nodes_number=32, inputs_number=128),   # Second Hidden Layer
            Layer(nodes_number=11, inputs_number=32),    # Output Layer
        ]

    def predict(self, pixel_grid: np.ndarray) -> np.ndarray:
        current_signals = np.asarray(pixel_grid, dtype=np.float64).flatten()

        for layer in self.layers:
            current_signals = layer.predict(current_signals)

        return current_signals

    def train(self, pixel_grid: np.ndarray, target_index: int):
        current_signals = self.predict(pixel_grid)

        targets = np.zeros(11, dtype=np.float64)
        targets[target_index] = 1.0

        output_errors = current_signals - targets

        current_errors = output_errors
        for layer in reversed(self.layers):
            current_errors = layer.train(current_errors)

    def save_model(self, filename: str = "model.h5"):
        with h5py.File(filename, 'w') as f:
            for i, layer in enumerate(self.layers):
                layer_group = f.create_group(f"layer_{i}")

                layer_weights = np.array([node.weights for node in layer.nodes])
                layer_biases = np.array([node.bias for node in layer.nodes])

                layer_group.create_dataset("weights", data=layer_weights)
                layer_group.create_dataset("biases", data=layer_biases)

        print(f"Model successfully saved to {filename}")

    def load_model(self, filename: str = "model.h5"):
        with h5py.File(filename, 'r') as f:
            self.layers = []

            i = 0
            while f"layer_{i}" in f:
                layer_weights = f[f"layer_{i}"]["weights"][:]
                layer_biases = f[f"layer_{i}"]["biases"][:]

                nodes_number = len(layer_weights)
                inputs_number = len(layer_weights[0])

                loaded_layer = Layer(
                    nodes_number=nodes_number,
                    inputs_number=inputs_number,
                    weights=layer_weights,
                    biases=layer_biases
                )
                self.layers.append(loaded_layer)
                i += 1

        print(f"Model successfully loaded from {filename}")