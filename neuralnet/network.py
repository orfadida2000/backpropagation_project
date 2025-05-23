import numpy as np

from neuralnet.assertions.network_assertion import network_assertion_fn
from neuralnet.layers import HiddenLayer, OutputLayer
from neuralnet.utilities import mse_loss_derivative
from neuralnet.utilities import is_numeric_array


class NeuralNetwork:
	def __init__(self,
				 layer_sizes,
				 use_bias_list,
				 weight_initializers=None,
				 bias_initializers=None,
				 activations=None,
				 activation_derivatives=None,
				 loss_derivative=None,
				 rng=None):
		"""
				Parameters:
					layer_sizes: list of ints
						Number of neurons per layer (input to output), length = M+1
					use_bias_list: list of bool or None
						Bias flag per layer. Index 0 must be None (input layer).
					activations: list of callables or None
						Activation function per layer. Index 0 must be None.
					activation_derivatives: list of callables or None
						Derivative of activation. Index 0 must be None.
					weight_initializers: list of callables or None
						Weight initializers for each layer. Index 0 must be None.
					bias_initializers: list of callables or None
						Bias initializers for each layer. Index 0 must be None.
					loss_derivative: callable or None
						Loss function derivative, used for output layer.
						Defaults to MSE derivative.
						Receives y_pred(a.k.a s1^{m}) and y_true.
					rng: np.random.Generator or None
						Random generator (optional). Passed to layers as-is.
				"""

		network_assertion_fn(layer_sizes,
							 use_bias_list,
							 weight_initializers,
							 bias_initializers,
							 activations,
							 activation_derivatives,
							 loss_derivative)

		self.L = len(layer_sizes) - 1  # number of trainable layers
		self.layers = [None]  # index 0 is reserved
		self.loss_derivative = loss_derivative or mse_loss_derivative
		self.input_size = layer_sizes[0]
		self.output_size = layer_sizes[-1]
		for l in range(1, self.L + 1):
			n_inputs = layer_sizes[l - 1]
			n_neurons = layer_sizes[l]
			use_bias = use_bias_list[l]

			activation = activations[l] if activations else None
			activation_derivative = activation_derivatives[l] if activation_derivatives else None
			weights_initializer = weight_initializers[l] if weight_initializers else None
			bias_initializer = bias_initializers[l] if bias_initializers else None

			layer_cls = OutputLayer if l == self.L else HiddenLayer
			layer = layer_cls(n_neurons,
							  n_inputs,
							  activation,
							  activation_derivative,
							  use_bias,
							  weights_initializer,
							  bias_initializer,
							  rng)
			self.layers.append(layer)

	@classmethod
	def from_weights(cls,
					 layer_sizes,
					 use_bias_list,
					 layers_initial_weights,
					 layers_initial_biases,
					 activations=None,
					 activation_derivatives=None,
					 loss_derivative=None):

		assert len(layers_initial_weights) == len(layer_sizes), "weight_initializers must match layer_sizes in length"
		assert len(layers_initial_biases) == len(layer_sizes), "bias_initializers must match layer_sizes in length"
		net = cls(layer_sizes,
				  use_bias_list,
				  activations=activations,
				  activation_derivatives=activation_derivatives,
				  loss_derivative=loss_derivative)
		for l in range(1, net.L + 1):
			layer = net.layers[l]
			layer.set_weights(layers_initial_weights[l], layers_initial_biases[l])
		return net

	def forward(self, s0):
		assert is_numeric_array(s0, 1), "s0 must be a 1D numeric array"
		assert len(s0) == self.input_size, "s0's length must match the input layer size"
		s = np.asarray(s0, dtype=np.float64)
		for l in range(1, self.L + 1):
			s = self.layers[l].forward(s)
		return s

	def predict_batch(self, S0):
		assert is_numeric_array(S0, 2), "S0 must be a 2D numeric array"
		assert S0.shape[0] == self.input_size, "S0's number of columns must match the input layer size"
		S = np.asarray(S0, dtype=np.float64)
		for l in range(1, self.L + 1):
			S = self.layers[l].forward_batch(S)
		return S.T

	def backward(self, y_true):
		assert is_numeric_array(y_true, 1), "y_true must be a 1D numeric array"
		assert len(y_true) == self.output_size, "y_true's length must match the output layer size"
		y_true = np.asarray(y_true, dtype=np.float64)
		self.layers[self.L].backward(y_true, self.loss_derivative)
		for l in reversed(range(1, self.L)):
			bp_errors_next = self.layers[l + 1].bp_errors
			W_next = self.layers[l + 1].W
			self.layers[l].backward(bp_errors_next, W_next)

	def update_weights(self, s0, learning_rate):
		assert is_numeric_array(s0, 1), "s0 must be a 1D numeric array"
		assert len(s0) == self.input_size, "s0's length must match the input layer size"
		assert isinstance(learning_rate, (float, int, np.floating, np.integer)) and learning_rate > 0, "Learning rate must be a positive number"
		s0 = np.asarray(s0, dtype=np.float64)
		learning_rate = np.float64(learning_rate)
		for l in range(1, self.L + 1):
			s_prev = s0 if l == 1 else self.layers[l - 1].s
			self.layers[l].update_weights(s_prev, learning_rate)

	def train_one_example(self, s0, y_true, learning_rate):
		"""
				Train the network for one step, on 1 example.

				Parameters:
					s0 : np.ndarray
						Input data (s^{(0)}).
					y_true : np.ndarray
						True label (y^{(L)}).
					learning_rate : float
						Learning rate for current update.
				"""
		self.forward(s0)
		self.backward(y_true)
		self.update_weights(s0, learning_rate)
