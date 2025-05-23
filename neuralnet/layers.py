import numpy as np

from neuralnet.assertions.layer_assertion import layer_assertion_fn
from neuralnet.utilities import *


class Layer:
	def __init__(self, n_neurons,
				 n_inputs,
				 activation=None,
				 activation_derivative=None,
				 use_bias=True,
				 weights_initializer=None,
				 bias_initializer=None,
				 rng=None):
		layer_assertion_fn(n_neurons,
						   n_inputs,
						   activation,
						   activation_derivative,
						   use_bias,
						   weights_initializer,
						   bias_initializer)
		self.N = n_neurons
		self.input_size = n_inputs
		self.use_bias = use_bias
		rng = rng or np.random.default_rng()

		self.activation = activation or tanh
		self.activation_derivative = activation_derivative or tanh_derivative
		weights_initializer = weights_initializer or xavier_uniform_initializer
		bias_initializer = bias_initializer or zero_initializer

		# Use the initializer function
		self.W = weights_initializer((self.N, self.input_size), rng)
		self.b = bias_initializer(self.N, rng) if self.use_bias else zero_initializer(self.N)
		assert is_numeric_array(self.W, 2) and is_numeric_array(self.b, 1), "Weights and biases must be 2d and 1d numeric arrays respectively"
		assert self.W.shape == (self.N, self.input_size), "Weights shape must be (n_neurons, n_inputs)"
		assert len(self.b) == self.N, "Bias shape must be (n_neurons,)"
		self.W = np.asarray(self.W, dtype=np.float64)
		self.b = np.asarray(self.b, dtype=np.float64)
		# Initialize fields
		self.h = None  # pre-activation field
		self.s = None  # output
		self.bp_errors = None  # error signals

	def set_weights(self, W, b=None):
		"""
		Set the weights and biases of the layer.

		Parameters:
			W : np.ndarray
				Weights matrix.
			b : np.ndarray or None
				Bias vector (optional).
		"""
		assert self.h is None and self.s is None and self.bp_errors is None, "Cannot set weights after forward or backward pass"
		assert is_numeric_array(W, 2) and W.shape == self.W.shape, "Weights must be a 2D numeric array with the same shape as the layer's weights"
		self.W = np.asarray(W, dtype=np.float64)
		if self.use_bias and b is not None:
			assert is_numeric_array(b, 1) and len(b) == len(self.b), "Bias must be a 1D numeric array with the same length as the layer's biases"
			self.b = np.asarray(b, dtype=np.float64)

	def forward(self, s_prev):
		self.h = self.W @ s_prev + self.b
		self.s = self.activation(self.h)
		return self.s

	def forward_batch(self, S_prev):
		H = self.W @ S_prev + self.b[:, np.newaxis]
		return self.activation(H)

	def backward(self, *args, **kwargs):
		"""
				Compute backpropagation error signals for this layer.

				Parameters:
					Depends on the layer type:
					- For OutputLayer: receives true label y_true
					- For HiddenLayer: receives bp_errors_next and W_next

				Updates:
					self.bp_errors : np.ndarray
						Backpropagation error signals -g'(h^{(l)})*(∂E/∂s^{(l)}).
				"""
		raise NotImplementedError("Must be implemented in subclass.")

	def update_weights(self, s_prev, learning_rate):
		"""
		Update weights and biases using computed backpropagation errors.

		Parameters:
			s_prev : np.ndarray
				Output from the previous layer (s^{(l-1)}).
			learning_rate : float
				Learning rate for current update.
		"""
		self.W += learning_rate * np.outer(self.bp_errors, s_prev)
		self.b += learning_rate * self.bp_errors * self.use_bias


class HiddenLayer(Layer):
	def backward(self, bp_errors_next, W_next):
		g_prime = self.activation_derivative(self.h)
		self.bp_errors = g_prime * (W_next.T @ bp_errors_next)


class OutputLayer(Layer):
	def backward(self, y_true, loss_derivative):
		g_prime = self.activation_derivative(self.h)
		self.bp_errors = -g_prime * loss_derivative(self.s, y_true)
