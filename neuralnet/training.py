from neuralnet.assertions.training_assertion import training_assertion_fn
from neuralnet.network import NeuralNetwork
from neuralnet.utilities import *
import numpy as np


def sample_training_example(rng):
	"""
	Samples a single (x, y) training pair.

	Returns:
		x: np.ndarray of shape (2,)
		y: np.ndarray of shape (1,)
	"""
	x = rng.uniform(-3, 3, size=2)  # shape: (2,)
	y = np.array([peaks(x[0], x[1])])  # shape: (1,)
	return x, y


def generate_validation_set(rng, m):
	"""
	Generates a validation set of S samples.

	Returns:
		X: np.ndarray of shape (2, m)
		Y: np.ndarray of shape (m, 1)
	"""
	X = rng.uniform(-3, 3, size=(2, m))  # shape: (2, m)
	x1 = X[0, :]  # shape: (m,)
	x2 = X[1, :]  # shape: (m,)
	Y = peaks(x1, x2).reshape(m, 1)  # shape: (m, 1)
	return X, Y


def train_one_network(
		net,
		training_steps,
		learning_rate_fn,
		validation_inputs,
		validation_targets,
		sample_training_example_fn,
		loss_fn,
		record_every=1000
		):
	"""
	Trains a single neural network using SGD and tracks generalization error.

	Parameters:
		net: NeuralNetwork
			An initialized network object (already has loss_derivative set).
		training_steps: int
			Total number of SGD updates (e.g., 1_000_000).
		learning_rate_fn: float
			Function that receives iteration number and returns learning rate .
		validation_inputs: np.ndarray
			Matrix of validation inputs, shape (2, num_examples).
		validation_targets: np.ndarray
			Vector of true outputs, shape (num_examples,).
		sample_training_example_fn: callable
			Function that returns (x, y) training pair.
		loss_fn: callable
			Scalar loss function (e.g. MSE).
		record_every: int
			Frequency for recording generalization error.

	Returns:
		trained_net: NeuralNetwork
			The same `net`, trained in-place.
		generalization_error: float
			Final validation error after the ast training step.
		error_curve: list of tuple[int, float]
				List of (step, error) for each checkpoint.
	"""
	training_assertion_fn(
			net,
			training_steps,
			learning_rate_fn,
			validation_inputs,
			validation_targets,
			sample_training_example_fn,
			loss_fn,
			record_every
			)
	error_curve = []
	for step in range(1, training_steps + 1):
		x, y = sample_training_example_fn()
		net.train_one_example(x, y, learning_rate_fn(step))

		if step % record_every == 0:
			predictions = net.predict_batch(validation_inputs)
			error = loss_fn(predictions, validation_targets)
			error_curve.append((step, error))
	if training_steps % record_every != 0:
		predictions = net.predict_batch(validation_inputs)
		error = loss_fn(predictions, validation_targets)
		error_curve.append((training_steps, error))
	generalization_error = error_curve[-1][1]
	return net, generalization_error, error_curve


def run_experiment(
		num_networks,
		training_steps,
		validation_size,
		record_every,
		learning_rate_fn
		):
	"""
	Trains multiple networks and returns the best one based on final generalization error.

	Parameters:
		num_networks: int
			How many different networks to train.
		training_steps: int
			How many SGD steps to train each network.
		validation_size: int
			Number of examples in the validation set.
		record_every: int
			Frequency (in steps) to record the generalization error.
		learning_rate_fn: callable
			A function that receives the step index and returns the learning rate.

	Returns:
		best_model: NeuralNetwork
			The trained model with the lowest final validation error.
		best_error_curve: list of tuple[int, float]
			The validation error recorded every `record_every` steps.
	"""

	rng = np.random.default_rng()
	X_valid, Y_valid = generate_validation_set(rng, validation_size)

	best_model = None
	lowest_error = float('inf')
	best_error_curve = None
	final_errors = []

	for i in range(num_networks):
		print(f"Network number {i+1}") # debugging
		net = NeuralNetwork(
				layer_sizes=[2, 100, 1],
				use_bias_list=[None, True, False],
				weight_initializers=[None, xavier_uniform_initializer, xavier_uniform_initializer],
				bias_initializers=[None, zero_initializer, None],
				activations=[None, tanh, tanh],
				activation_derivatives=[None, tanh_derivative, tanh_derivative],
				loss_derivative=mse_loss_derivative,
				rng=rng
				)

		trained_net, final_error, error_curve = train_one_network(
				net=net,
				training_steps=training_steps,
				learning_rate_fn=learning_rate_fn,
				validation_inputs=X_valid,
				validation_targets=Y_valid,
				sample_training_example_fn=lambda: sample_training_example(rng),
				loss_fn=mse_loss,
				record_every=record_every
				)
		final_errors.append(final_error) # debugging

		if final_error < lowest_error:
			lowest_error = final_error
			best_model = trained_net
			best_error_curve = error_curve

	print(f"Lowest error: {lowest_error}")  # debugging
	print(f"Final errors for all networks: {final_errors}")  # debugging

	return best_model, best_error_curve

