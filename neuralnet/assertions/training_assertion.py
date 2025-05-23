import numpy as np

from neuralnet.network import NeuralNetwork
from neuralnet.utilities import is_numeric_array

def training_assertion_fn(
		net,
		training_steps,
		learning_rate_fn,
		validation_inputs,
		validation_targets,
		sample_training_example_fn,  # <-- you need to define
		loss_fn,  # <-- you need to define (e.g. MSE)
		record_every
		):
	assert isinstance(net, NeuralNetwork), "net must be an instance of NeuralNetwork"
	assert is_numeric_array(validation_inputs,2) and is_numeric_array(validation_targets,2), "validation sets must be a 2d numeric arrays"
	assert validation_inputs.shape[1] == validation_targets.shape[
		0], "validation_inputs and validation_targets must have the same number of examples"
	assert validation_inputs.shape[
			   0] == net.input_size, "validation_inputs must have the same number of features as the input layer"
	assert validation_targets.shape[
			   1] == net.output_size, "validation_targets must have the same number of features as the output layer"
	assert callable(sample_training_example_fn), "sample_training_example_fn must be a callable function"
	assert callable(loss_fn), "loss_fn must be a callable function"
	assert callable(learning_rate_fn), "learning_rate_fn must be a callable function"
	assert isinstance(record_every, (int, np.integer)) and record_every > 0, "record_every must be a positive integer"
	assert isinstance(training_steps, (int, np.integer)) and training_steps > 0, "training_steps must be a positive integer"
