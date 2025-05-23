import numpy as np


def layer_assertion_fn(n_neurons,
					n_inputs,
					activation,
					activation_derivative,
					use_bias,
					weights_initializer,
					bias_initializer):
	assert isinstance(n_neurons, (int, np.integer)) and n_neurons > 0, "n_neurons must be a positive integer"
	assert isinstance(n_inputs, (int, np.integer)) and n_inputs > 0, "n_inputs must be a positive integer"
	assert activation is None or callable(activation), "activation must be a callable function or None"
	assert activation_derivative is None or callable(activation_derivative), "activation_derivative must be a callable function or None"
	assert (activation and activation_derivative) or (
			not activation and not activation_derivative), "both activation and activation_derivative need to be either initialized, or None"
	assert isinstance(use_bias, (bool, np.bool_)), "use_bias must be a boolean value"
	assert weights_initializer is None or callable(weights_initializer), "weights_initializer must be a callable function or None"
	assert bias_initializer is None or callable(bias_initializer), "bias_initializer must be a callable function or None"
