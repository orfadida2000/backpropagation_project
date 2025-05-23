def network_assertion_fn(layer_sizes,
						 use_bias_list,
						 weight_initializers,
						 bias_initializers,
						 activations,
						 activation_derivatives,
						 loss_derivative):
	assert len(layer_sizes) == len(use_bias_list), "use_bias_list must match layer_sizes in length"
	assert (activations and activation_derivatives) or (
			not activations and not activation_derivatives), "both activations and activation_derivatives need to be either initialized, or None"
	if activations is not None:
		assert len(activations) == len(layer_sizes), "activations must match layer_sizes in length"
	if activation_derivatives is not None:
		assert len(activation_derivatives) == len(layer_sizes), "activation_derivatives must match layer_sizes in length"
	if weight_initializers is not None:
		assert len(weight_initializers) == len(layer_sizes), "weight_initializers must match layer_sizes in length"
	if bias_initializers is not None:
		assert len(bias_initializers) == len(layer_sizes), "bias_initializers must match layer_sizes in length"
	assert loss_derivative is None or callable(loss_derivative), "loss_derivative must be a callable function or None"
