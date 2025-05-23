import numpy as np

from neuralnet.network import NeuralNetwork
from neuralnet.utilities import tanh, tanh_derivative, mse_loss_derivative


def main():
	P = np.array([1, 1, 1])
	y_true = np.array([-1])
	learning_rate = 1
	V = np.array([[np.log(4 / 3), -np.log(2), -np.log(2)],
				  [np.log(2), -np.log(2), -np.log(2)]])
	W = np.array([[5 / 8, 0]])
	bias = np.array([np.log(2) + 1 / 2])
	layer_sizes = [3, 2, 1]
	use_bias_list = [None, False, True]
	layers_initial_weights = [None, V, W]
	layers_initial_biases = [None, None, bias]
	activations = [None, tanh, tanh]
	activation_derivatives = [None, tanh_derivative, tanh_derivative]

	net = NeuralNetwork.from_weights(layer_sizes,
									 use_bias_list,
									 layers_initial_weights,
									 layers_initial_biases,
									 activations,
									 activation_derivatives,
									 mse_loss_derivative)
	# Forward pass
	a = net.forward(P)
	print(f"The output of the network is: {a}")

	# Backward pass
	net.backward(y_true)

	w_p2_to_b1_old = net.layers[1].W[0,1]
	# Print the weight before update
	print(f"Weight before update: {w_p2_to_b1_old}")
	# Update weights
	net.update_weights(P, learning_rate)
	w_p2_to_b1_new = net.layers[1].W[0,1]
	delta_w = w_p2_to_b1_new - w_p2_to_b1_old
	# Print the weight change
	print(f"Weight change: {delta_w}")
	# Print the updated weight
	print(f"Updated weight: {w_p2_to_b1_new}")

if __name__ == "__main__":
	main()
