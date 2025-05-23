import numpy as np

def xavier_uniform_initializer(shape, rng):
	fan_out, fan_in = shape
	limit = np.sqrt(6 / (fan_in + fan_out))
	return rng.uniform(-limit, limit, size=shape)


def zero_initializer(shape, rng=None):
	return np.zeros(shape, dtype=np.float64)


def mse_loss(predictions, targets):
	return np.mean(0.5 * (predictions - targets) ** 2)


def mse_loss_derivative(y_pred, y_true):
	return y_pred - y_true


def peaks(x1, x2):
	"""
	Approximates the MATLAB 'peaks' function, scaled down by 10.
	Accepts scalars or 1D NumPy arrays.
	"""
	z = (
				3 * (1 - x1) ** 2 * np.exp(-x1 ** 2 - (x2 + 1) ** 2)
				- 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-x1 ** 2 - x2 ** 2)
				- 1 / 3 * np.exp(-(x1 + 1) ** 2 - x2 ** 2)
		) / 10
	return z


def tanh(h):
	"""
	Applies the hyperbolic tangent activation function elementwise.
	Input:
		h: np.ndarray or float
	Returns:
		np.ndarray or float with tanh applied
	"""
	return np.tanh(h)


def tanh_derivative(h):
	"""
	Computes the derivative of tanh activation function elementwise.
	Input:
		h: np.ndarray or float
	Returns:
		np.ndarray or float with derivative values
	"""
	return 1 - np.tanh(h) ** 2

def is_numeric_array(arr, n_dim):
	"""
	Check if the input is a numeric array (int or float).
	"""
	if not isinstance(arr, np.ndarray):
		return False
	return arr.ndim == n_dim and (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer))
