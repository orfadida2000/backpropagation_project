import matplotlib.pyplot as plt
import numpy as np
from neuralnet.utilities import peaks


def plot_func_surface_k(ax, X1, X2, Z):
	"""
	Plots a surface with black mesh lines.
	"""
	ax.plot_surface(X1, X2, Z, cmap='jet', edgecolor='k', alpha=1.0)


def plot_func_surface_none(ax, X1, X2, Z):
	"""
	Plots a surface without mesh lines.
	"""
	ax.plot_surface(X1, X2, Z, cmap='jet', edgecolor='none', alpha=1.0)


def plot_func_wireframe_true(ax, X1, X2, Z):
	"""
	Plots a wireframe with black mesh lines.
	"""
	ax.plot_wireframe(X1, X2, Z, rstride=2, cstride=2, color='black', linewidth=0.7)


def plot_func_wireframe_pred(ax, X1, X2, Z):
	"""
	Plots a wireframe with red mesh lines.
	"""
	ax.plot_wireframe(X1, X2, Z, rstride=2, cstride=2, color='darkgreen', linewidth=0.7)


def plot_error_curve(error_curve, label_size=11, title_size=13):
	"""
	Plots the generalization error over time.
	"""
	steps, errors = zip(*error_curve)
	plt.figure(figsize=(8, 5))
	plt.plot(steps, errors, linewidth=2)
	plt.xlabel('Training Steps', fontsize=label_size)
	plt.ylabel('Generalization Error', fontsize=label_size)
	plt.title('Generalization Error Over Time (Best Model)', fontsize=title_size)
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def draw_subplot(fig, subplot_index, X1, X2, Z, title, zlabel, plot_func, label_size, title_size):
	ax = fig.add_subplot(1, 2, subplot_index, projection='3d')
	plot_func(ax, X1, X2, Z)
	ax.set_title(title, fontsize=title_size)
	ax.set_xlabel("x₁", fontsize=label_size)
	ax.set_ylabel("x₂", fontsize=label_size)
	ax.set_zlabel(zlabel, fontsize=label_size)
	ax.view_init(elev=30, azim=45)


def show_subplots_pair(suptitle, X1, X2, Z_true, Z_pred,
					   plot_func_true, plot_func_pred,
					   label_size, title_size, suptitle_size):
	"""
	General-purpose 3D pair plotting (true vs predicted).
	"""
	fig = plt.figure(figsize=(16, 7))

	draw_subplot(fig,
				 1,
				 X1,
				 X2,
				 Z_true,
				 "Target Function (peaks)",
				 "peaks(x₁, x₂)",
				 plot_func_true,
				 label_size,
				 title_size)
	draw_subplot(fig,
				 2,
				 X1,
				 X2,
				 Z_pred,
				 "Neural Network Approximation",
				 "f̂(x₁, x₂)",
				 plot_func_pred,
				 label_size,
				 title_size)

	fig.suptitle(suptitle, fontsize=suptitle_size)
	plt.tight_layout()
	plt.show()


def plot_comparisons(best_model, grid_size=100, label_size=11, title_size=13, suptitle_size=14):
	"""
	Generates three mesh comparison plots:
	1. Surface with colormap + black mesh
	2. Surface with colormap + no mesh
	3. Wireframe mesh only
	"""
	x1 = np.linspace(-3, 3, grid_size)
	x2 = np.linspace(-3, 3, grid_size)
	X1, X2 = np.meshgrid(x1, x2)
	X_flat = np.vstack([X1.ravel(), X2.ravel()])
	Z_true = peaks(X1, X2)  # shape: (grid_size, grid_size)
	Z_pred = best_model.predict_batch(X_flat).reshape(grid_size, grid_size)

	subplots_pairs = ({
							  "suptitle":       "Surface Plot with Jet Colormap (with mesh)",
							  "plot_func_true": plot_func_surface_k,
							  "plot_func_pred": plot_func_surface_k},
					  {
							  "suptitle":       "Surface Plot with Jet Colormap (no mesh)",
							  "plot_func_true": plot_func_surface_none,
							  "plot_func_pred": plot_func_surface_none},
					  {
							  "suptitle":       "Wireframe Mesh Plot",
							  "plot_func_true": plot_func_wireframe_true,
							  "plot_func_pred": plot_func_wireframe_pred})

	for subplots_pair in subplots_pairs:
		show_subplots_pair(
				suptitle=subplots_pair["suptitle"],
				X1=X1,
				X2=X2,
				Z_true=Z_true,
				Z_pred=Z_pred,
				plot_func_true=subplots_pair["plot_func_true"],
				plot_func_pred=subplots_pair["plot_func_pred"],
				label_size=label_size,
				title_size=title_size,
				suptitle_size=suptitle_size
				)
