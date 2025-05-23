from neuralnet.training import run_experiment
from neuralnet.visualization import plot_error_curve, plot_comparisons

NUM_NETWORKS = 100
OVERALL_TRAINING_STEPS = 1_000_000
VALIDATION_SIZE = 1000
RECORD_EVERY = 1000
LEARNING_RATE_FN = lambda step: 0.001

def main():
    # === Run the experiment ===
    steps_per_network = OVERALL_TRAINING_STEPS
    best_model, best_error_curve = run_experiment(
        num_networks=NUM_NETWORKS,
        training_steps=steps_per_network,
        validation_size=VALIDATION_SIZE,
        record_every=RECORD_EVERY,
        learning_rate_fn=LEARNING_RATE_FN
    )

    # === Visualize results ===
    plot_error_curve(best_error_curve)
    plot_comparisons(best_model)

if __name__ == "__main__":
    main()
