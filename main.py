from data import get_data
from model import LinearLearner
from train import train_model
from plot import plot_loss, plot_predictions


def main():

    import torch
    torch.manual_seed(42)

    # Load data
    x, y = get_data()

    # Create model
    model = LinearLearner()

    # Train model
    losses, snapshots = train_model(model, x, y)

    # Show learned parameters
    print("\nLearned Weight:")
    print(model.linear.weight)

    print("\nLearned Bias:")
    print(model.linear.bias)

    print("\nPredictions:")
    print(model(x))
    print(f"\nInitial Loss: {losses[0]:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")

    true_weight, true_bias = 2.0, 1.0
    learned_weight = model.linear.weight.item()
    learned_bias = model.linear.bias.item()

    print(f"\nTrue weight: {true_weight}, Learned weight: {learned_weight:.4f}")
    print(f"True bias: {true_bias}, Learned bias: {learned_bias:.4f}")

    weight_error = abs(learned_weight - true_weight) / true_weight * 100
    bias_error = abs(learned_bias - true_bias) / true_bias * 100
    print(f"Weight error: {weight_error:.2f}%")
    print(f"Bias error: {bias_error:.2f}%")

    convergence_threshold = 0.01
    for i, l in enumerate(losses):
        if l < convergence_threshold:
            print(f"\nConverged (loss < {convergence_threshold}) at epoch {i}")
            break
    else:
        print(f"\nDid not converge below {convergence_threshold} within {len(losses)} epochs")

    # Create plots
    plot_loss(losses)
    plot_predictions(model, x, y)
    from plot import create_training_gif

    create_training_gif(
    model,
    snapshots,
    x,
    y
)


if __name__ == "__main__":
    main()