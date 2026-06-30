from data import get_data
from model import LinearLearner
from train import train_model
from plot import plot_loss, plot_predictions


def main():

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