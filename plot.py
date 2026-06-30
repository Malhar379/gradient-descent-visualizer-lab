import matplotlib.pyplot as plt
import torch
import imageio.v2 as imageio
import os


def plot_loss(losses):

    plt.figure(figsize=(6,4))
    plt.plot(losses)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig("outputs/loss_curve.png")
    plt.close()


def plot_predictions(model, x, y):

    with torch.no_grad():
        prediction = model(x)

    plt.figure(figsize=(6,4))

    plt.scatter(x.numpy(), y.numpy(), s=80, label="True Data")
    plt.plot(x.numpy(), prediction.numpy(), linewidth=2, label="Prediction")

    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/prediction_vs_truth.png")
    plt.close()


def create_training_gif(model, snapshots, x, y):

    os.makedirs("outputs/temp", exist_ok=True)

    frames = []

    for epoch, state in enumerate(snapshots):

        # Restore model at that epoch
        model.load_state_dict(state)

        with torch.no_grad():
            prediction = model(x)

        plt.figure(figsize=(6,4))

        plt.scatter(
            x.numpy(),
            y.numpy(),
            s=80,
            color="black",
            label="True"
        )

        plt.plot(
            x.numpy(),
            prediction.numpy(),
            color="red",
            linewidth=2,
            label="Prediction"
        )

        weight = model.linear.weight.item()
        bias = model.linear.bias.item()

        # Calculate current loss
        with torch.no_grad():
            current_loss = torch.mean((prediction - y) ** 2).item()

        plt.title(f"Epoch {epoch+1}")

        plt.text(
             0.02,
             0.98,
            f"Loss   : {current_loss:.4f}\n"
            f"Weight : {weight:.4f}\n"
            f"Bias   : {bias:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85)
        )
        plt.xlim(-0.5, 3.5)
        plt.ylim(-1, 8)

        plt.grid(True)
        plt.legend()

        filename = f"outputs/temp/frame_{epoch:03d}.png"

        plt.savefig(filename)
        plt.close()

        frames.append(imageio.imread(filename))

    imageio.mimsave(
        "outputs/training.gif",
        frames,
        duration=0.08
    )

    # Clean temporary images
    for file in os.listdir("outputs/temp"):
        os.remove(os.path.join("outputs/temp", file))

    os.rmdir("outputs/temp")