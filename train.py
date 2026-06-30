import torch
import torch.nn as nn
import copy


def train_model(model, x, y, epochs=100, learning_rate=0.01):

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    snapshots = []

    for epoch in range(epochs):

        prediction = model(x)

        loss = criterion(prediction, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        # Save model at this epoch
        snapshots.append(copy.deepcopy(model.state_dict()))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss.item():.4f}")

    return losses, snapshots