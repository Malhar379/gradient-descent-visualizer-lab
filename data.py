import torch


def get_data():
    x = torch.tensor([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])

    y = torch.tensor([
        [1.0],
        [3.0],
        [5.0],
        [7.0]
    ])

    return x, y