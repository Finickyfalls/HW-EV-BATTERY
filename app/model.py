# This file recreates the model architecture and loads the saved weights,
# so the deployed app can make predictions without needing to retrain.

import torch
import torch.nn as nn
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class BatteryFailureClassifier(nn.Module):
    # Must match the training architecture exactly, or the saved weights won't load correctly.
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_artifacts():
    # Loads the scaler and the trained model weights from disk.
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    input_dim = scaler.mean_.shape[0]  # infer number of features from the scaler
    model = BatteryFailureClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "battery_model.pth"),
        map_location=torch.device("cpu"),
        weights_only=True
    ))
    model.eval()  # set to evaluation mode

    return model, scaler
