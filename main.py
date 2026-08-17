# This is the actual web API. It loads the model once at startup, then
# exposes endpoints for making predictions.

import numpy as np
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.model import load_artifacts

app = FastAPI(title="EV Battery Failure Predictor")
templates = Jinja2Templates(directory="app/templates")

model, scaler = load_artifacts()  # load once when the app starts, not per-request

# Must match the exact order of features used during training.
FEATURE_ORDER = [
    "state_of_health", "cell_voltage_std", "charge_efficiency",
    "cycle_count", "internal_resistance", "thermal_runaway_risk"
]


class PredictRequest(BaseModel):
    # Defines what a valid JSON request to /predict must look like.
    state_of_health: float
    cell_voltage_std: float
    charge_efficiency: float
    cycle_count: float
    internal_resistance: float
    thermal_runaway_risk: float


def predict(features: dict):
    # Shared prediction logic used by both the API and the web form.
    x = np.array([[features[f] for f in FEATURE_ORDER]])
    x_scaled = scaler.transform(x)          # scale exactly like training data was scaled
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(x_tensor)
        prob = torch.sigmoid(logit).item()  # convert raw output into a 0-1 probability

    pred_class = "Failure" if prob >= 0.5 else "Healthy"
    return pred_class, prob


# --- JSON API endpoint (for other programs to call) ---
@app.post("/predict")
def predict_api(payload: PredictRequest):
    pred_class, prob = predict(payload.dict())
    return {"prediction": pred_class, "probability": round(prob, 4)}


# --- Simple web form (for a human to test in a browser) ---
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    state_of_health: float = Form(...),
    cell_voltage_std: float = Form(...),
    charge_efficiency: float = Form(...),
    cycle_count: float = Form(...),
    internal_resistance: float = Form(...),
    thermal_runaway_risk: float = Form(...),
):
    features = {
        "state_of_health": state_of_health,
        "cell_voltage_std": cell_voltage_std,
        "charge_efficiency": charge_efficiency,
        "cycle_count": cycle_count,
        "internal_resistance": internal_resistance,
        "thermal_runaway_risk": thermal_runaway_risk,
    }
    pred_class, prob = predict(features)
    result = {"prediction": pred_class, "probability": f"{prob:.4f}"}
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


# Railway (and most hosts) ping this to check the app is alive.
@app.get("/health")
def health():
    return {"status": "ok"}
