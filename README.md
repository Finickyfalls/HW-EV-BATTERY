# EV Battery Failure Predictor

A PyTorch neural network classifier that predicts whether an EV battery is
healthy or heading toward failure, served with FastAPI and deployed via Docker on Railway.

## Features used
- state_of_health (%)
- cell_voltage_std
- charge_efficiency (%)
- cycle_count
- internal_resistance
- thermal_runaway_risk

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit http://localhost:8000

## Run with Docker
```bash
docker build -t ev-battery-predictor .
docker run -p 8000:8000 ev-battery-predictor
```
Visit http://localhost:8000

## API
POST /predict
```json
{
  "state_of_health": 68.0,
  "cell_voltage_std": 6.2,
  "charge_efficiency": 78.5,
  "cycle_count": 950,
  "internal_resistance": 0.85,
  "thermal_runaway_risk": 74.0
}
```

## Live Deployment
hw-ev-battery-production.up.railway.app
