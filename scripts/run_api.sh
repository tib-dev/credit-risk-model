#!/usr/bin/env bash
echo "Starting FastAPI server..."
uvicorn credit_risk_model.api.main:app --host 0.0.0.0 --port 8000 --reload
