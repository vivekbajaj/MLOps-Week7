import logging
import json
import time
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- Configuration ---
LOCAL_MODEL_PATH = "model.joblib"
IRIS_CLASS_NAMES = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# --- Setup Tracer ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# --- Custom JSON Formatter for Logging ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "severity": record.levelname,
            "message": record.getMessage(),
            "jsonPayload": getattr(record, 'json_fields', {})
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# --- Setup Structured Logging ---
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# --- FastAPI App ---
app = FastAPI()
app.state.model = None 
app_state = {"is_ready": False, "is_alive": True}

# --- Input Schema ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- App Events ---
@app.on_event("startup")
async def startup_event():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, LOCAL_MODEL_PATH)
        logger.info(f"Attempting to load model from: {model_path}")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}")
        app.state.model = joblib.load(model_path)
        app_state["is_ready"] = True
        logger.info("Model loaded successfully. Service is ready.")
    except Exception as e:
        app_state["is_ready"] = False
        logger.error("Failed to load model on startup.", exc_info=True, extra={"json_fields": {"error": str(e)}})

# --- Probes ---
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"] and app.state.model:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# --- Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# --- Exception Handler ---
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    log_payload = {
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc),
    }
    logger.error("Unhandled exception occurred", exc_info=True, extra={"json_fields": log_payload})
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "trace_id": trace_id})

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(input: IrisInput):
    if not app.state.model:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. Check server logs.")

    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            # Prepare input using a pandas DataFrame
            input_data = input.dict()
            input_df = pd.DataFrame([input_data])
            
            # Get probabilities first
            probabilities = app.state.model.predict_proba(input_df)[0]

            # Find the index of the highest probability. This IS the predicted index.
            prediction_idx = int(np.argmax(probabilities))
            
            confidence = float(probabilities[prediction_idx])
            prediction_class = IRIS_CLASS_NAMES.get(prediction_idx, "Unknown")
            
            result = {
                "prediction": prediction_class,
                "prediction_index": prediction_idx,
                "confidence": round(confidence, 4)
            }
            latency = round((time.time() - start_time) * 1000, 2)

            log_payload = {
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }
            logger.info("Prediction successful", extra={"json_fields": log_payload})
            
            return result

        except Exception as e:
            log_payload = {
                "event": "prediction_error",
                "trace_id": trace_id,
                "input": input.dict(),
                "error": str(e)
            }
            logger.error("Prediction failed", exc_info=True, extra={"json_fields": log_payload})
            raise HTTPException(status_code=500, detail="Prediction failed")
