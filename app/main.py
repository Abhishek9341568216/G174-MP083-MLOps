import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .model_loader import load_model
from .schemas import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
)

logger = logging.getLogger("mp083.api")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(logging.StreamHandler())


def log_event(level, event, **fields):
    logger.log(level, json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
        **fields,
    }))


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    try:
        app.state.model = load_model()
    except Exception as error:
        log_event(logging.ERROR, "model_load_failed", error_type=type(error).__name__)
    else:
        log_event(logging.INFO, "model_loaded", version=app.state.model[1]["version"])
    yield


app = FastAPI(title="Mobile Data Consumption Prediction", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = uuid4().hex
    start = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    fields = {}
    if route_path == "/predict":
        fields["prediction_status"] = "succeeded" if response.status_code == 200 else "failed"
    if hasattr(request.state, "error_type"):
        fields["error_type"] = request.state.error_type
    log_event(logging.INFO, "request_completed", request_id=request_id, route=route_path,
              status_code=response.status_code, duration_ms=round((perf_counter() - start) * 1000, 2), **fields)
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    request.state.error_type = "RequestValidationError"
    return JSONResponse(status_code=422, content={"detail": "Invalid prediction input"})


def ready_model(request):
    if request.app.state.model is None:
        raise HTTPException(503, "Model unavailable")
    return request.app.state.model


@app.get("/")
def root():
    return {"service": app.title, "docs": "/docs"}


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/health")
def health(request: Request):
    ready = request.app.state.model is not None
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ok" if ready else "degraded",
                                                                   "model_ready": ready})


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request):
    pipeline, metadata = ready_model(request)
    try:
        prediction = pipeline.predict(pd.DataFrame([payload.model_dump()]))[0]
        return PredictionOutput(gb_per_day=prediction, model_version=metadata["version"])
    except Exception as error:
        request.state.error_type = type(error).__name__
        raise HTTPException(500, "Prediction failed") from None
@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
)
def predict_batch(payload: BatchPredictionInput, request: Request):
    pipeline, metadata = ready_model(request)

    try:
        records = [
            record.model_dump()
            for record in payload.records
        ]

        predictions = pipeline.predict(pd.DataFrame(records))

        if len(predictions) != len(records):
            raise ValueError("Unexpected prediction count")

        if not np.isfinite(predictions).all():
            raise ValueError("Invalid prediction values")

        results = [
            PredictionOutput(
                gb_per_day=float(value),
                model_version=metadata["version"],
            )
            for value in predictions
        ]

        return BatchPredictionOutput(predictions=results)

    except Exception as error:
        request.state.error_type = type(error).__name__
        raise HTTPException(
            500,
            "Batch prediction failed",
        ) from None



@app.get("/model-info")
def model_info(request: Request):
    _, metadata = ready_model(request)
    return {key: value for key, value in metadata.items() if key != "training_keys"}
