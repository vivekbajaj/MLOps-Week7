# Demo Iris ML Service with Structured Logging and OpenTelemetry Tracing

## Overview

This repository contains a FastAPI-based machine learning microservice for Iris flower classification. The service loads a scikit-learn model and exposes a `/predict` endpoint for inference. It features structured JSON logging and is integrated with OpenTelemetry for distributed tracing, exporting spans to Google Cloud Trace.

The project includes Kubernetes deployment manifests with autoscaling and health probes, and a Dockerfile for containerized deployment.

## Project Files

- `demo_log.py`: Main FastAPI application module that loads the ML model, defines API endpoints, handles prediction requests, and integrates logging and OpenTelemetry tracing.
- `Dockerfile`: Container specification to build the application image, installing dependencies, and preparing the runtime environment.
- `deployment-1.yaml`: Kubernetes Deployment manifest describing pods, replicas, and liveness/readiness probes for app deployment.
- `service-1.yaml`: Kubernetes Service manifest exposing the deployment as a LoadBalancer service.
- `hpa.yaml`: Kubernetes Horizontal Pod Autoscaler manifest for autoscaling pods based on CPU utilization.
- `requirements.txt`: Python package dependencies needed to run the application.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Logging and Tracing](#logging-and-tracing)
- [Deployment](#deployment)
- [Autoscaling](#autoscaling)
- [Health Checks](#health-checks)
- [Contributing](#contributing)
- [License](#license)

## Features

- **FastAPI**: Lightweight REST API framework for serving the ML model.
- **ML Model**: Scikit-learn Iris classifier loaded with `joblib`.
- **Endpoints**:
  - `/predict`: Accepts Iris flower features and returns predictions.
  - `/live_check` & `/ready_check`: Kubernetes health probes.
- **Structured JSON Logging**: Logs enriched events in JSON format for easier analysis.
- **OpenTelemetry Integration**: Traces request lifecycle and exports to Google Cloud Trace.
- **Kubernetes Deployment**: Full manifests for deployment, service, and Horizontal Pod Autoscaler (HPA).
- **Dockerized**: Includes Dockerfile for easy container builds.

## Architecture

Client
│
├─> Kubernetes Service (LoadBalancer)
│
├─> FastAPI Pod(s) w/ Iris ML Model
│
├─> Structured Logs (stdout)
└─> OpenTelemetry Tracing → Google Cloud Trace

text

## Getting Started

### Prerequisites

- Python 3.11
- Docker
- Kubernetes cluster (with GCP integration for Cloud Trace exporter)
- `kubectl` CLI configured
- `make` (optional, for convenience)

### Installing Dependencies (Local Development)

pip install -r requirements.txt

text

### Running Locally

Make sure you have your trained model saved as `model.joblib` in the same directory.

Run the FastAPI service:

uvicorn demo_log:app --host 0.0.0.0 --port 8200

text

The API will be available at `http://localhost:8200`.

## Usage

### Request Format for `/predict` (POST)

{
"sepal_length": 5.1,
"sepal_width": 3.5,
"petal_length": 1.4,
"petal_width": 0.2
}

text

### Sample cURL

curl -X POST "http://localhost:8200/predict"
-H "Content-Type: application/json"
-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

text

### Response

{
"prediction": "setosa",
"prediction_index": 0,
"confidence": 0.9876
}

text

## Logging and Tracing

- Logs are emitted in structured JSON format to stdout/stderr.
- Includes trace IDs for correlation with OpenTelemetry traces.
- Errors and exceptions contain detailed diagnostic information.
- Traces are exported to Google Cloud Trace via the `CloudTraceSpanExporter`.
- Every prediction is traced under a span named `model_inference`.

## Deployment

### Docker

Build the Docker image:

docker build -t demo_log:latest .

text

Run the container:

docker run -p 8200:8200 demo_log:latest

text

### Kubernetes

Apply the manifests:

kubectl apply -f deployment-1.yaml
kubectl apply -f service-1.yaml
kubectl apply -f hpa.yaml

text

- The `deployment-1.yaml` deploys the app with liveness and readiness probes.
- The `service-1.yaml` exposes the deployment using a LoadBalancer.
- The `hpa.yaml` enables autoscaling based on CPU utilization.

## Autoscaling

The Horizontal Pod Autoscaler (HPA) configuration will automatically scale between 2 and 10 replicas depending on CPU load (target 60% utilization).

## Health Checks

Two endpoints are included for Kubernetes probes:

- **Liveness Probe**: `/live_check`  
  Returns 200 if the app is alive.

- **Readiness Probe**: `/ready_check`  
  Returns 200 only if the ML model is loaded and the app is ready to serve.

These help Kubernetes manage pod lifecycle.

## Contributing

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

If you have any questions or need assistance, please open an issue or contact the maintainer.


