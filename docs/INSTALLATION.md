# 📥 Platform Installation & Deployment Guide

This guide details the steps required to set up and run the AI Digital Media Forensics & Verification Platform in different environments.

---

## 💻 1. Local Development Setup

### System Prerequisites
*   **Python**: Version `3.10` or `3.11` (Python 3.12+ is supported but dependencies like PyTorch are most stable on 3.10/3.11).
*   **FFmpeg**: Required on the host machine for extracting audio and cutting video frames.
    *   *Windows*: Install via Chocolatey (`choco install ffmpeg`) or download from official site and add to system `PATH`.
    *   *macOS*: Install via Homebrew (`brew install ffmpeg`).
    *   *Linux*: Install via apt (`sudo apt install ffmpeg`).

### Installation Sequence
1.  **Initialize Dev Environment**:
    Run the automated platform setup script depending on your OS:
    *   *Windows (PowerShell)*:
        ```powershell
        powershell -ExecutionPolicy Bypass -File scripts/setup_dev.ps1
        ```
    *   *Linux / macOS (Bash)*:
        ```bash
        chmod +x scripts/setup_dev.sh
        ./scripts/setup_dev.sh
        ```
2.  **Activate Virtual Environment**:
    *   *Windows*: `venv\Scripts\activate`
    *   *Linux / macOS*: `source venv/bin/activate`
3.  **Verify Setup**:
    Run the unit test suite to verify the REST API routers and SQLite connections:
    ```bash
    python -m pytest -o addopts="" tests/
    ```

---

## 🐳 2. Containerized Deployment (Docker Compose)

The platform includes a production-grade multi-stage `Dockerfile` and a `docker-compose.yml` service profile that bundles the FastAPI app, Prometheus, and Grafana.

### CPU Deployment
To launch the platform in CPU-only mode:
1.  Set `USE_GPU=False` in your `.env` file.
2.  Launch the services:
    ```bash
    docker-compose up --build -d
    ```

### GPU-Accelerated Deployment (NVIDIA CUDA)
To utilize host GPUs inside Docker containers:
1.  Ensure you have **NVIDIA Container Toolkit** installed on your host.
2.  Verify `USE_GPU=True` and configure your target `CUDA_DEVICE_ID` in `.env`.
3.  Launch the services:
    ```bash
    docker-compose up --build -d
    ```
    *Docker Compose will automatically reserve host GPUs and pass them through to the container.*

---

## 📊 3. Telemetry & Telemetry Dashboards

Once Docker Compose is running, the following internal endpoints become active:

| Endpoint | Target URL | Description |
| :--- | :--- | :--- |
| **Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API exploration and testing dashboard. |
| **Prometheus Metrics** | [http://localhost:8000/metrics](http://localhost:8000/metrics) | Scrape target exposed by FastAPI middleware. |
| **Prometheus UI** | [http://localhost:9090](http://localhost:9090) | Telemetry querying database page. |
| **Grafana UI** | [http://localhost:3000](http://localhost:3000) | Live dashboards (Credentials: `admin`/`admin-secure-pass`). |
