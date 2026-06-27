# 🚀 Production Readiness, Security & Future Roadmap

This document maps out the system production checklists, security policies, horizontal scaling strategies, and the roadmap for upcoming development phases.

---

## 1. Production Checklist

Ensure these validation checks are met before deploying the platform nodes to staging or production:

*   [ ] **Secret Rotation**: Change the default `SECRET_KEY` in the `.env` production profile. Do not commit `.env` to git.
*   [ ] **Production Database**: Switch the default database URL setting from SQLite to a highly-available **PostgreSQL** cluster.
*   [ ] **Upload Buffer Cleanups**: Configure cron tasks or daemon services on the host to periodically clear old files inside `storage/uploads/` and `storage/reports/`.
*   [ ] **Rate Limiting**: Integrate rate-limiting policies in API gateways (e.g. Nginx or Cloudflare) to prevent brute-force request exhaustion.
*   [ ] **Docker Security**: Run the Docker containers under non-root user execution contexts (e.g. configuring `USER python` in the final stages of the Dockerfile).

---

## 2. Horizontal Scaling & High-Availability Plan

```
                   ┌───────────────────┐
                   │   Load Balancer   │
                   └─────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ FastAPI App │  │ FastAPI App │  │ FastAPI App │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Postgres Master │
                    └─────────────────┘
```

*   **Stateless Scaling**: Since API controllers do not hold state locally, multiple container replicas of the FastAPI application can run behind an Nginx load balancer or inside a Kubernetes replica set.
*   **Database Partitioning**: As predictions grow, partition SQL tables (e.g., partitioning `predictions` table by date blocks) and scale read queries using PostgreSQL read replicas.
*   **Distributed Task Queues**: For very long video analysis, replace synchronous `/predict` calls with an asynchronous **Celery + Redis** task distribution system.

---

## 3. Future Development Roadmap

### Phase 2: Bulk Preprocessing & Feature Extraction
*   *Milestones*: Create multi-threaded OpenCV/FFmpeg video parsing workers; batch process datasets and generate dataset indices.

### Phase 3: Model Training & Validation
*   *Milestones*: Write PyTorch training loops for Video, Audio, and Multimodal architectures; integrate experiment tracking using MLflow.

### Phase 4: Streamlit Frontend & User Dashboards
*   *Milestones*: Launch a web UI dashboard allowing analysts to upload media, view verification scores, visual attribution heatmaps (Grad-CAM), and download PDF audits.

### Phase 5: Browser Extension & API Integrations
*   *Milestones*: Launch a Chrome/Firefox extension that checks embedded web page videos on the fly using API calls to the forensics server.
