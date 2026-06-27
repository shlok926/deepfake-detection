# 💻 Developer & Contribution Guide

This guide defines coding standards, testing patterns, and procedures for contributing to the AI Forensics Platform.

---

## 1. Coding Standards & Tooling

The platform enforces PEP 8 and strict typing rules. Ensure your IDE is configured to use the workspace formatters.

*   **Code Formatter**: We use **Black** (Line length limit: `120` characters).
*   **Import Sorting**: We use **isort** (Profile configured to match Black).
*   **Static Linters**: We use **Flake8** and **mypy** for strict typing checks.

### Utility Formatting Commands
Run these tasks using the `Makefile` command profiles:
*   **Format entire project**:
    ```bash
    make format
    ```
*   **Check code quality and types**:
    ```bash
    make lint
    ```

---

## 2. Guide: Adding a New Forensic Detection Model

To add a new detection model (e.g., an Image Compression Artifact detector, or a Temporal LSTM network):

1.  **Add Model Architecture**:
    Define your network class in `ai_engine/video/` or create a new modality directory under `ai_engine/`. Ensure it inherits from `nn.Module` and implements `forward`.
2.  **Expose Feature Extraction Modes**:
    Implement a `feature_extraction` boolean flag in your class constructor so that your model can return intermediate embeddings (required for late fusion classifiers).
3.  **Register Model Checkpoint**:
    Register your new model name, version, and default path in the SQLAlchemy `ModelRegistryModel` schema in `app/database/models.py`.
4.  **Inject into Inference Controller**:
    Update `app/routes/predict.py` to allow execution of the new model.

---

## 3. Contribution Rules (Branching & PRs)

*   **Branching Pattern**: Use standard feature branches: `feature/your-feature-name`.
*   **Merge Target**: All pull requests must target the `develop` branch.
*   **Testing Requirement**: PRs cannot be merged unless they pass all unit tests in the CI/CD pipeline and have 100% linter compilation approval.
*   **Git Hooks**: Enable pre-commit hooks before committing:
    ```bash
    pre-commit install
    ```
