# tests/conftest.py
"""Pytest configuration for Breast Cancer Risk prototype.

Ensures that the pre‑risk model pipeline exists before any test that
loads it runs. If the file is missing (e.g., fresh checkout), the
training script is executed to create the artifact in the project's
`models/` directory.
"""

import subprocess
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "risk_pre_pipeline.pkl"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train" / "train_pre_model.py"



@pytest.fixture(autouse=True, scope="session")
def ensure_pre_risk_model():
    """Make sure the pre‑risk model file exists.

    The fixture runs once per test session. If ``MODEL_PATH`` is not
    present, it invokes the training script. Any failure is raised as a
    ``RuntimeError`` so that tests abort clearly.
    """
    if not MODEL_PATH.exists():
        try:
            subprocess.check_call(["python", str(TRAIN_SCRIPT)], cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to train pre‑risk model: {exc}")
    # Yield control to the tests – nothing to clean up.
    yield
