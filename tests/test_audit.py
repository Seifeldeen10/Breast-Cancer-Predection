import json
import pathlib

import pytest

from src.audit.audit import audit_file

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"

@pytest.fixture
def synthetic_csv_path() -> pathlib.Path:
    return TEST_DATA_DIR / "synthetic.csv"

def _ensure_output_dir() -> pathlib.Path:
    """Create a writable temporary directory within the repository.
    Returns the path to the directory.
    """
    out_dir = pathlib.Path(__file__).parent / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def test_audit_generates_reports(synthetic_csv_path):
    # Run audit on synthetic CSV, output to a repo‑local temporary directory
    out_dir = _ensure_output_dir()
    reports = audit_file(str(synthetic_csv_path), output_dir=str(out_dir))
    # Verify both JSON and MD reports exist
    assert reports["json"].exists()
    assert reports["md"].exists()
    # Load JSON and check expected keys
    data = json.loads(reports["json"].read_text())
    assert "shape" in data
    assert data["shape"]["rows"] == 5
    assert data["shape"]["columns"] == 4
    # Missing values should be detected (age for row 3, outcome for row 5)
    assert "age" in data["missing"]
    assert data["missing"]["age"]["count"] == 1
    # Duplicate rows (row 2 appears twice)
    assert data["duplicate_rows"] == 1
    # Identifier flag is optional based on explicit ID column detection; no hard assertion needed

def test_missing_file_error():
    out_dir = _ensure_output_dir()
    with pytest.raises(FileNotFoundError):
        audit_file("nonexistent_file.csv", output_dir=str(out_dir))

def test_unsupported_extension_error():
    out_dir = _ensure_output_dir()
    # Create a dummy .txt file
    txt_path = out_dir / "dummy.txt"
    txt_path.write_text("just some text")
    with pytest.raises(ValueError) as exc:
        audit_file(str(txt_path), output_dir=str(out_dir))
    assert "Unsupported file extension" in str(exc.value)

