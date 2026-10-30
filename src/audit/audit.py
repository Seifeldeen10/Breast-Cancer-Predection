import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml


def load_tabular(file_path: Path) -> pd.DataFrame:
    """Load CSV, TSV, or Excel file into a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or required
            optional dependencies (e.g., openpyxl) are missing.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if ext in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(file_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to read Excel file (ensure 'openpyxl' is installed for .xlsx): {exc}"
            ) from exc
    raise ValueError(f"Unsupported file extension: {ext}")


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute required audit statistics for a DataFrame."""
    stats: Dict[str, Any] = {}
    stats["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    stats["columns"] = list(df.columns)
    # Data types
    stats["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    # Missing values
    missing = df.isnull().sum()
    stats["missing"] = {
        col: {"count": int(cnt), "percent": round(cnt / len(df) * 100, 2)}
        for col, cnt in missing.items()
        if cnt > 0
    }
    # Duplicate rows
    dup_count = df.duplicated().sum()
    stats["duplicate_rows"] = int(dup_count)
    # Unique values for object / categorical columns
    cat_stats = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        uniques = df[col].dropna().unique()
        cat_stats[col] = {
            "unique_count": int(len(uniques)),
            "unique_values": list(map(str, uniques))[:20],  # limit for brevity
        }
    stats["categorical"] = cat_stats
    # Numerical summaries
    num_stats = {}
    for col in df.select_dtypes(include=["number"]).columns:
        desc = df[col].describe()
        num_stats[col] = {
            "count": int(desc["count"]),
            "mean": round(desc["mean"], 4),
            "std": round(desc["std"], 4),
            "min": round(desc["min"], 4),
            "25%": round(desc["25%"], 4),
            "50%": round(desc["50%"], 4),
            "75%": round(desc["75%"], 4),
            "max": round(desc["max"], 4),
        }
    stats["numerical"] = num_stats
    # Candidate target distributions – simple heuristic: low‑cardinality columns (< 10 uniq)
    target_dist = {}
    for col in df.columns:
        uniq = df[col].nunique(dropna=True)
        if uniq > 1 and uniq <= 10:
            dist = df[col].value_counts(dropna=False).to_dict()
            target_dist[col] = {str(k): int(v) for k, v in dist.items()}
    stats["candidate_targets"] = target_dist
    # Flags for manual review
    flags: List[str] = []
    # Potential identifier leakage – only columns that explicitly look like IDs (e.g., end with '_id' or are 'patient_id')
    id_like = [c for c in df.columns if str(c).lower().endswith('_id') or str(c).lower() == 'patient_id']
    if id_like:
        flags.append(f"Potential identifier columns: {', '.join(id_like)}")
    # Deterministic columns (single unique value)
    deterministic = [c for c in df.columns if df[c].nunique(dropna=True) == 1]
    if deterministic:
        flags.append(f"Deterministic columns (single value): {', '.join(deterministic)}")
    # Suspiciously deterministic relationships – very simple: any column that is the same as another
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 >= col2:
                continue
            if df[col1].equals(df[col2]):
                flags.append(f"Columns '{col1}' and '{col2}' are identical – possible leakage")
    stats["flags"] = flags
    return stats


def generate_markdown_report(stats: Dict[str, Any], dataset_path: Path) -> str:
    """Create a human‑readable Markdown report from statistics."""
    lines = [f"# Data Audit Report for `{dataset_path.name}`", ""]
    shape = stats["shape"]
    lines.append(f"**Rows:** {shape['rows']}, **Columns:** {shape['columns']}  \n")
    lines.append("## Column Overview")
    for col in stats["columns"]:
        dtype = stats["dtypes"][col]
        lines.append(f"- `{col}`: {dtype}")
    lines.append("\n## Missing Values")
    if stats["missing"]:
        for col, info in stats["missing"].items():
            lines.append(f"- `{col}`: {info['count']} missing ({info['percent']}%)")
    else:
        lines.append("_No missing values detected._")
    lines.append(f"\n## Duplicate Rows\n- {stats['duplicate_rows']} duplicate rows detected")
    lines.append("\n## Categorical Columns Summary")
    if stats["categorical"]:
        for col, info in stats["categorical"].items():
            lines.append(
                f"- `{col}`: {info['unique_count']} unique values (sample: {', '.join(info['unique_values'])})"
            )
    else:
        lines.append("_No categorical columns detected._")
    lines.append("\n## Numerical Columns Summary")
    if stats["numerical"]:
        for col, info in stats["numerical"].items():
            lines.append(
                f"- `{col}`: count={info['count']}, mean={info['mean']}, std={info['std']}, min={info['min']}, 25%={info['25%']}, 50%={info['50%']}, 75%={info['75%']}, max={info['max']}"
            )
    else:
        lines.append("_No numerical columns detected._")
    lines.append("\n## Candidate Target Distributions (Low‑cardinality columns)")
    if stats["candidate_targets"]:
        for col, dist in stats["candidate_targets"].items():
            lines.append(f"- `{col}` distribution:")
            for val, cnt in dist.items():
                lines.append(f"  - {val}: {cnt}")
    else:
        lines.append("_No low‑cardinality columns found._")
    lines.append("\n## Flags for Manual Review")
    if stats["flags"]:
        for f in stats["flags"]:
            lines.append(f"- {f}")
    else:
        lines.append("_No immediate concerns detected._")
    # Dataset provenance placeholder – will be filled manually after inspection
    lines.append("\n## Provenance & License\n- Source: unknown (to be filled after manual inspection)\n- License: unknown\n")
    # Suitability conclusion placeholder
    lines.append("\n## Suitability Assessment\n- Risk modeling: **requires further investigation**\n- Stage modeling: **requires further investigation**\n- Exploratory analysis: **possible**\n")
    return "\n".join(lines)


def save_reports(stats: Dict[str, Any], dataset_path: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    json_path = out_dir / f"{stem}_report.json"
    md_path = out_dir / f"{stem}_report.md"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=2)
    md_content = generate_markdown_report(stats, dataset_path)
    md_path.write_text(md_content, encoding="utf-8")
    return {"json": json_path, "md": md_path}


def audit_file(input_path: str, output_dir: str = "reports") -> Dict[str, Path]:
    """High‑level helper used by both CLI and tests.

    Returns a mapping with keys ``json`` and ``md`` pointing to the generated files.
    """
    dataset_path = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    df = load_tabular(dataset_path)
    stats = compute_statistics(df)
    return save_reports(stats, dataset_path, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a tabular breast‑cancer dataset.")
    parser.add_argument("input", help="Path to CSV/TSV/Excel file to audit")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="reports",
        help="Directory where JSON/MD reports will be written (default: reports)",
    )
    args = parser.parse_args()
    try:
        paths = audit_file(args.input, args.output_dir)
        print(f"JSON report written to: {paths['json']}")
        print(f"Markdown report written to: {paths['md']}")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
