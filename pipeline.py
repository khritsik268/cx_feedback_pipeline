from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

# ----------------------------
# Configuration (override via CLI or env if desired)
# ----------------------------
FEEDBACK_URL = os.getenv(
    "FEEDBACK_URL",
    "https://undsicdm3f.execute-api.us-east-2.amazonaws.com/prod/feedback",
)
CUSTOMER_URL = os.getenv(
    "CUSTOMER_URL",
    "https://undsicdm3f.execute-api.us-east-2.amazonaws.com/prod/customers",
)
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
OUTPUT_DIR_DEFAULT = os.getenv("OUTPUT_DIR", "output")
OUTPUT_CSV_DEFAULT = os.getenv("OUTPUT_CSV", "customer_feedback.csv")
DIAG_FILENAME = "run_diagnostics.json"


# ----------------------------
# Logging
# ----------------------------
def setup_logging(verbosity: int) -> None:
    """
    Configure logging.
    verbosity: 0=INFO, 1=DEBUG
    """
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------
# Data Structures
# ----------------------------
@dataclass
class FetchResult:
    ok: bool
    data: Any
    status: int | None
    error: str | None


# ----------------------------
# HTTP & Validation
# ----------------------------
def fetch_json(url: str, timeout: float) -> FetchResult:
    """
    GET a JSON payload with basic error handling.
    Returns a FetchResult with ok flag and details for diagnostics.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        status = resp.status_code
        resp.raise_for_status()
        try:
            return FetchResult(ok=True, data=resp.json(), status=status, error=None)
        except ValueError as ve:
            logging.error("Invalid JSON from %s: %s", url, ve)
            return FetchResult(ok=False, data=None, status=status, error="invalid_json")
    except requests.RequestException as re:
        logging.error("HTTP error for %s: %s", url, re)
        return FetchResult(ok=False, data=None, status=None, error=str(re))


def expect_list_under_key(obj: Any, key: str) -> Tuple[bool, List[Dict[str, Any]] | None, str | None]:
    """
    Validate that `obj` is a dict and contains a list-of-dicts under `key`.
    Returns (ok, list, error_message).
    """
    if not isinstance(obj, dict):
        return False, None, "top_level_not_dict"
    if key not in obj:
        return False, None, f"missing_key:{key}"
    val = obj[key]
    if not isinstance(val, list):
        return False, None, f"value_not_list:{key}"
    # Optionally ensure elements are dicts
    if val and not all(isinstance(x, dict) for x in val):
        return False, None, f"list_elements_not_dict:{key}"
    return True, val, None


# ----------------------------
# Transformations
# ----------------------------
def normalize_feedback_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure feedback DF has 'customer_id' column:
    - If there's an 'id' column, rename to 'customer_id'.
    - Otherwise, leave as-is and rely on presence check later.
    """
    if "customer_id" in df.columns:
        return df
    if "id" in df.columns:
        df = df.rename(columns={"id": "customer_id"})
        logging.debug("Renamed feedback.id -> customer_id")
    else:
        logging.warning("Feedback data lacks 'customer_id' (or 'id'). Join may fail.")
    return df


def normalize_customer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure customer DF has 'customer_id' column:
    - If 'customerId' exists, rename to 'customer_id'.
    - Remove literal 'cid' prefix if present (non-regex).
    - Cast to string for safe joining.
    """
    if "customer_id" not in df.columns and "customerId" in df.columns:
        df = df.rename(columns={"customerId": "customer_id"})
        logging.debug("Renamed customer.customerId -> customer_id")

    if "customer_id" not in df.columns:
        logging.warning("Customer data lacks 'customer_id' (or 'customerId'). Join may fail.")
        return df

    # Clean literal 'cid' prefix if present (e.g., "cid123" -> "123")
    df["customer_id"] = df["customer_id"].astype(str).str.replace("cid", "", regex=False)
    return df


def standardize_join_keys(feedback_df: pd.DataFrame, customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coerce join keys to string for both frames.
    """
    if "customer_id" in feedback_df.columns:
        feedback_df["customer_id"] = feedback_df["customer_id"].astype(str)
    if "customer_id" in customer_df.columns:
        customer_df["customer_id"] = customer_df["customer_id"].astype(str)
    return feedback_df, customer_df


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally compute an average score if survey_q1 and survey_q2 exist.
    """
    needed = {"survey_q1", "survey_q2"}
    if needed.issubset(df.columns):
        df = df.copy()
        df["survey_q1"] = pd.to_numeric(df["survey_q1"], errors="coerce")
        df["survey_q2"] = pd.to_numeric(df["survey_q2"], errors="coerce")
        df["avg_survey_score"] = df[["survey_q1", "survey_q2"]].mean(axis=1, skipna=True)
        logging.info("Derived column 'avg_survey_score'.")
    else:
        missing = needed - set(df.columns)
        logging.debug("Skipping derived fields; missing columns: %s", sorted(missing))
    return df


# ----------------------------
# Join & Diagnostics
# ----------------------------
def merge_with_diagnostics(
        feedback_df: pd.DataFrame, customer_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform an inner join by 'customer_id' and compute diagnostics:
    - row counts before/after
    - key coverage (unique ids)
    - left/right-only counts and up to 5 sample keys
    """
    diagnostics: Dict[str, Any] = {}

    # Preconditions
    if "customer_id" not in feedback_df.columns or "customer_id" not in customer_df.columns:
        diagnostics["missing_join_key"] = {
            "feedback_has_customer_id": "customer_id" in feedback_df.columns,
            "customer_has_customer_id": "customer_id" in customer_df.columns,
        }
        logging.error("Missing 'customer_id' in one or both datasets; cannot merge.")
        return pd.DataFrame(), diagnostics

    # Shape stats
    diagnostics["pre_merge"] = {
        "feedback_rows": int(len(feedback_df)),
        "customer_rows": int(len(customer_df)),
        "feedback_unique_ids": int(feedback_df["customer_id"].nunique(dropna=True)),
        "customer_unique_ids": int(customer_df["customer_id"].nunique(dropna=True)),
    }

    # Mismatch analysis
    _left = feedback_df[["customer_id"]].assign(_src="left")
    _right = customer_df[["customer_id"]].assign(_src="right")
    union = pd.concat([_left, _right], ignore_index=True)
    counts = union.groupby(["customer_id"], dropna=False)["_src"].nunique()
    only_left_ids = counts[counts == 1].index.intersection(_left["customer_id"])
    only_right_ids = counts[counts == 1].index.intersection(_right["customer_id"])

    diagnostics["key_mismatches"] = {
        "left_only_count": int(len(only_left_ids)),
        "right_only_count": int(len(only_right_ids)),
        "left_only_sample": list(map(str, list(only_left_ids)[:5])),
        "right_only_sample": list(map(str, list(only_right_ids)[:5])),
    }

    if diagnostics["key_mismatches"]["left_only_count"] or diagnostics["key_mismatches"]["right_only_count"]:
        logging.warning(
            "Join key mismatches detected: left_only=%d, right_only=%d",
            diagnostics["key_mismatches"]["left_only_count"],
            diagnostics["key_mismatches"]["right_only_count"],
        )

    # Merge
    merged = feedback_df.merge(
        customer_df, on="customer_id", how="inner", validate="many_to_many"
    )

    diagnostics["post_merge"] = {
        "merged_rows": int(len(merged)),
        "merge_ratio_vs_feedback": float(len(merged)) / max(1, len(feedback_df)),
        "merge_ratio_vs_customer": float(len(merged)) / max(1, len(customer_df)),
    }

    logging.info(
        "Merged rows: %d (feedback=%d, customers=%d)",
        diagnostics["post_merge"]["merged_rows"],
        diagnostics["pre_merge"]["feedback_rows"],
        diagnostics["pre_merge"]["customer_rows"],
    )

    return merged, diagnostics


# ----------------------------
# Orchestration
# ----------------------------
def run(output_dir: str, output_csv: str) -> int:
    """
    Main orchestration with explicit exit codes.
    Writes diagnostics JSON alongside the CSV for quick triage.
    """
    diag: Dict[str, Any] = {
        "feedback_url": FEEDBACK_URL,
        "customer_url": CUSTOMER_URL,
    }

    # Fetch
    logging.info("Fetching feedback data…")
    feedback_res = fetch_json(FEEDBACK_URL, HTTP_TIMEOUT_SECONDS)
    logging.info("Fetching customer data…")
    customer_res = fetch_json(CUSTOMER_URL, HTTP_TIMEOUT_SECONDS)
    diag["http"] = {
        "feedback": {"ok": feedback_res.ok, "status": feedback_res.status, "error": feedback_res.error},
        "customer": {"ok": customer_res.ok, "status": customer_res.status, "error": customer_res.error},
    }

    if not (feedback_res.ok and customer_res.ok):
        _write_diagnostics(output_dir, diag)
        return 1  # network error

    # Shape validation
    ok_f, feedback_list, err_f = expect_list_under_key(feedback_res.data, "feedback")
    ok_c, customer_list, err_c = expect_list_under_key(customer_res.data, "customers")
    diag["shape"] = {"feedback": err_f or "ok", "customers": err_c or "ok"}

    if not (ok_f and ok_c):
        logging.error("Unexpected response shapes: feedback=%s, customers=%s", err_f, err_c)
        _write_diagnostics(output_dir, diag)
        return 2  # shape error

    # DataFrames
    feedback_df = pd.DataFrame(feedback_list or [])
    customer_df = pd.DataFrame(customer_list or [])

    logging.debug("Feedback columns: %s", feedback_df.columns.tolist())
    logging.debug("Customer columns: %s", customer_df.columns.tolist())

    feedback_df = normalize_feedback_df(feedback_df)
    customer_df = normalize_customer_df(customer_df)
    feedback_df, customer_df = standardize_join_keys(feedback_df, customer_df)

    merged_df, join_diag = merge_with_diagnostics(feedback_df, customer_df)
    diag.update(join_diag)

    if merged_df.empty:
        logging.error("Merge produced 0 rows; check join keys and source coverage.")
        _write_diagnostics(output_dir, diag)
        return 3  # useless/empty merge

    merged_df = derive_fields(merged_df)

    # Output
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_csv)
        merged_df.to_csv(out_path, index=False)
        diag["output_csv"] = out_path
        _write_diagnostics(output_dir, diag)
        logging.info("CSV written: %s", out_path)
        return 0
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to write CSV: %s", e)
        _write_diagnostics(output_dir, diag)
        return 4  # write failure


def _write_diagnostics(output_dir: str, diag: Dict[str, Any]) -> None:
    """Write a small JSON diagnostics file to help operators triage quickly."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, DIAG_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        logging.debug("Diagnostics written: %s", path)
    except Exception as e:  # noqa: BLE001
        logging.warning("Could not write diagnostics file: %s", e)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join feedback with customers and export CSV.")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory for outputs (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV_DEFAULT,
        help=f"CSV filename (default: {OUTPUT_CSV_DEFAULT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v for DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    code = run(args.output_dir, args.output_csv)
    sys.exit(code)


if __name__ == "__main__":
    main()
