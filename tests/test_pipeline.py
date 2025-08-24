# test_pipeline.py

from unittest.mock import patch, MagicMock

import pandas as pd

# Import functions from your pipeline module
# (adjust the import if your file has a different name, e.g. `pipeline.py`)
from pipeline import (
    run,
    merge_with_diagnostics,
    normalize_feedback_df,
    normalize_customer_df,
    derive_fields,
)


# ---------------------------------------------------
# 1. Happy-path end-to-end run
# ---------------------------------------------------
def test_run_happy_path(tmp_path):
    """End-to-end test of run() with mocked HTTP returning valid JSON."""

    feedback_payload = {"feedback": [{"id": "1", "survey_q1": 5, "survey_q2": 3}]}
    customer_payload = {"customers": [{"customerId": "1", "name": "Alice"}]}

    with patch("requests.get") as mock_get:
        mock_resp_feedback = MagicMock()
        mock_resp_feedback.status_code = 200
        mock_resp_feedback.json.return_value = feedback_payload

        mock_resp_customer = MagicMock()
        mock_resp_customer.status_code = 200
        mock_resp_customer.json.return_value = customer_payload

        mock_get.side_effect = [mock_resp_feedback, mock_resp_customer]

        exit_code = run(str(tmp_path), "test.csv")

    # Assert success
    assert exit_code == 0
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "run_diagnostics.json").exists()

    # Check derived column
    df = pd.read_csv(tmp_path / "test.csv")
    assert "avg_survey_score" in df.columns
    assert df["avg_survey_score"].iloc[0] == 4.0


# ---------------------------------------------------
# 2. Merge diagnostics with mismatched keys
# ---------------------------------------------------
def test_merge_with_diagnostics_key_mismatches():
    """Verify mismatched keys are detected and reported in diagnostics."""

    feedback_df = pd.DataFrame([{"customer_id": "1"}, {"customer_id": "2"}])
    customer_df = pd.DataFrame([{"customer_id": "2"}, {"customer_id": "3"}])

    merged, diag = merge_with_diagnostics(feedback_df, customer_df)

    # Merge should only match id=2
    assert len(merged) == 1
    assert merged["customer_id"].iloc[0] == "2"

    # Diagnostics should report 1 left-only and 1 right-only key
    assert diag["key_mismatches"]["left_only_count"] == 1
    assert diag["key_mismatches"]["right_only_count"] == 1
    assert "1" in diag["key_mismatches"]["left_only_sample"]
    assert "3" in diag["key_mismatches"]["right_only_sample"]


# ---------------------------------------------------
# 3. Normalization + derived fields logic
# ---------------------------------------------------
def test_normalization_and_derived_fields():
    """Check feedback/customer normalization and derived column creation."""

    # Feedback: id should rename -> customer_id
    feedback_df = pd.DataFrame([{"id": "cid123", "survey_q1": "5", "survey_q2": "7"}])
    norm_feedback = normalize_feedback_df(feedback_df)
    assert "customer_id" in norm_feedback.columns
    assert norm_feedback["customer_id"].iloc[0] == "cid123"

    # Customer: customerId should rename and strip cid prefix
    customer_df = pd.DataFrame([{"customerId": "cid123"}])
    norm_customer = normalize_customer_df(customer_df)
    assert "customer_id" in norm_customer.columns
    assert norm_customer["customer_id"].iloc[0] == "123"

    # Derived field average
    derived = derive_fields(norm_feedback)
    assert "avg_survey_score" in derived.columns
    assert derived["avg_survey_score"].iloc[0] == 6.0
