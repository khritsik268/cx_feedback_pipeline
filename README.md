# Farlinium Technical Case Study: Data Pipeline for CX Feedback

## Overview
This Python script automates the collection and consolidation of customer feedback and customer metadata. It pulls data from two REST APIs using HTTP GET requests, merges the datasets on customer ID, and outputs a clean CSV ready for business analysis. Additionally, it calculates an average survey score for each customer.

---

## Requirements
- Python 3.13 or higher
  - Python packages:
  ```bash
  pip install -r requirements.txt
  ```
  
### Running the script
1. Run the script:
    ```
    python pipeline.py
   ```
2. The output CSV will be generated at:
    ```
   output/customer_feedback.csv
   ```

### Assumptions & Design Decisions
- The feedback API returns JSON with a `feedback` key containing survey responses.
- The customer API returns JSON with a `customers` key containing customer metadata.
- Customer IDs in the customer dataset have a `"cid"` prefix; this is removed to match the numeric IDs in the feedback dataset.
- Both survey responses (`survey_q1`,`survey_q2`) are numeric (1-10).
- Merge type: **inner join** to include only customers with both feedback and metadata.
- Output folder `output/` is automatically created if it doesn't exist.

### Additional Notes
- Error Handling:
  - HTTP errors and timeouts are logged.
  - Missing or malformed data is handled gracefully.
- Calculated Fields:
  - `avg_survey_score` = mean of `survey_q1` and `survey_q2`.
- The CSV includes relevant fields from both datasets and is ready for downstream reporting.

### Sample Output Columns
    customer_id,survey_q1,survey_q2,free_text,account_age_days,geographic_region,customer_segment,avg_survey_score




