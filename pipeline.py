import requests
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API endpoints
FEEDBACK_URL = "https://undsicdm3f.execute-api.us-east-2.amazonaws.com/prod/feedback"
CUSTOMER_URL = "https://undsicdm3f.execute-api.us-east-2.amazonaws.com/prod/customers"

def fetch_data(url):
    # Fetch JSON data from API endpoint with error handling
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

def main():
    # Step 1: Get data
    logging.info("Fetching feedback data...")
    feedback_data = fetch_data(FEEDBACK_URL)
    logging.info("Fetching customer data...")
    customer_data = fetch_data(CUSTOMER_URL)

    if not feedback_data or not customer_data:
        logging.error("One or both datasets could not be retrieved.")
        return

    # Step 2: Convert to DataFrames (nested under dict keys)
    feedback_df = pd.DataFrame(feedback_data["feedback"])
    customer_df = pd.DataFrame(customer_data["customers"])

    # Debug: check column names
    print("Feedback columns:", feedback_df.columns.tolist())
    print("Customer columns:", customer_df.columns.tolist())
    print("Feedback sample:")
    print(feedback_df.head())
    print("Customer sample:")
    print(customer_df.head())

    # Step 3: Normalize join keys
    feedback_df.rename(columns={"id": "customer_id"}, inplace=True)  # in case 'id' exists
    customer_df.rename(columns={"customerId": "customer_id"}, inplace=True)  # in case 'customerId' exists

    # Normalize IDs so they can match: remove 'cid' from customer_df
    customer_df["customer_id"] = customer_df["customer_id"].str.replace("cid", "", regex=False)

    # Ensure both columns are strings
    feedback_df["customer_id"] = feedback_df["customer_id"].astype(str)
    customer_df["customer_id"] = customer_df["customer_id"].astype(str)

    # Step 4: Merge on normalized key
    merged_df = feedback_df.merge(customer_df, on="customer_id", how="inner")

    print("Merged rows:", len(merged_df))
    print(merged_df.head())

    # Step 5: Add calculated field: average of the two ratings
    if {"survey_q1", "survey_q2"}.issubset(merged_df.columns):
        merged_df["avg_survey_score"] = merged_df[["survey_q1", "survey_q2"]].mean(axis=1)

    # Step 6: Save to CSV file (create output folder if it doesn't already exist)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "customer_feedback.csv")

    merged_df.to_csv(output_file, index=False)
    logging.info(f"CSV file generated: {output_file}")


if __name__ == "__main__":
    main()