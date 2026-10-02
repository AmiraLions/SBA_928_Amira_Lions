"""
Dataset Builder for SBA 928: Fine-Tuning google/flan-t5-small
Author: Amira Lions
Description: Loads customer support ticket data, cleans records, and formats 
             them into strict instruction-context-target training pairs.
"""

import pandas as pd
from pathlib import Path

def build_training_dataset():
    csv_path = Path("SBA_928_customer_support_tickets.csv")

    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}. Ensure it is in the project root.")
        return

    print("Loading customer support dataset...")
    df = pd.read_csv(csv_path)

    df_clean = df.dropna(subset=['Ticket Subject', 'Ticket Description', 'Resolution']).copy()
    df_clean = df_clean[df_clean['Resolution'].str.len() > 5]

    formatted_data = []
    for _, row in df_clean.iterrows():
        instruction = "Analyze the customer support ticket, identify the issue, and summarize the resolution."
        context = f"Product: {row['Product Purchased']} | Subject: {row['Ticket Subject']} | Description: {row['Ticket Description']}"
        target = f"Resolution: {row['Resolution']}"

        formatted_data.append({
            "instruction": instruction,
            "context": context,
            "target": target
        })

    output_df = pd.DataFrame(formatted_data)
    output_path = Path("processed_support_dataset.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(output_df)} formatted examples to {output_path}")

if __name__ == "__main__":
    build_training_dataset()