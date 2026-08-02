import os
import pandas as pd
import numpy as np

VAGUE_REASONS = ['Defective', 'Not as described']
MISMATCH_REASONS = ['Wrong item']
RISKY_CATEGORIES = ['Fashion', 'Shoes', 'Clothing', 'Electronics']


def build_customer_features(sim_data_path: str, output_path: str) -> pd.DataFrame:
    """
    Aggregates order-level records into a customer-level feature dataset.
    Includes historical behavioral attributes and an independent ground-truth target variable `flagged_by_company`.
    
    Target variable (`flagged_by_company` / `is_fraud`) is an independent ground-truth audit label
    and is strictly NOT derived from input features like return_ratio.
    """
    df = pd.read_csv(sim_data_path)
    print(f"Loaded simulated order-level data: {df.shape}")

    # Base aggregation per customer
    grouped = df.groupby('Customer_ID').agg(
        total_orders=('Order_ID', 'count'),
        total_returns=('Is_Returned', 'sum'),
        avg_return_window=('Days_to_Return', lambda x: float(x[x > 0].mean()) if (x > 0).any() else 0.0),
        most_common_category=('Product_Category', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        account_age_days=('account_age_days', 'first'),
        customer_rating_behavior=('customer_rating_behavior', 'first'),
        previous_fraud_flags=('previous_fraud_flags', 'first'),
        flagged_by_company=('flagged_by_company', 'first')
    ).reset_index()

    grouped['return_ratio'] = (grouped['total_returns'] / grouped['total_orders']).round(4)
    grouped['avg_return_window'] = grouped['avg_return_window'].fillna(0.0).round(2)

    # Calculate product category risk: proportion of customer's orders in high-risk categories
    risky_order_counts = df[df['Product_Category'].isin(RISKY_CATEGORIES)].groupby('Customer_ID').size()
    grouped['risky_category_orders'] = grouped['Customer_ID'].map(risky_order_counts).fillna(0).astype(int)
    grouped['product_category_risk'] = (grouped['risky_category_orders'] / grouped['total_orders']).round(4)
    grouped.drop(columns=['risky_category_orders'], inplace=True)

    # Vague reason count per customer
    vague_count = df[df['Return_Reason'].isin(VAGUE_REASONS)].groupby('Customer_ID').size()
    grouped['vague_reason_count'] = grouped['Customer_ID'].map(vague_count).fillna(0).astype(int)

    # Mismatch flag history per customer (e.g. 'Wrong item')
    mismatch_flag = df[df['Return_Reason'].isin(MISMATCH_REASONS)].groupby('Customer_ID').size() > 0
    grouped['mismatch_flag_history'] = grouped['Customer_ID'].map(mismatch_flag).fillna(False).astype(int)

    # Alias target label as is_fraud for clarity
    grouped['is_fraud'] = grouped['flagged_by_company'].astype(int)

    print(f"Customer-level feature table shape: {grouped.shape}")
    print(f"Independent target `flagged_by_company` value counts:\n{grouped['flagged_by_company'].value_counts()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False)
    print(f"Saved customer features to: {output_path}")

    return grouped


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_simulated_customers.csv')
    out_csv = os.path.join(base_dir, 'data', 'customer_features.csv')

    build_customer_features(sim_csv, out_csv)
