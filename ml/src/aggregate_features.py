import os
import pandas as pd

VAGUE_REASONS = ['Defective', 'Not as described']
MISMATCH_REASONS = ['Wrong item']
RISKY_CATEGORIES = ['Fashion', 'Shoes', 'Clothing']


def build_customer_features(sim_data_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(sim_data_path)
    print(f"Loaded simulated order-level data: {df.shape}")

    grouped = df.groupby('Customer_ID').agg(
        total_orders=('Order_ID', 'count'),
        total_returns=('Is_Returned', 'sum'),
        avg_return_window=('Days_to_Return', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        most_common_category=('Product_Category', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
    ).reset_index()

    grouped['return_ratio'] = (grouped['total_returns'] / grouped['total_orders']).round(3)
    grouped['avg_return_window'] = grouped['avg_return_window'].fillna(0).round(1)

    vague_count = df[df['Return_Reason'].isin(VAGUE_REASONS)].groupby('Customer_ID').size()
    grouped['vague_reason_count'] = grouped['Customer_ID'].map(vague_count).fillna(0).astype(int)

    mismatch_flag = df[df['Return_Reason'].isin(MISMATCH_REASONS)].groupby('Customer_ID').size() > 0
    grouped['mismatch_flag_history'] = grouped['Customer_ID'].map(mismatch_flag).fillna(False)

    print(f"Customer-level feature table shape: {grouped.shape}  <-- must be ~1385 rows, NOT 5542")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    return grouped


def calculate_rule_based_score(row) -> int:
    score = 0
    if row['return_ratio'] > 0.3:
        score += 30
    elif row['return_ratio'] > 0.15:
        score += 15
    if 0 < row['avg_return_window'] < 2:
        score += 15
    if row['most_common_category'] in RISKY_CATEGORIES:
        score += 20
    if row['vague_reason_count'] >= 2:
        score += 25
    elif row['vague_reason_count'] == 1:
        score += 10
    if row['mismatch_flag_history']:
        score += 15
    return min(score, 100)


def bucket(score):
    if score <= 40:
        return 'Low'
    elif score <= 70:
        return 'Medium'
    return 'High'


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_simulated_customers.csv')
    out_csv = os.path.join(base_dir, 'data', 'customer_features.csv')

    customer_df = build_customer_features(sim_csv, out_csv)
    customer_df['risk_score_label'] = customer_df.apply(calculate_rule_based_score, axis=1)
    customer_df['risk_level_label'] = customer_df['risk_score_label'].apply(bucket)
    customer_df.to_csv(out_csv, index=False)

    print("\nRisk label distribution:")
    print(customer_df['risk_level_label'].value_counts())
