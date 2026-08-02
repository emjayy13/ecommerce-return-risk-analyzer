import os
import pandas as pd
import numpy as np


def clean_and_preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Loads raw synthetic e-commerce returns data, performs data cleaning,
    anomaly handling, and order-level feature engineering.

    Preserves ALL original columns so downstream code can use demographics,
    payment, and shipping features that were previously discarded.
    """
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Initial raw data shape: {df.shape}")

    df.columns = df.columns.str.strip()

    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Return_Date'] = pd.to_datetime(df['Return_Date'], errors='coerce')

    df['Is_Returned'] = (df['Return_Status'] == 'Returned').astype(int)

    df['Days_to_Return_Calc'] = (df['Return_Date'] - df['Order_Date']).dt.days

    invalid_date_mask = (df['Is_Returned'] == 1) & (df['Days_to_Return_Calc'] < 0)
    invalid_count = invalid_date_mask.sum()
    print(f"Removing {invalid_count} records with negative return durations...")
    df = df[~invalid_date_mask].copy()

    outlier_mask = (df['Is_Returned'] == 1) & (df['Days_to_Return_Calc'] > 90)
    outlier_count = outlier_mask.sum()
    print(f"Removing {outlier_count} return outlier records (> 90 days)...")
    df = df[~outlier_mask].copy()

    df['Return_Reason'] = df['Return_Reason'].fillna('Not Returned')
    df['Return_Date'] = df['Return_Date'].astype(object).fillna('N/A')
    df['Days_to_Return'] = df['Days_to_Return_Calc'].fillna(0)
    df.drop(columns=['Days_to_Return_Calc'], inplace=True)

    df['Order_Year'] = df['Order_Date'].dt.year
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_Day'] = df['Order_Date'].dt.day
    df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
    df['Is_Weekend'] = df['Order_DayOfWeek'].isin([5, 6]).astype(int)

    df['Total_Order_Value'] = (df['Product_Price'] * df['Order_Quantity']).round(2)
    df['Effective_Order_Value'] = (
        (df['Total_Order_Value'] - df['Discount_Applied']).clip(lower=0).round(2)
    )
    df['Discount_Percentage'] = np.where(
        df['Total_Order_Value'] > 0,
        (df['Discount_Applied'] / df['Total_Order_Value'] * 100).round(2),
        0.0,
    )

    df['Is_Clothing_Category'] = df['Product_Category'].isin(
        ['Clothing', 'Shoes', 'Fashion']
    ).astype(int)

    df['High_Discount_Flag'] = (df['Discount_Percentage'] > 30).astype(int)

    df['Is_Express_Shipping'] = df['Shipping_Method'].isin(
        ['Next-Day', 'Express']
    ).astype(int)

    df['Order_Date'] = df['Order_Date'].dt.strftime('%Y-%m-%d')

    cleaned_count = len(df)
    retention_rate = (cleaned_count / initial_count) * 100
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Data retention rate: {retention_rate:.2f}% ({cleaned_count}/{initial_count} rows)")
    print(f"Return distribution:\n{df['Is_Returned'].value_counts()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_synthetic_data.csv')
    clean_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_clean.csv')

    clean_and_preprocess_data(raw_csv, clean_csv)
