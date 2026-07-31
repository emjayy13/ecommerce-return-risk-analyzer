import os
import numpy as np
import pandas as pd


def simulate_customer_history(clean_data_path: str, output_path: str,
                                avg_orders_per_customer: int = 4,
                                random_seed: int = 42) -> pd.DataFrame:
    """
    The raw dataset has one order per User_ID (no repeat customers),
    which makes customer-level history features meaningless.

    This function simulates realistic repeat-customer behavior by
    reassigning orders into synthetic 'Customer_ID' groups, so each
    fake customer has multiple orders (like a real customer would).

    NOTE: This is a synthetic simulation layer, done transparently
    because the source dataset lacked genuine repeat-purchase history.
    """
    np.random.seed(random_seed)
    df = pd.read_csv(clean_data_path)
    n_orders = len(df)
    n_customers = max(1, n_orders // avg_orders_per_customer)

    print(f"Total orders: {n_orders}")
    print(f"Simulating {n_customers} synthetic customers "
          f"(~{avg_orders_per_customer} orders each on average)")

    df['Customer_ID'] = np.random.randint(1, n_customers + 1, size=n_orders)
    df['Customer_ID'] = 'CUST' + df['Customer_ID'].astype(str).str.zfill(6)

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    order_counts = df['Customer_ID'].value_counts()
    print(f"Customers with >1 order: {(order_counts > 1).sum()}")
    print(f"Max orders for a single customer: {order_counts.max()}")
    print(f"Avg orders per customer: {order_counts.mean():.2f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved simulated-customer dataset to: {output_path}")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_clean.csv')
    simulated_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_simulated_customers.csv')

    simulate_customer_history(clean_csv, simulated_csv)
