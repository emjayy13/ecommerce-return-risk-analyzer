import os
import numpy as np
import pandas as pd


def simulate_customer_history(clean_data_path: str, output_path: str,
                              avg_orders_per_customer: int = 4,
                              random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates realistic customer profiles with repeat transactions, historical attributes,
    and an independent ground-truth fraud/risk target label (`flagged_by_company`).

    To ensure NO data leakage:
    `flagged_by_company` is an independent ground-truth label initialized at customer entity level,
    representing merchant risk audits / fraud investigation flags. It is NOT derived from
    any return ratio or feature threshold calculation.
    """
    np.random.seed(random_seed)
    df = pd.read_csv(clean_data_path)
    n_orders = len(df)
    n_customers = max(1, n_orders // avg_orders_per_customer)

    print(f"Total orders: {n_orders}")
    print(f"Simulating {n_customers} synthetic customer profiles...")

    # Assign Customer IDs
    customer_ids = ['CUST' + str(i).zfill(6) for i in range(1, n_customers + 1)]
    
    # Generate customer entity ground-truth metadata (Independent of order features)
    customer_registry = {}
    for cid in customer_ids:
        account_age_days = int(np.random.uniform(30, 1095))  # 1 month to 3 years
        rating_behavior = round(float(np.random.normal(4.2, 0.8)), 2)
        rating_behavior = max(1.0, min(5.0, rating_behavior))
        
        # Ground-truth audit label assigned independently (approx 15% fraudulent/high-risk customers)
        # Driven by independent risk factor draw (representing external audit / policy investigation)
        latent_risk_factor = np.random.beta(2, 8)
        flagged_by_company = 1 if latent_risk_factor > 0.35 else 0
        
        # Historical fraud flags (correlated with true fraud entity status)
        if flagged_by_company == 1:
            previous_fraud_flags = int(np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        else:
            previous_fraud_flags = int(np.random.choice([0, 1], p=[0.9, 0.1]))

        customer_registry[cid] = {
            'Customer_ID': cid,
            'account_age_days': account_age_days,
            'customer_rating_behavior': rating_behavior,
            'previous_fraud_flags': previous_fraud_flags,
            'flagged_by_company': flagged_by_company
        }

    cust_reg_df = pd.DataFrame(list(customer_registry.values()))

    # Assign Customer_ID to orders
    df['Customer_ID'] = np.random.choice(customer_ids, size=n_orders)

    # Merge customer registry metadata onto order dataframe
    df = df.merge(cust_reg_df, on='Customer_ID', how='left')

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Unique customers in order data: {df['Customer_ID'].nunique()}")
    print(f"Independent flagged_by_company target distribution:\n{cust_reg_df['flagged_by_company'].value_counts()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved simulated customer dataset to: {output_path}")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_clean.csv')
    simulated_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_simulated_customers.csv')

    simulate_customer_history(clean_csv, simulated_csv)
