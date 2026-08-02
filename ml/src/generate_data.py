import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(output_path: str, n_orders: int = 5400, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic e-commerce returns dataset with realistic distributions.
    This is the raw input for the ML pipeline preprocessing step.
    """
    np.random.seed(random_seed)

    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Shoes', 'Fashion',
                   'Books', 'Sports', 'Toys', 'Beauty', 'Grocery']
    category_weights = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]

    shipping_methods = ['Standard', 'Express', 'Next-Day', 'Free Shipping']
    shipping_weights = [0.45, 0.20, 0.10, 0.25]

    return_reasons = ['Changed mind', 'Defective', 'Not as described', 'Wrong item',
                      'Better price found', 'No longer needed', 'Damaged in transit']
    return_reason_weights = [0.30, 0.18, 0.15, 0.08, 0.10, 0.12, 0.07]

    base_date = datetime(2024, 1, 1)
    order_dates = [base_date + timedelta(days=int(d)) for d in np.random.uniform(0, 540, n_orders)]
    order_dates.sort()

    categories_sampled = np.random.choice(categories, size=n_orders, p=category_weights)
    shipping_sampled = np.random.choice(shipping_methods, size=n_orders, p=shipping_weights)

    base_prices = {
        'Electronics': (25, 500), 'Clothing': (15, 120), 'Home & Kitchen': (10, 200),
        'Shoes': (20, 180), 'Fashion': (15, 150), 'Books': (5, 40),
        'Sports': (10, 150), 'Toys': (8, 80), 'Beauty': (5, 60), 'Grocery': (3, 30)
    }
    product_prices = [
        round(np.random.uniform(*base_prices[cat]), 2) for cat in categories_sampled
    ]
    order_quantities = np.random.choice([1, 1, 1, 2, 2, 3], size=n_orders)

    overall_return_rate = 0.22
    category_return_modifiers = {
        'Electronics': 1.2, 'Clothing': 1.5, 'Shoes': 1.4, 'Fashion': 1.3,
        'Home & Kitchen': 0.9, 'Books': 0.6, 'Sports': 0.8, 'Toys': 0.7,
        'Beauty': 0.9, 'Grocery': 0.5
    }
    return_probs = [overall_return_rate * category_return_modifiers.get(c, 1.0) for c in categories_sampled]
    return_probs = [min(p, 0.55) for p in return_probs]
    is_returned = np.random.binomial(1, return_probs)

    return_dates = []
    return_reasons_sampled = []
    for i in range(n_orders):
        if is_returned[i]:
            days_after = int(np.random.exponential(scale=12)) + 1
            days_after = min(days_after, 85)
            return_dates.append(order_dates[i] + timedelta(days=days_after))
            return_reasons_sampled.append(
                np.random.choice(return_reasons, p=return_reason_weights)
            )
        else:
            return_dates.append(None)
            return_reasons_sampled.append('Not Returned')

    return_statuses = ['Returned' if r == 1 else 'Not Returned' for r in is_returned]

    max_discount_rate = {
        'Electronics': 0.15, 'Clothing': 0.30, 'Shoes': 0.25, 'Fashion': 0.25,
        'Home & Kitchen': 0.20, 'Books': 0.10, 'Sports': 0.15, 'Toys': 0.20,
        'Beauty': 0.15, 'Grocery': 0.05
    }
    discount_applied = [
        round(np.random.uniform(0, max_discount_rate.get(c, 0.2)) * p * q, 2)
        for c, p, q in zip(categories_sampled, product_prices, order_quantities)
    ]

    order_ids = [f'ORD{str(i).zfill(8)}' for i in range(1, n_orders + 1)]

    df = pd.DataFrame({
        'Order_ID': order_ids,
        'Order_Date': [d.strftime('%Y-%m-%d') for d in order_dates],
        'Return_Date': [d.strftime('%Y-%m-%d') if d else 'N/A' for d in return_dates],
        'Return_Status': return_statuses,
        'Product_Category': categories_sampled,
        'Product_Price': product_prices,
        'Order_Quantity': order_quantities,
        'Discount_Applied': discount_applied,
        'Shipping_Method': shipping_sampled,
        'Return_Reason': return_reasons_sampled
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset: {df.shape[0]} orders, {df.shape[1]} columns")
    print(f"Saved to: {output_path}")
    print(f"Return rate: {is_returned.mean():.2%}")
    print(f"Category distribution:\n{df['Product_Category'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_synthetic_data.csv')
    generate_synthetic_data(raw_csv)
