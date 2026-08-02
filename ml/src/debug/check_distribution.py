import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'customer_features.csv')

if not os.path.exists(data_path):
    print("Features file not found. Run: python ml/src/train.py")
    exit(1)

df = pd.read_csv(data_path)

print("return_ratio stats:")
print(df['return_ratio'].describe())
print("\nvague_reason_count stats:")
print(df['vague_reason_count'].describe())
print("\nvague_reason_count value counts:")
print(df['vague_reason_count'].value_counts())
print("\nmismatch_flag_history counts:")
print(df['mismatch_flag_history'].value_counts())
print("\nmost_common_category counts:")
print(df['most_common_category'].value_counts())
