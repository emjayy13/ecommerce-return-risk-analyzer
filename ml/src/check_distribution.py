import pandas as pd

df = pd.read_csv('../data/customer_features.csv')

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
