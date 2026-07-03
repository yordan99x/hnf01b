from data_loader import load_data
from categorizer import categorize_dataframe

df = load_data()

df = categorize_dataframe(df)

print("="*60)
print(df.head())

print("\n")

print("="*60)
print("CALL CATEGORY")

print(df["Call Category"].value_counts().sort_index())

print("\n")

print("="*60)
print("PRICE CATEGORY")

print(df["Price Category"].value_counts().sort_index())