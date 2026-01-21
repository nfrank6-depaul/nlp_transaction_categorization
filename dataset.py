import pandas as pd

# load dataset as a pandas DataFrame
df = pd.read_csv("dataset.csv")
# print(df.head())
print()
print(df.describe(include="all"))

# remove memo from dataset
transactions_df = df.drop(columns=["Memo"])

# find any columns with missing values
missing_values = transactions_df.isnull().sum()
print()
print("Missing values per column:")
print(missing_values)

# show the category of rows with missing values
rows_with_missing_values = transactions_df[transactions_df.isnull().any(axis=1)]
print()
print("Rows with missing values:")
print(rows_with_missing_values)

# fill missing values in the Category column with "Credit Card Payment"
transactions_df["Category"] = transactions_df["Category"].fillna("Credit Card Payment")
print()
print("Dataset after filling missing Category values:")
print(transactions_df.describe(include="all"))

# show the category of rows with missing values
rows_with_missing_values = transactions_df[transactions_df.isnull().any(axis=1)]
print()
print("Rows with missing values:")
if rows_with_missing_values.shape[0] > 0:
    print(rows_with_missing_values)
else:
    print("No rows with missing values.")

# # remove rows with missing values
# transactions_df = transactions_df.dropna()
# print()
# print("Dataset after removing rows with missing values:")
# print(transactions_df.describe(include="all"))

# # print how many rows were removed
# rows_removed = df.shape[0] - transactions_df.shape[0]
# print()
# print(f"Rows removed: {rows_removed}")

