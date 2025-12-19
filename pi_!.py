import pandas as pd

# Path to your pickle file
file_path = "new_data_full_df_processed.pkl"

try:
    df = pd.read_pickle(file_path)

    print("\n=== DataFrame Loaded Successfully ===")
    print(df.info())  # Structure, columns, dtypes
    print("\n=== First 5 Rows ===")
    print(df.head())  # Quick peek at data
    print("\n=== Column List ===")
    print(df.columns.tolist())
    print("\n=== Basic Stats (numerical columns) ===")
    print(df.describe().T)

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print("Error loading DataFrame:", e)
