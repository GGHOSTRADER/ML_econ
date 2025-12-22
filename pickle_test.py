import pandas as pd

# -------------------------------------

# LOAD PICKLES TO CHECK USING ipython interactive enviroment

# -------------------------------------


file_path = "transformer_rv15_predictions_new_data.pkl"  # <--- Path to your pickle file


try:
    df = pd.read_pickle(file_path)

    print("\n=== DataFrame Loaded Successfully ===")
    print(df.info())  # Structure, columns, dtypes
    print("\n=== First 5 Rows ===")
    print(df.head())  # Quick peek at data
    print("\n=== Last 5 Rows ===")
    print(df.tail())  # Quick peek at data
    print("\n=== Column List ===")
    print(df.columns.tolist())
    print("\n=== Basic Stats (numerical columns) ===")
    print(df.describe().T)

    for col in df.columns:
        print(f"\n=== Stats for column: {col} ===")
        try:
            print(f" skew : {df[col].skew()}")
            print(f" kurtosis : {df[col].kurtosis()}")
        except Exception as e:
            print(f" Could not compute stats for column {col}: {e}")

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print("Error loading DataFrame:", e)


# =========================
# CSV
csv_1 = pd.read_csv("data_ml_final.txt")
print(csv_1.shape)
print(csv_1.iloc[:2])
print(csv_1.iloc[-2:])
