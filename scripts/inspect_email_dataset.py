import pandas as pd

path = "data/email_dataset/PhishingEmailData.csv"

for encoding in ["cp1252", "latin1"]:
    try:
        df = pd.read_csv(path, encoding=encoding, on_bad_lines="warn")
        print("Encoding:", encoding)
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        print("\nMissing values:")
        print(df.isnull().sum().to_string())
        break
    except Exception as e:
        print(f"{encoding} failed: {e}")
