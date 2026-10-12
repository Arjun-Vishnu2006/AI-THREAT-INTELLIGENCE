import pandas as pd

path = "data/email_dataset/PhishingEmailData.csv"
df = pd.read_csv(path, encoding="cp1252")

for col in ["Coined.Word", "URL_Title", "Email_Subject", "Email_Content", "To"]:
    print("\n" + "=" * 60)
    print("COLUMN:", col)
    print("Unique values:", df[col].nunique(dropna=False))
    print(df[col].value_counts(dropna=False).head(15).to_string())

print("\nAll column names:")
for col in df.columns:
    print(repr(col))
