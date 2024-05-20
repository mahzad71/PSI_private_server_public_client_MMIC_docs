import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('diagnosis.csv')

# Filter out ICD-9 and ICD-10 codes into separate DataFrames
df_icd9 = df[df['icd_version'] == 9]
df_icd10 = df[df['icd_version'] == 10]

# Save the filtered DataFrames to new CSV files
df_icd9.to_csv('diagnosis_icd9.csv', index=False)
df_icd10.to_csv('diagnosis_icd10.csv', index=False)

print("Files have been saved successfully.")
