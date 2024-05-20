import pandas as pd

# Load the data from CSV file
df = pd.read_csv('diagnosis_icd10.csv')

# Check the column names to ensure they are correct
print(df.columns)

# Count the occurrences of each unique ICD code
icd_counts = df['icd_code'].value_counts()

# Print the counts
print(icd_counts)
