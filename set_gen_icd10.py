import pandas as pd
import numpy as np
import random
import csv
import os

# Load the data from a CSV file
df = pd.read_csv('diagnosis_icd10.csv')

def original_encode(icd_code):
    return hash(icd_code) % (10**9)

def fixed_length_encode(icd_code):
    base_value = 0
    for char in icd_code:
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char.upper()) - ord('A') + 1
        else:
            value = 0
        base_value = (base_value * 36 + value) % (10**12)
    return base_value

# Apply both encoding functions
df['encoded_icd_codes'] = df['icd_code'].apply(original_encode)
df['fixed_length_encoded_icd_codes'] = df['icd_code'].apply(fixed_length_encode)

# Create a mapping from fixed length codes to unique numbers
unique_codes = pd.Series(df['fixed_length_encoded_icd_codes'].unique()).reset_index().rename(columns={0: 'fixed_length_encoded_icd_codes', 'index': 'unique_number'})
df = df.merge(unique_codes, on='fixed_length_encoded_icd_codes')
df['unique_number'] += 1  # Optional: start numbering from 1

# Calculate the counts for each ICD code
code_counts = df['icd_code'].value_counts()

# Create dataframes for server and client
server_df = pd.DataFrame()
client_df = pd.DataFrame()

np.random.seed(42)

for code, count in code_counts.items():
    code_df = df[df['icd_code'] == code].sample(frac=1)
    server_count = int(np.ceil(0.75 * count))
    client_count = count - server_count
    server_df = pd.concat([server_df, code_df[:server_count]])
    client_df = pd.concat([client_df, code_df[server_count:]])

server_df = server_df.sample(frac=1, random_state=42).reset_index(drop=True)
client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)

intersection_df = pd.merge(server_df, client_df, on='unique_number')
intersection_df.to_csv('intersection_dataset.csv', index=False)
client_df.to_csv('client_dataset.csv', index=False)
server_df.to_csv('server_dataset.csv', index=False)

print(f"Total records: {df.shape[0]}")
print(f"Server dataset records: {server_df.shape[0]}")
print(f"Client dataset records: {client_df.shape[0]}")
print("Intersection records saved: ", intersection_df.shape[0])

print("Server data has been successfully split and saved.")

