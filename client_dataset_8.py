import csv
import os

# Define the input file and output base settings
input_file_name = 'client_dataset.csv'
output_file_base_name = 'client_file_'
summary_file_name = 'client_dataset_NC.csv'

# Determine number of records per file
records_per_file = 3
extra_records_in_last_file = 4  # The last file will have 4 records

try:
    with open(input_file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Capture the header row, now includes the unique number
        data = list(reader)  # Read the rest of the data

    # Calculate total records needed for each file
    total_records = len(data)
    files_needed = (total_records + records_per_file - 1) // records_per_file  # Calculate how many files are needed

    # Create the output files
    start_index = 0
    for i in range(files_needed):
        # Determine the number of records for the current file
        current_records_per_file = extra_records_in_last_file if i == files_needed - 1 else records_per_file
        end_index = start_index + current_records_per_file

        # Create and write to each output file
        output_file_name = f'{output_file_base_name}{i+1}.csv'
        with open(output_file_name, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)  # Write the header, including the unique number
            writer.writerows(data[start_index:end_index])  # Write the data slice

        # Update start_index for the next file
        start_index = end_index

    # After all files have been created, read them and create a summary file
    with open(summary_file_name, 'w', newline='') as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(['file_number', 'encoded_icd_codes', 'icd_code', 'unique_number'])  # Adjusted header to include ICD code and unique numbers

        # Read each created file and extract necessary data
        for j in range(files_needed):
            part_file_name = f'{output_file_base_name}{j+1}.csv'
            with open(part_file_name, 'r', newline='') as part_file:
                reader = csv.reader(part_file)
                next(reader)  # Skip the header
                for row in reader:
                    # Assuming unique number is the last column, and ICD code is the fourth column
                    summary_writer.writerow([j + 1, row[-3], row[3], row[-1]])  # Adjust indices according to actual data layout

    print("Data has been successfully split into {} files and summarized in the summary file.".format(files_needed))

except FileNotFoundError:
    print(f"Error: The file '{input_file_name}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
