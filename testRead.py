import os
import pandas as pd

# Relative path from the Python script to the CSV file
relative_path_to_csv = "output_video1.mp4/flow_vectors.csv"

# Construct the absolute path to the CSV file
script_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_directory, relative_path_to_csv)

# Read the CSV file using pandas
try:
    df = pd.read_csv(csv_file_path)
    # Print the first row
    print("First row of the CSV file:")
    print(df.iloc[0])
except FileNotFoundError:
    print("File not found:", csv_file_path)
except pd.errors.EmptyDataError:
    print("CSV file is empty:", csv_file_path)