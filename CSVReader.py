import os
import time
import pandas as pd

# Define constants
WINDOW_SIZE = 10  # Size of the sliding window
OUTPUT_FOLDER = "output_webcam"  # Folder where CSV file is expected

# Function to read sliding window of flow vectors from CSV and print them
def read_and_print_csv_values(csv_file, window_size):
    df = pd.read_csv(csv_file)
    num_rows = len(df)
    if num_rows < window_size:
        print("CSV file does not contain enough rows for the sliding window.")
        return
    print("Sliding window of CSV values:")
    for i in range(num_rows - window_size + 1):
        window_data = df.iloc[i:i+window_size]
        print(window_data)
        print()  # Add a newline for readability

# Main function
def main():
    while True:
        csv_file = find_csv_file()
        if csv_file:
            try:
                read_and_print_csv_values(csv_file, WINDOW_SIZE)
                break
            except pd.errors.EmptyDataError:
                print("CSV file is empty.")
        else:
            print("Waiting for the output_webcam folder...")
            time.sleep(1)  # Wait for 1 second before retrying

# Function to find the CSV file within the output_webcam folder
def find_csv_file():
    if not os.path.exists(OUTPUT_FOLDER):
        return None
    for file in os.listdir(OUTPUT_FOLDER):
        if file.endswith(".csv"):
            return os.path.join(OUTPUT_FOLDER, file)
    return None

# Entry point of the program
if __name__ == "__main__":
    main()