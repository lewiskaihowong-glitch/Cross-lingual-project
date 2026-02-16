import pandas as pd
import numpy as np

def take_first_n_rows(data, n):
    return data.head(n)

def take_random_n_rows(data, n):
    return data.sample(n=n, random_state=42)

def truncate_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    print(f"Loaded data with {len(data)} rows")
    print(data.head())

    print(data[data["status"] == "Green"].shape[0])
    groups = data.index // 25

    new_data = data.groupby(groups).apply(lambda x: take_first_n_rows(x, 14)).reset_index(drop=True)
    remaining_data = data.groupby(groups).tail(-2)
    additional_data = remaining_data.apply(lambda x: take_random_n_rows(x, 2))
    final_data = pd.concat([new_data, additional_data], ignore_index=True)
    print(f"New dataset has {len(final_data)} rows")
    final_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = "data/final/final_dataset_Korean.csv"
    output_path = "data/final/final_dataset_Korean_100.csv"
    truncate_dataset(input_path, output_path)