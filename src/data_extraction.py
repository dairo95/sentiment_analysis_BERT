import pandas as pd


def load_data(file_path: str):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
  
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        raise


df = load_data("data/dataset.csv")
print(df.head())
