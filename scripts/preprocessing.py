import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import logging
from typing import Union, List
import argparse
import json

def configure_logging(log_dir: Union[str, Path], log_level: str = 'INFO') -> None:
    """
    Configure logging settings.

    Parameters:
    log_dir (Union[str, Path]): The directory where log files will be stored.
    log_level (str): The logging level (e.g., 'INFO', 'DEBUG').
    """
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'preprocessing.log'),
                logging.StreamHandler()  # Log to console as well
            ]
        )
        logging.info("Logging configured.")
    except Exception as e:
        print(f"Error configuring logging: {e}")
        raise

def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (Union[str, Path]): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"File is empty: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate the input data to ensure required columns are present.

    Parameters:
    df (pd.DataFrame): The input dataset.
    required_columns (List[str]): List of required column names.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_message = f"Missing required columns: {missing_columns}"
        logging.error(error_message)
        raise ValueError(error_message)
    logging.info("Input data validation passed.")

def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load the configuration from a JSON file.

    Parameters:
    config_path (Union[str, Path]): Path to the JSON configuration file.

    Returns:
    dict: The configuration as a dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocess the dataset based on the configuration.

    Parameters:
    df (pd.DataFrame): The input dataset.
    config (dict): The configuration dictionary.

    Returns:
    pd.DataFrame: The preprocessed dataset.
    """
    numeric_features = config['feature_engineering']['scaling']['features']
    categorical_features = config['feature_engineering']['encoding']['categorical_features']

    logging.info("Starting data preprocessing...")

    # Preprocessing pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Normalize numerical features
    ])
    logging.info("Numeric transformer pipeline created.")

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('onehot', OneHotEncoder(sparse_output=False))  # One-hot encode without dropping any category
    ])
    logging.info("Categorical transformer pipeline created.")

    # Combine both transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    logging.info("Column transformer created.")

    # Apply transformations
    processed_data = preprocessor.fit_transform(df.drop(columns='target'))
    logging.info("Transformations applied to data.")

    # Convert the processed data back to a DataFrame
    processed_columns = (
        numeric_features +
        list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
    )
    processed_df = pd.DataFrame(processed_data, columns=processed_columns)
    logging.info("Processed data converted back to DataFrame.")

    # Add the target column back to the DataFrame
    processed_df['target'] = df['target'].values
    logging.info("Target column added back to DataFrame.")

    logging.info("Data preprocessing completed.")
    return processed_df

def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (Union[str, Path]): The path to the output CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess heart disease dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to store log files.")
    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_dir)

    # Load configuration
    config = load_config(args.config)

    # Define file paths
    input_file_path = Path(args.input)
    output_file_path = Path(args.output)

    # Load, validate, preprocess, and save data
    df = load_data(input_file_path)
    validate_data(df, config['eda']['numeric_features'] + config['eda']['categorical_features'] + ['target'])
    processed_df = preprocess_data(df, config)
    save_data(processed_df, output_file_path)

