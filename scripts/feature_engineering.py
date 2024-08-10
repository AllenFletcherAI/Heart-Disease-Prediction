import pandas as pd
from pathlib import Path
import json
import logging
import argparse

def configure_logging(log_dir: Path) -> None:
    """
    Configure logging settings.

    Parameters:
    log_dir (Path): The directory where log files will be stored.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)  # Create the log directory if it doesn't exist
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'feature_engineering.log'),
                logging.StreamHandler()  # Log to console as well
            ]
        )
        logging.info("Logging configured.")
    except Exception as e:
        print(f"Error configuring logging: {e}")
        raise

def load_config(config_path: Path) -> dict:
    """
    Load configuration settings from a JSON file.

    Parameters:
    config_path (Path): The path to the config JSON file.

    Returns:
    dict: The loaded configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (Path): The path to the CSV file.

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

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with engineered features.
    """
    # Example Feature Engineering:
    # 1. Creating binary columns for categorical features if not already present
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(int)
    
    if 'chest pain type' in df.columns:
        df = pd.get_dummies(df, columns=['chest pain type'], prefix='chest pain type')
    
    if 'fasting blood sugar' in df.columns:
        df['fasting blood sugar'] = df['fasting blood sugar'].astype(int)
    
    if 'resting ecg' in df.columns:
        df = pd.get_dummies(df, columns=['resting ecg'], prefix='resting ecg')
    
    if 'exercise angina' in df.columns:
        df['exercise angina'] = df['exercise angina'].astype(int)
    
    if 'ST slope' in df.columns:
        df = pd.get_dummies(df, columns=['ST slope'], prefix='ST slope')

    # Feature engineering for numerical features:
    # 1. Normalization or standardization of numeric features
    numerical_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

    # Drop the original categorical columns if needed
    columns_to_drop = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Add more feature engineering steps as required

    logging.info("Feature engineering completed.")
    return df

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save the processed dataset to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (Path): The path to the output CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform feature engineering on heart disease dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the processed CSV file.")
    parser.add_argument('--config', type=str, default='config/config.json', help="Path to the config JSON file.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to store log files.")
    args = parser.parse_args()

    # Configure logging
    configure_logging(Path(args.log_dir))

    # Load configuration
    config = load_config(Path(args.config))

    # Load the data
    df = load_data(Path(args.input))

    # Perform feature engineering
    df_processed = feature_engineering(df)

    # Save the processed data
    save_data(df_processed, Path(args.output))

