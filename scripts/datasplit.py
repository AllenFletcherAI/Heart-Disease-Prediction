import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def load_config(config_path: Path) -> dict:
    """
    Load configuration from a JSON file.

    Parameters:
    config_path (Path): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def validate_config(config: dict) -> None:
    """
    Validate the configuration dictionary.

    Parameters:
    config (dict): Configuration dictionary.

    Raises:
    ValueError: If required configuration keys are missing.
    """
    required_keys = ['datasplit']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

def configure_logging(log_dir: Path, log_level: str) -> None:
    """
    Configure logging settings.

    Parameters:
    log_dir (Path): Directory to store log files.
    log_level (str): Logging level (e.g., 'INFO', 'DEBUG').
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_split.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging configured.")
    except Exception as e:
        sys.stderr.write(f"Error configuring logging: {e}\n")
        raise

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    input_path (Path): Path to the input CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded from {input_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Split data into training and testing sets.

    Parameters:
    df (pd.DataFrame): Data to split.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed.

    Returns:
    tuple: Split data (X_train, X_test, y_train, y_test).
    """
    try:
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(X_train, X_test, y_train, y_test, output_dir: Path) -> None:
    """
    Save split data to disk.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Testing labels.
    output_dir (Path): Directory to save the split datasets.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(output_dir / 'X_train.csv', index=False)
        X_test.to_csv(output_dir / 'X_test.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
        y_test.to_csv(output_dir / 'y_test.csv', index=False)
        logging.info(f"Split data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error saving split data: {e}")
        raise

def main() -> None:
    """
    Main function to execute the data splitting process.

    Parses command-line arguments, configures logging, loads configuration and data,
    splits the data into training and testing sets, and saves the split data to disk.
    """
    parser = argparse.ArgumentParser(description="Split feature-engineered data into training and testing sets.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split datasets.")
    parser.add_argument('--config', type=str, default='config/config.json', help="Path to the config JSON file.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to store log files.")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(Path(args.config))

        # Validate configuration
        validate_config(config)

        # Configure logging
        configure_logging(Path(args.log_dir), config.get('log_level', 'INFO'))

        # Debugging: Log that logging configuration is complete
        logging.debug("Logging configuration complete.")

        # Load the data
        df = load_data(Path(args.input))

        # Split the data
        datasplit_config = config.get('datasplit', {})
        test_size = datasplit_config.get('test_size')
        random_state = datasplit_config.get('random_state')
        
        # Debugging: Log the retrieved configuration values
        logging.debug(f"Retrieved configuration values for data split: test_size={test_size}, random_state={random_state}")

        X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

        # Save the split data
        save_data(X_train, X_test, y_train, y_test, Path(args.output_dir))

        logging.info("Data splitting process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}")
        sys.exit(1)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()

