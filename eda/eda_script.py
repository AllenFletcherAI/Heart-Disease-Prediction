import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Union, List
import argparse
import shutil

def configure_logging(log_dir: Union[str, Path]) -> None:
    """
    Configure logging settings.

    Parameters:
    log_dir (Union[str, Path]): The directory where log files will be stored.
    """
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)  # Create the log directory if it doesn't exist
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'eda.log'),
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
        df = pd.read_csv(file_path, na_values=[''])  # Treat only blank entries as NaN
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

def setup_plot_directory(plot_dir: Union[str, Path]) -> None:
    """
    Delete the existing plot directory and recreate it.

    Parameters:
    plot_dir (Union[str, Path]): The directory to be deleted and recreated.
    """
    try:
        if Path(plot_dir).exists():
            shutil.rmtree(plot_dir)  # Delete the directory and its contents
        Path(plot_dir).mkdir(parents=True, exist_ok=True)  # Recreate the directory
        logging.info(f"Plot directory {plot_dir} is ready.")
    except Exception as e:
        logging.error(f"Error setting up plot directory: {e}")
        raise

def plot_histograms(df: pd.DataFrame, features: List[str], plot_dir: Path) -> None:
    """
    Plot histograms for numerical features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    features (List[str]): List of features to plot.
    plot_dir (Path): Directory to save the plots.
    """
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[feature].dropna(), kde=True)  # Drop NaN values for plotting
            plt.title(f'Histogram of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.savefig(plot_dir / f'histogram_{feature}.png')
            plt.close()
            logging.info(f"Histogram for {feature} saved.")
        else:
            logging.warning(f"Column {feature} not found in the DataFrame.")

def plot_countplots(df: pd.DataFrame, features: List[str], plot_dir: Path) -> None:
    """
    Plot countplots for categorical features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    features (List[str]): List of features to plot.
    plot_dir (Path): Directory to save the plots.
    """
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=feature, data=df)
            plt.title(f'Countplot of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.savefig(plot_dir / f'countplot_{feature}.png')
            plt.close()
            logging.info(f"Countplot for {feature} saved.")
        else:
            logging.warning(f"Column {feature} not found in the DataFrame.")

def plot_boxplots(df: pd.DataFrame, features: List[str], plot_dir: Path) -> None:
    """
    Plot boxplots for numerical features grouped by the target.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    features (List[str]): List of features to plot.
    plot_dir (Path): Directory to save the plots.
    """
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='target', y=feature, data=df)
            plt.title(f'Boxplot of {feature} by Target')
            plt.xlabel('Target')
            plt.ylabel(feature)
            plt.savefig(plot_dir / f'boxplot_{feature}_by_target.png')
            plt.close()
            logging.info(f"Boxplot for {feature} by target saved.")
        else:
            logging.warning(f"Column {feature} not found in the DataFrame.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on heart disease dataset.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to store log files.")
    parser.add_argument('--plot_dir', type=str, default='eda/plots', help="Directory to store plots.")
    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_dir)

    # Set up the plot directory
    setup_plot_directory(args.plot_dir)

    # Define file paths
    input_file_path = Path(args.input)
    plot_dir = Path(args.plot_dir)

    # Load the data
    df = load_data(input_file_path)

    # Define features for EDA
    numeric_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    categorical_features = ['sex_1', 'chest pain type_2', 'chest pain type_3', 'chest pain type_4',
                            'fasting blood sugar_1', 'resting ecg_1', 'resting ecg_2', 
                            'exercise angina_1', 'ST slope_2', 'ST slope_3']

    # Create plots
    plot_histograms(df, numeric_features, plot_dir)
    plot_countplots(df, categorical_features, plot_dir)
    plot_boxplots(df, numeric_features, plot_dir)

