import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from joblib import dump
from pathlib import Path
import logging
import sys
import argparse
from typing import Tuple

def configure_logging(log_dir: Path, log_level: str = 'INFO') -> None:
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
                logging.FileHandler(log_dir / 'model_training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging configured.")
    except Exception as e:
        sys.stderr.write(f"Error configuring logging: {e}\n")
        raise

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load training and testing data from CSV files.

    Parameters:
    data_dir (Path): Directory containing the data files.

    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()  # Convert to Series
        y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()  # Convert to Series
        logging.info(f"Data loaded successfully. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Validate the loaded data.

    Parameters:
    X_train, X_test (pd.DataFrame): Training and testing features.
    y_train, y_test (pd.Series): Training and testing labels.
    """
    try:
        assert not X_train.empty and not X_test.empty, "Feature data is empty"
        assert not y_train.empty and not y_test.empty, "Label data is empty"
        assert X_train.shape[0] == y_train.shape[0], "Mismatch in training data size"
        assert X_test.shape[0] == y_test.shape[0], "Mismatch in testing data size"
        logging.info("Data validation successful.")
    except AssertionError as e:
        logging.error(f"Data validation error: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, max_iter: int = 1000) -> LogisticRegression:
    """
    Train a Logistic Regression model using Grid Search for hyperparameter tuning.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    max_iter (int): Maximum number of iterations for the solver.

    Returns:
    LogisticRegression: Trained model.
    """
    try:
        # Define the parameter grid for Grid Search
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'solver': ['liblinear', 'saga'],  # Optimization algorithm
        }
        
        # Initialize Logistic Regression
        log_reg = LogisticRegression(max_iter=max_iter)
        
        # Initialize Grid Search
        grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, 
                                   scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
        
        # Fit Grid Search to the data
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info("Model trained successfully.")
        return best_model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the trained model and print performance metrics.

    Parameters:
    model (LogisticRegression): Trained model.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing labels.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{report}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_model(model: LogisticRegression, model_dir: Path) -> None:
    """
    Save the trained model to a file.

    Parameters:
    model (LogisticRegression): Trained model.
    model_dir (Path): Directory to save the model.
    """
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        dump(model, model_dir / 'logistic_regression_model.joblib')
        logging.info(f"Model saved to {model_dir / 'logistic_regression_model.joblib'}")
        print(f"Model saved to {model_dir / 'logistic_regression_model.joblib'}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    """
    Main function to execute the model training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a logistic regression model.")
    parser.add_argument('--data_dir', type=str, default='data/split_data', help='Directory containing the data files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for the solver')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)

    configure_logging(log_dir, args.log_level)

    try:
        X_train, X_test, y_train, y_test = load_data(data_dir)
        validate_data(X_train, X_test, y_train, y_test)
        model = train_model(X_train, y_train, args.max_iter)
        evaluate_model(model, X_test, y_test)
        save_model(model, model_dir)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()

