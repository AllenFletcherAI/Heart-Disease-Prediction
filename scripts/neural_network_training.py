import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import logging
import sys
import argparse
import json
from typing import Tuple, Dict, Any
import keras_tuner as kt


def configure_logging(log_dir: Path, log_level: str = 'INFO') -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'neural_network_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging configured.")


def load_config(config_file: Path) -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        config = json.load(file)
    logging.info("Configuration loaded.")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ['preprocessing', 'feature_engineering', 'neural_network']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    nn_config = config['neural_network']
    if not isinstance(nn_config.get('layers'), list) or not nn_config['layers']:
        raise ValueError("Neural network configuration must include a non-empty 'layers' list.")
    if not isinstance(nn_config.get('optimizer'), str):
        raise ValueError("Neural network configuration must include an 'optimizer' string.")
    if not isinstance(nn_config.get('loss_function'), str):
        raise ValueError("Neural network configuration must include a 'loss_function' string.")
    if not isinstance(nn_config.get('metrics'), list) or not nn_config['metrics']:
        raise ValueError("Neural network configuration must include a non-empty 'metrics' list.")
    if not isinstance(nn_config.get('epochs'), int) or nn_config['epochs'] <= 0:
        raise ValueError("Neural network configuration must include a positive integer 'epochs'.")
    if not isinstance(nn_config.get('batch_size'), int) or nn_config['batch_size'] <= 0:
        raise ValueError("Neural network configuration must include a positive integer 'batch_size'.")


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
        y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()
        logging.info(f"Data loaded successfully. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    logging.getLogger().setLevel(logging.DEBUG)
    preprocessing_config = config['preprocessing']
    feature_engineering_config = config['feature_engineering']

    numeric_features = preprocessing_config['missing_values']['numeric_columns']
    categorical_features = preprocessing_config['missing_values']['categorical_columns']

    logging.debug(f"Columns in X_train: {list(X_train.columns)}")
    logging.debug(f"Columns in X_test: {list(X_test.columns)}")

    for col in numeric_features:
        if col in X_train.columns and col in X_test.columns:
            X_train[col] = X_train[col].fillna(X_train[col].mean())
            X_test[col] = X_test[col].fillna(X_test[col].mean())
        else:
            logging.warning(f"Column '{col}' is missing in the dataset.")

    for col in categorical_features:
        if col in X_train.columns and col in X_test.columns:
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
            X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
        else:
            logging.warning(f"Column '{col}' is missing in the dataset.")

    numeric_features = [col for col in numeric_features if col in X_train.columns]
    categorical_features = [col for col in categorical_features if col in X_train.columns]

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    try:
        X_train_transformed = column_transformer.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)
    except Exception as e:
        logging.error(f"Error during transformation: {e}")
        raise

    logging.getLogger().setLevel(logging.INFO)
    return X_train_transformed, X_test_transformed


def build_model(hp) -> tf.keras.Model:
    model = Sequential()
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                     min_value=64,
                                     max_value=512,
                                     step=64),
                        activation=hp.Choice('activation_' + str(i), ['relu', 'tanh'])))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), 0.2, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model


def tune_hyperparameters(X_train: np.ndarray, y_train: pd.Series, tuning_dir: str = 'my_dir') -> Dict[str, Any]:
    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=3,
                         directory=tuning_dir,
                         project_name='intro_to_kt')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: pd.Series, config: Dict[str, Any]) -> tf.keras.Model:
    nn_config = config['neural_network']
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    model.fit(
        X_train, y_train,
        epochs=nn_config['epochs'],
        batch_size=nn_config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights
    )
    
    logging.info("Model training completed.")
    return model


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: pd.Series) -> None:
    try:
        # Unpack the metrics returned by model.evaluate
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
        logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        logging.info(f"Classification Report:\n{report}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Classification Report:\n{report}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise


def save_model(model: tf.keras.Model, model_dir: Path) -> None:
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / 'neural_network_model')
        logging.info(f"Model saved to {model_dir / 'neural_network_model'}")
        print(f"Model saved to {model_dir / 'neural_network_model'}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network model.")
    parser.add_argument('--data_dir', type=str, default='data/split_data', help='Directory containing the data files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to configuration file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--tuning_dir', type=str, default='my_dir', help='Directory for hyperparameter tuning')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    config_file = Path(args.config)
    tuning_dir = args.tuning_dir

    configure_logging(log_dir, args.log_level)
    config = load_config(config_file)
    validate_config(config)

    try:
        X_train, X_test, y_train, y_test = load_data(data_dir)
        X_train_transformed, X_test_transformed = preprocess_data(X_train, X_test, config)
        
        best_hps = tune_hyperparameters(X_train_transformed, y_train, tuning_dir)
        logging.info(f"Best hyperparameters: {best_hps}")

        model = build_model(best_hps)
        model = train_model(model, X_train_transformed, y_train, config)
        evaluate_model(model, X_test_transformed, y_test)
        save_model(model, model_dir)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()

