---

# Heart Disease Prediction Application

## Overview

The Heart Disease Prediction Application is a machine learning project designed to predict the risk of heart disease based on patient data. This application leverages various machine learning models, including Logistic Regression, Random Forest, and a Neural Network, to provide accurate risk assessments.

## Features

- **Predictive Models**: Utilizes Logistic Regression, Random Forest, and a Neural Network for heart disease prediction.
- **Web Interface**: A user-friendly web interface for inputting patient data and receiving predictions.
- **Data Processing**: Includes scripts for data preprocessing, feature engineering, and model training.

## Project Structure

```
HDD/
├── commands/
│   └── commands
├── config/
│   └── config.json
├── data/
│   ├── split_data/
│   ├── feature_engineered_data.csv
│   ├── heart_disease.csv
│   └── processed_heart_disease.csv
├── eda/
│   ├── plots/
│   └── eda.py
├── logs/
│   ├── data_split.log
│   ├── eda.log
│   ├── feature_engineering.log
│   ├── preprocessing.log
│   ├── model_training.log
│   └── neural_network_training.log
├── models/
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── neural_network_model/
│       ├── assets/
│       ├── keras_metadata.pb
│       ├── saved_model.pb
│       ├── fingerprint.pb
│       └── neural_network_model.h5
├── scripts/
│   ├── preprocessing.py
│   ├── datasplit.py
│   ├── random_forest.py
│   ├── feature_engineering.py
│   ├── logistic_regression.py
│   └── neural_network_training.py
├── app.py
├── frontend/
│   ├── assets/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_<module>.py
├── .gitignore
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AllenFLetcherAI/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. **Set Up a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask Application**:
    ```bash
    python app.py
    ```

2. **Access the Web Interface**:
    Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

3. **Input Patient Data**:
    - Fill in the form with patient data.
    - Click the "Predict" button to get the risk prediction.

## Configuration

Configuration settings are managed in the `config/config.json` file. It includes settings for data preprocessing, feature engineering, and neural network configurations.

## Logging

Logs for data splitting, feature engineering, preprocessing, model training, and neural network training can be found in the `logs/` directory.

## Contributing

If you would like to contribute to this project, please refer to the `CONTRIBUTING.md` file for guidelines on how to get started.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or feedback, please contact:

**Allen Fletcher**  

Email: allensimeonfletcher@gmail.com  

GitHub: AllenFletcherAI; https://github.com/AllenFletcherAI

LinkedIn: https://www.linkedin.com/in/allen-fletcher/

---
