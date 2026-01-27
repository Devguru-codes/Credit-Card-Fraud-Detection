# Credit Card Fraud Detection with Quantum Machine Learning

## Overview

This project implements a comprehensive pipeline for credit card fraud detection, combining classical machine learning techniques with quantum machine learning (QML) using a Variational Quantum Classifier (VQC). The workflow covers data preprocessing, feature engineering, class imbalance handling, feature selection (K-Best and PSO), and hybrid quantum-classical modeling.

## Dataset

- The dataset (`fraud test.csv`) contains credit card transaction records, including features such as transaction amount, location, merchant, category, and a binary target `is_fraud`.
- Personally Identifiable Information (PII) columns are removed to ensure privacy.

## Data Preprocessing

- **Missing Values:** Rows with missing values are dropped.
- **Datetime Conversion:** Transaction timestamps are converted to datetime objects.
- **PII Removal:** Columns like `first`, `last`, `street`, `dob`, `cc_num`, and `trans_num` are dropped.

## Feature Engineering

- **Temporal Features:** Extracted `hour_of_day`, `day_of_week`, and `month` from transaction timestamps.
- **Categorical Encoding:** Features such as `merchant`, `category`, `gender`, `city`, `state`, and `job` are encoded using `OrdinalEncoder`.
- **Scaling:** Numerical features (`amt`, `city_pop`) are normalized using `StandardScaler`.

## Handling Class Imbalance

- **SMOTE-ENN:** Applied to the training set to balance the minority fraud class and clean noisy samples.

## Feature Selection

- **K-Best:** Selects top features based on statistical tests.
- **PSO (Particle Swarm Optimization):** Meta-heuristic search for optimal feature subsets, maximizing F1-score.

## Quantum Machine Learning

- **VQC (Variational Quantum Classifier):** Built using PennyLane and TensorFlow, leveraging a hybrid quantum-classical architecture.
- **Pipeline:** Classical dense layer → Quantum circuit (angle embedding + strongly entangling layers) → Classical sigmoid output.
- **Training:** Performed on a balanced subset due to quantum simulation constraints.

## Results & Metrics

### Metrics Computed
- **Accuracy**: Evaluated on both training and validation sets for classical and quantum models.
- **Loss**: Binary cross-entropy loss tracked during training and validation.
- **F1-score, Precision, Recall, AUC**: Computed using scikit-learn for imbalanced classification.
- **Classification Report**: Includes precision, recall, F1-score for each class.
- **Model Training Curves**: Plots of loss and accuracy over epochs for both classical and quantum models.

> *Note: Exact metric values depend on the specific run and are output in the notebook cells after model training and evaluation.*

### Results & Insights

- Significant feature space expansion after encoding (over 2000 features).
- SMOTE-ENN effectively balances the dataset for robust model training.
- Feature selection (K-Best, PSO) reduces dimensionality and improves model focus.
- Quantum ML demonstrates the integration of QML in real-world fraud detection.
- Further improvements can be made with dimensionality reduction and advanced quantum circuits.

## How to Run

1. **Requirements:**
   - Python 3.x
   - pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, pyswarms
   - PennyLane, TensorFlow

2. **Setup:**
   - Install dependencies:
     ```
     pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn pyswarms pennylane tensorflow
     ```
   - Place `fraud test.csv` in the project directory.

3. **Usage:**
   - Open and run the notebook:  
     `ML_PROJECT_CREDIT_CARD_FRAUD_DETECTION (Only Quantum) (1).ipynb`
   - Follow the notebook cells for step-by-step execution.

## Repository

- GitHub: [https://github.com/Devguru-codes/Credit-Card-Fraud-Detection](https://github.com/Devguru-codes/Credit-Card-Fraud-Detection)

## Credits

- Developed by Akshat Shukla
- Quantum ML section inspired by PennyLane documentation

## License

This project is for educational and research purposes.