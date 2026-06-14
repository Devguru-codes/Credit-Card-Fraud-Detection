# Credit Card Fraud Detection with Quantum Machine Learning

## Overview

This project implements a comprehensive and advanced pipeline for credit card fraud detection. It seamlessly bridges classical machine learning techniques with Quantum Machine Learning (QML) by employing a Variational Quantum Classifier (VQC). 

The workflow is designed to handle the complexities of real-world financial data, covering everything from detailed feature engineering and handling severe class imbalance to meta-heuristic feature selection (Particle Swarm Optimization) and hybrid quantum-classical modeling using PennyLane and TensorFlow.

---

## 1. Data Preprocessing & Feature Engineering

The dataset contains credit card transactions with features like amount, location, merchant details, and timestamps. Personally Identifiable Information (PII) such as names, credit card numbers, and exact addresses are removed to ensure privacy.

We extract meaningful temporal features from the transaction timestamp, which often carry strong signals for fraud detection:

```python
# Extract temporal features from datetime
df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['month'] = df['trans_date_trans_time'].dt.month
```

Categorical variables are encoded, and numerical variables are scaled using `OrdinalEncoder` and `StandardScaler` within a `ColumnTransformer` pipeline:

```python
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Categorical Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ('passthrough', 'passthrough', passthrough_features)
    ], remainder='drop'
)
X_train_transformed = preprocessor.fit_transform(X_train)

# Numerical Scaling
scaler_preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), features_to_scale),
        ('passthrough', 'passthrough', passthrough_cols)
    ], remainder='drop'
)
X_train_scaled = scaler_preprocessor.fit_transform(X_train_transformed)
```

---

## 2. Handling Class Imbalance

Fraud datasets are notoriously imbalanced. To prevent the model from becoming biased towards the majority class (non-fraud), we utilize **SMOTE-ENN** (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors). This approach oversamples the minority class while cleaning up noisy samples in the overlapping regions.

```python
from imblearn.combine import SMOTEENN

# Initialize SMOTEENN
smote_enn = SMOTEENN(random_state=42)

# Apply SMOTE-ENN to the training data
X_resampled, y_resampled = smote_enn.fit_resample(X_train_reduced, y_train_reduced)
```

---

## 3. Advanced Feature Selection

To reduce dimensionality, eliminate noise, and make quantum simulation computationally tractable, we employ a two-step feature selection process.

First, we use **K-Best** for a baseline reduction:
```python
from sklearn.feature_selection import SelectKBest, f_classif

kbest_selector = SelectKBest(score_func=f_classif, k=50)
X_resampled_selected = kbest_selector.fit_transform(X_resampled, y_resampled)
```

Next, we apply **Particle Swarm Optimization (PSO)**, a meta-heuristic algorithm, to find the optimal subset of features that maximizes the F1-score of a baseline classifier.

```python
import pyswarms as ps

# Define PSO optimizer for discrete binary search space
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 1, 'p': 1}
optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=X_resampled.shape[1], options=options)

# Optimize to find the best feature subset
cost, pos = optimizer.optimize(f_objective, iters=10, X_train_data=X_resampled_np, y_train_data=y_resampled)
best_features_pso_binary = pos.astype(int)
```

---

## 4. Classical & Deep Learning Baseline Models

To evaluate the effectiveness of the Quantum model, several classical machine learning and deep learning algorithms are used as baselines:

*   **Logistic Regression**: A linear model serving as the foundational baseline. It is fast, interpretable, and used inside our PSO fitness function to select the optimal feature subset.
*   **Random Forest Classifier**: An ensemble learning method that constructs a multitude of decision trees. It is highly robust against overfitting and handles non-linear relationships well.
*   **XGBoost & LightGBM**: Advanced gradient boosting ensemble techniques that build trees sequentially. They typically achieve state-of-the-art performance on tabular data like fraud detection.
*   **Isolation Forest**: An unsupervised learning algorithm for anomaly detection that works well for identifying rare fraud cases.
*   **Neural Networks (MLP & Dense NN)**: Standard Deep Learning feedforward architectures to provide a direct baseline against the Variational Quantum circuit.

---

## 5. Hybrid Quantum-Classical Modeling (VQC)

The core innovation of this project is the integration of a **Variational Quantum Classifier (VQC)**. We utilize `PennyLane` to construct parameterized quantum circuits and integrate them as custom trainable layers within a `TensorFlow/Keras` classical neural network architecture.

### Quantum Circuit Definition
We encode our classical data into quantum states using `AngleEmbedding` and process it using `StronglyEntanglingLayers`.

```python
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    # Data encoding layer
    qml.templates.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))
    # Parameterized entangling layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measurement
    return qml.expval(qml.PauliZ(0))
```

### Keras Integration
The quantum circuit is seamlessly wrapped inside a custom Keras layer, allowing end-to-end backpropagation using classical optimizers like Adam.

```python
import tensorflow as tf
from tensorflow import keras

class QuantumLayer(keras.layers.Layer):
    def __init__(self, qnode, weight_shape, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.qnode = qnode
        self.weight_shape = weight_shape

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=self.weight_shape,
            initializer=keras.initializers.RandomUniform(minval=0, maxval=2*np.pi),
            trainable=True,
            dtype=tf.float32
        )
        super(QuantumLayer, self).build(input_shape)

    def call(self, inputs):
        # Maps the quantum function over the batch of inputs
        return tf.map_fn(
            lambda x: self.qnode(x, self.q_weights), 
            inputs, 
            dtype=tf.float32
        )

# Hybrid Model Architecture
model = keras.models.Sequential([
    keras.Input(shape=(n_input_features,)),
    keras.layers.Dense(n_qubits, activation="relu"), # Classical dimensionality reduction
    QuantumLayer(qnode, weight_shape),               # Quantum processing
    keras.layers.Dense(1, activation="sigmoid")      # Classical output
])

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
```

---

## 6. Model Comparison

The following table outlines the comprehensive evaluation of all models on the test set, demonstrating how the classical baselines compare with deep learning approaches:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9729 | 0.0553 | 0.9184 | 0.1043 | 0.9702 |
| **Isolation Forest** | 0.9976 | 0.3113 | 0.3367 | 0.3235 | 0.9602 |
| **LightGBM** | 0.9981 | 0.4677 | 0.8878 | 0.6127 | 0.9735 |
| **Neural Network (MLP)** | 0.9987 | 0.5793 | 0.8571 | 0.6914 | 0.9726 |
| **XGBoost** | 0.9989 | 0.6385 | 0.8469 | 0.7281 | 0.9814 |
| **TensorFlow (Dense NN)** | 0.9990 | 0.6641 | 0.8673 | 0.7522 | 0.9749 |
| **Random Forest** | **0.9993** | **0.7788** | 0.8265 | **0.8020** | 0.9577 |
| **VQC (Hybrid)** | *(Run nb)* | *(Run nb)* | *(Run nb)* | *(Run nb)* | *(Run nb)* |

---

## How to Run

### 1. Requirements & Setup
Ensure you have Python 3.x installed, then install the necessary dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn pyswarms pennylane tensorflow
```

Place your `fraud test.csv` in the root of the project directory.

### 2. Execution
Launch Jupyter and open the main notebook to execute the pipeline cell by cell:
`ML_PROJECT_CREDIT_CARD_FRAUD_DETECTION.ipynb`

---

## Credits
- Developed by Devguru Tiwari
- Quantum ML architecture inspired by PennyLane documentation.

## License
This project is for educational and research purposes.
