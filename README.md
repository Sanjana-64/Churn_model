# Customer Churn Prediction using Neural Networks

A machine learning project that predicts whether a bank customer is likely to leave the bank using customer demographic and account information.

## Project Overview

Customer churn prediction is a classification problem where the objective is to identify customers who are likely to leave a service.

In this project, a **Deep Learning neural network** built with **TensorFlow/Keras** is used to predict customer churn. The dataset contains 10,000 customer records with demographic, financial, and account-related information.

The project covers:

* Data loading and exploration
* Data preprocessing
* Feature selection
* Categorical variable encoding
* Train-test splitting
* Feature scaling
* Neural network construction
* Model training
* Model saving
* Customer churn prediction

## Dataset

The project uses the `Churn_Modelling (1).csv` dataset.

The original dataset contains **10,000 customer records**.

### Features

| Feature         | Description                                          |
| --------------- | ---------------------------------------------------- |
| CreditScore     | Customer credit score                                |
| Geography       | Customer's country                                   |
| Gender          | Customer gender                                      |
| Age             | Customer age                                         |
| Tenure          | Number of years with the bank                        |
| Balance         | Customer account balance                             |
| NumOfProducts   | Number of bank products used                         |
| HasCrCard       | Whether the customer has a credit card               |
| IsActiveMember  | Whether the customer is an active member             |
| EstimatedSalary | Estimated customer salary                            |
| Exited          | Target variable indicating whether the customer left |

The original `RowNumber`, `CustomerId`, and `Surname` columns are removed during preprocessing because they are not used as model features.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* Jupyter Notebook / Google Colab

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("Churn_Modelling (1).csv")
```

### 2. Data Preprocessing

The following columns are removed:

```text
RowNumber
CustomerId
Surname
```

The `Gender` and `Geography` categorical features are encoded using `LabelEncoder`.

### 3. Feature and Target Separation

The target variable is:

```text
Exited
```

The remaining features are used as input variables.

The data is divided into:

* **80% training data**
* **20% testing data**

using `train_test_split` with `random_state=42`.

### 4. Feature Scaling

`StandardScaler` is used to scale the input features before training the neural network.

## Neural Network Architecture

The project uses a Sequential neural network with the following architecture:

```text
Input Layer
    ↓
Dense Layer - 64 neurons - ReLU
    ↓
Dense Layer - 32 neurons - ReLU
    ↓
Output Layer - 1 neuron - Sigmoid
```

The network contains **2,817 trainable parameters**.

### Model Configuration

* Optimizer: Adam
* Loss function: Binary Crossentropy
* Metric: Accuracy
* Epochs: 100
* Output activation: Sigmoid

## Model Training

The model is trained using the training dataset with the test dataset supplied as validation data:

```python
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=100
)
```

The notebook records approximately:

```text
Training accuracy:   48.46%
Validation accuracy: 47.50%
```

These results remain essentially unchanged throughout the recorded 100 epochs.

> **Note:** The current notebook uses an Adam learning rate of `0.0`. This prevents the model weights from updating during training and is the main reason the recorded training metrics remain unchanged. For a meaningful trained model, the learning rate should be corrected before using this project as a production-quality predictor.

## Model Saving

The trained Keras model is saved as:

```text
mymodel.keras
```

using:

```python
model.save("mymodel.keras")
```

## Prediction

The saved model can be loaded and used to predict churn probability for a new customer.

The model produces a probability between 0 and 1. A threshold of `0.5` is then used to convert the probability into a binary prediction:

```text
0 → Customer is predicted not to exit
1 → Customer is predicted to exit
```

The example included in the notebook produces a prediction probability of approximately `0.7413` and a binary prediction of `1`.

## Project Structure

```text
customer-churn-prediction/
│
├── Churn_model.ipynb
├── Churn_Modelling (1).csv
├── mymodel.keras
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

## Running the Project

1. Clone the repository.
2. Install the required dependencies.
3. Place the dataset in the project directory.
4. Open `Churn_model.ipynb`.
5. Run the notebook cells sequentially.

The notebook can be run using either Jupyter Notebook or Google Colab.

## Future Improvements

Possible improvements to this project include:

* Correcting the optimizer learning rate.
* Using `OneHotEncoder` for categorical variables such as Geography.
* Creating a proper preprocessing pipeline.
* Evaluating the model using precision, recall, F1-score, and ROC-AUC.
* Adding a confusion matrix.
* Handling class imbalance.
* Performing hyperparameter tuning.
* Comparing the neural network with Logistic Regression, Random Forest, and XGBoost.
* Adding early stopping and model checkpoints.
* Building a simple web interface for real-time churn prediction.

## Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using Python, Scikit-learn, and TensorFlow/Keras. It covers data preprocessing, feature scaling, neural network development, model training, model saving, and inference.

The current notebook is primarily a learning/demo implementation. The model configuration should be improved before relying on its predictions for real-world decision-making.

## Author

Sanjana T
If you found this project useful, consider giving the repository a ⭐ on GitHub.
