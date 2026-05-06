import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

print("==================== EXERCICE 2: CLASSIFICATION BINAIRE SUR LES DONNÉES IRIS SETOSA ====================")

# ====================== CHARGEMENT ET SÉLECTION DES CARACTÉRISTIQUES ======================
# Load Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Select Petal Length and Petal Width
X_selected = X[:, [2, 3]]

# Create binary labels: Setosa (1) vs Others (0)
y_binary = (y == 0).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== IMPLEMENTATION FROM SCRATCH ======================
print("\n------------------ FROM SCRATCH ------------------")

def perceptron_train_scratch(X, y, learning_rate=0.1, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for epoch in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = 1 if linear_output >= 0 else 0
            update = learning_rate * (y[idx] - y_predicted)
            weights += update * x_i
            bias += update
            
    return weights, bias

def perceptron_predict_scratch(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.where(linear_output >= 0, 1, 0)

# Train perceptron from scratch
weights_scratch, bias_scratch = perceptron_train_scratch(X_train_scaled, y_train)
y_pred_train_scratch = perceptron_predict_scratch(X_train_scaled, weights_scratch, bias_scratch)
y_pred_test_scratch = perceptron_predict_scratch(X_test_scaled, weights_scratch, bias_scratch)

train_accuracy_scratch = accuracy_score(y_train, y_pred_train_scratch)
test_accuracy_scratch = accuracy_score(y_test, y_pred_test_scratch)

print(f"Training Accuracy (Scratch): {train_accuracy_scratch:.2%}")
print(f"Test Accuracy (Scratch): {test_accuracy_scratch:.2%}")

print("\nConfusion Matrix (Test Data) - Scratch:")
cm_scratch = confusion_matrix(y_test, y_pred_test_scratch)
print(cm_scratch)

print("\nClassification Report - Scratch:")
print(classification_report(y_test, y_pred_test_scratch,
      target_names=['Non-Setosa', 'Setosa']))

# ====================== IMPLEMENTATION SCIKIT-LEARN ======================
print("\n------------- SCIKIT-LEARNING METHOD ------------")

# Train Perceptron
clf = Perceptron(eta0=0.1, max_iter=100, random_state=42, tol=1e-3)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

# Evaluate model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy (Sklearn): {train_accuracy:.2%}")
print(f"Test Accuracy (Sklearn): {test_accuracy:.2%}")

print("\nConfusion Matrix (Test Data) - Sklearn:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

print("\nClassification Report - Sklearn:")
print(classification_report(y_test, y_pred_test,
      target_names=['Non-Setosa', 'Setosa']))

# ====================== IMPLEMENTATION KERAS ======================
print("\n------------------ KERAS METHOD ------------------")

# Create and train Perceptron using Keras
def create_perceptron_model(input_dim):
    model = Sequential([
        Dense(1, input_dim=input_dim, activation='sigmoid',  # Using sigmoid for binary classification
              kernel_initializer='zeros', bias_initializer='zeros')
    ])
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_perceptron_model(X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    verbose=0,
                    validation_split=0.1)

# Make predictions
y_pred_train_prob = model.predict(X_train_scaled)
y_pred_train = (y_pred_train_prob > 0.5).astype(int).flatten()
y_pred_test_prob = model.predict(X_test_scaled)
y_pred_test = (y_pred_test_prob > 0.5).astype(int).flatten()

# Evaluate model
train_accuracy_keras = accuracy_score(y_train, y_pred_train)
test_accuracy_keras = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy (Keras): {train_accuracy_keras:.2%}")
print(f"Test Accuracy (Keras): {test_accuracy_keras:.2%}")

print("\nConfusion Matrix (Test Data) - Keras:")
cm_keras = confusion_matrix(y_test, y_pred_test)
print(cm_keras)

print("\nClassification Report - Keras:")
print(classification_report(y_test, y_pred_test,
      target_names=['Non-Setosa', 'Setosa']))

# Get weights and bias (for comparison with sklearn)
weights_keras = model.get_weights()[0].flatten()  # weights for the two features
bias_keras = model.get_weights()[1][0]           # bias
print(f"\nLearned weights (Keras): {weights_keras}")
print(f"Learned bias (Keras): {bias_keras}")

# Visualization (similar to original)
h = 0.02
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(12, 5))

# Training data plot
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1],
            c='red', marker='o', label='Non-Setosa', edgecolors='k')
plt.scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1],
            c='blue', marker='s', label='Setosa', edgecolors='k')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title(f'Training Data (Accuracy: {train_accuracy_keras:.1%})')
plt.legend()
plt.grid(True, alpha=0.3)

# Test data plot
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
correct = y_test == y_pred_test
plt.scatter(X_test_scaled[correct & (y_test == 0), 0],
            X_test_scaled[correct & (y_test == 0), 1],
            c='red', marker='o', label='Non-Setosa (Correct)', edgecolors='k')
plt.scatter(X_test_scaled[correct & (y_test == 1), 0],
            X_test_scaled[correct & (y_test == 1), 1],
            c='blue', marker='s', label='Setosa (Correct)', edgecolors='k')
incorrect = y_test != y_pred_test
plt.scatter(X_test_scaled[incorrect, 0], X_test_scaled[incorrect, 1],
            c='yellow', marker='x', s=200, label='Misclassified', linewidths=3)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title(f'Test Data (Accuracy: {test_accuracy_keras:.1%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_classification.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SUMMARY OF ALL THREE IMPLEMENTATIONS")
print("="*60)
print(f"From Scratch - Train Acc: {train_accuracy_scratch:.2%}, Test Acc: {test_accuracy_scratch:.2%}")
print(f"Sklearn      - Train Acc: {train_accuracy:.2%}, Test Acc: {test_accuracy:.2%}")
print(f"Keras        - Train Acc: {train_accuracy_keras:.2%}, Test Acc: {test_accuracy_keras:.2%}")
print("="*60)