import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

print("==================== EXERCICE 3: RÉGRESSION LINÉAIRE AVEC LE PERCEPTRON ====================")

# ====================== CRÉATION DES DONNÉES ======================
# Create dataset: y = 2x + 1 with some noise
np.random.seed(42)
X = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 2, 100)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== IMPLEMENTATION FROM SCRATCH ======================
print("\n------------------ FROM SCRATCH ------------------")

def perceptron_train_scratch(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for epoch in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            # For regression, we use linear activation (no threshold)
            y_predicted = linear_output
            # Error for regression is the difference
            error = y[idx] - y_predicted
            # Update weights and bias (similar to perceptron rule but for regression)
            weights += learning_rate * error * x_i
            bias += learning_rate * error
            
    return weights, bias

def perceptron_predict_scratch(X, weights, bias):
    return np.dot(X, weights) + bias

# Train perceptron from scratch
weights_scratch, bias_scratch = perceptron_train_scratch(X_train_scaled, y_train)
y_pred_train_scratch = perceptron_predict_scratch(X_train_scaled, weights_scratch, bias_scratch)
y_pred_test_scratch = perceptron_predict_scratch(X_test_scaled, weights_scratch, bias_scratch)

# Evaluate model
mse_train_scratch = mean_squared_error(y_train, y_pred_train_scratch)
mse_test_scratch = mean_squared_error(y_test, y_pred_test_scratch)
mae_test_scratch = mean_absolute_error(y_test, y_pred_test_scratch)
r2_test_scratch = r2_score(y_test, y_pred_test_scratch)

print(f"Training MSE (Scratch): {mse_train_scratch:.4f}")
print(f"Test MSE (Scratch): {mse_test_scratch:.4f}")
print(f"Test MAE (Scratch): {mae_test_scratch:.4f}")
print(f"R² Score (Scratch): {r2_test_scratch:.4f}")

print(f"\nLearned weight (Scratch): {weights_scratch[0]:.4f}")
print(f"Learned bias (Scratch): {bias_scratch:.4f}")

# ====================== IMPLEMENTATION SCIKIT-LEARN ======================
print("\n------------- SCIKIT-LEARNING METHOD ------------")

# Train Perceptron (SGDRegressor for regression)
regressor = SGDRegressor(loss='squared_error', eta0=0.01,
                         max_iter=1000, random_state=42)
regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = regressor.predict(X_train_scaled)
y_pred_test = regressor.predict(X_test_scaled)

# Evaluate model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MSE (Sklearn): {mse_train:.4f}")
print(f"Test MSE (Sklearn): {mse_test:.4f}")
print(f"Test MAE (Sklearn): {mae_test:.4f}")
print(f"R² Score (Sklearn): {r2_test:.4f}")

print(f"\nLearned weight (Sklearn): {regressor.coef_[0]:.4f}")
print(f"Learned bias (Sklearn): {regressor.intercept_[0]:.4f}")

# ====================== IMPLEMENTATION KERAS ======================
print("\n------------------ KERAS METHOD ------------------")

# Create and train Perceptron using Keras (for regression, we use linear activation)
def create_perceptron_model(input_dim):
    model = Sequential([
        Dense(1, input_dim=input_dim, activation='linear',  # Linear activation for regression
              kernel_initializer='zeros', bias_initializer='zeros')
    ])
    optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Train the model
model = create_perceptron_model(X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train, 
                    epochs=1000, 
                    verbose=0,
                    validation_split=0.1)

# Make predictions
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

# Evaluate model
mse_train_keras = mean_squared_error(y_train, y_pred_train)
mse_test_keras = mean_squared_error(y_test, y_pred_test)
mae_test_keras = mean_absolute_error(y_test, y_pred_test)
r2_test_keras = r2_score(y_test, y_pred_test)

print(f"Training MSE (Keras): {mse_train_keras:.4f}")
print(f"Test MSE (Keras): {mse_test_keras:.4f}")
print(f"Test MAE (Keras): {mae_test_keras:.4f}")
print(f"R² Score (Keras): {r2_test_keras:.4f}")

print(f"\nLearned weight (Keras): {model.get_weights()[0][0][0]:.4f}")
print(f"Learned bias (Keras): {model.get_weights()[1][0]:.4f}")

# Visualization
plt.figure(figsize=(12, 5))

# Training data plot
plt.subplot(1, 2, 1)
plt.scatter(X_train_scaled, y_train, c='blue',
            alpha=0.6, label='Training data')
plt.plot(X_train_scaled, y_pred_train, c='red',
         linewidth=2, label='Fitted line')
plt.xlabel('X (standardized)')
plt.ylabel('y')
plt.title(f'Training Data (MSE: {mse_train_keras:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Test data plot
plt.subplot(1, 2, 2)
plt.scatter(X_test_scaled, y_test, c='blue', alpha=0.6, label='Test data')
plt.plot(X_test_scaled, y_pred_test, c='red', linewidth=2, label='Fitted line')
plt.xlabel('X (standardized)')
plt.ylabel('y')
plt.title(f'Test Data (MSE: {mse_test_keras:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SUMMARY OF ALL THREE IMPLEMENTATIONS")
print("="*60)
print(f"From Scratch - Train MSE: {mse_train_scratch:.4f}, Test MSE: {mse_test_scratch:.4f}")
print(f"Sklearn      - Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
print(f"Keras        - Train MSE: {mse_train_keras:.4f}, Test MSE: {mse_test_keras:.4f}")
print("="*60)