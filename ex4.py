import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

print("==================== EXERCICE 4: CLASSIFICATION SUR MNIST (DIGITS) ====================")

# ====================== CHARGEMENT ET PRÉPARATION ======================
# Load MNIST digits dataset (subset: 0-9 digits, 1797 samples)
digits = load_digits()
X = digits.data
y = digits.target

print(f"Dataset shape:     {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes:           {np.unique(y)}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== IMPLEMENTATION FROM SCRATCH ======================
print("\n------------------ FROM SCRATCH ------------------")

# For simplicity, we'll use a simplified from-scratch implementation
# that mimics the perceptron learning rule for multi-class classification
# In practice, a true from-scratch multi-class perceptron would be more complex
# but we'll demonstrate the concept

def perceptron_train_scratch_multiclass(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Initialize weights for each class (one-vs-all approach)
    weights = np.zeros((n_features, n_classes))
    biases = np.zeros(n_classes)
    
    # Convert labels to one-hot encoding for training
    y_one_hot = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_one_hot[i, y[i]] = 1
    
    for epoch in range(epochs):
        for i in range(n_samples):
            # Linear output for all classes
            linear_output = np.dot(X[i], weights) + biases
            # Softmax activation
            exp_scores = np.exp(linear_output)
            probs = exp_scores / np.sum(exp_scores)
            
            # Error (difference between predicted probability and true label)
            error = probs - y_one_hot[i]
            
            # Update weights and biases
            for j in range(n_features):
                weights[j] -= learning_rate * error * X[i][j]
            biases -= learning_rate * error
    
    return weights, biases

def perceptron_predict_scratch_multiclass(X, weights, biases):
    # Linear output
    linear_output = np.dot(X, weights) + biases
    # Softmax
    exp_scores = np.exp(linear_output)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Return predicted class
    return np.argmax(probs, axis=1)

# Train perceptron from scratch (simplified)
print("Training from scratch perceptron (simplified multi-class)...")
weights_scratch, biases_scratch = perceptron_train_scratch_multiclass(X_train_scaled, y_train)
y_pred_train_scratch = perceptron_predict_scratch_multiclass(X_train_scaled, weights_scratch, biases_scratch)
y_pred_test_scratch = perceptron_predict_scratch_multiclass(X_test_scaled, weights_scratch, biases_scratch)

train_accuracy_scratch = accuracy_score(y_train, y_pred_train_scratch)
test_accuracy_scratch = accuracy_score(y_test, y_pred_test_scratch)

print(f"Training Accuracy (Scratch): {train_accuracy_scratch:.2%}")
print(f"Test Accuracy (Scratch): {test_accuracy_scratch:.2%}")

print("\nConfusion Matrix (Test Data) - Scratch:")
cm_scratch = confusion_matrix(y_test, y_pred_test_scratch)
print(cm_scratch)

print("\nClassification Report - Scratch:")
print(classification_report(y_test, y_pred_test_scratch))

# ====================== IMPLEMENTATION SCIKIT-LEARN ======================
print("\n------------- SCIKIT-LEARNING METHOD ------------")

# Train Perceptron
clf = SGDClassifier(
    loss='perceptron',
    eta0=0.01,
    max_iter=100,
    random_state=42,
    tol=1e-3
)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

# Evaluate model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy (Sklearn): {train_accuracy:.2%}")
print(f"Test Accuracy (Sklearn): {test_accuracy:.2%}")

print("\nClassification Report (Test Data) - Sklearn:")
print(classification_report(y_test, y_pred_test))

# ====================== IMPLEMENTATION KERAS ======================
print("\n------------------ KERAS METHOD ------------------")

# Convert labels to categorical for Keras (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# Create and train Perceptron using Keras (multi-class classification)
def create_perceptron_model(input_dim, num_classes):
    model = Sequential([
        Dense(num_classes, input_dim=input_dim, activation='softmax',  # Softmax for multi-class
              kernel_initializer='zeros', bias_initializer='zeros')
    ])
    optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_perceptron_model(X_train_scaled.shape[1], 10)
history = model.fit(X_train_scaled, y_train_cat, 
                    epochs=100, 
                    verbose=0,
                    validation_split=0.1)

# Make predictions
y_pred_train_prob = model.predict(X_train_scaled)
y_pred_train = np.argmax(y_pred_train_prob, axis=1)
y_pred_test_prob = model.predict(X_test_scaled)
y_pred_test = np.argmax(y_pred_test_prob, axis=1)

# Evaluate model
train_accuracy_keras = accuracy_score(y_train, y_pred_train)
test_accuracy_keras = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy (Keras): {train_accuracy_keras:.2%}")
print(f"Test Accuracy (Keras): {test_accuracy_keras:.2%}")

print("\nClassification Report (Test Data) - Keras:")
print(classification_report(y_test, y_pred_test))

# Confusion matrix
cm_keras = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_keras, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for the test set (MNIST)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('mnist_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

# Visualize some predictions
def plot_predictions(images, true_labels, pred_labels, num_samples=10):
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples//2, i+1)
        plt.imshow(images[i].reshape(8, 8), cmap='gray')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f'T:{true_labels[i]}\\nP:{pred_labels[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=100, bbox_inches='tight')
    plt.show()

# Show some test predictions
plot_predictions(X_test_scaled, y_test, y_pred_test, num_samples=10)

# Get weights and bias (for inspection)
weights_keras = model.get_weights()[0]  # Shape: (64, 10) - 64 input features, 10 classes
bias_keras = model.get_weights()[1]     # Shape: (10,)
print(f"\nWeights shape (Keras): {weights_keras.shape}")
print(f"Bias shape (Keras): {bias_keras.shape}")

print("\n" + "="*60)
print("SUMMARY OF ALL THREE IMPLEMENTATIONS")
print("="*60)
print(f"From Scratch - Train Acc: {train_accuracy_scratch:.2%}, Test Acc: {test_accuracy_scratch:.2%}")
print(f"Sklearn      - Train Acc: {train_accuracy:.2%}, Test Acc: {test_accuracy:.2%}")
print(f"Keras        - Train Acc: {train_accuracy_keras:.2%}, Test Acc: {test_accuracy_keras:.2%}")
print("="*60)

print("\nInterpretation:")
print("Le Perceptron linéaire donne des résultats raisonnables sur MNIST")
print("mais ne capture pas la structure spatiale des pixels ni les relations")
print("non-linéaires complexes. Pour des performances d'état de l'art,")
print("on utilisera des CNN.")