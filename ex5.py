import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

print("==================== EXERCICE 5: SEGMENTATION DES DONNÉES SYNTHÉTIQUES (make_blobs) ====================")

# ====================== GÉNÉRATION ET VISUALISATION DES DONNÉES ======================
X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                  random_state=42, cluster_std=1.5)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data with 3 Clusters (make_blobs)')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('ex5_initial_data.png', dpi=100, bbox_inches='tight')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ====================== IMPLEMENTATION FROM SCRATCH ======================
print("\n------------------ FROM SCRATCH ------------------")

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)
                
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
    def _activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

mask_train = y_train < 2
X_train_binary = X_train[mask_train]
y_train_binary = y_train[mask_train]

mask_test = y_test < 2
X_test_binary = X_test[mask_test]
y_test_binary = y_test[mask_test]

perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)
perceptron.fit(X_train_binary, y_train_binary)

y_pred_train = perceptron.predict(X_train_binary)
y_pred_test = perceptron.predict(X_test_binary)

train_accuracy = np.mean(y_pred_train == y_train_binary)
test_accuracy = np.mean(y_pred_test == y_test_binary)

print(f"\n--- Perceptron Results (Clusters 0 vs 1) ---")
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")

plt.figure(figsize=(10, 6))
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X_train_binary[y_train_binary == 0, 0], 
            X_train_binary[y_train_binary == 0, 1],
            c='blue', marker='o', label='Cluster 0', edgecolors='k')
plt.scatter(X_train_binary[y_train_binary == 1, 0], 
            X_train_binary[y_train_binary == 1, 1],
            c='red', marker='s', label='Cluster 1', edgecolors='k')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('Perceptron Decision Boundary (Clusters 0 vs 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ex5_perceptron_boundary.png', dpi=100, bbox_inches='tight')
plt.show()

# ====================== IMPLEMENTATION KERAS ======================
print("\n------------------ KERAS METHOD ------------------")

# Create and train Perceptron using Keras
def create_perceptron_model(input_dim):
    model = Sequential([
        Dense(1, input_dim=input_dim, activation='sigmoid',  # Sigmoid for binary classification
              kernel_initializer='zeros', bias_initializer='zeros')
    ])
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
model = create_perceptron_model(X_train_binary.shape[1])
history = model.fit(X_train_binary, y_train_binary, 
                    epochs=1000, 
                    verbose=0,
                    validation_split=0.1)

# Make predictions
y_pred_train_prob = model.predict(X_train_binary)
y_pred_train = (y_pred_train_prob > 0.5).astype(int).flatten()
y_pred_test_prob = model.predict(X_test_binary)
y_pred_test = (y_pred_test_prob > 0.5).astype(int).flatten()

# Evaluate model
train_accuracy_keras = np.mean(y_pred_train == y_train_binary)
test_accuracy_keras = np.mean(y_pred_test == y_test_binary)

print(f"Training Accuracy (Keras): {train_accuracy_keras:.2%}")
print(f"Test Accuracy (Keras): {test_accuracy_keras:.2%}")

# Get weights and bias (for comparison with sklearn)
weights_keras = model.get_weights()[0].flatten()  # weights for the two features
bias_keras = model.get_weights()[1][0]           # bias
print(f"\nLearned weights (Keras): {weights_keras}")
print(f"Learned bias (Keras): {bias_keras}")

# Visualization
plt.figure(figsize=(10, 6))
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X_train_binary[y_train_binary == 0, 0], 
            X_train_binary[y_train_binary == 0, 1],
            c='blue', marker='o', label='Cluster 0', edgecolors='k')
plt.scatter(X_train_binary[y_train_binary == 1, 0], 
            X_train_binary[y_train_binary == 1, 1],
            c='red', marker='s', label='Cluster 1', edgecolors='k')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('Perceptron Decision Boundary (Keras) - Clusters 0 vs 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ex5_perceptron_boundary_keras.png', dpi=100, bbox_inches='tight')
plt.show()

# ====================== IMPLEMENTATION KMEANS (UNSUPERVISED) ======================
print("\n------------- KMEANS CLUSTERING (UNSUPERVISED) ------------")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
y_kmeans = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', 
            edgecolors='k', s=50, marker='o')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
            s=200, edgecolors='k', label='Centroids')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('KMeans Clustering (3 Clusters)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ex5_kmeans_clusters.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n--- KMeans Results ---")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")

# ====================== COMPARISON VISUAL ======================
print("\n------------------ COMPARISON VISUELLE ------------------")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', 
                edgecolors='k', s=50)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Original Data (Ground Truth)')
axes[0].grid(True, alpha=0.3)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
axes[1].scatter(X_train_binary[y_train_binary == 0, 0], 
                X_train_binary[y_train_binary == 0, 1],
                c='blue', marker='o', label='Cluster 0', edgecolors='k')
axes[1].scatter(X_train_binary[y_train_binary == 1, 0], 
                X_train_binary[y_train_binary == 1, 1],
                c='red', marker='s', label='Cluster 1', edgecolors='k')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title(f'Perceptron (Binary)\nTest Accuracy: {test_accuracy:.1%}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', 
                edgecolors='k', s=50)
axes[2].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                s=200, edgecolors='k', label='Centroids')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].set_title('KMeans (3 Clusters)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex5_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SUMMARY OF ALL IMPLEMENTATIONS")
print("="*60)
print(f"Perceptron from Scratch - Train Acc: {train_accuracy:.2%}, Test Acc: {test_accuracy:.2%}")
print(f"Perceptron Keras        - Train Acc: {train_accuracy_keras:.2%}, Test Acc: {test_accuracy_keras:.2%}")
print(f"KMeans (Unsupervised)   - Inertia: {kmeans.inertia_:.2f}")
print("="*60)

print("\n--- Analyse des résultats ---")
print("1. Perceptron (supervisé): Sépare les données en 2 classes avec une frontière linéaire")
print("2. KMeans (non supervisé): Segmente les données en 3 groupes sans labels")
print("3. Différence clé: Le Perceptron nécessite des labels (apprentissage supervisé)")
print("   tandis que KMeans découvre les clusters automatiquement (apprentissage non supervisé)")