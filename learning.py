import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from collections import Counter

def get_seed(student_id):
    return abs(hash(student_id)) % (2**32)

student_id = "20201234"
seed = get_seed(student_id)
print(f"Random Seed: {seed}\n")

def generate_data(seed):
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=seed
    )
    return X, y

X, y = generate_data(seed)

def split_data(X, y, train_ratio, seed):
    total_samples = X.shape[0]
    train_size = int(train_ratio * total_samples)
    indices = np.arange(total_samples)
    #[2,5,4,3,7,1]
    np.random.seed(seed)
    np.random.shuffle(indices)
    #[7'5'3'1'2]
    train_idx = indices[:train_size]
    test_idx  = indices[train_size:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_data(X, y, 0.7, seed)

print("===== First 5 Training Samples (Features + Label) =====")
for i in range(5):
    print(f"{i+1}. Features: {X_train[i]} | Label: {y_train[i]}")
print("\n")

def euclidean_distance(test_point, train_point):
    return np.sqrt(np.sum((test_point - train_point)**2))

def compute_distance_matrix(X_test, X_train):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    distance_matrix = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            distance_matrix[i, j] = euclidean_distance(X_test[i], X_train[j])
    return distance_matrix

distance_matrix = compute_distance_matrix(X_test, X_train)

print("===== Distance Matrix (First 5 Test Rows x First 5 Train Columns) =====")
print(np.round(distance_matrix[:5, :5],3))
print("\n")

def predict_single(distances, train_labels, k):
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels  = train_labels[nearest_indices]
    vote_counts = Counter(nearest_labels)
    predicted_label = vote_counts.most_common(1)[0][0]
    return predicted_label, nearest_indices, distances[nearest_indices], nearest_labels

def predict_all(distance_matrix, train_labels, k):
    predictions = []
    neighbors_info = []
    for i in range(distance_matrix.shape[0]):
        label, indices, dists, labels = predict_single(distance_matrix[i], train_labels, k)
        predictions.append(label)
        neighbors_info.append((indices, dists, labels))
    return np.array(predictions), neighbors_info

k_values = [1,3,5,7,9]

def evaluate_knn(distance_matrix, y_train, y_test, k_values):
    accuracies = []
    all_neighbors_info = {}
    for k in k_values:
        preds, neighbors_info = predict_all(distance_matrix, y_train, k)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)
        all_neighbors_info[k] = neighbors_info
        print(f"===== k = {k} =====")
        print(f"Accuracy = {acc:.2f}\n")
        for idx in range(2):
            indices, dists, labels = neighbors_info[idx]
            print(f"Test Point {idx+1}:")
            print(f"  Neighbor Indices: {indices}")
            print(f"  Distances: {np.round(dists,3)}")
            print(f"  Neighbor Labels: {labels}")
            print(f"  Predicted Label: {Counter(labels).most_common(1)[0][0]}")
            print("\n")
    return accuracies, all_neighbors_info

accuracies, neighbors_info_dict = evaluate_knn(distance_matrix, y_train, y_test, k_values)

plt.figure(figsize=(6,4))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("k vs Testing Accuracy")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Testing Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()

