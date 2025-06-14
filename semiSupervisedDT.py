import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load your preprocessed data
features = np.load("X_preprocessed.npy")
labels = np.load("y_labels.npy")

# First split: train+validation (60%) and test (40%)
X_temp, X_test, y_temp, y_test = train_test_split(
    features, labels, test_size=0.4, stratify=labels, random_state=42
)

# From training data: keep 50% labeled, 50% unlabeled (simulate semi-supervised setup)
X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Starting with {len(y_labeled)} labeled samples")
print(f"Remaining {len(y_unlabeled_true)} samples are unlabeled")

# Normalize features
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)
X_test_scaled = scaler.transform(X_test)

# Optional dimensionality reduction for better generalization
pca = PCA(n_components=8)
X_labeled = pca.fit_transform(X_labeled)
X_unlabeled = pca.transform(X_unlabeled)
X_test_scaled = pca.transform(X_test_scaled)

# Initialize the model
model = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)

# Semi-supervised parameters
confidence_threshold = 0.8
threshold_decay = 0.995
min_confidence = 0.6
max_iterations = 200

# Early stopping
best_accuracy = 0
no_improvement_counter = 0
early_stop_patience = 10

# Start semi-supervised learning loop
for iteration in range(max_iterations):
    print(f"\nIteration {iteration + 1} - Threshold: {confidence_threshold:.3f}")
    
    model.fit(X_labeled, y_labeled)
    test_predictions = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Early stopping logic
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        
    if no_improvement_counter >= early_stop_patience:
        print("No improvement. Stopping early.")
        break

    # Predict on unlabeled data
    probabilities = model.predict_proba(X_unlabeled)
    predicted_labels = model.predict(X_unlabeled)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Select high-confidence predictions
    confident_mask = confidence_scores >= confidence_threshold
    confident_indices = np.where(confident_mask)[0]
    
    if len(confident_indices) == 0:
        if confidence_threshold > min_confidence:
            confidence_threshold = max(confidence_threshold - 0.05, min_confidence)
            print(f"No confident samples. Lowering threshold to {confidence_threshold:.2f}.")
            continue
        else:
            print("No confident samples and minimum threshold reached. Stopping.")
            break

    # Move high-confidence samples from unlabeled to labeled set
    X_labeled = np.vstack([X_labeled, X_unlabeled[confident_indices]])
    y_labeled = np.concatenate([y_labeled, predicted_labels[confident_indices]])
    X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)

    print(f"Added {len(confident_indices)} new samples. Total labeled samples: {len(y_labeled)}")

    # Gradually reduce threshold
    confidence_threshold = max(confidence_threshold * threshold_decay, min_confidence)
    
    if len(X_unlabeled) == 0:
        print("All unlabeled data has been used.")
        break

# Final evaluation
final_predictions = model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"\nFinal test accuracy: {final_accuracy:.4f}")
print(classification_report(y_test, final_predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, final_predictions))

# Save results to file
with open("results.txt", "w") as file:
    file.write(f"Final test accuracy: {final_accuracy:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_report(y_test, final_predictions))
    file.write("\nConfusion Matrix:\n")
    file.write(np.array2string(confusion_matrix(y_test, final_predictions)))
