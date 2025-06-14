import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load preprocessed data
X = np.load("X_preprocessed.npy")
y = np.load("y_labels.npy")

# Split into temp + test sets (60% temp, 40% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# Use 50% labeled to start
X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Initial labeled size: {len(y_labeled)}")
print(f"Initial unlabeled size: {len(y_unlabeled_true)}")

# Scale features
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)
X_test_scaled = scaler.transform(X_test)

# Optional: PCA to reduce noise / dimensionality
pca = PCA(n_components=8)
X_labeled = pca.fit_transform(X_labeled)
X_unlabeled = pca.transform(X_unlabeled)
X_test_scaled = pca.transform(X_test_scaled)

# Stronger model
model = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)

confidence_threshold = 0.8
threshold_decay = 0.995
min_threshold = 0.6
max_iter = 200

best_acc = 0
no_improve_count = 0
early_stop_limit = 10

for i in range(max_iter):
    print(f"\nIteration {i+1} - Confidence Threshold: {confidence_threshold:.3f}")

    model.fit(X_labeled, y_labeled)

    final_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, final_preds)
    print(f"Test accuracy at iteration {i+1}: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        no_improve_count = 0
    else:
        no_improve_count += 1
    if no_improve_count >= early_stop_limit:
        print("Early stopping due to no improvement.")
        break

    probs = model.predict_proba(X_unlabeled)
    preds = model.predict(X_unlabeled)
    confidences = np.max(probs, axis=1)

    high_conf_mask = confidences >= confidence_threshold
    indices = np.where(high_conf_mask)[0]

    if len(indices) == 0:
        if confidence_threshold > min_threshold:
            confidence_threshold = max(confidence_threshold - 0.05, min_threshold)
            print(f"No confident samples, lowering threshold to {confidence_threshold:.2f} and continuing.")
            continue
        else:
            print("No confident samples and threshold at minimum. Stopping semi-supervised learning.")
            break

    X_labeled = np.vstack([X_labeled, X_unlabeled[indices]])
    y_labeled = np.concatenate([y_labeled, preds[indices]])

    X_unlabeled = np.delete(X_unlabeled, indices, axis=0)

    print(f"Added {len(indices)} samples; labeled set size is now {len(y_labeled)}")

    confidence_threshold = max(confidence_threshold * threshold_decay, min_threshold)

    if len(X_unlabeled) == 0:
        print("All unlabeled samples have been labeled.")
        break

# Final evaluation
final_preds = model.predict(X_test_scaled)
acc = accuracy_score(y_test, final_preds)
print(f"\nFinal accuracy on test set: {acc:.4f}")
print(classification_report(y_test, final_preds))
print("Confusion matrix:")
print(confusion_matrix(y_test, final_preds))

with open("results.txt", "w") as f:
    f.write(f"Final accuracy on test set: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, final_preds))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(confusion_matrix(y_test, final_preds)))
