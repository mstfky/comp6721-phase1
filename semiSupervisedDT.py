import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load preprocessed data
X = np.load("X_preprocessed.npy")
y = np.load("y_labels.npy")

# Final test set (same as before)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Among remaining, split into 20% labeled, 80% unlabeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
    X_temp, y_temp, test_size=0.8, stratify=y_temp, random_state=42
)

print(f"Initial labeled size: {len(y_labeled)}")
print(f"Unlabeled size: {len(y_unlabeled_true)}")

# Initialize Decision Tree
dt = DecisionTreeClassifier()

# Start iterations
for i in range(10): #Optional safety limit "while True"
    print(f"\n Iteration {i+1}")

    # Train on labeled data
    dt.fit(X_labeled, y_labeled)

    # Predict on unlabeled
    probs = dt.predict_proba(X_unlabeled)
    preds = dt.predict(X_unlabeled)
    confidences = np.max(probs, axis=1)

    # High confidence threshold
    high_confidence_mask = confidences >= 0.85
    if not np.any(high_confidence_mask):
        print(" No high-confidence predictions found. Stopping.")
        break

    # Add confident samples to labeled set
    X_confident = X_unlabeled[high_confidence_mask]
    y_confident = preds[high_confidence_mask]

    X_labeled = np.vstack([X_labeled, X_confident])
    y_labeled = np.concatenate([y_labeled, y_confident])

    # Remove added samples from unlabeled set
    X_unlabeled = X_unlabeled[~high_confidence_mask]

    print(f"Added {len(y_confident)} new samples. New labeled size: {len(y_labeled)}")

    if len(X_unlabeled) == 0:
        print(" All unlabeled samples have been processed.")
        break

# Final model evaluation
final_dt = DecisionTreeClassifier()
final_dt.fit(X_labeled, y_labeled)
final_preds = final_dt.predict(X_test)

print("\n Final Evaluation on Test Set:")
print(classification_report(y_test, final_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))

from sklearn.metrics import accuracy_score, f1_score

# Extract metrics
dt_accuracy = accuracy_score(y_test, final_preds)
dt_f1 = f1_score(y_test, final_preds, average='macro')

# Store results for plotting
semi_supervised_results = {
    "Semi-Supervised DT": {
        "accuracy": dt_accuracy,
        "f1": dt_f1
    }
}

# Save to .npy file
np.save("semi_supervised_results.npy", semi_supervised_results)

# Print summary
print("\n✅ Final Scores Summary (Semi-Supervised DT):")
print(f"Accuracy = {dt_accuracy:.4f}")
print(f"Macro F1-Score = {dt_f1:.4f}")
