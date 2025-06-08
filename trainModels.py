import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Step 1: Load preprocessed data
X = np.load("X_preprocessed.npy")
y = np.load("y_labels.npy")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("✅ Data loaded and split successfully.")
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# --- Random Forest Hyperparameter Tuning ---
print("\n🌲 Random Forest Hyperparameter Tuning")
depth_values = [5, 10, 20]
best_rf_score = 0
best_rf_model = None

print("Depth\tAccuracy\tF1-score")
for depth in depth_values:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')

    print(f"{depth}\t{acc:.4f}\t\t{f1:.4f}")

    if f1 > best_rf_score:
        best_rf_score = f1
        best_rf_model = rf

print("\n✅ Best Random Forest model selected.")


# --- SVM Hyperparameter Tuning ---
print("\n🌀 SVM Hyperparameter Tuning")
C_values = [0.1, 1, 10]
best_svm_score = 0
best_svm_model = None

print("C\tAccuracy\tF1-score")
for C in C_values:
    svm = SVC(C=C, kernel='rbf', gamma='scale')
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')

    print(f"{C}\t{acc:.4f}\t\t{f1:.4f}")

    if f1 > best_svm_score:
        best_svm_score = f1
        best_svm_model = svm

print("\n✅ Best SVM model selected.")

# --- Final Evaluation Reports ---
# Evaluate best SVM
print("\n📊 Final Evaluation: Best SVM Model")
svm_preds = best_svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

# Evaluate best RF
print("\n📊 Final Evaluation: Best Random Forest Model")
rf_preds = best_rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))