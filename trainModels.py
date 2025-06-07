import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2.1: Load preprocessed data
X = np.load("X_preprocessed.npy")
y = np.load("y_labels.npy")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(" Data loaded and split successfully.")
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Step 2.2: Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
print("\n SVM model trained.")

# Step 2.2: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
print(" Random Forest model trained.")

# Step 2.3: Evaluate SVM
svm_preds = svm.predict(X_test)
print("\n SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

# Step 2.3: Evaluate Random Forest
rf_preds = rf.predict(X_test)
print("\n Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
