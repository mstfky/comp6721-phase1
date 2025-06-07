# COMP6721 Phase I - Venue Classification Project

## 🧠 Project Overview

This project tackles a venue classification task using machine learning models on image data. The goal is to classify indoor scenes into one of three categories:

- Museum
- Library
- Shopping Mall

We implemented and evaluated the following models:

- **SVM (Support Vector Machine)** - Supervised
- **Random Forest** - Supervised
- **Decision Tree (Semi-Supervised)** - Iterative pseudo-labeling approach

---

## 📁 Project Structure

```
├── resizeImages.py                # Image loading, resizing, preprocessing
├── train_models.py                # Train and evaluate SVM & Random Forest
├── semi_supervised_dt.py         # Semi-supervised Decision Tree implementation
├── produceResultsOfComparision.py# Accuracy & F1 bar charts
├── X_preprocessed.npy            # Preprocessed image features
├── y_labels.npy                  # Corresponding labels
├── model_accuracy_comparison.png # Plot comparing model accuracies
├── model_f1_comparison.png       # Plot comparing F1 scores
```

---

## 🛠 How to Run

### 1. Preprocess the images:
```bash
python resizeImages.py
```

### 2. Train and evaluate supervised models:
```bash
python train_models.py
```

### 3. Run semi-supervised decision tree:
```bash
python semi_supervised_dt.py
```

### 4. Generate comparison graphs:
```bash
python produceResultsOfComparision.py
```

---

## 🔍 Dataset

We used a subset of the MIT Places2 dataset:

- Link: http://places2.csail.mit.edu/
- Classes: `museum-indoor`, `library-indoor`, `shopping_mall-indoor`
- Images were resized to 64x64 and flattened for classic ML models.

---

## 📊 Evaluation Metrics

All models were evaluated using the same test set. Metrics include:

- Accuracy
- Precision, Recall
- F1-Score (Macro + Per Class)
- Confusion Matrix

Visual comparison included in:
- `model_accuracy_comparison.png`
- `model_f1_comparison.png`

---

## 📚 References

- scikit-learn documentation: https://scikit-learn.org/
- MIT Places2 dataset: http://places2.csail.mit.edu/
- COMP6721 Summer 2025 Project Outline