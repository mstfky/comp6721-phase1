import matplotlib.pyplot as plt

# === Your actual results ===
models = ['SVM', 'Random Forest', 'Semi-Supervised DT']
accuracy = [0.53, 0.52, 0.35]
macro_f1 = [0.53, 0.52, 0.34]

# === Accuracy Bar Chart ===
plt.figure(figsize=(6, 4))
plt.bar(models, accuracy, color=['goldenrod', 'coral', 'crimson'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")  # Saves image in same folder
plt.show()

# === F1-Score Bar Chart ===
plt.figure(figsize=(6, 4))
plt.bar(models, macro_f1, color=['goldenrod', 'coral', 'crimson'])
plt.title("Model Macro F1-Score Comparison")
plt.ylabel("Macro F1-Score")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("model_f1_comparison.png")
plt.show()
