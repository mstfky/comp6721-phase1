import numpy as np
import matplotlib.pyplot as plt

# === Load results ===
supervised_results = np.load("final_model_results.npy", allow_pickle=True).item()
semi_supervised_results = np.load("semi_supervised_results.npy", allow_pickle=True).item()

# === Combine results ===
combined_results = {**supervised_results, **semi_supervised_results}

# === Prepare data ===
models = list(combined_results.keys())
accuracy = [combined_results[m]["accuracy"] for m in models]
macro_f1 = [combined_results[m]["f1"] for m in models]

# === Accuracy Plot ===
plt.figure(figsize=(6, 4))
plt.bar(models, accuracy, color=['goldenrod', 'coral', 'crimson'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# === F1-Score Plot ===
plt.figure(figsize=(6, 4))
plt.bar(models, macro_f1, color=['goldenrod', 'coral', 'crimson'])
plt.title("Model Macro F1-Score Comparison")
plt.ylabel("Macro F1-Score")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("model_f1_comparison.png")
plt.show()
