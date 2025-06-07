import os
import numpy as np
from PIL import Image


base_path = r"C:\Users\camus\Desktop\mustafakaya\Concordia\Meng Software Engineering\Comp6721\Comp6721_Project\Comp6721_Project_Dataset\Test"
categories = ["museum-indoor", "library-indoor", "shopping_mall-indoor"]
image_size = (64, 64)

X = []
y = []

# Loop through each category
for label, category in enumerate(categories):
    folder_path = os.path.join(base_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize(image_size)
            img_array = np.array(img).flatten() / 255.0
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save for later use
np.save("X_preprocessed.npy", X)
np.save("y_labels.npy", y)
print("✅ Preprocessing complete. Saved to 'X_preprocessed.npy' and 'y_labels.npy'.")
