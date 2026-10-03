import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Project paths
BASE_DIR = os.path.expanduser(
    "~/brain_robot_ws/src/brain_robot_controller/data/hapt_dataset"
)

TRAIN_X = os.path.join(BASE_DIR, "Train", "X_train.txt")
TRAIN_Y = os.path.join(BASE_DIR, "Train", "y_train.txt")
TEST_X = os.path.join(BASE_DIR, "Test", "X_test.txt")
TEST_Y = os.path.join(BASE_DIR, "Test", "y_test.txt")

OUTPUT_DIR = os.path.expanduser(
    "~/brain_robot_ws/src/brain_robot_controller/data/processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading HAPT dataset...")

# Load data
X_train = np.loadtxt(TRAIN_X)
y_train = np.loadtxt(TRAIN_Y, dtype=int)

X_test = np.loadtxt(TEST_X)
y_test = np.loadtxt(TEST_Y, dtype=int)

print(f"Training samples: {X_train.shape}")
print(f"Testing samples:  {X_test.shape}")

# Convert labels from 1-12 to 0-11
y_train = y_train - 1
y_test = y_test - 1

# Normalize features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# Save normalization parameters
np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(OUTPUT_DIR, "scaler_scale.npy"), scaler.scale_)

print("\nPreprocessing complete.")
print(f"Features: {X_train.shape[1]}")
print(f"Classes:  {len(np.unique(y_train))}")
print(f"Processed data saved to: {OUTPUT_DIR}")

print("\nClass distribution:")
for class_id in range(12):
    count = np.sum(y_train == class_id)
    print(f"Class {class_id + 1:2d}: {count}")
