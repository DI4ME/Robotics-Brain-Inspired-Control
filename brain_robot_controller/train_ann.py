import os
import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.expanduser(
    "~/brain_robot_ws/src/brain_robot_controller"
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# LOAD PREPROCESSED HAPT DATA
# ============================================================

print("=" * 60)
print("BRAIN-INSPIRED ROBOT CONTROL - ANN TRAINING")
print("=" * 60)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"\nOriginal training data : {X_train.shape}")
print(f"Testing data           : {X_test.shape}")
print(f"Number of classes      : {len(np.unique(y_train))}")


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print("\nBalancing training classes...")

classes, counts = np.unique(y_train, return_counts=True)

print("\nOriginal class distribution:")
for c, count in zip(classes, counts):
    print(f"Class {c + 1:2d}: {count}")

# Use the median class size as the target.
# This prevents extreme oversampling of very small classes.
target_count = int(np.median(counts))

rng = np.random.default_rng(42)

X_balanced_parts = []
y_balanced_parts = []

for c in classes:

    class_indices = np.where(y_train == c)[0]

    if len(class_indices) >= target_count:
        selected = rng.choice(
            class_indices,
            size=target_count,
            replace=False
        )
    else:
        selected = rng.choice(
            class_indices,
            size=target_count,
            replace=True
        )

    X_balanced_parts.append(X_train[selected])
    y_balanced_parts.append(y_train[selected])


X_train_balanced = np.concatenate(X_balanced_parts, axis=0)
y_train_balanced = np.concatenate(y_balanced_parts, axis=0)

# Shuffle balanced data
shuffle_indices = rng.permutation(len(y_train_balanced))

X_train_balanced = X_train_balanced[shuffle_indices]
y_train_balanced = y_train_balanced[shuffle_indices]

print("\nBalanced training data:")
print(f"Samples : {X_train_balanced.shape[0]}")
print(f"Features: {X_train_balanced.shape[1]}")


# ============================================================
# CREATE ANN
# ============================================================

print("\nCreating ANN...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)

print("\nANN architecture:")
print("Input layer  : 561 neurons")
print("Hidden layer : 128 neurons - ReLU")
print("Hidden layer : 64 neurons  - ReLU")
print("Output layer : 12 classes")


# ============================================================
# TRAIN ANN
# ============================================================

print("\nStarting ANN training...")
print("-" * 60)

model.fit(X_train_balanced, y_train_balanced)

print("-" * 60)
print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating ANN...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")

plt.title("HAPT ANN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()

plt.xticks(range(12), range(1, 13))
plt.yticks(range(12), range(1, 13))

plt.tight_layout()

confusion_path = os.path.join(
    PLOT_DIR,
    "ann_confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=200)
plt.close()

print(f"\nConfusion matrix saved:")
print(confusion_path)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    model.loss_curve_,
    label="ANN Training Loss"
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    PLOT_DIR,
    "ann_training_loss.png"
)

plt.savefig(loss_path, dpi=200)
plt.close()

print(f"Training loss graph saved:")
print(loss_path)


# ============================================================
# SAVE COMPLETE ANN MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "ann_model.joblib"
)

joblib.dump(model, model_path)

print(f"\nComplete ANN model saved:")
print(model_path)


# ============================================================
# SAVE ANN WEIGHTS AND BIASES
# ============================================================

weights_path = os.path.join(
    MODEL_DIR,
    "ann_weights.npz"
)

np.savez(
    weights_path,

    # Layer 1
    W1=model.coefs_[0],
    b1=model.intercepts_[0],

    # Layer 2
    W2=model.coefs_[1],
    b2=model.intercepts_[1],

    # Output layer
    W3=model.coefs_[2],
    b3=model.intercepts_[2]
)

print("\nANN weights and biases saved:")
print(weights_path)

print("\nWeight shapes:")
print("W1:", model.coefs_[0].shape)
print("b1:", model.intercepts_[0].shape)
print("W2:", model.coefs_[1].shape)
print("b2:", model.intercepts_[1].shape)
print("W3:", model.coefs_[2].shape)
print("b3:", model.intercepts_[2].shape)


# ============================================================
# SAVE ANN INFORMATION
# ============================================================

info_path = os.path.join(
    MODEL_DIR,
    "ann_info.txt"
)

with open(info_path, "w") as f:

    f.write("Brain-Inspired Robotic Control System\n")
    f.write("ANN Model Information\n")
    f.write("=" * 50 + "\n\n")

    f.write("Dataset: HAPT\n")
    f.write("Input features: 561\n")
    f.write("Output classes: 12\n\n")

    f.write("Architecture:\n")
    f.write("561 -> 128 -> 64 -> 12\n\n")

    f.write("Hidden activation: ReLU\n")
    f.write("Optimizer: Adam\n")
    f.write("Loss: Cross Entropy\n")
    f.write(f"Test accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Training iterations: {model.n_iter_}\n")

print("\nANN information saved:")
print(info_path)

print("\n" + "=" * 60)
print("ANN TRAINING PIPELINE COMPLETE")
print("=" * 60)
