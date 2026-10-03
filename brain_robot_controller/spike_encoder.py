import os
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# BRAIN-INSPIRED ROBOT CONTROL
# ANN OUTPUT -> SPIKE GENERATION
#
# Rate-based spike encoding:
# Higher ANN probability -> higher spike frequency
# ============================================================


BASE_DIR = os.path.expanduser(
    "~/brain_robot_ws/src/brain_robot_controller"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# SPIKE ENCODER
# ============================================================

class RateSpikeEncoder:

    def __init__(
        self,
        simulation_time=1.0,
        time_step=0.001,
        max_rate=100.0,
        random_seed=42
    ):
        """
        Rate-based Poisson spike encoder.

        simulation_time:
            Total simulation time in seconds.

        time_step:
            Time resolution in seconds.

        max_rate:
            Maximum spike frequency in Hz.
        """

        self.simulation_time = simulation_time
        self.time_step = time_step
        self.max_rate = max_rate

        self.num_steps = int(
            simulation_time / time_step
        )

        self.rng = np.random.default_rng(
            random_seed
        )


    def encode(self, probabilities):

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        # Keep probabilities inside valid range
        probabilities = np.clip(
            probabilities,
            0.0,
            1.0
        )

        num_neurons = len(probabilities)

        # Convert ANN probabilities to firing rates
        firing_rates = (
            probabilities * self.max_rate
        )

        # Probability of a spike during one timestep
        spike_probability = (
            firing_rates * self.time_step
        )

        # Generate stochastic spikes
        random_values = self.rng.random(
            (self.num_steps, num_neurons)
        )

        spikes = (
            random_values < spike_probability
        ).astype(np.int8)

        return spikes, firing_rates


# ============================================================
# LOAD ANN
# ============================================================

print("=" * 60)
print("ANN -> SPIKE ENCODING")
print("=" * 60)

model_path = os.path.join(
    MODEL_DIR,
    "ann_model.joblib"
)

model = joblib.load(model_path)

print("\nANN model loaded.")

print("Input features:", model.n_features_in_)
print("Output classes:", model.n_outputs_)

print("Activation:", model.activation)


# ============================================================
# LOAD TEST SAMPLE
# ============================================================

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)


# Select one real-world HAPT sample
sample_index = 0

sample = X_test[
    sample_index:sample_index + 1
]

true_class = int(
    y_test[sample_index]
) + 1


# ============================================================
# ANN PREDICTION
# ============================================================

probabilities = model.predict_proba(
    sample
)[0]

predicted_class = int(
    np.argmax(probabilities)
) + 1

confidence = float(
    np.max(probabilities)
)


print("\nReal HAPT sample:")
print("True activity class     :", true_class)

print("ANN predicted class     :", predicted_class)

print(
    "ANN confidence          :",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# PRINT ANN OUTPUT
# ============================================================

print("\nANN output probabilities:")

for i, probability in enumerate(
    probabilities
):

    print(
        f"Class {i + 1:2d}: "
        f"{probability:.4f}"
    )


# ============================================================
# CREATE SPIKES
# ============================================================

encoder = RateSpikeEncoder(
    simulation_time=1.0,
    time_step=0.001,
    max_rate=100.0,
    random_seed=42
)

spikes, firing_rates = encoder.encode(
    probabilities
)


# ============================================================
# SPIKE STATISTICS
# ============================================================

spike_counts = spikes.sum(axis=0)

print("\nSpike generation completed.")

print(
    "Simulation time:",
    encoder.simulation_time,
    "seconds"
)

print(
    "Time step:",
    encoder.time_step,
    "seconds"
)

print(
    "Total simulation steps:",
    encoder.num_steps
)

print("\nSpike statistics:")

for i in range(len(probabilities)):

    print(
        f"Class {i + 1:2d}: "
        f"{spike_counts[i]:3d} spikes | "
        f"rate = {firing_rates[i]:6.2f} Hz"
    )


# ============================================================
# SPIKE RASTER GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

for neuron in range(
    spikes.shape[1]
):

    spike_times = np.where(
        spikes[:, neuron] == 1
    )[0] * encoder.time_step

    plt.scatter(
        spike_times,
        np.full_like(
            spike_times,
            neuron + 1
        ),
        s=8
    )


plt.xlabel("Time (seconds)")
plt.ylabel("ANN Output Neuron / Activity Class")

plt.title(
    "ANN Output Spike Raster - HAPT Sample"
)

plt.yticks(
    range(1, len(probabilities) + 1)
)

plt.grid(True, alpha=0.3)

plt.tight_layout()


raster_path = os.path.join(
    PLOT_DIR,
    "spike_raster.png"
)

plt.savefig(
    raster_path,
    dpi=200
)

plt.close()


print("\nSpike raster saved:")
print(raster_path)


# ============================================================
# FIRING RATE GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    range(1, len(firing_rates) + 1),
    firing_rates
)

plt.xlabel("Activity Class")

plt.ylabel("Firing Rate (Hz)")

plt.title(
    "ANN Probability to Spike Firing Rate"
)

plt.xticks(
    range(1, len(firing_rates) + 1)
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

plt.tight_layout()


rate_path = os.path.join(
    PLOT_DIR,
    "spike_firing_rates.png"
)

plt.savefig(
    rate_path,
    dpi=200
)

plt.close()


print("Firing-rate graph saved:")
print(rate_path)


# ============================================================
# SAVE SPIKE DATA
# ============================================================

spike_data_path = os.path.join(
    DATA_DIR,
    "spike_data.npz"
)

np.savez(
    spike_data_path,
    spikes=spikes,
    probabilities=probabilities,
    firing_rates=firing_rates,
    true_class=true_class,
    predicted_class=predicted_class
)

print("\nSpike data saved:")
print(spike_data_path)


print("\n" + "=" * 60)
print("SPIKE ENCODING COMPLETE")
print("=" * 60)
