import os
import numpy as np
import joblib
import matplotlib.pyplot as plt


class TiO2Memristor:
    """
    Simplified voltage-driven TiO2 memristor model.

    The internal state w represents the normalized conductive
    filament/doped-region state:

        0 <= w <= 1

    w = 0  -> high resistance / low conductance
    w = 1  -> low resistance / high conductance
    """

    def __init__(
        self,
        g_on=1e-3,
        g_off=1e-5,
        voltage_threshold=0.20,
        k_positive=1.2,
        k_negative=1.2,
        initial_state=0.05,
    ):
        self.g_on = g_on
        self.g_off = g_off
        self.voltage_threshold = voltage_threshold
        self.k_positive = k_positive
        self.k_negative = k_negative
        self.w = initial_state

    def conductance(self):
        return self.g_off + self.w * (self.g_on - self.g_off)

    def step(self, voltage, dt):
        if voltage > self.voltage_threshold:
            dw = (
                self.k_positive
                * (voltage - self.voltage_threshold)
                * (1.0 - self.w)
            )

        elif voltage < -self.voltage_threshold:
            dw = (
                self.k_negative
                * (voltage + self.voltage_threshold)
                * self.w
            )

        else:
            # Small natural relaxation toward the previous state.
            dw = 0.0

        self.w += dw * dt
        self.w = float(np.clip(self.w, 0.0, 1.0))

        current = self.conductance() * voltage

        return self.w, self.conductance(), current


def main():

    package_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(
        package_root, "models", "ann_model.joblib"
    )

    x_test_path = os.path.join(
        package_root, "data", "processed", "X_test.npy"
    )

    spike_path = os.path.join(
        package_root, "data", "processed", "spike_data.npz"
    )

    plots_dir = os.path.join(package_root, "plots")
    data_dir = os.path.join(package_root, "data", "processed")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load trained ANN
    # ---------------------------------------------------------

    model = joblib.load(model_path)
    x_test = np.load(x_test_path)

    sample = x_test[0].reshape(1, -1)

    probabilities = model.predict_proba(sample)[0]

    predicted_class = int(np.argmax(probabilities))
    dominant_rate = float(probabilities[predicted_class] * 100.0)

    print("ANN model loaded.")
    print(f"Predicted activity class: {predicted_class + 1}")
    print(f"Dominant ANN firing rate: {dominant_rate:.2f} Hz")

    # ---------------------------------------------------------
    # Generate spike train
    # ---------------------------------------------------------

    simulation_time = 1.0
    dt = 0.001

    time = np.arange(
        0.0,
        simulation_time,
        dt
    )

    rng = np.random.default_rng(42)

    spike_probability = (
        dominant_rate * dt
    )

    spikes = (
        rng.random(len(time)) < spike_probability
    ).astype(float)

    # ---------------------------------------------------------
    # TiO2 memristor
    # ---------------------------------------------------------

    memristor = TiO2Memristor(
        g_on=1e-3,
        g_off=1e-5,
        voltage_threshold=0.20,
        k_positive=1.2,
        k_negative=1.2,
        initial_state=0.05,
    )

    spike_voltage = 0.50

    voltage = spikes * spike_voltage

    states = []
    conductances = []
    currents = []

    for v in voltage:

        state, conductance, current = memristor.step(
            v,
            dt
        )

        states.append(state)
        conductances.append(conductance)
        currents.append(current)

    states = np.asarray(states)
    conductances = np.asarray(conductances)
    currents = np.asarray(currents)

    # ---------------------------------------------------------
    # Save numerical results
    # ---------------------------------------------------------

    np.savez(
        os.path.join(
            data_dir,
            "tio2_memristor_results.npz"
        ),
        time=time,
        voltage=voltage,
        spikes=spikes,
        state=states,
        conductance=conductances,
        current=currents,
        predicted_class=predicted_class,
        firing_rate=dominant_rate,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("TiO2 memristor simulation completed.")
    print(f"Initial state       : {states[0]:.6f}")
    print(f"Final state         : {states[-1]:.6f}")
    print(f"Minimum state       : {states.min():.6f}")
    print(f"Maximum state       : {states.max():.6f}")
    print(f"Initial conductance : {conductances[0]:.8f} S")
    print(f"Final conductance   : {conductances[-1]:.8f} S")
    print(f"Maximum current     : {np.max(np.abs(currents)):.8f} A")
    print(f"Number of spikes    : {int(spikes.sum())}")

    # ---------------------------------------------------------
    # Plot 1: input voltage
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        voltage
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("TiO2 Memristor Input Voltage")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "tio2_input_voltage.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 2: memristor state
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        states
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Memristor State (w)")
    plt.title("TiO2 Memristor Internal State")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "tio2_memristor_state.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 3: conductance
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        conductances
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Conductance (S)")
    plt.title("TiO2 Memristor Conductance")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "tio2_memristor_conductance.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Plot 4: current
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        currents
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.title("TiO2 Memristor Current Response")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "tio2_memristor_current.png"
        ),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # Combined response
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        states,
        label="Memristor state"
    )

    plt.plot(
        time,
        spikes,
        label="Input spikes"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized value")
    plt.title("ANN Spike Input and TiO2 Memristor Response")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "tio2_memristor_response.png"
        ),
        dpi=150
    )

    plt.close()

    print()
    print("Plots saved successfully.")


if __name__ == "__main__":
    main()
