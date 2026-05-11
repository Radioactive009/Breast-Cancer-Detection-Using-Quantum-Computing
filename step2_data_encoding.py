"""
============================================================
  STEP 2: Quantum Data Encoding
  Quantum Machine Learning Project
  Inspired by: "Circuit-centric quantum classifiers" (Schuld et al.)
============================================================

WHY DATA ENCODING MATTERS:
--------------------------
To process classical data on a quantum computer, we must first map classical 
values (like pixel intensities or Iris measurements) into a quantum state.
This process is called "Quantum Data Encoding" or "State Preparation".

In the paper "Circuit-centric quantum classifiers", this is the first layer
of the Variational Quantum Classifier (VQC). It transforms our input features
into a high-dimensional Hilbert space where a quantum circuit can then find 
non-linear decision boundaries.

ANGLE ENCODING:
---------------
Angle encoding uses the features as rotation angles for quantum gates.
A feature value 'x' is encoded as:
    |psi(x)> = RY(x)|0> = cos(x/2)|0> + sin(x/2)|1>

- Advantage: Requires only 1 qubit per feature.
- Limitation: Features should be scaled to fit within [0, pi] or [0, 2pi].
"""

import sys
import io

# Force UTF-8 output on Windows (handles Qiskit's box-drawing characters)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# Qiskit components
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

print("=" * 60)
print("  [OK]  Step 2 Environment Ready!")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2.1: LOAD AND PREPROCESS DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Loading Iris Dataset...")

# 1. Load Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Filter for Binary Classification (Classes 0 and 1)
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# 3. Use only the first 2 features (Sepal length, Sepal width) for 2 qubits
X = X[:, :2]

# 4. Normalize Features to [0, pi] for Angle Encoding
# This ensures the features map cleanly to the rotation range of RY gates.
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_normalized = scaler.fit_transform(X)

# Select one sample to encode
sample_idx = 0
sample_features = X_normalized[sample_idx]
original_features = X[sample_idx]

print(f"  - Original Features (cm)  : {original_features}")
print(f"  - Normalized (for angles) : {sample_features}")
print(f"  - Target Class            : {y[sample_idx]} ({iris.target_names[y[sample_idx]]})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2.2: BUILD THE ENCODING CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Building Angle Encoding Circuit...")

# Create a circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply Angle Encoding using RY gates
# RY(theta) rotates the qubit around the Y-axis of the Bloch Sphere.
# feature[0] -> qubit 0
# feature[1] -> qubit 1
qc.ry(sample_features[0], 0)
qc.ry(sample_features[1], 1)

# Add measurements to see the results of our encoding
qc.measure([0, 1], [0, 1])

print("  [OK] Encoding circuit built.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2.3: DISPLAY THE CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[DIAGRAM] Circuit Diagram (text form):")
print("-" * 50)
print(qc.draw(output="text"))
print("-" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2.4: SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[SIM] Running simulation on AerSimulator...")

simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

SHOTS = 2048
result = simulator.run(compiled_circuit, shots=SHOTS).result()
counts = result.get_counts()

print(f"  [OK] Simulation complete! ({SHOTS} shots)")
print(f"\n[RESULTS] Measurement Counts:")
print(f"  {counts}")

# Calculate theoretical probabilities for comparison
# Prob(1) = sin(theta/2)^2
prob_0_is_1 = np.sin(sample_features[0] / 2)**2
prob_1_is_1 = np.sin(sample_features[1] / 2)**2

print(f"\n  Theoretical Probabilities for |1>:")
print(f"  - Qubit 0: {prob_0_is_1:.3f}")
print(f"  - Qubit 1: {prob_1_is_1:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2.5: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[PLOT] Generating visualizations...")

fig = plt.figure(figsize=(15, 8), facecolor="#0f0f1a")
fig.suptitle(
    "Step 2: Quantum Data Encoding (Angle Encoding)\n"
    "Mapping Classical Iris Features to Quantum States",
    fontsize=14,
    color="white",
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

# --- Panel 1: Circuit Visualization ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#1a1a2e")
ax1.set_title("Angle Encoding Circuit", color="#a78bfa", fontsize=12, pad=15)
ax1.axis("off")

# Render MPL circuit
circuit_fig = qc.draw(
    output="mpl",
    fold=-1,
    cregbundle=False,
    style={
        "backgroundcolor": "#1a1a2e",
        "textcolor": "#e2e8f0",
        "gatefacecolor": "#7c3aed",
        "gatetextcolor": "#ffffff",
        "subtextcolor": "#94a3b8",
        "linecolor": "#64748b",
        "creglinecolor": "#38bdf8",
        "measurearrowcolor": "#38bdf8",
    }
)

circuit_fig.set_facecolor("#1a1a2e")
circuit_fig.canvas.draw()

# Convert circuit fig to image for main plot
import io
buf = io.BytesIO()
circuit_fig.savefig(buf, format="png", facecolor="#1a1a2e", bbox_inches="tight", dpi=120)
buf.seek(0)
import matplotlib.image as mpimg
img = mpimg.imread(buf)
ax1.imshow(img)
plt.close(circuit_fig)

# --- Panel 2: Measurement Histogram ---
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#1a1a2e")
ax2.set_title("Measurement Outcome Distribution", color="#a78bfa", fontsize=12, pad=15)

states = sorted(counts.keys())
values = [counts[s] for s in states]
colors = ["#7c3aed", "#38bdf8", "#10b981", "#f59e0b"][:len(states)]

bars = ax2.bar(states, values, color=colors, width=0.4,
               edgecolor="#ffffff", linewidth=0.8, alpha=0.9)

for bar, val in zip(bars, values):
    pct = val / SHOTS * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (SHOTS * 0.02),
        f"{val}\n({pct:.1f}%)",
        ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )

ax2.set_xlabel("Quantum State Outcome", color="#94a3b8", fontsize=10)
ax2.set_ylabel(f"Frequency (Total Shots: {SHOTS})", color="#94a3b8", fontsize=10)
ax2.set_ylim(0, max(values) * 1.25)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#334155")

# Explanation box
explanation = (
    f"Features: [{original_features[0]:.1f}, {original_features[1]:.1f}] cm\n"
    f"Angles (rad): [{sample_features[0]:.2f}, {sample_features[1]:.2f}]\n"
    "Encoding: |psi> = RY(x1)|0> \u2297 RY(x2)|0>\n"
    "This creates a unique quantum signature for the data."
)
ax2.text(
    0.5, 0.05, explanation,
    transform=ax2.transAxes,
    ha="center", va="bottom",
    fontsize=9, color="#94a3b8",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#334155", alpha=0.8)
)

plt.savefig("step2_data_encoding_output.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
print("  [OK] Plot saved as 'step2_data_encoding_output.png'")

# Non-blocking show
plt.show(block=False)
plt.pause(3)
plt.close("all")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & CONNECTION TO THE PAPER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  STEP 2 SUMMARY: Quantum Data Encoding")
print("=" * 60)
print("""
  1. Data Prepared : Iris dataset filtered for binary classification.
  2. Features Scaled: Mapped features to [0, pi] range.
  3. Method Used   : Angle Encoding (using RY rotation gates).
  4. Quantum State : Classical data now exists as a qubit rotation angle.

  CONNECTION TO "CIRCUIT-CENTRIC QUANTUM CLASSIFIERS":
  - The paper views the entire quantum circuit as a model: f(x) = M(U(x)|0>).
  - This script implements U(x), the 'Feature Map'.
  - By rotating the qubits based on features, we lift the data into
    a complex Hilbert space.
  - In the next step, we will add 'Trainable Layers' (W) to transform this
    encoded state and find the best way to separate the classes.

  NEXT STEP -> Parameterized Quantum Circuit (PQC) & Trainable Weights.
""")
print("=" * 60)
