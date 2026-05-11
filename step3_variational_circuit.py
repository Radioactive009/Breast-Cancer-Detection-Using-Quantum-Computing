"""
============================================================
  STEP 3: Variational Quantum Circuit (Trainable Layer)
  Quantum Machine Learning Project
  Inspired by: "Circuit-centric quantum classifiers" (Schuld et al.)
============================================================

WHAT IS A VARIATIONAL QUANTUM CIRCUIT (VQC)?
--------------------------------------------
A VQC is a quantum circuit that contains "trainable" parameters (theta). 
These parameters are adjusted by a classical optimizer to minimize a cost 
function, much like weights in a classical Neural Network.

In the paper "Circuit-centric quantum classifiers", the model is defined as:
    f(x, theta) = Measurement( W(theta) * U(x) |0> )

Where:
  - U(x): The Feature Map (Step 2: Data Encoding).
  - W(theta): The Model Circuit (Step 3: This script).

THE TRAINABLE LAYER (W):
------------------------
This layer consists of parameterized rotation gates (RX, RY, RZ). By changing 
the rotation angles (theta), we change how the encoded data is transformed 
in the Hilbert space.

THE ENTANGLEMENT LAYER:
-----------------------
Entanglement (using CNOT gates) allows qubits to interact. In a classifier, 
this is crucial because it allows the model to capture correlations between 
different input features that a single-qubit model would miss. This increases 
the "expressibility" of the quantum model.
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
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

print("=" * 60)
print("  [OK]  Step 3 Environment Ready!")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.1: PREPARE DATA (Same as Step 2)
# ─────────────────────────────────────────────────────────────────────────────

iris = datasets.load_iris()
X = iris.data[iris.target < 2][:, :2]  # 2 classes, 2 features
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_norm = scaler.fit_transform(X)

# Select one sample for the demonstration
sample_idx = 0
features = X_norm[sample_idx]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2: BUILD THE FULL VARIATIONAL CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Constructing the Full Model Architecture...")

num_qubits = 2
qc = QuantumCircuit(num_qubits, num_qubits)

# --- LAYER 1: DATA ENCODING (U_x) ---
# This encodes our classical features into the quantum state.
qc.barrier() # Barrier for visual separation
qc.ry(features[0], 0)
qc.ry(features[1], 1)
qc.label = "Data Encoding"

# --- LAYER 2: VARIATIONAL LAYER (W_theta) ---
# These are the "trainable weights" of our Quantum Neural Network.
# We use RX and RY gates to rotate the state in multiple directions.
theta = ParameterVector('θ', length=4) 

qc.barrier()
qc.rx(theta[0], 0)
qc.ry(theta[1], 0)
qc.rx(theta[2], 1)
qc.ry(theta[3], 1)

# --- LAYER 3: ENTANGLEMENT LAYER ---
# CNOT gates connect the qubits, allowing for non-linear feature correlation.
qc.barrier()
qc.cx(0, 1)

# --- LAYER 4: MEASUREMENT ---
qc.barrier()
qc.measure([0, 1], [0, 1])

print("  [OK] Variational circuit architecture defined.")
print(f"  - Parameters count: {qc.num_parameters}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.3: BIND PARAMETERS (Initial Random Weights)
# ─────────────────────────────────────────────────────────────────────────────

# In a real training loop, these weights would be updated by an optimizer.
# For this step, we initialize them with random values.
np.random.seed(42)
initial_weights = np.random.uniform(0, 2 * np.pi, len(theta))

# Bind the parameters to the circuit
qc_bound = qc.assign_parameters({theta: initial_weights})

print("\n[*] Parameters Bound (Initial Weights):")
for i, val in enumerate(initial_weights):
    print(f"  - \u03b8[{i}] : {val:.4f} rad")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.4: SIMULATION & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[DIAGRAM] Circuit Diagram (text form):")
print("-" * 70)
print(qc_bound.draw(output="text"))
print("-" * 70)

print("\n[SIM] Running simulation on AerSimulator...")
simulator = AerSimulator()
compiled_circuit = transpile(qc_bound, simulator)

SHOTS = 2048
counts = simulator.run(compiled_circuit, shots=SHOTS).result().get_counts()

# --- Plotting ---
print("\n[PLOT] Generating visualizations...")

fig = plt.figure(figsize=(16, 9), facecolor="#0f0f1a")
fig.suptitle(
    "Step 3: Variational Quantum Circuit (VQC)\n"
    "Data Encoding (U) + Trainable Weights (W) + Entanglement",
    fontsize=14, color="white", fontweight="bold", y=0.97
)

gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.3)

# Panel 1: The Circuit (MPL)
ax1 = fig.add_subplot(gs[0, :]) # Top row spans both columns
ax1.set_facecolor("#1a1a2e")
ax1.set_title("Full Model Circuit Architecture", color="#a78bfa", fontsize=12)
ax1.axis("off")

circuit_fig = qc.draw(
    output="mpl", 
    style={
        "backgroundcolor": "#1a1a2e", "textcolor": "#e2e8f0",
        "gatefacecolor": "#7c3aed", "gatetextcolor": "#ffffff",
        "linecolor": "#64748b", "creglinecolor": "#38bdf8"
    }
)
circuit_fig.savefig("temp_circuit.png", bbox_inches="tight", facecolor="#1a1a2e")
img = plt.imread("temp_circuit.png")
ax1.imshow(img)
plt.close(circuit_fig)

# Panel 2: Measurement Histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#1a1a2e")
ax2.set_title("Initial Prediction Distribution", color="#a78bfa", fontsize=12)

states = sorted(counts.keys())
values = [counts[s] for s in states]
ax2.bar(states, values, color="#38bdf8", edgecolor="white", alpha=0.8)
ax2.tick_params(colors="white")
ax2.set_ylabel("Counts", color="#94a3b8")

# Panel 3: Technical Explanation
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0f172a")
ax3.axis("off")

explanation_text = (
    "HOW THIS ACTS AS A NEURAL NETWORK:\n\n"
    "1. INPUT: Features are rotated (RY) into qubits.\n"
    "2. WEIGHTS: RX/RY gates transform the state.\n"
    "   Changing \u03b8 changes the classification boundary.\n"
    "3. INTERACTION: CNOT creates entanglement,\n"
    "   allowing features to 'talk' to each other.\n"
    "4. OUTPUT: Measurement counts determine the\n"
    "   probability of class 0 vs class 1.\n\n"
    "In Step 4, we will use a classical optimizer to\n"
    "find the best \u03b8 values to match the Iris labels."
)
ax3.text(0.05, 0.5, explanation_text, color="#e2e8f0", fontsize=10, 
         va="center", family="monospace",
         bbox=dict(boxstyle="round,pad=1", facecolor="#1e293b", edgecolor="#334155"))

plt.savefig("step3_variational_circuit_output.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
print("  [OK] Plot saved as 'step3_variational_circuit_output.png'")

plt.show(block=False)
plt.pause(3)
plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  STEP 3 SUMMARY: Variational Quantum Circuit")
print("=" * 60)
print("""
  1. Full Pipeline : Encoding -> Variational -> Entanglement -> Measure.
  2. Parameters    : Created a ParameterVector for 4 trainable weights (\u03b8).
  3. Entanglement  : Added a CNOT gate to enable complex correlations.
  4. Simulation    : Simulated the circuit with random initial weights.

  CONNECTION TO "CIRCUIT-CENTRIC QUANTUM CLASSIFIERS":
  - This script implements W(\u03b8), the model circuit described in the paper.
  - The combination of single-qubit rotations (RX, RY) and CNOT gates is
    a standard "Ansatz" that provides high expressibility.
  - We have now successfully built the 'Quantum Neural Network' structure.

  NEXT STEP -> Training! (Using an Optimizer to update \u03b8).
""")
print("=" * 60)
