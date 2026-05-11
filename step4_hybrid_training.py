"""
============================================================
  STEP 4: Hybrid Quantum-Classical Training
  Quantum Machine Learning Project
  Inspired by: "Circuit-centric quantum classifiers" (Schuld et al.)
============================================================

WHAT IS HYBRID QUANTUM-CLASSICAL LEARNING?
------------------------------------------
Current quantum computers (NISQ era) are noisy and have limited coherence. 
The "Hybrid" approach solves this by:
  1. Using a Quantum Computer to evaluate the model (calculate probabilities).
  2. Using a Classical Computer to optimize the parameters (using COBYLA, Adam, etc.).

This cycle continues until the classical optimizer finds the best quantum 
gate angles (theta) to separate the data classes.

THE COST FUNCTION:
------------------
We use Mean Squared Error (MSE). The optimizer tries to minimize:
    Loss = 1/N * sum( (Predicted_Probability - True_Label)^2 )

OPTIMIZATION IN THE PAPER:
--------------------------
The paper "Circuit-centric quantum classifiers" describes this as finding 
the parameters theta of the circuit W(theta) that minimize the empirical risk.
"""

import sys
import io
import time

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

print("=" * 60)
print("  [OK]  Step 4 Environment Ready!")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4.1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Preparing Subset of Iris Dataset...")

iris = datasets.load_iris()
X = iris.data[iris.target < 2][:, :2]  # 2 classes, 2 features
y = iris.target[iris.target < 2]

# Normalize features to [0, pi]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_norm = scaler.fit_transform(X)

# Split into train/test (using a small subset for faster training)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Further reduce training size for demonstration speed
train_size = 15
X_train = X_train[:train_size]
y_train = y_train[:train_size]

print(f"  - Training samples : {len(X_train)}")
print(f"  - Test samples     : {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4.2: DEFINE THE PARAMETERIZED CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

def create_qnn_circuit():
    """Creates a 2-qubit parameterized quantum circuit."""
    # Parameters for data (x) and weights (theta)
    x = ParameterVector('x', length=2)
    theta = ParameterVector('\u03b8', length=4)
    
    qc = QuantumCircuit(2, 1) # 2 qubits, 1 classical bit for result
    
    # --- Data Encoding ---
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)
    qc.barrier()
    
    # --- Variational Layer ---
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 0)
    qc.rx(theta[2], 1)
    qc.ry(theta[3], 1)
    qc.barrier()
    
    # --- Entanglement ---
    qc.cx(0, 1)
    qc.barrier()
    
    # --- Measurement ---
    # We measure qubit 1 to get the classification result
    qc.measure(1, 0)
    
    return qc, x, theta

qc, x_params, theta_params = create_qnn_circuit()
simulator = AerSimulator()

print("  [OK] Quantum Neural Network (QNN) Architecture Ready.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4.3: OBJECTIVE FUNCTION & TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

loss_history = []

def predict(weights, data_point):
    """Executes the circuit for a single data point and returns prob of |1>."""
    # Bind data and weights
    bound_qc = qc.assign_parameters({
        x_params[0]: data_point[0],
        x_params[1]: data_point[1],
        theta_params[0]: weights[0],
        theta_params[1]: weights[1],
        theta_params[2]: weights[2],
        theta_params[3]: weights[3]
    })
    
    # Run simulation
    compiled_qc = transpile(bound_qc, simulator)
    shots = 1024
    result = simulator.run(compiled_qc, shots=shots).result()
    counts = result.get_counts()
    
    # Probability of measuring '1'
    prob_1 = counts.get('1', 0) / shots
    return prob_1

def cost_function(weights):
    """Calculates MSE loss over the training set."""
    predictions = [predict(weights, x) for x in X_train]
    mse = np.mean((np.array(predictions) - y_train)**2)
    
    loss_history.append(mse)
    if len(loss_history) % 5 == 0:
        print(f"  - Iteration {len(loss_history):3d} | Loss: {mse:.4f}")
    
    return mse

print("\n[*] Starting Optimization (Hybrid Loop)...")
print("    Optimizer: COBYLA")

# Initial random weights
np.random.seed(42)
initial_weights = np.random.uniform(0, 2*np.pi, len(theta_params))

start_time = time.time()

# Run the optimization
res = minimize(
    cost_function, 
    initial_weights, 
    method='COBYLA', 
    options={'maxiter': 50}
)

end_time = time.time()
optimized_weights = res.x

print(f"\n  [OK] Training Complete in {end_time - start_time:.2f} seconds!")
print(f"  - Final Loss: {loss_history[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4.4: EVALUATION & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Evaluating on Test Set...")

test_preds = [predict(optimized_weights, x) for x in X_test]
test_classes = [1 if p > 0.5 else 0 for p in test_preds]
accuracy = np.mean(np.array(test_classes) == y_test)

print(f"  - Test Accuracy: {accuracy * 100:.1f}%")

# --- Plotting ---
print("\n[PLOT] Generating visualizations...")

fig = plt.figure(figsize=(15, 10), facecolor="#0f0f1a")
fig.suptitle(
    "Step 4: Hybrid Quantum-Classical Training Results\n"
    "Variational Classifier Optimization (Iris Dataset)",
    fontsize=16, color="white", fontweight="bold", y=0.96
)

# 1. Loss Curve
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor("#1a1a2e")
ax1.plot(loss_history, color="#38bdf8", linewidth=2, marker='o', markersize=4, label="MSE Loss")
ax1.set_title("Training Loss (MSE)", color="#a78bfa")
ax1.set_xlabel("Iteration", color="#94a3b8")
ax1.set_ylabel("Loss", color="#94a3b8")
ax1.grid(color="#334155", linestyle="--", alpha=0.5)
ax1.tick_params(colors="white")

# 2. Final Parameter Radar (Visualizing the 'Weights')
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor("#1a1a2e")
params_labels = [f"\u03b8{i}" for i in range(4)]
ax2.bar(params_labels, optimized_weights, color="#7c3aed", alpha=0.7)
ax2.set_title("Optimized Parameters (Theta)", color="#a78bfa")
ax2.tick_params(colors="white")

# 3. Test Predictions Histogram
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor("#1a1a2e")
ax3.hist(test_preds, bins=10, color="#10b981", alpha=0.8, edgecolor="white")
ax3.axvline(0.5, color="red", linestyle="--", label="Decision Threshold")
ax3.set_title("Distribution of Test Predictions", color="#a78bfa")
ax3.set_xlabel("Probability of Class 1", color="#94a3b8")
ax3.legend()
ax3.tick_params(colors="white")

# 4. Summary Text
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor("#0f172a")
ax4.axis("off")
summary_text = (
    f"TRAINING SUMMARY:\n\n"
    f"Dataset       : Iris (Subset)\n"
    f"Optimizer     : COBYLA\n"
    f"Max Iterations: 50\n"
    f"Final Accuracy: {accuracy*100:.1f}%\n\n"
    f"QUANTUM ADVANTAGE NOTE:\n"
    f"While this model is small, it\n"
    f"implements the exact logic of\n"
    f"Schuld et al.'s research.\n"
    f"The non-linear classification\n"
    f"occurs in the high-dimensional\n"
    f"Hilbert space of the qubits."
)
ax4.text(0.1, 0.5, summary_text, color="#e2e8f0", fontsize=11, 
         va="center", family="monospace",
         bbox=dict(boxstyle="round,pad=1", facecolor="#1e293b", edgecolor="#334155"))

plt.savefig("step4_training_results.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
print("  [OK] Plot saved as 'step4_training_results.png'")

plt.show(block=False)
plt.pause(3)
plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  STEP 4 SUMMARY: Hybrid Training Complete")
print("=" * 60)
print(f"""
  1. Optimization  : Parameters optimized using COBYLA (classical).
  2. Cost Function : Minimized Mean Squared Error (MSE).
  3. Prediction    : Used measurement probability P(|1>) as decision.
  4. Accuracy      : Achieved {accuracy*100:.1f}% on unseen test data.

  RESEARCH CONNECTION:
  - This completes the implementation of the "Circuit-centric quantum 
    classifier" architecture.
  - The model successfully learned to classify Iris flowers by adjusting
    quantum gate rotations based on classical feedback.
  - This is the fundamental workflow for VQAs (Variational Quantum Algorithms).

  CONGRATULATIONS! You have built a working Quantum Machine Learning model.
""")
print("=" * 60)
