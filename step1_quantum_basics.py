"""
============================================================
  STEP 1: Environment Setup + First Quantum Circuit
  Quantum Machine Learning Project
  Inspired by: "Circuit-centric quantum classifiers" (Schuld et al.)
============================================================

Paper Reference:
    Schuld, M., Bocharov, A., Svore, K., & Wiebe, N. (2018).
    "Circuit-centric quantum classifiers."
    arXiv:1804.00633 — https://arxiv.org/abs/1804.00633

WHY THIS CIRCUIT MATTERS FOR VARIATIONAL QUANTUM CLASSIFIERS:
--------------------------------------------------------------
The paper proposes a model where a parameterized quantum circuit (PQC) —
also called an "Ansatz" — is used to classify data encoded as quantum states.

The core idea is:
  1. Encode classical data into a quantum state via rotation gates.
  2. Apply a trainable circuit (like our entangling CNOT + Hadamard here).
  3. Measure the output to get a class prediction.

This Bell-state circuit (H + CNOT) is the foundation of that entangling
structure. It demonstrates the two most critical concepts:
  - Superposition: mapping inputs to a quantum feature space.
  - Entanglement: allowing qubits to share information, giving the model
    correlational power that is classically hard to replicate.

Understanding this circuit = understanding why quantum circuits can act as
powerful, non-linear classifiers on quantum-encoded data.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Force UTF-8 output on Windows (handles Qiskit's box-drawing characters)
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Qiskit — IBM's open-source quantum computing SDK
from qiskit import QuantumCircuit

# Qiskit Aer — High-performance quantum circuit simulator (runs on classical CPU/GPU)
from qiskit_aer import AerSimulator

# Qiskit's result visualization tools
from qiskit.visualization import plot_histogram

# SciPy's transpile for compiling circuits to a backend's native gate set
from qiskit import transpile

# NumPy — Standard scientific computing library
import numpy as np

# Matplotlib — For plotting results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=" * 60)
print("  [OK]  All packages imported successfully!")
print("  Qiskit + Aer + NumPy + Matplotlib are ready.")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT EXPLANATIONS (in-code reference)
# ─────────────────────────────────────────────────────────────────────────────

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KEY QUANTUM CONCEPTS (Beginner Guide)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 WHAT IS A QUBIT?
   A classical bit is either 0 or 1.
   A qubit (quantum bit) can be BOTH 0 and 1 simultaneously — this is
   called "superposition". Mathematically:
       |ψ⟩ = α|0⟩ + β|1⟩
   where α and β are complex probability amplitudes, and |α|² + |β|² = 1.
   Only when measured does it "collapse" to a definite 0 or 1.
   This probabilistic nature is what gives quantum computers their power.

📌 WHAT DOES THE HADAMARD (H) GATE DO?
   The Hadamard gate puts a qubit into EQUAL superposition:
       H|0⟩ = (|0⟩ + |1⟩) / √2   →  50% chance of measuring 0, 50% chance of 1
       H|1⟩ = (|0⟩ - |1⟩) / √2
   In matrix form:
       H = (1/√2) * [[1,  1],
                      [1, -1]]
   In the paper, rotation gates (Rx, Ry) play a similar role for data encoding.

📌 WHAT IS ENTANGLEMENT?
   When two qubits are ENTANGLED, measuring one instantly determines the
   outcome of the other — no matter how far apart they are.
   The CNOT (Controlled-NOT) gate creates entanglement:
       - Control qubit = 0 → Target qubit unchanged
       - Control qubit = 1 → Target qubit is flipped
   After H + CNOT, our 2-qubit system is in a Bell state:
       |Φ+⟩ = (|00⟩ + |11⟩) / √2
   This means we'll ONLY ever see "00" or "11" — NEVER "01" or "10".
   Entanglement is crucial in variational classifiers because it captures
   correlations between data features across qubits.

📌 WHAT IS MEASUREMENT?
   Measurement collapses the quantum state to a classical bit (0 or 1).
   The probability of each outcome is determined by |α|² and |β|².
   Running the circuit many times (shots) gives us an empirical distribution.
   In classifiers, the measurement result encodes the predicted class label.
"""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BUILD THE QUANTUM CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[*] Building 2-qubit quantum circuit...")

# Create a quantum circuit with:
#   - 2 quantum registers (qubits)
#   - 2 classical registers (to store measurement results)
qc = QuantumCircuit(2, 2)

# ── Gate 1: Hadamard on Qubit 0 ──────────────────────────────────────────────
# Puts qubit 0 into superposition: |0⟩ → (|0⟩ + |1⟩)/√2
# This is analogous to the data-encoding rotation in variational classifiers.
qc.h(0)

# ── Gate 2: CNOT (Controlled-NOT) from Qubit 0 → Qubit 1 ────────────────────
# Creates quantum entanglement between the two qubits.
# Combined with H, this produces the Bell state: (|00⟩ + |11⟩)/√2
# In variational classifiers, CNOT layers create the "entangling" structure
# that allows the circuit to learn complex, non-linear decision boundaries.
qc.cx(0, 1)

# ── Gate 3: Measure both qubits ──────────────────────────────────────────────
# Measure qubit 0 → classical bit 0
# Measure qubit 1 → classical bit 1
# The results collapse the quantum state to definite 0s and 1s.
qc.measure(0, 0)
qc.measure(1, 1)

print("  [OK] Circuit built successfully!")
print(f"  - Number of qubits    : {qc.num_qubits}")
print(f"  - Number of gates     : {qc.size()}")
print(f"  - Circuit depth       : {qc.depth()}")
print(f"  - Classical registers : {qc.num_clbits}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DISPLAY THE CIRCUIT DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

print("\n[DIAGRAM] Circuit Diagram (text form):")
print("-" * 50)
print(qc.draw(output="text"))
print("-" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SIMULATE WITH AER SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

print("\n[SIM] Running simulation on AerSimulator...")

# Initialize the Aer statevector/shot-based simulator
simulator = AerSimulator()

# Transpile the circuit for the target backend
# This converts our high-level circuit to native hardware-compatible gates.
compiled_circuit = transpile(qc, simulator)

# Run the circuit for 1024 "shots" (repeated trials)
# More shots = more accurate empirical probability distribution
SHOTS = 1024
job = simulator.run(compiled_circuit, shots=SHOTS)
result = job.result()

# Extract the measurement counts (e.g., {'00': 512, '11': 512})
counts = result.get_counts(compiled_circuit)

print(f"  [OK] Simulation complete! ({SHOTS} shots)")
print(f"\n[RESULTS] Measurement Counts:")
print(f"  {counts}")
print()

# Interpret the results
for state, count in sorted(counts.items()):
    probability = count / SHOTS * 100
    print(f"  |{state}>  ->  {count:4d} times  ({probability:.1f}%)")

print()
print("  [!] Notice: Only '00' and '11' appear -- this is ENTANGLEMENT at work!")
print("     The qubits are perfectly correlated: measuring one tells you the other.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: VISUALIZE — CIRCUIT + HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

print("\n[PLOT] Generating visualizations...")

fig = plt.figure(figsize=(14, 7), facecolor="#0f0f1a")
fig.suptitle(
    "Step 1: Bell State Quantum Circuit\n"
    "Foundation for Circuit-Centric Quantum Classifiers",
    fontsize=14,
    color="white",
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Panel 1: Circuit Diagram ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#1a1a2e")
ax1.set_title("Quantum Circuit (Bell State)", color="#a78bfa", fontsize=12, pad=12)
ax1.axis("off")

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

# Render the circuit diagram as image and embed into our figure
circuit_fig.canvas.draw()
import io
buf = io.BytesIO()
circuit_fig.savefig(buf, format="png", facecolor="#1a1a2e", bbox_inches="tight", dpi=120)
buf.seek(0)
import matplotlib.image as mpimg
img = mpimg.imread(buf)
ax1.imshow(img)
plt.close(circuit_fig)

# ── Panel 2: Measurement Histogram ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#1a1a2e")
ax2.set_title("Measurement Outcome Distribution", color="#a78bfa", fontsize=12, pad=12)

states = sorted(counts.keys())
values = [counts[s] for s in states]
colors = ["#7c3aed", "#38bdf8", "#10b981", "#f59e0b"][:len(states)]

bars = ax2.bar(states, values, color=colors, width=0.4,
               edgecolor="#ffffff", linewidth=0.8, alpha=0.92)

# Add count + probability labels above each bar
for bar, val in zip(bars, values):
    pct = val / SHOTS * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{val}\n({pct:.1f}%)",
        ha="center", va="bottom",
        color="white", fontsize=11, fontweight="bold"
    )

ax2.set_xlabel("Measurement Outcome (Classical Bits)", color="#94a3b8", fontsize=10)
ax2.set_ylabel(f"Count (out of {SHOTS} shots)", color="#94a3b8", fontsize=10)
ax2.set_ylim(0, max(values) * 1.25)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#334155")
ax2.set_facecolor("#1a1a2e")
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color("#e2e8f0")

# Annotation box
annotation = (
    "Bell State: |Φ+⟩ = (|00⟩ + |11⟩)/√2\n"
    "Only |00⟩ and |11⟩ are possible outcomes.\n"
    "This is quantum entanglement."
)
ax2.text(
    0.5, 0.04, annotation,
    transform=ax2.transAxes,
    ha="center", va="bottom",
    fontsize=9, color="#94a3b8",
    style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#334155", alpha=0.8)
)

plt.savefig("step1_quantum_circuit_output.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f1a")
print("  [OK] Plot saved as 'step1_quantum_circuit_output.png'")
# Non-blocking show: displays the window briefly in interactive environments
plt.show(block=False)
plt.pause(3)
plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  STEP 1 SUMMARY")
print("=" * 60)
print("""
  Circuit Built   : 2-qubit Bell State Circuit
  Gates Applied   : H (Hadamard) on q0 → CNOT (q0 → q1) → Measure
  Simulator Used  : Qiskit Aer (AerSimulator)
  Shots Run       : 1024

  Key Concepts Demonstrated:
    [+] Qubit superposition via Hadamard gate
    [+] Quantum entanglement via CNOT gate
    [+] Measurement collapse and probabilistic outcomes
    [+] Bell state |Phi+> -- one of the four maximally entangled states

  Connection to "Circuit-Centric Quantum Classifiers" (Schuld et al.):
    [+] The H + CNOT structure IS the entangling ansatz used in PQCs
    [+] Superposition enables exploration of high-dimensional feature spaces
    [+] Entanglement enables cross-qubit feature correlations (non-linearity)
    [+] Measurement maps quantum state -> classical prediction (class label)

  NEXT STEP → Data encoding + parameterized rotation gates (Rx, Ry, Rz)
              to build the actual variational quantum classifier circuit.
""")
print("=" * 60)
