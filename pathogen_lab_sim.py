
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(page_title="Pathogen Simulator Dashboard", layout="wide")
st.title("🧬 Pathogen on the Loose: Lab Simulation Dashboard")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
num_particles = st.sidebar.slider("Number of Particles", 50, 300, 100, step=10)
steps = st.sidebar.slider("Simulation Steps", 100, 500, 300, step=50)
dt = st.sidebar.slider("Time Step (dt)", 0.01, 0.5, 0.1, step=0.01)
sequencer_radius = st.sidebar.slider("Sequencer Radius", 1.0, 5.0, 2.0, step=0.5)
aerosol_risk_radius = st.sidebar.slider("Aerosol Risk Radius", 2.0, 6.0, 3.5, step=0.5)
breach_threshold = st.sidebar.slider("Breach Threshold", 10.0, 20.0, 12.0, step=0.5)
boundary = 10
num_workers = 3

# Define pathogen types
pathogen_types = {
    'virus': {'color': 'orange', 'D': 1.2, 'breach_bias': 1.0},
    'plasmid': {'color': 'purple', 'D': 0.8, 'breach_bias': 0.7},
    'dna_fragment': {'color': 'blue', 'D': 0.5, 'breach_bias': 0.5}
}

types = np.random.choice(['virus', 'plasmid', 'dna_fragment'], size=num_particles, p=[0.5, 0.3, 0.2])
positions = np.zeros((num_particles, steps, 2))
escaped = np.zeros(num_particles, dtype=bool)
sequenced = np.zeros(num_particles, dtype=bool)
aerosolized = np.zeros(num_particles, dtype=bool)
escaped_types = {k: 0 for k in pathogen_types}
sequenced_types = {k: 0 for k in pathogen_types}
aerosolized_types = {k: 0 for k in pathogen_types}
worker_positions = np.random.uniform(-boundary, boundary, size=(num_workers, steps, 2))

# Simulation loop
for n in range(num_particles):
    p_type = types[n]
    D_eff = pathogen_types[p_type]['D']
    bias = pathogen_types[p_type]['breach_bias']
    for i in range(1, steps):
        dx = np.sqrt(2 * D_eff * dt) * np.random.randn()
        dy = np.sqrt(2 * D_eff * dt) * np.random.randn()

        for w in range(num_workers):
            w_dx = np.sqrt(2 * 1.5 * dt) * np.random.randn()
            w_dy = np.sqrt(2 * 1.5 * dt) * np.random.randn()
            worker_positions[w, i] = worker_positions[w, i-1] + [w_dx, w_dy]

            dist = np.linalg.norm(positions[n, i-1] - worker_positions[w, i-1])
            if dist < 2.0:
                dx += np.random.normal(0, 0.5)
                dy += np.random.normal(0, 0.5)

        new_x = positions[n, i-1, 0] + dx
        new_y = positions[n, i-1, 1] + dy

        dist_sq = new_x**2 + new_y**2
        if not sequenced[n] and dist_sq <= sequencer_radius**2:
            sequenced[n] = True
            sequenced_types[p_type] += 1
        if not aerosolized[n] and sequencer_radius**2 < dist_sq <= aerosol_risk_radius**2:
            aerosolized[n] = True
            aerosolized_types[p_type] += 1

        if (np.abs(new_x) > breach_threshold and np.random.rand() < bias) or            (np.abs(new_y) > breach_threshold and np.random.rand() < bias):
            escaped[n] = True
            escaped_types[p_type] += 1
            positions[n, i:] = positions[n, i-1]
            break

        positions[n, i] = [new_x, new_y]

# Plotting trajectories
fig, ax = plt.subplots(figsize=(8, 8))
for n in range(num_particles):
    p_type = types[n]
    if escaped[n]:
        ax.plot(positions[n, :, 0], positions[n, :, 1], color='red', lw=1)
    elif aerosolized[n]:
        ax.plot(positions[n, :, 0], positions[n, :, 1], color='gray', lw=1)
    else:
        ax.plot(positions[n, :, 0], positions[n, :, 1], color=pathogen_types[p_type]['color'], lw=1)

# Draw containment and sequencer zones
ax.axhline(boundary, color='gray', linestyle='--')
ax.axhline(-boundary, color='gray', linestyle='--')
ax.axvline(boundary, color='gray', linestyle='--')
ax.axvline(-boundary, color='gray', linestyle='--')
ax.axhline(breach_threshold, color='black', linestyle=':', lw=1)
ax.axhline(-breach_threshold, color='black', linestyle=':', lw=1)
ax.axvline(breach_threshold, color='black', linestyle=':', lw=1)
ax.axvline(-breach_threshold, color='black', linestyle=':', lw=1)
circle1 = plt.Circle((0, 0), sequencer_radius, color='green', fill=False, linewidth=2, label='Sequencer')
circle2 = plt.Circle((0, 0), aerosol_risk_radius, color='cyan', fill=False, linestyle='--', linewidth=2, label='Aerosol Zone')
ax.add_patch(circle1)
ax.add_patch(circle2)
for w in range(num_workers):
    ax.plot(worker_positions[w, :, 0], worker_positions[w, :, 1], linestyle=':', color='brown', alpha=0.5, label='Worker' if w == 0 else "")
ax.set_title("Lab Simulation: Trajectories")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.legend()
ax.axis('equal')
st.pyplot(fig)
