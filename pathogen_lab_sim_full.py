
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Page config
st.set_page_config(page_title="Simulating Pathogen movement in metagenomic sequencing to quantify biocontainment risks through incorporating breach scenarios", layout="wide")
st.title(" Simulating Pathogen movement in metagenomic sequencing to quantify biocontainment risks through incorporating breach scenarios")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
num_particles = st.sidebar.slider("Number of Particles", 50, 300, 100, step=10)
steps = st.sidebar.slider("Simulation Steps", 100, 500, 300, step=50)
dt = st.sidebar.slider("Time Step (dt)", 0.01, 0.5, 0.1, step=0.01)
sequencer_radius = st.sidebar.slider("Sequencer Radius", 1.0, 5.0, 2.0, step=0.5)
aerosol_risk_radius = st.sidebar.slider("Aerosol Risk Radius", 2.0, 6.0, 3.5, step=0.5)
breach_threshold = st.sidebar.slider("Breach Threshold", 10.0, 20.0, 12.0, step=0.5)
boundary = 10
num_workers = 3

# Pathogen types and traits
pathogen_types = {
    'virus': {'color': 'orange', 'D': 1.2, 'breach_bias': 1.0},
    'plasmid': {'color': 'purple', 'D': 0.8, 'breach_bias': 0.7},
    'dna_fragment': {'color': 'blue', 'D': 0.5, 'breach_bias': 0.5}
}

# Initialize data
types = np.random.choice(list(pathogen_types.keys()), size=num_particles, p=[0.5, 0.3, 0.2])
positions = np.zeros((num_particles, steps, 2))
escaped = np.zeros(num_particles, dtype=bool)
sequenced = np.zeros(num_particles, dtype=bool)
aerosolized = np.zeros(num_particles, dtype=bool)
worker_positions = np.random.uniform(-boundary, boundary, size=(num_workers, steps, 2))
escaped_types = {k: 0 for k in pathogen_types}
sequenced_types = {k: 0 for k in pathogen_types}
aerosolized_types = {k: 0 for k in pathogen_types}

# Run simulation
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
        elif not aerosolized[n] and sequencer_radius**2 < dist_sq <= aerosol_risk_radius**2:
            aerosolized[n] = True
            aerosolized_types[p_type] += 1

        if (np.abs(new_x) > breach_threshold and np.random.rand() < bias) or            (np.abs(new_y) > breach_threshold and np.random.rand() < bias):
            escaped[n] = True
            escaped_types[p_type] += 1
            positions[n, i:] = positions[n, i-1]
            break

        positions[n, i] = [new_x, new_y]

# Create plot
fig, ax = plt.subplots(figsize=(8, 8))
for n in range(num_particles):
    p_type = types[n]
    col = 'red' if escaped[n] else 'gray' if aerosolized[n] else pathogen_types[p_type]['color']
    ax.plot(positions[n, :, 0], positions[n, :, 1], color=col, lw=1)

circle1 = plt.Circle((0, 0), sequencer_radius, color='green', fill=False, linewidth=2)
circle2 = plt.Circle((0, 0), aerosol_risk_radius, color='cyan', fill=False, linestyle='--', linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.axhline(boundary, color='gray', linestyle='--')
ax.axhline(-boundary, color='gray', linestyle='--')
ax.axvline(boundary, color='gray', linestyle='--')
ax.axvline(-boundary, color='gray', linestyle='--')
ax.set_title("Trajectory Map")
ax.axis('equal')
st.pyplot(fig)

# Heatmap
heatmap_res = 100
heatmap = np.zeros((heatmap_res, heatmap_res))
for n in range(num_particles):
    if not escaped[n]:
        final_pos = positions[n, -1]
        x_idx = int((final_pos[0] + boundary) / (2 * boundary) * heatmap_res)
        y_idx = int((final_pos[1] + boundary) / (2 * boundary) * heatmap_res)
        if 0 <= x_idx < heatmap_res and 0 <= y_idx < heatmap_res:
            heatmap[y_idx, x_idx] += 1
st.subheader("Heatmap of Final Particle Positions (Non-breached)")
fig2, ax2 = plt.subplots()
sns.heatmap(heatmap, cmap="viridis", ax=ax2)
st.pyplot(fig2)

# Stats
st.subheader("📊 Simulation Outcomes")
st.markdown(f"**Total Breaches:** {np.sum(escaped)} ({100*np.mean(escaped):.2f}%)")
st.markdown(f"**Sequenced:** {np.sum(sequenced)}")
st.markdown(f"**Aerosolized near sequencer:** {np.sum(aerosolized)}")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Escaped by type**")
    st.json(escaped_types)
with col2:
    st.write("**Sequenced by type**")
    st.json(sequenced_types)
with col3:
    st.write("**Aerosolized by type**")
    st.json(aerosolized_types)


escaped_count = np.sum(escaped)
aerosol_count = np.sum(aerosolized)
unsequenced_breaches = np.sum((escaped) & (~sequenced))
mean_diffusion = np.mean([pathogen_types[t]['D'] for t in types])

# Normalize components to 0–1
escaped_ratio = escaped_count / num_particles
aerosol_ratio = aerosol_count / num_particles
unseq_breach_ratio = unsequenced_breaches / (escaped_count if escaped_count > 0 else 1)
norm_diffusion = (mean_diffusion - 0.5) / (1.5 - 0.5)  # assuming D ranges from 0.5 to 1.5

# CRI: weighted average of components
CRI = 0.4 * escaped_ratio + 0.2 * aerosol_ratio + 0.3 * unseq_breach_ratio + 0.1 * norm_diffusion
CRI = round(CRI, 3)

# Sequenced-before-escape (SBE)
sequenced_and_escaped = np.sum((escaped) & (sequenced))
SBE = sequenced_and_escaped / (escaped_count if escaped_count > 0 else 1)
SBE = round(SBE, 3)

# Display results
st.subheader("🧯 Biosecurity Metrics")
st.markdown(f"**Contamination Risk Index (CRI):** {CRI}  {'🟢 Low' if CRI < 0.3 else '🟡 Medium' if CRI < 0.6 else '🔴 High'}")
st.markdown(f"**Sequenced-Before-Escape Score (SBE):** {SBE}  {'✅ Efficient' if SBE > 0.7 else '⚠️ Needs Improvement' if SBE > 0.3 else '❌ Critical'}")

# === PHASE 2: Worker-Pathogen Proximity Risk ===
proximity_events = []
for w in range(num_workers):
    for i in range(steps):
        for n in range(num_particles):
            dist = np.linalg.norm(positions[n, i] - worker_positions[w, i])
            if dist < 2.0:
                proximity_events.append((i, w, n))

# Calculate time-based overlap count
overlap_series = pd.Series([e[0] for e in proximity_events])
if not overlap_series.empty:
    time_bins = np.arange(0, steps+1, steps//20)
    overlap_counts = overlap_series.value_counts().reindex(time_bins, fill_value=0).sort_index()
else:
    time_bins = np.arange(0, steps+1, steps//20)
    overlap_counts = pd.Series(0, index=time_bins)

st.subheader("Worker-Pathogen Proximity Risk")
st.markdown(f"**Total Close Contact Events (<2μm):** {len(proximity_events)}")
fig3, ax3 = plt.subplots()
ax3.plot(overlap_counts.index, overlap_counts.values, marker='o')
ax3.set_title("Time-Series of Proximity Events")
ax3.set_xlabel("Time Step")
ax3.set_ylabel("Close Contacts")
st.pyplot(fig3)



# CSV log
df_log = pd.DataFrame({
    "Type": types,
    "Escaped": escaped,
    "Sequenced": sequenced,
    "Aerosolized": aerosolized
})
csv = df_log.to_csv(index=False).encode("utf-8")
st.download_button(" Download Simulation Log", csv, "simulation_log.csv", "text/csv")
