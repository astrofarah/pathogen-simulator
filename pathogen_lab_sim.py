

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
boundary = 10
steps = 300
breach_threshold = 12.0
num_particles = 100
num_workers = 3
dt = 0.1
sequencer_radius = 2.0
aerosol_risk_radius = 3.5  # aerosol risk zone beyond sequencer

# Define pathogen types with properties
pathogen_types = {
    'virus': {'color': 'orange', 'D': 1.2, 'breach_bias': 1.0},
    'plasmid': {'color': 'purple', 'D': 0.8, 'breach_bias': 0.7},
    'dna_fragment': {'color': 'blue', 'D': 0.5, 'breach_bias': 0.5}
}

# Assign types to particles
types = np.random.choice(['virus', 'plasmid', 'dna_fragment'], size=num_particles, p=[0.5, 0.3, 0.2])
positions = np.zeros((num_particles, steps, 2))
escaped = np.zeros(num_particles, dtype=bool)
sequenced = np.zeros(num_particles, dtype=bool)
aerosolized = np.zeros(num_particles, dtype=bool)
escaped_types = {k: 0 for k in pathogen_types}
sequenced_types = {k: 0 for k in pathogen_types}
aerosolized_types = {k: 0 for k in pathogen_types}

# Initialize lab worker positions and paths
worker_positions = np.random.uniform(-boundary, boundary, size=(num_workers, steps, 2))

# Simulate motion
for n in range(num_particles):
    p_type = types[n]
    D_eff = pathogen_types[p_type]['D']
    bias = pathogen_types[p_type]['breach_bias']
    for i in range(1, steps):
        dx = np.sqrt(2 * D_eff * dt) * np.random.randn()
        dy = np.sqrt(2 * D_eff * dt) * np.random.randn()

        # Add disturbance from nearby lab workers
        for w in range(num_workers):
            w_dx = np.sqrt(2 * 1.5 * dt) * np.random.randn()
            w_dy = np.sqrt(2 * 1.5 * dt) * np.random.randn()
            worker_positions[w, i] = worker_positions[w, i-1] + [w_dx, w_dy]

            dist_to_worker = np.linalg.norm(positions[n, i-1] - worker_positions[w, i-1])
            if dist_to_worker < 2.0:
                dx += np.random.normal(0, 0.5)
                dy += np.random.normal(0, 0.5)

        new_x = positions[n, i-1, 0] + dx
        new_y = positions[n, i-1, 1] + dy

        # ONT Sequencer interaction
        dist_to_sequencer = new_x**2 + new_y**2
        if not sequenced[n] and dist_to_sequencer <= sequencer_radius**2:
            sequenced[n] = True
            sequenced_types[p_type] += 1

        # Aerosol risk zone
        if not aerosolized[n] and dist_to_sequencer <= aerosol_risk_radius**2 and dist_to_sequencer > sequencer_radius**2:
            aerosolized[n] = True
            aerosolized_types[p_type] += 1

        # Breach detection
        if (np.abs(new_x) > breach_threshold and np.random.rand() < bias) or \
           (np.abs(new_y) > breach_threshold and np.random.rand() < bias):
            escaped[n] = True
            escaped_types[p_type] += 1
            positions[n, i:] = positions[n, i-1]
            break

        positions[n, i] = [new_x, new_y]

# Plot trajectories
plt.figure(figsize=(8, 8))
for n in range(num_particles):
    p_type = types[n]
    if escaped[n]:
        plt.plot(positions[n, :, 0], positions[n, :, 1], color='red', lw=1)
    elif aerosolized[n]:
        plt.plot(positions[n, :, 0], positions[n, :, 1], color='gray', lw=1)
    else:
        plt.plot(positions[n, :, 0], positions[n, :, 1], color=pathogen_types[p_type]['color'], lw=1)

# Draw containment, breach, sequencer zones
plt.axhline(boundary, color='gray', linestyle='--')
plt.axhline(-boundary, color='gray', linestyle='--')
plt.axvline(boundary, color='gray', linestyle='--')
plt.axvline(-boundary, color='gray', linestyle='--')
plt.axhline(breach_threshold, color='black', linestyle=':', lw=1)
plt.axhline(-breach_threshold, color='black', linestyle=':', lw=1)
plt.axvline(breach_threshold, color='black', linestyle=':', lw=1)
plt.axvline(-breach_threshold, color='black', linestyle=':', lw=1)

# Sequencer zones
circle1 = plt.Circle((0, 0), sequencer_radius, color='green', fill=False, linestyle='-', linewidth=2, label='ONT Sequencer')
circle2 = plt.Circle((0, 0), aerosol_risk_radius, color='cyan', fill=False, linestyle='--', linewidth=2, label='Aerosol Risk Zone')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

# Plot workers
for w in range(num_workers):
    plt.plot(worker_positions[w, :, 0], worker_positions[w, :, 1], linestyle=':', color='brown', alpha=0.6, label='Lab Worker' if w == 0 else "")

plt.title("Simulation: Sequencing, Breach & Aerosol Spread\nRed = Breach | Gray = Aerosol | Orange = Virus | Purple = Plasmid | Blue = DNA | Green = Sequencer")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

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

plt.figure(figsize=(6, 5))
sns.heatmap(heatmap, cmap='viridis', cbar_kws={'label': 'Particle Density'})
plt.title("Heatmap of Final Contained Pathogens")
plt.xlabel("X Position Bin")
plt.ylabel("Y Position Bin")
plt.show()

# Final report
print(f"\n=== Final Report ===")
print(f"Total Breach Risk: {np.sum(escaped)} / {num_particles} ({np.sum(escaped)/num_particles*100:.1f}%)")
print(f"Total Sequenced: {np.sum(sequenced)} / {num_particles} ({np.sum(sequenced)/num_particles*100:.1f}%)")
print(f"Total Aerosolized (Near Sequencer): {np.sum(aerosolized)} / {num_particles} ({np.sum(aerosolized)/num_particles*100:.1f}%)")
for k in pathogen_types:
    print(f"\n→ {k.capitalize()}:")
    print(f"  Escaped: {escaped_types[k]}")
    print(f"  Sequenced: {sequenced_types[k]}")
    print(f"  Aerosolized: {aerosolized_types[k]}")
