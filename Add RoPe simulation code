
"""
sim.py — Minimal experimental simulation for RoPE-based Sentient Agent
Author: [Your Name]
Date: 2025-11-07

This script simulates N agents, each with a state vector.
Each agent’s state is rotated (RoPE-inspired) and passed to the next agent.
We measure how well the original context is preserved after multiple hops.
"""

import numpy as np

# -----------------------------
# Basic simulation parameters
# -----------------------------
N_AGENTS = 8          # number of agents
D_MODEL = 32          # vector dimension (should be even)
N_HOPS = 10           # number of message hops
ALPHA = 0.1           # update rate
THETA_STEP = 0.2      # base rotation step per agent

np.random.seed(42)    # reproducibility


# -----------------------------
# Utility functions
# -----------------------------
def rotary_matrix(theta, dim):
    """Constructs a RoPE-like rotation matrix."""
    R = np.eye(dim)
    for i in range(0, dim, 2):
        c, s = np.cos(theta), np.sin(theta)
        R[i, i], R[i, i + 1] = c, -s
        R[i + 1, i], R[i + 1, i + 1] = s, c
    return R


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# -----------------------------
# Initialize agents
# -----------------------------
states = np.random.randn(N_AGENTS, D_MODEL)
thetas = np.linspace(0, THETA_STEP * (N_AGENTS - 1), N_AGENTS)

# -----------------------------
# Simulation loop
# -----------------------------
def simulate_transfer(states, thetas):
    """Pass state information through the network with RoPE rotations."""
    current = states[0].copy()  # start with agent 0
    similarities = []

    for hop in range(1, N_AGENTS):
        sender_theta = thetas[hop - 1]
        receiver_theta = thetas[hop]

        # Apply rotation to sender’s state
        R_sender = rotary_matrix(sender_theta, D_MODEL)
        rotated_state = R_sender @ current

        # Receiver updates its state
        s_rec = states[hop]
        s_rec = s_rec + ALPHA * (rotated_state - s_rec)
        states[hop] = s_rec

        # Measure similarity with original
        sim = cosine_similarity(states[0], s_rec)
        similarities.append(sim)

        current = s_rec.copy()

    return similarities


# -----------------------------
# Run experiment
# -----------------------------
similarities = simulate_transfer(states, thetas)

print("\n=== RoPE Simulation Results ===")
for i, sim in enumerate(similarities, start=1):
    print(f"Hop {i:2d} → Similarity to original: {sim:.4f}")

print("\nAverage similarity:", np.mean(similarities))
print("Final similarity:", similarities[-1])
