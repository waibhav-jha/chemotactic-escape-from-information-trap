import numpy as np

def policy_0bit_value(c_now, c_ref=0.5):
    # Illustrative value-only policy.
    # Replace with the exact policy in the analytical benchmark.
    return np.where(c_now > c_ref, 1, -1)

def policy_1bit(c_now, c_previous, previous_action):
    dc = c_now-c_previous
    return np.where(dc > 0, previous_action, -previous_action)
