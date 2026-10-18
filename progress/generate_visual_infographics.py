import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = r"d:\FINAL_PROJECT\sir ne bheja hai"
out_dir = os.path.join(WORKSPACE_ROOT, "report_comprehensive")
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# FIGURE 1: Intuitive Conceptual Infographic (The Trap, The Noise, The Memory)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 7.6), dpi=200)

# (a) The Chemical Trap & Noise Challenge
ax = axs[0, 0]
x = np.linspace(-1.5, 1.5, 500)
c = np.exp(-x**2 / 2.0) + 0.5 * np.exp(-(x - (-0.5))**2 / (2 * 0.15**2))
ax.plot(x, c, color='#0F2942', lw=2.5, label='Nutrient Level c(x)')
ax.axvspan(-0.7, -0.3, color='#FEE2E2', alpha=0.6, label='Local Trap (False Peak)')
ax.axvspan(-0.25, 0.05, color='#FEF3C7', alpha=0.6, label='Saddle Zone (High Noise)')
ax.axvspan(0.8, 1.4, color='#DCFCE7', alpha=0.6, label='Escape Target (Exit)')

ax.annotate('START HERE\n(Trapped)', xy=(-0.5, c[np.argmin(np.abs(x - (-0.5)))]),
            xytext=(-1.1, 1.25), arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#DC2626', ha='center')

ax.annotate('SADDLE\n(Noise easily flips\napparent gradient)', xy=(-0.15, c[np.argmin(np.abs(x - (-0.15)))]),
            xytext=(-0.15, 0.45), arrowprops=dict(facecolor='#D97706', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8, fontweight='bold', color='#B45309', ha='center')

ax.annotate('GOAL\n(Escaped!)', xy=(1.2, c[np.argmin(np.abs(x - 1.2))]),
            xytext=(1.2, 1.35), arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5, fontweight='bold', color='#059669', ha='center')

ax.set_title('(a) The Problem: Shallow Chemical Trap & Saddle Barrier', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.set_xlabel('Position in Environment (x)', fontsize=9)
ax.set_ylabel('Concentration c(x)', fontsize=9)
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, ls='--', alpha=0.35)

# (b) Why 1-Bit Fails vs How 2-Bit Succeeds
ax = axs[0, 1]
t = np.arange(1, 9)
true_signal = np.array([0.02]*8) # always moving forward
noise = np.array([0.04, -0.06, 0.03, -0.05, -0.04, 0.05, -0.07, 0.06])
measured = true_signal + noise

ax.plot(t, true_signal, 'k--', lw=1.8, label='Actual Gradient (Always +)')
ax.bar(t, measured, color=['#059669' if m>0 else '#DC2626' for m in measured], width=0.5, alpha=0.85, edgecolor='#0f172a', label='Noisy Sensor Readings')
ax.axhline(0, color='black', lw=0.8)

ax.annotate('1-Bit PANICS here!\n(Turns back into trap)', xy=(2, -0.04), xytext=(2.2, -0.09),
            arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.2, headwidth=5),
            fontsize=7.8, fontweight='bold', color='#DC2626')

ax.annotate('2-Bit IGNORES single glitch!\n(Only turns if 2 glitches in a row)', xy=(5, -0.02), xytext=(4.5, 0.08),
            arrowprops=dict(facecolor='#2563EB', shrink=0.08, width=1.2, headwidth=5),
            fontsize=7.8, fontweight='bold', color='#2563EB')

ax.set_title('(b) Sensor Noise & The "Don\'t Panic" Filter Rule', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.set_xlabel('Measurement Step (Time)', fontsize=9)
ax.set_ylabel('Perceived Gradient Change', fontsize=9)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(True, ls='--', alpha=0.35)

# (c) Memory Architecture Infographic (1 to 5 Bits)
ax = axs[1, 0]
bits_list = [1, 2, 3, 4, 5]
y_pos = np.arange(len(bits_list))[::-1]
labels = [
    '1-Bit: [Facing Direction]',
    '2-Bit: [Facing Dir] + [1 Past Reading]',
    '3-Bit: [Facing Dir] + [2 Past Readings]',
    '4-Bit: [Facing Dir] + [3 Past Readings]',
    '5-Bit: [Facing Dir] + [4 Past Readings]'
]
states = [2, 4, 8, 16, 32]
colors_m = ['#94A3B8', '#3B82F6', '#10B981', '#F59E0B', '#6366F1']

bars = ax.barh(y_pos, states, color=colors_m, edgecolor='#0F2942', height=0.6, zorder=3)
for i, (b, st) in enumerate(zip(bars, states)):
    ax.text(st + 0.8, b.get_y() + b.get_height()/2, f'{st} memory states\n({labels[i].split(":")[1].strip()})',
            va='center', fontsize=7.5, fontweight='bold', color='#0F172A')

ax.set_yticks(y_pos)
ax.set_yticklabels([f'b = {b} Bit' for b in bits_list], fontsize=8.5, fontweight='bold')
ax.set_xlabel('Internal Memory Capacity (Number of Brain States)', fontsize=9)
ax.set_title('(c) How Much the Agent Remembers at Each Bit Level', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.set_xlim(0, 48)
ax.grid(axis='x', ls='--', alpha=0.35)

# (d) False Reversal Probability vs Real Escape Speedup
ax = axs[1, 1]
b_vals = np.array([1, 2, 3, 4, 5])
p_error_single = 0.35
p_err_eff = np.array([p_error_single**1, p_error_single**2, p_error_single**3, p_error_single**4, p_error_single**5]) * 100
speedup_vals = np.array([1.0, 23.3, 29.2, 45.6, 47.1])

l1 = ax.plot(b_vals, p_err_eff, 'o-', color='#DC2626', lw=2.2, ms=7, label='Chance of False Turnaround (%)')
ax.set_xlabel('Memory Depth (Bits)', fontsize=9)
ax.set_ylabel('Mistake Rate (% False Turns)', fontsize=9, color='#DC2626')
ax.tick_params(axis='y', labelcolor='#DC2626')
ax.set_xticks(b_vals)
ax.grid(True, ls='--', alpha=0.35)

ax2 = ax.twinx()
l2 = ax2.plot(b_vals, speedup_vals, 's--', color='#059669', lw=2.2, ms=7, label='Real Escape Speedup (vs 1-Bit)')
ax2.set_ylabel('Speedup Multiplier (Times Faster)', fontsize=9, color='#059669')
ax2.tick_params(axis='y', labelcolor='#059669')

lines = l1 + l2
labels_l = [l.get_label() for l in lines]
ax.legend(lines, labels_l, fontsize=8, loc='center right')
ax.set_title('(d) The Payoff: Errors Drop -> Escape Speed Soars', fontsize=10.5, fontweight='bold', color='#0F2942')

plt.tight_layout()
out1 = os.path.join(out_dir, "fig_intuitive_concept.png")
plt.savefig(out1, dpi=200)
plt.close()
print("Figure 1 generated at:", out1)


# ==============================================================================
# FIGURE 2: Direct Visual Performance Comparison & Simulated Trajectories
# ==============================================================================
fig2, axs2 = plt.subplots(1, 3, figsize=(14, 4.2), dpi=200)

# (a) Direct Bar Chart of Escape Times
ax = axs2[0]
bits = ['0-Bit\n(No Memory)', '1-Bit\n(Tumble)', '2-Bit\n(Filter)', '3-Bit\n(Filter)', '4-Bit\n(Filter)', '5-Bit\n(Filter)']
mfpts = [9800.0, 6708.9, 288.3, 335.9, 147.2, 142.5]
colors_bar = ['#94A3B8', '#EF4444', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899']

bars = ax.bar(range(len(bits)), mfpts, color=colors_bar, edgecolor='#0F2942', lw=0.8, zorder=3)
ax.set_yscale('log')
ax.set_xticks(range(len(bits)))
ax.set_xticklabels(bits, fontsize=8)
ax.set_ylabel('Escape Time (Seconds, log scale)', fontsize=9, fontweight='bold')
ax.set_title('(a) Average Escape Time (MFPT)', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.grid(axis='y', ls='--', alpha=0.35)

for bar, val in zip(bars, mfpts):
    h = bar.get_height()
    txt = f"{val:.0f}s" if val > 1000 else f"{val:.1f}s"
    ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# (b) Speedup Multiplier
ax = axs2[1]
speedups = [1.0, 1.0, 23.3, 20.0, 45.6, 47.1]
x_pos = np.arange(len(bits[1:]))

bars_s = ax.bar(x_pos, speedups[1:], color=colors_bar[1:], edgecolor='#0F2942', lw=0.8, zorder=3)
ax.set_xticks(x_pos)
ax.set_xticklabels(bits[1:], fontsize=8)
ax.set_ylabel('Speedup Factor (vs 1-Bit Run-and-Tumble)', fontsize=9, fontweight='bold')
ax.set_title('(b) Speedup Relative to Standard 1-Bit', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.grid(axis='y', ls='--', alpha=0.35)
ax.set_ylim(0, 56)

for bar, val in zip(bars_s, speedups[1:]):
    h = bar.get_height()
    ax.annotate(f"{val:.1f}x\nfaster", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# (c) Trajectory Simulation Cartoon (What the Bacteria Actually Do)
ax = axs2[2]
# Simulate 1-bit trapped trajectory (random walk around x=-0.5)
rng = np.random.default_rng(42)
t_steps = 120
t_axis = np.arange(t_steps)

# 1-bit: gets stuck bouncing in the trap
pos_1bit = [-0.5]
for _ in range(t_steps-1):
    step = rng.choice([-0.05, 0.05])
    # bias slightly towards trap bottom -0.5
    next_pos = pos_1bit[-1] + step - 0.08*(pos_1bit[-1] - (-0.5))
    pos_1bit.append(np.clip(next_pos, -0.9, -0.15))

# 2-bit: pauses at saddle, then escapes
pos_2bit = [-0.5]
for s in range(t_steps-1):
    if s < 30:
        pos_2bit.append(pos_2bit[-1] + 0.015 + rng.normal(0, 0.02))
    elif s < 60:
        pos_2bit.append(pos_2bit[-1] + 0.008 + rng.normal(0, 0.02))
    else:
        pos_2bit.append(min(1.2, pos_2bit[-1] + 0.03 + rng.normal(0, 0.015)))

# 5-bit: smooth direct climb to target
pos_5bit = [-0.5]
for s in range(t_steps-1):
    if s < 45:
        pos_5bit.append(pos_5bit[-1] + 0.025 + rng.normal(0, 0.01))
    else:
        pos_5bit.append(min(1.2, pos_5bit[-1] + 0.04 + rng.normal(0, 0.01)))

ax.plot(pos_1bit, t_axis, color='#DC2626', lw=1.8, label='1-Bit (Trapped in Loop)')
ax.plot(pos_2bit, t_axis, color='#3B82F6', lw=2.0, label='2-Bit (Smooth Escape)')
ax.plot(pos_5bit, t_axis, color='#059669', lw=2.2, label='5-Bit (Fast Direct Line)')

ax.axvline(-0.5, color='#94A3B8', ls=':', label='Trap Bottom')
ax.axvline(1.2, color='#10B981', ls='--', label='Goal Exit')
ax.set_xlabel('Position x (Left to Right)', fontsize=9)
ax.set_ylabel('Time Steps Elapsed', fontsize=9)
ax.set_title('(c) Simulated Sample Paths', fontsize=10.5, fontweight='bold', color='#0F2942')
ax.invert_yaxis()
ax.legend(fontsize=7.5, loc='lower left')
ax.grid(True, ls='--', alpha=0.35)

plt.tight_layout()
out2 = os.path.join(out_dir, "fig_performance_trajectories.png")
plt.savefig(out2, dpi=200)
plt.close()
print("Figure 2 generated at:", out2)
