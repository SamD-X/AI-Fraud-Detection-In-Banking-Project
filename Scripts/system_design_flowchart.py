import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Create a figure
fig, ax = plt.subplots(figsize=(6, 8))

# Flowchart steps
steps = [
    "1. Data Loading\n(Load CSV Files)",
    "2. Preprocessing\n(Scale Features, Split Data)",
    "3. SMOTE\n(Balance Training Data)",
    "4. Model Training\n(Logistic Regression)",
    "5. Threshold Tuning\n(Threshold = 0.999)",
    "6. Evaluation\n(Metrics, Visualizations)",
    "7. Result Storage\n(Save Metrics, Plots)"
]

# Plot each step as a box
for i, step in enumerate(steps):
    y_position = 0.9 - (i * 0.12)
    ax.text(0.5, y_position, step, fontsize=10, ha='center', va='center',
            bbox=dict(facecolor='lightgreen', edgecolor='black'))

# Add arrows between steps
for i in range(len(steps) - 1):
    y_start = 0.9 - (i * 0.12) - 0.05
    y_end = 0.9 - ((i + 1) * 0.12) + 0.05
    ax.add_patch(FancyArrowPatch((0.5, y_start), (0.5, y_end), mutation_scale=20, color='black'))

# Title
plt.title('Figure 2: System Design Flowchart', fontsize=14)

# Hide axes
ax.axis('off')

# Save the figure
plt.savefig('C:/Users/Asus/Desktop/Final Project/system_design_flowchart.png', bbox_inches='tight')
plt.show()