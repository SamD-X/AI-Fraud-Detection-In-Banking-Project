import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Existing System Workflow
ax.text(0.2, 0.9, 'Existing System\n(Rule-Based)', fontsize=12, ha='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
ax.text(0.2, 0.7, '1. Define Rules\n(e.g., Amount > $10,000)', fontsize=10, ha='center')
ax.text(0.2, 0.5, '2. Apply Rules to\nTransactions', fontsize=10, ha='center')
ax.text(0.2, 0.3, '3. Flag Suspicious\nTransactions', fontsize=10, ha='center')
ax.text(0.2, 0.1, '4. Manual Review', fontsize=10, ha='center')

# Proposed System Workflow
ax.text(0.8, 0.9, 'Proposed System\n(AI-Based)', fontsize=12, ha='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
ax.text(0.8, 0.7, '1. Preprocess Data\n(Scale Features)', fontsize=10, ha='center')
ax.text(0.8, 0.5, '2. Apply SMOTE\n(Balance Data)', fontsize=10, ha='center')
ax.text(0.8, 0.3, '3. Train Logistic\nRegression Model', fontsize=10, ha='center')
ax.text(0.8, 0.1, '4. Tune Threshold\n(0.999)', fontsize=10, ha='center')

# Arrows for Existing System
ax.add_patch(FancyArrowPatch((0.2, 0.65), (0.2, 0.55), mutation_scale=20, color='black'))
ax.add_patch(FancyArrowPatch((0.2, 0.45), (0.2, 0.35), mutation_scale=20, color='black'))
ax.add_patch(FancyArrowPatch((0.2, 0.25), (0.2, 0.15), mutation_scale=20, color='black'))

# Arrows for Proposed System
ax.add_patch(FancyArrowPatch((0.8, 0.65), (0.8, 0.55), mutation_scale=20, color='black'))
ax.add_patch(FancyArrowPatch((0.8, 0.45), (0.8, 0.35), mutation_scale=20, color='black'))
ax.add_patch(FancyArrowPatch((0.8, 0.25), (0.8, 0.15), mutation_scale=20, color='black'))

# Title
plt.title('Figure 1: Workflow Comparison (Existing vs. Proposed System)', fontsize=14)

# Hide axes
ax.axis('off')

# Save the figure
plt.savefig('C:/Users/Asus/Desktop/Final Project/workflow_comparison.png', bbox_inches='tight')
plt.show()