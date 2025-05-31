import pandas as pd
import matplotlib.pyplot as plt

# Figure 7: Class Distribution Before and After SMOTE
# Original dataset
original_counts = [284315, 492]  # Legit, Fraud
smote_counts = [227451, 227451]  # After SMOTE (training set)
labels = ['Legitimate', 'Fraudulent']

plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.bar(labels, original_counts, color=['blue', 'red'])
plt.title('Original Class Distribution')
plt.ylabel('Count')
for i, v in enumerate(original_counts):
    plt.text(i, v + 5000, str(v), ha='center')
plt.subplot(1, 2, 2)
plt.bar(labels, smote_counts, color=['blue', 'red'])
plt.title('After SMOTE (Training Set)')
plt.ylabel('Count')
for i, v in enumerate(smote_counts):
    plt.text(i, v + 5000, str(v), ha='center')
plt.tight_layout()
plt.savefig('C:/Users/Asus/Desktop/Final Project/class_distribution.png')
plt.show()

# Figure 8: Metrics vs. Threshold
thresholds = [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]
precision = [0.0589, 0.1972, 0.2707, 0.6296, 0.7297, 0.7429]
recall = [0.9184, 0.8673, 0.8673, 0.8673, 0.8265, 0.7959]
f1 = [0.1107, 0.3214, 0.4126, 0.7296, 0.7751, 0.7685]

plt.figure(figsize=(6, 4))
plt.plot(thresholds, precision, label='Precision', marker='o', color='blue')
plt.plot(thresholds, recall, label='Recall', marker='o', color='red')
plt.plot(thresholds, f1, label='F1 Score', marker='o', color='green')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs. Threshold (SMOTE Model)')
plt.legend()
plt.grid(True)
plt.savefig('C:/Users/Asus/Desktop/Final Project/metrics_vs_threshold.png')
plt.show()