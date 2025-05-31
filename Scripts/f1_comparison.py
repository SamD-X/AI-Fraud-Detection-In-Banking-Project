import matplotlib.pyplot as plt

# F1 scores from models
models = ['Initial', 'SMOTE', 'Threshold 0.999']
f1_scores = [0.6951, 0.1107, 0.7751]

# Create bar chart
plt.figure(figsize=(6, 4))
plt.bar(models, f1_scores, color=['blue', 'orange', 'green'])
plt.title('F1 Score Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.savefig('C:/Users/Asus/Desktop/Final Project/f1_comparison.png')
plt.show()