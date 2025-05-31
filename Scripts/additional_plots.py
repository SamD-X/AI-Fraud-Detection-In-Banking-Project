import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the preprocessed data
X_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv')
X_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_test.csv')
y_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv')
y_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_test.csv')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train.values.ravel())

# Train the Logistic Regression model on SMOTE data
model = LogisticRegression(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for Class 1 (fraud)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='blue', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Final Model)')
plt.legend()
plt.grid(True)
plt.savefig('C:/Users/Asus/Desktop/Final Project/precision_recall_curve.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='green', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Final Model)')
plt.legend()
plt.grid(True)
plt.savefig('C:/Users/Asus/Desktop/Final Project/roc_curve.png')
plt.show()