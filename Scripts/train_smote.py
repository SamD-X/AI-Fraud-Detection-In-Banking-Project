import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed data
X_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv')
X_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_test.csv')
y_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv')
y_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_test.csv')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train.values.ravel())

# Check the new class distribution
print("Class Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Train the Logistic Regression model on SMOTE data
model = LogisticRegression(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('C:/Users/Asus/Desktop/Final Project/confusion_matrix_smote.png')
plt.show()