import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/Asus/Desktop/Final Project/creditcard.csv')

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Separate features (X) and target (y)
X = data.drop('Class', axis=1)  # Features: Time, V1-V28, Amount
y = data['Class']  # Target: Class (0 or 1)

# Scale the features (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print("\nTraining Set Shape (X_train):", X_train.shape)
print("Testing Set Shape (X_test):", X_test.shape)
print("Training Labels Shape (y_train):", y_train.shape)
print("Testing Labels Shape (y_test):", y_test.shape)

# Save the preprocessed data for later use
X_train.to_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv', index=False)
X_test.to_csv('C:/Users/Asus/Desktop/Final Project/X_test.csv', index=False)
y_train.to_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv', index=False)
y_test.to_csv('C:/Users/Asus/Desktop/Final Project/y_test.csv', index=False)
print("\nPreprocessed files saved in Final Project folder.")