import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Section 1: Data Loading and Preprocessing (from preprocess.py)
def preprocess_data():
    # Load the dataset
    data = pd.read_csv('C:/Users/Asus/Desktop/Final Project/creditcard.csv')
    
    # Check for missing values
    print("Missing Values:")
    print(data.isnull().sum())
    
    # Separate features (X) and target (y)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Scale the features
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
    
    # Save the preprocessed data
    X_train.to_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv', index=False)
    X_test.to_csv('C:/Users/Asus/Desktop/Final Project/X_test.csv', index=False)
    y_train.to_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv', index=False)
    y_test.to_csv('C:/Users/Asus/Desktop/Final Project/y_test.csv', index=False)
    print("\nPreprocessed files saved in Final Project folder.")
    return X_train, X_test, y_train, y_test

# Section 2: Train Initial Model (from train_model.py)
def train_initial_model(X_train, X_test, y_train, y_test):
    # Train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Initial Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    # Confusion matrix (renamed to avoid overwrite)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Initial Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('C:/Users/Asus/Desktop/Final Project/confusion_matrix_initial.png')
    plt.close()

# Section 3: Apply SMOTE and Train Model (from train_smote.py)
def train_smote_model():
    # Load preprocessed data
    X_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv')
    y_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv')
    y_train = y_train.values.ravel()
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_smote, y_train_smote)
    print("SMOTE model trained with default threshold (0.5).")
    return model

# Section 4: Threshold Tuning (from train_smote_threshold.py)
def tune_threshold():
    # Load preprocessed data
    X_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_train.csv')
    y_train = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_train.csv')
    X_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/X_test.csv')
    y_test = pd.read_csv('C:/Users/Asus/Desktop/Final Project/y_test.csv')
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    # Apply SMOTE and train model
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_smote, y_train_smote)
    
    # Get prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Save probabilities for visualization
    np.save('C:/Users/Asus/Desktop/Final Project/y_prob.npy', y_prob)
    
    # Apply threshold
    threshold = 0.999
    y_pred = (y_prob >= threshold).astype(int)
    
    # Evaluate
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Final Model Performance (Threshold 0.999):")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return y_test, y_prob

# Section 5: F1 Score Comparison (from f1_comparison.py)
def plot_f1_comparison():
    # F1 scores for different models
    models = ['Initial Model', 'SMOTE (Threshold 0.5)', 'Final Model (Threshold 0.999)']
    f1_scores = [0.6951, 0.1107, 0.7751]
    
    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(models, f1_scores, color=['blue', 'orange', 'green'])
    plt.ylim(0, 1)
    plt.title('F1 Score Comparison Across Models')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    plt.savefig('C:/Users/Asus/Desktop/Final Project/f1_comparison.png')
    plt.close()

# Section 6: Generate Multiple Visualizations (from more_plots.py)
def generate_plots(y_test, y_prob):
    y_pred = (y_prob >= 0.999).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Final Model, Threshold 0.999)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('C:/Users/Asus/Desktop/Final Project/confusion_matrix.png')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve (Final Model)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig('C:/Users/Asus/Desktop/Final Project/precision_recall_curve.png')
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Final Model)')
    plt.legend(loc="lower right")
    plt.savefig('C:/Users/Asus/Desktop/Final Project/roc_curve.png')
    plt.close()
    # Note: Class distribution and metrics vs. threshold plots are omitted for brevity

# Section 7: Workflow Comparison Diagram (from workflow_comparison.py)
def plot_workflow_comparison():
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
    plt.close()

# Section 8: System Design Flowchart (from system_design_flowchart.py)
def plot_system_design_flowchart():
    # Simplified placeholder for system design flowchart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.9, 'System Design Flowchart', fontsize=12, ha='center', bbox=dict(facecolor='lightyellow', edgecolor='black'))
    ax.text(0.5, 0.7, '1. Load Data', fontsize=10, ha='center')
    ax.text(0.5, 0.5, '2. Preprocess Data', fontsize=10, ha='center')
    ax.text(0.5, 0.3, '3. Train Model', fontsize=10, ha='center')
    ax.text(0.5, 0.1, '4. Evaluate & Visualize', fontsize=10, ha='center')
    ax.add_patch(FancyArrowPatch((0.5, 0.65), (0.5, 0.55), mutation_scale=20, color='black'))
    ax.add_patch(FancyArrowPatch((0.5, 0.45), (0.5, 0.35), mutation_scale=20, color='black'))
    ax.add_patch(FancyArrowPatch((0.5, 0.25), (0.5, 0.15), mutation_scale=20, color='black'))
    plt.title('Figure 2: System Design Flowchart', fontsize=14)
    ax.axis('off')
    plt.savefig('C:/Users/Asus/Desktop/Final Project/system_design_flowchart.png', bbox_inches='tight')
    plt.close()

# Main execution block
if __name__ == "__main__":
    print("Starting Fraud Detection Project...")
    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Step 2: Train initial model
    train_initial_model(X_train, X_test, y_train, y_test)
    
    # Step 3: Train SMOTE model
    train_smote_model()
    
    # Step 4: Tune threshold and get probabilities
    y_test, y_prob = tune_threshold()
    
    # Step 5: Plot F1 score comparison
    plot_f1_comparison()
    
    # Step 6: Generate other visualizations
    generate_plots(y_test, y_prob)
    
    # Step 7: Plot workflow comparison
    plot_workflow_comparison()
    
    # Step 8: Plot system design flowchart
    plot_system_design_flowchart()
    
    print("Fraud Detection Project completed. Check the Final Project folder for outputs.")