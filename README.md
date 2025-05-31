# Fraud Detection Project

This is my MCA project to find fraud in credit card transactions using AI. I used Python with Logistic Regression and SMOTE. The final F1 score is 77.51%, which means it catches most frauds with fewer mistakes.

## Files in This Project
- **scripts/**: Has the code files (`.py`) to run the project.
- **outputs/**: Has the pictures (`.png`) like graphs and charts showing the results.
- **about.txt**: Explains more about the project.

## How to Get the Data File (creditcard.csv)
The project needs a file called `creditcard.csv`, but it’s too big to upload here. You can download it from this website:
- Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Download the `creditcard.csv` file (you might need a free Kaggle account).
- Save the `creditcard.csv` file in the main folder of this project (not inside `scripts/` or `outputs/`, but next to them).

## How to Run the Project
1. Download all the files from this repository.
2. Put the `creditcard.csv` file in the main folder.
3. Install Python on your computer.
4. Install these tools using the terminal : pip install pandas pip install numpy pip install scikit-learn pip install imblearn pip install matplotlib pip install seaborn
5. Go to the `scripts/` folder and run the files in this order : python preprocess.py python train_model.py python train_smote.py python train_smote_threshold.py python f1_comparison.py python more_plots.py python workflow_comparison.py python system_design_flowchart.py
6. Check the `outputs/` folder for the pictures showing the results !
