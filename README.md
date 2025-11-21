
# Loan Default Prediction Project

## Introduction
This project aims to develop a machine learning model to predict borrower default for a micro-finance institution. The goal is to enable better risk assessment and decision-making during the loan application process, ultimately contributing to financial stability and minimizing losses.

## Project Structure
All project files, including Jupyter notebooks, datasets (`.csv`), and generated plots (`.png` or other image formats), should be organized within a consistent directory structure. For instance, all plots generated during the analysis should be saved into a dedicated `plots/` directory. This organization ensures easy referencing, reproducibility, and maintainability of the project.

## Data Analysis
The 'loan_borowwer_data.csv' dataset was loaded into a pandas DataFrame. Initial data exploration involved displaying the first few rows and descriptive statistics. Categorical features, specifically the 'purpose' column, were one-hot encoded to convert them into numerical representations. Numerical features such as `int.rate`, `installment`, `log.annual.inc`, `dti`, `fico`, `days.with.cr.line`, `revol.bal`, `revol.util`, and `inq.last.6mths` were scaled using StandardScaler to standardize their values. Finally, the preprocessed data was split into training (70%) and testing (30%) sets, ensuring stratification for the target variable 'not.fully.paid' to maintain class proportions.

## Model Development
A RandomForestClassifier was chosen and trained on the preprocessed training data (`X_train`, `y_train`) to predict borrower default. The model was initialized with `random_state=42` for reproducibility and trained on the prepared features and target variable.

## Results
The trained RandomForestClassifier achieved an overall accuracy of 83.89% on the test set. However, a significant disparity in performance was observed for the two classes:
*   **Class 0 (No Default)**: The model showed strong performance with Precision 0.84, Recall 1.00, and F1-score 0.91, indicating high proficiency in identifying non-defaulting borrowers.
*   **Class 1 (Default)**: Performance for identifying defaulting borrowers was significantly lower, with Precision 0.41, Recall 0.02, and F1-score 0.03, highlighting a challenge in detecting actual defaults.

The Area Under the Receiver Operating Characteristic (AUC-ROC) score was 0.6541, suggesting moderate discriminative power but also reflecting the difficulty in separating the two classes. Key features influencing default prediction included `installment`, `log.annual.inc`, `revol.bal`, `days.with.cr.line`, `dti`, `revol.util`, `int.rate`, `fico`, `inq.last.6mths`, and `credit.policy`.

### Visualizations
Key model outcomes and feature importances are visualized through:
*   **Confusion Matrix**: Illustrates the number of correct and incorrect predictions made by the classification model.
*   **ROC Curve**: Shows the trade-off between the true positive rate and false positive rate at various threshold settings.
*   **Feature Importance Plot**: Displays the relative importance of each feature in the model's predictions.

_Note: These visualizations should be saved to a dedicated `plots/` directory within the project for easy access and reference._

## Recommendations
To improve the model's predictive power for defaulting borrowers, techniques to address class imbalance such as oversampling (e.g., SMOTE), undersampling, or using class weights should be explored and implemented. Micro-finance institutions should leverage the identified high-impact features like `installment`, `log.annual.inc`, and `revol.bal` during loan application assessment to better evaluate repayment capacity and financial stability, potentially leading to adjusted loan terms or targeted financial literacy support. Further model tuning and exploration of other algorithms could also enhance predictive performance for the minority class.
