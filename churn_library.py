"""
Churn Library Script

This script allows the user to perform a series of data science tasks associated with
predicting customer churn. It includes data import, exploratory data analysis (EDA),
feature engineering, model training, and evaluation.

Functions included allow for:
- Importing data from a CSV file.
- Performing EDA with various visualizations such as histograms, bar plots, heatmaps,
  boxplots, count plots, pair plots, and violin plots.
- Encoding categorical variables and preparing data for modeling.
- Training Random Forest and Logistic Regression models.
- Generating and saving classification reports and ROC curves for model evaluation.
- Creating and storing feature importance plots using traditional methods and SHAP values.

The script is structured to run in a sequential manner, where each step of the data
processing and analysis pipeline is executed in order.

Author: Jeremy Gu
Date: 1/21/2024
Update: 1/22/2024 include all parameters and constants in constants.py.

"""

# import libraries
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
import joblib

# Import constants from constants.py
import constants as c

# Configure seaborn
sns.set()

# Set environment variable for matplotlib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# # Configure logging
# logging.basicConfig(
#     filename=c.LOG_FILE_PATH,
#     level=logging.getLevelName(
#         c.LOG_LEVEL),
#     filemode=c.LOG_FILE_MODE,
#     format=c.LOG_FORMAT)


def ensure_dir(file_path):
    """
    ensures the directory exists for a file path
    input:
        file_path: string of file path
    output:
        None
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Part 1. checked


def import_data(pth):
    """
    returns dataframe for the csv found at pth
    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    """
    logging.info("Importing data from %s", pth)
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    logging.info("Data imported successfully and 'Churn' column created.")
    return df

# part 2. checked


def perform_eda(df):
    """
    Perform EDA on df and save figures to images/eda folder
    input:
        df: pandas dataframe
    output:
        None
    """
    if not os.path.exists(c.EDA_FOLDER):
        os.makedirs(c.EDA_FOLDER)

    logging.info("Performing Exploratory Data Analysis (EDA)")

    # Histogram of Churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(c.EDA_FOLDER, 'churn_hist.png'))

    # Histogram of Customer Age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(c.EDA_FOLDER, 'customer_age_hist.png'))

    # Bar plot of Marital Status
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(c.EDA_FOLDER, 'marital_status_bar.png'))

    # Distribution plot of Total Transactions Count
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], kde=True)
    plt.savefig(os.path.join(c.EDA_FOLDER, 'total_trans_ct_dist.png'))

    # Heatmap of Correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(os.path.join(c.EDA_FOLDER, 'heatmap.png'))

    # Additional Plots
    # Boxplot for Quantitative Features
    plt.figure(figsize=(20, 10))
    df.boxplot(column=['Customer_Age', 'Total_Trans_Ct', 'Credit_Limit'])
    plt.savefig(os.path.join(c.EDA_FOLDER, 'quantitative_boxplots.png'))

    # Count plot for a Categorical Feature (e.g., Education Level)
    plt.figure(figsize=(20, 10))
    sns.countplot(x='Education_Level', data=df)
    plt.savefig(os.path.join(c.EDA_FOLDER, 'education_level_count.png'))

    # Pairplot for first few Numeric Features
    plt.figure(figsize=(20, 20))
    sns.pairplot(df[['Customer_Age', 'Total_Relationship_Count',
                 'Months_Inactive_12_mon', 'Credit_Limit']])
    plt.savefig(os.path.join(c.EDA_FOLDER, 'pairplot.png'))

    # Violin Plot for Credit Limit
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='Churn', y='Credit_Limit', data=df)
    plt.savefig(os.path.join(c.EDA_FOLDER, 'credit_limit_violin.png'))

    logging.info("EDA completed and figures saved in the images/eda folder.")

# part 3. checked


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category
    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name
    output:
        df: pandas dataframe with new columns for
    """
    logging.info("Encoding categorical features.")
    for category in category_lst:
        cat_list = []
        cat_groups = df.groupby(category)[response].mean()

        for val in df[category]:
            cat_list.append(cat_groups.loc[val])

        df[f'{category}_{response}'] = cat_list
    logging.info("Categorical features encoded successfully.")
    return df

# part 4. checked


def perform_feature_engineering(dataframe, response_column):
    """
    input:
        df: pandas dataframe
        response: string of response name
    output:
        X_train, X_test, y_train, y_test
    """
    logging.info("Performing feature engineering...")

    # Create new features
    for col in c.CATEGORY_COLS:
        dataframe[col + '_Churn'] = dataframe.groupby(col)[response_column].transform('mean')

    x = dataframe[c.KEEP_COLS]
    y = dataframe[response_column]

    # Split data into training and test sets
    logging.info("Splitting data into training and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE)
    logging.info("Feature engineering completed.")
    return x_train, x_test, y_train, y_test


# part 5. checked
def classification_report_image(
        rfc_model,
        lrc_model,
        X_train,
        X_test,
        y_train,
        y_test):
    """
    Generates and saves classification reports and ROC curve images for trained models.
    """
    logging.info("Generating classification reports and ROC curves...")

    # Ensure the directories exist
    ensure_dir(os.path.join(c.RESULTS_FOLDER, 'rfc_classification_report.png'))
    ensure_dir(os.path.join(c.RESULTS_FOLDER, 'lrc_classification_report.png'))
    ensure_dir(os.path.join(c.RESULTS_FOLDER, 'lrc_roc_curve.png'))
    ensure_dir(os.path.join(c.RESULTS_FOLDER, 'rfc_roc_curve.png'))

    # Generate predictions
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)
    y_train_preds_lr = lrc_model.predict(X_train)
    y_test_preds_lr = lrc_model.predict(X_test)

    # Generate and save classification report for Random Forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, 'Random Forest Train', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, classification_report(
            y_train, y_train_preds_rf), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Random Forest Test', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, classification_report(
            y_test, y_test_preds_rf), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        os.path.join(
            c.RESULTS_FOLDER,
            'rfc_classification_report.png'))
    logging.info("Classification report for Random Forest saved.")

    # Generate and save classification report for Logistic Regression
    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, 'Logistic Regression Train', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, classification_report(
            y_train, y_train_preds_lr), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Logistic Regression Test', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, classification_report(
            y_test, y_test_preds_lr), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        os.path.join(
            c.RESULTS_FOLDER,
            'lrc_classification_report.png'))
    logging.info("Classification report for Logistic Regression saved.")

    # Generate and save ROC curve for Logistic Regression
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(lrc_model, X_test, y_test, ax=ax, alpha=0.8)
    plt.title('ROC Curve - Logistic Regression')
    plt.savefig(os.path.join(c.RESULTS_FOLDER, 'lrc_roc_curve.png'))
    logging.info("ROC curve for Logistic Regression saved.")

    # Generate and save ROC curve for Random Forest Classifier
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    plt.title('ROC Curve - Random Forest')
    plt.savefig(os.path.join(c.RESULTS_FOLDER, 'rfc_roc_curve.png'))
    logging.info("ROC curve for Random Forest saved.")

# part 6.


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances using traditional method and SHAP values.
    """
    # Ensure the directory exists
    output_dir = os.path.dirname(output_pth)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Creating and saving traditional feature importance plot.")
    # Traditional feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importances")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_pth}_traditional.png')

    logging.info(
        f"Traditional feature importance plot saved in {output_pth}_traditional.png.")

    logging.info("Creating and saving SHAP feature importance plot.")
    # SHAP feature importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.tight_layout()
    plt.savefig(f'{output_pth}_shap.png')

    logging.info(
        f"SHAP feature importance plot saved in {output_pth}_shap.png.")


# Part 7. chekced
def train_models(X_train, y_train):
    """
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        y_train: y training data

    output:
        None
    """
    logging.info("Training models started.")

    # Random Forest Classifier
    logging.info("Training Random Forest Classifier...")
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=c.RF_PARAM_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info("Random Forest training completed.")

    # Logistic Regression Classifier
    logging.info("Training Logistic Regression Classifier...")
    lrc = LogisticRegression(**c.LR_PARAMS)
    lrc.fit(X_train, y_train)
    logging.info("Logistic Regression training completed.")
    logging.info("Training models completed.")

    # Saving models
    logging.info("Saving trained models...")
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            c.MODEL_FOLDER,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(c.MODEL_FOLDER, 'logistic_model.pkl'))
    logging.info("Models saved.")

    return cv_rfc.best_estimator_, lrc


if __name__ == "__main__":

    print("Running churn_library_solution...")
    logging.info("Running churn_library_solution...")
    # Step 1: Data Import and Exploratory Data Analysis
    logging.info("Step 1: Data Import and Exploratory Data Analysis")
    df = import_data(c.DATA_PATH)
    perform_eda(df)
    logging.info("Step 1 completed.")
    # Step 2: Encoding and Feature Engineering
    logging.info("Step 2: Encoding and Feature Engineering")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, category_lst, 'Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    logging.info("Step 2 completed.")
    # Step 3: Model Training
    logging.info("Step 3: Model Training")
    rfc_model, lrc_model = train_models(X_train, y_train)
    logging.info("Step 3 completed.")

    # Step 4: Generating and Saving Classification Reports and ROC Curves
    logging.info(
        "Step 4: Generating and Saving Classification Reports and ROC Curves")
    classification_report_image(
        rfc_model,
        lrc_model,
        X_train,
        X_test,
        y_train,
        y_test)
    logging.info("Step 4 completed.")
    # Step 5: Generating and Saving Feature Importance Plot
    logging.info("Step 5: Generating and Saving Feature Importance Plot")
    feature_importance_plot(rfc_model, X_train, c.BASE_OUTPUT_PATH)
    logging.info("rfc_model feature importance plot saved.")
    logging.info("Step 5 completed.")
    print("Script execution completed.")
    logging.info("Script execution completed.")
