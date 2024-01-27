"""
Module to test the churn_script_solution.py script.

To run the tests, use the following command:
pytest -v test_churn_script_solution.py

# To run the tests with logging enabled, use the following command:
# pytest.ini

[pytest]
log_cli = true
log_cli_level = INFO
log_file = logs/churn_library.log
log_file_level = DEBUG
log_file_format = %(asctime)s %(levelname)s %(message)s
log_file_date_format = %Y-%m-%d %H:%M:%S

"""
from unittest.mock import patch
import shutil
import os
import pandas as pd
import pytest
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models, classification_report_image, feature_importance_plot
from constants import EDA_FOLDER, MODEL_FOLDER, RESULTS_FOLDER
import logging


@pytest.fixture
def sample_data():
    return "./data/bank_data.csv"

# Test for import_data function


def test_import_data(sample_data):
    # Call the function with the path of the sample CSV
    logging.info("Testing import_data")
    df = import_data(sample_data)
    logging.info("Checking if the returned object is a DataFrame")

    # Check if the returned object is a DataFrame
    logging.info("Checking if the returned object is a DataFrame")
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"


# ---- CONTINUE TESTING ONE BY ONE ----

# # Test for perform_eda function

# Test for perform_eda function

def test_perform_eda(sample_data):
    # Call the function with the path of the sample CSV
    logging.info("Testing perform_eda")

    # Load the data
    df = import_data(sample_data)

    # Ensure EDA folder is clean before test
    if os.path.exists(EDA_FOLDER):
        # Remove the EDA folder
        shutil.rmtree(EDA_FOLDER)

    # Perform EDA
    perform_eda(df)

    # Check if EDA_FOLDER is created
    # assert os.path.exists(EDA_FOLDER), "EDA folder was not created"
    assert os.path.isdir(EDA_FOLDER), "EDA folder was not created"

    # List of expected output files
    expected_files = [
        'churn_hist.png',
        'customer_age_hist.png',
        'marital_status_bar.png',
        'total_trans_ct_dist.png',
        'heatmap.png',
        'quantitative_boxplots.png',
        'education_level_count.png',
        'pairplot.png',
        'credit_limit_violin.png'
    ]

    # Verify that each file is created
    for file in expected_files:
        file_path = os.path.join(EDA_FOLDER, file)
        assert os.path.isfile(file_path), f"File {file} was not created"

        # Optionally check file size (not zero)
        assert os.path.getsize(file_path) > 0, f"File {file} is empty"

    logging.info("perform_eda test passed successfully")

    # Clean up: Remove the EDA folder after the test
    shutil.rmtree(EDA_FOLDER)

# Test for encoder_helper function


def test_encoder_helper(sample_data):
    logging.info("Testing encoder_helper")

    # Load the data
    df = import_data(sample_data)

    # Define the categorical columns and response
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = 'Churn'

    # Original number of columns
    original_num_columns = df.shape[1]

    # Call encoder_helper
    df_encoded = encoder_helper(df, category_lst, response)

    # Check if new columns are added
    assert df_encoded.shape[1] == original_num_columns + \
        len(category_lst), "New columns were not added correctly"

    # Verify that each new column has correct name and data
    for category in category_lst:
        new_column_name = f'{category}_{response}'
        assert new_column_name in df_encoded.columns, f"{new_column_name} is not in the dataframe"

        # Check if the new column values are between 0 and 1 (as they represent
        # proportions)
        assert all(0 <= x <= 1 for x in df_encoded[new_column_name]
                   ), f"Values in {new_column_name} are not valid proportions"

    logging.info("encoder_helper test passed successfully")

#
    # Test for perform_feature_engineering function


def test_perform_feature_engineering(sample_data):
    logging.info("Testing perform_feature_engineering")

    # Load the data
    df = import_data(sample_data)
    response_column = 'Churn'

    # Original number of columns
    original_num_columns = df.shape[1]

    # Call perform_feature_engineering
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df, response_column)

    # Check if new columns are added
    assert df.shape[1] > original_num_columns, "New columns were not added correctly"

    # Check the size of the splits
    assert x_train.shape[0] < df.shape[0] and x_test.shape[0] < df.shape[0], "Data splitting is incorrect"

    logging.info("perform_feature_engineering test passed successfully")


# Test for train_models function

def test_train_models(sample_data):
    logging.info("Testing train_models")

    # Load a small subset of data for testing
    df = import_data(sample_data)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df.head(200), 'Churn')

    # Mock the actual training process
    with patch('joblib.dump') as mock_dump:
        # Call train_models
        rfc_model, lrc_model = train_models(x_train, y_train)

        # Ensure models are saved (mocking joblib.dump)
        assert mock_dump.call_count == 2, "Models were not saved correctly"

    logging.info("train_models test passed successfully")


# Test for classification_report_image function
def test_classification_report_image(sample_data):
    logging.info("Testing classification_report_image")

    # Load the data and train models for testing
    df = import_data(sample_data)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df.head(100), 'Churn')
    rfc_model, lrc_model = train_models(x_train, y_train)

    # Call classification_report_image
    classification_report_image(
        rfc_model,
        lrc_model,
        x_train,
        x_test,
        y_train,
        y_test)

    # Expected output files
    expected_files = [
        'rfc_classification_report.png',
        'lrc_classification_report.png',
        'lrc_roc_curve.png',
        'rfc_roc_curve.png'
    ]

    # Check if files are created
    for file in expected_files:
        file_path = os.path.join(RESULTS_FOLDER, file)
        assert os.path.isfile(file_path), f"{file} was not created"

    logging.info("classification_report_image test passed successfully")


# Test for feature_importance_plot function
def test_feature_importance_plot(sample_data):
    logging.info("Testing feature_importance_plot")

    # Load data and train a model for testing
    df = import_data(sample_data)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df.head(100), 'Churn')
    rfc_model, _ = train_models(x_train, y_train)

    # Define output path
    output_pth = os.path.join(RESULTS_FOLDER, 'feature_importance')

    # Call feature_importance_plot
    feature_importance_plot(rfc_model, x_train, output_pth)

    # Expected output files
    expected_files = [
        f'{output_pth}_traditional.png',
        f'{output_pth}_shap.png'
    ]

    # Check if files are created
    for file in expected_files:
        assert os.path.isfile(file), f"{file} was not created"

    logging.info("feature_importance_plot test passed successfully")
