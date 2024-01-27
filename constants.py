# constants.py

# Paths
DATA_PATH = './data/bank_data.csv'
EDA_FOLDER = 'images/eda'
RESULTS_FOLDER = 'images/results'
MODEL_FOLDER = './models'
BASE_OUTPUT_PATH = 'images/feature_importance'

# Logging Configuration
LOG_FILE_PATH = './results.log'
LOG_LEVEL = 'INFO'
LOG_FILE_MODE = 'w'
LOG_FORMAT = '%(name)s - %(levelname)s - %(message)s'


# Model Training Constants
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Model Parameters
RF_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
LR_PARAMS = {'solver': 'lbfgs', 'max_iter': 3000}


# Categorical Columns
CATEGORY_COLS = [
    'Gender', 
    'Education_Level', 
    'Marital_Status', 
    'Income_Category', 
    'Card_Category'
]


# Feature Engineering
KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]
