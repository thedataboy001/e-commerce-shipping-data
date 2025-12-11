DROP_COLUMNS = "ID"
TARGET_COLUMN = 'Reached.on.Time_Y.N'
RANDOM_STATE = 24
TEST_SIZE = 0.3
RAW_DATA_PATH = "data/01-raw/raw_data.csv"
PROCESSED_DATA_PATH = "data/02-preprocessed/processed_data.csv"
NUMERICAL_FEATURES = ["Cost_of_the_Product", "Discount_offered", "Weight_in_gms", "Prior_purchases"]
CATEGORICAL_FEATURES = ["Warehouse_block", "Mode_of_Shipment"]


from pathlib import Path

ARTIFACTS_DIR = Path(
    r"C:\Users\eeluf\Desktop\Dev\e-commerce shipping data (classification model)\models"
)
