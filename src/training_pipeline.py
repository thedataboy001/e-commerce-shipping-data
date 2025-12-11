import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.inference_pipeline import train_random_forest, plot_confusion_metrix
from src.utils_and_constants import (PROCESSED_DATA_PATH, 
                                 TARGET_COLUMN, RANDOM_STATE, 
                                 TEST_SIZE)

from pandas.errors import EmptyDataError, ParserError


# def load_transformed_data(filepath):
#     df = pd.read_csv(filepath)
#     X = df.drop(TARGET_COLUMN, axis=1)
#     y = df[TARGET_COLUMN]

#     return X, y


def load_transformed_data(filepath: str = PROCESSED_DATA_PATH):
    """
    Load preprocessed data and return (X, y).

    Raises ParserError (or KeyError) for empty or invalid CSV,
    matching what the tests expect.
    """
    try:
        df = pd.read_csv(filepath)
    except EmptyDataError as e:
        # Re-raise as ParserError so tests see the expected exception type
        raise ParserError("Empty or invalid CSV file") from e

    # Normal path: split into X and y
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in data.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y

def main():
    X, y = load_transformed_data()

        # Decide whether to stratify or not
    value_counts = y.value_counts()

    if len(value_counts) > 1 and value_counts.min() >= 2:
        stratify_arg = y
    else:
        # Too few samples per class (like in the unit test),
        # fall back to non-stratified split
        stratify_arg = None 

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size= TEST_SIZE, 
                                                        random_state= RANDOM_STATE, 
                                                        stratify=stratify_arg)
    
    model = train_random_forest(X_train, y_train, X_test, y_test)
    plot_confusion_metrix(model, X_test, y_test)

if __name__ == "__main__":
    main()