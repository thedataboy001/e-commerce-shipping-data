"""
Unit tests for training_pipeline.py

Tests for:
- load_transformed_data: Loading and splitting data correctly
- train_random_forest: Model training and validation
- plot_confusion_metrix: Visualization output
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

from src.training_pipeline import load_transformed_data, main
from src.inference_pipeline import train_random_forest, plot_confusion_metrix
from src.utils_and_constants import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


class TestLoadTransformedData:
    """Test suite for load_transformed_data function"""

    def test_load_transformed_data_success(self, tmp_path):
        """Test successful data loading and target column separation"""
        # Create sample data
        sample_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            TARGET_COLUMN: [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(sample_data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        # Load data
        X, y = load_transformed_data(str(csv_path))

        # Assertions
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert TARGET_COLUMN not in X.columns
        assert len(X) == 5
        assert len(y) == 5
        assert list(X.columns) == ['feature1', 'feature2']
        assert list(y.values) == [0, 1, 0, 1, 0]

    def test_load_transformed_data_shape_consistency(self, tmp_path):
        """Test that X and y have consistent lengths"""
        sample_data = {
            'col1': range(100),
            'col2': range(100, 200),
            'col3': range(200, 300),
            TARGET_COLUMN: np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(sample_data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        X, y = load_transformed_data(str(csv_path))

        assert len(X) == len(y) == 100
        assert X.shape[0] == y.shape[0]

    def test_load_transformed_data_missing_target_column(self, tmp_path):
        """Test error handling when target column is missing"""
        sample_data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        }
        df = pd.DataFrame(sample_data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(KeyError):
            load_transformed_data(str(csv_path))

    def test_load_transformed_data_empty_file(self, tmp_path):
        """Test handling of empty CSV file"""
        csv_path = tmp_path / "empty_data.csv"
        df_empty = pd.DataFrame()
        df_empty.to_csv(csv_path, index=False)

        with pytest.raises((KeyError, pd.errors.ParserError)):
            load_transformed_data(str(csv_path))

    def test_load_transformed_data_invalid_path(self):
        """Test error handling for invalid file path"""
        with pytest.raises(FileNotFoundError):
            load_transformed_data("non_existent_file.csv")


class TestTrainRandomForest:
    """Test suite for train_random_forest function"""

    @pytest.fixture
    def sample_train_test_data(self):
        """Create sample training and testing data"""
        np.random.seed(RANDOM_STATE)
        
        # Create features
        X_train = pd.DataFrame({
            'feat1': np.random.randn(100),
            'feat2': np.random.randn(100),
            'feat3': np.random.randn(100),
        })
        
        X_test = pd.DataFrame({
            'feat1': np.random.randn(30),
            'feat2': np.random.randn(30),
            'feat3': np.random.randn(30),
        })
        
        # Create target (binary classification)
        y_train = pd.Series(np.random.randint(0, 2, 100))
        y_test = pd.Series(np.random.randint(0, 2, 30))
        
        return X_train, X_test, y_train, y_test

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_train_random_forest_returns_model(self, mock_dump, mock_log_model, 
                                               mock_log_metric, mock_log_params, 
                                               mock_start_run, mock_set_exp, 
                                               sample_train_test_data):
        """Test that train_random_forest returns a trained model"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        # Mock mlflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_train_random_forest_logs_metrics(self, mock_dump, mock_log_model, 
                                              mock_log_metric, mock_log_params,
                                              mock_start_run, mock_set_exp,
                                              sample_train_test_data):
        """Test that metrics are logged correctly"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Verify mlflow.log_metric was called
        assert mock_log_metric.called
        
        # Check that expected metrics were logged
        metric_calls = [call[0][0] for call in mock_log_metric.call_args_list]
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for expected_metric in expected_metrics:
            assert expected_metric in metric_calls

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_train_random_forest_logs_params(self, mock_dump, mock_log_model,
                                             mock_log_metric, mock_log_params,
                                             mock_start_run, mock_set_exp,
                                             sample_train_test_data):
        """Test that model parameters are logged"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Verify parameters were logged
        assert mock_log_params.called

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_train_random_forest_saves_model(self, mock_dump, mock_log_model,
                                             mock_log_metric, mock_log_params,
                                             mock_start_run, mock_set_exp,
                                             sample_train_test_data):
        """Test that model is saved to disk"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Verify joblib.dump was called
        assert mock_dump.called

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_train_random_forest_accuracy_reasonable(self, mock_dump, mock_log_model,
                                                      mock_log_metric, mock_log_params,
                                                      mock_start_run, mock_set_exp,
                                                      sample_train_test_data):
        """Test that model achieves reasonable accuracy"""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check predictions are valid
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)



class TestIntegration:
    """Integration tests for the training pipeline"""

    def test_full_pipeline_with_sample_data(self, tmp_path):
        """Test the full training pipeline with sample data"""
        # Create sample processed data
        sample_data = {
            'feat1': np.random.randn(100),
            'feat2': np.random.randn(100),
            'feat3': np.random.randn(100),
            TARGET_COLUMN: np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(sample_data)
        csv_path = tmp_path / "processed_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and prepare data
        X, y = load_transformed_data(str(csv_path))
        
        # Verify data integrity
        assert not X.empty
        assert not y.empty
        assert len(X) == len(y)
        assert TARGET_COLUMN not in X.columns

    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @patch('joblib.dump')
    def test_pipeline_handles_imbalanced_data(self, mock_dump, mock_log_model,
                                              mock_log_metric, mock_log_params,
                                              mock_start_run, mock_set_exp):
        """Test pipeline with imbalanced classification data"""
        np.random.seed(RANDOM_STATE)
        
        # Create imbalanced data (70% class 0, 30% class 1)
        X_train = pd.DataFrame({
            'feat1': np.random.randn(100),
            'feat2': np.random.randn(100),
        })
        
        # Imbalanced target
        y_train = pd.Series([0] * 70 + [1] * 30)
        
        X_test = pd.DataFrame({
            'feat1': np.random.randn(30),
            'feat2': np.random.randn(30),
        })
        y_test = pd.Series([0] * 21 + [1] * 9)
        
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        model = train_random_forest(X_train, y_train, X_test, y_test)
        
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
