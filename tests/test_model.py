import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestration import preprocess_data, load_data
from web_service import preprocess_input
import model

class TestModelFunctions:
    """Test model-related functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'age': [45, 60, 35],
            'bmi': [25.5, 30.2, 22.1],
            'sleep_hours': [7.5, 6.0, 8.0],
            'family_history': ['Yes', 'No', 'Yes'],
            'diet': ['Healthy', 'Moderate', 'Unhealthy'],
            'exercise_frequency': ['Daily', 'Weekly', 'Rarely'],
            'smoking_status': ['Never', 'Former', 'Current'],
            'alcohol_consumption': ['Light', 'None', 'Moderate'],
            'risk_level': [0, 1, 2]
        })
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input for prediction"""
        return {
            'age': 45.0,
            'bmi': 25.5,
            'sleep_hours': 7.5,
            'family_history': 'Yes',
            'diet': 'Healthy',
            'exercise_frequency': 'Daily',
            'smoking_status': 'Never',
            'alcohol_consumption': 'Light'
        }
    
    def test_preprocess_data_with_target(self, sample_data):
        """Test preprocessing with target column"""
        X, y = preprocess_data(sample_data)
        
        # Check that target is separated
        assert y is not None
        assert len(y) == 3
        assert 'risk_level' not in X.columns
        assert 'id' not in X.columns
        
        # Check categorical encoding
        assert X['family_history'].dtype in ['int64', 'int32']
        assert X['diet'].dtype in ['int64', 'int32']
    
    def test_preprocess_data_without_target(self):
        """Test preprocessing without target column"""
        data = pd.DataFrame({
            'id': [1, 2],
            'age': [45, 60],
            'bmi': [25.5, 30.2],
            'sleep_hours': [7.5, 6.0],
            'family_history': ['Yes', 'No'],
            'diet': ['Healthy', 'Moderate'],
            'exercise_frequency': ['Daily', 'Weekly'],
            'smoking_status': ['Never', 'Former'],
            'alcohol_consumption': ['Light', 'None']
        })
        
        X, y = preprocess_data(data)
        
        # Check that no target is returned
        assert y is None
        assert 'id' not in X.columns
        assert len(X) == 2
    
    def test_preprocess_input(self, sample_input):
        """Test web service input preprocessing"""
        processed = preprocess_input(sample_input)
        
        # Check DataFrame structure
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 1
        
        # Check categorical mappings
        assert processed['family_history'].iloc[0] == 1  # 'Yes' -> 1
        assert processed['diet'].iloc[0] == 0  # 'Healthy' -> 0
        assert processed['smoking_status'].iloc[0] == 2  # 'Never' -> 2
    
    def test_categorical_mappings(self):
        """Test that categorical mappings are consistent"""
        test_data = {
            'age': 45.0,
            'bmi': 25.5,
            'sleep_hours': 7.5,
            'family_history': 'No',
            'diet': 'Unhealthy',
            'exercise_frequency': 'Rarely',
            'smoking_status': 'Current',
            'alcohol_consumption': 'Heavy'
        }
        
        processed = preprocess_input(test_data)
        
        # Check all mappings
        assert processed['family_history'].iloc[0] == 0  # 'No' -> 0
        assert processed['diet'].iloc[0] == 2  # 'Unhealthy' -> 2
        assert processed['exercise_frequency'].iloc[0] == 2  # 'Rarely' -> 2
        assert processed['smoking_status'].iloc[0] == 0  # 'Current' -> 0
        assert processed['alcohol_consumption'].iloc[0] == 0  # 'Heavy' -> 0
    
    def test_model_encode_data(self, sample_data):
        """Test the model's encode_data function"""
        X, y = model.encode_data(sample_data)
        
        # Check output structure
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 3
        
        # Check that categorical columns are encoded
        for col in X.columns:
            if col not in ['age', 'bmi', 'sleep_hours']:  # numerical columns
                assert X[col].dtype in ['int64', 'int32', 'int8']

class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        incomplete_data = {
            'age': 45.0,
            'bmi': 25.5,
            # Missing other required columns
        }
        
        with pytest.raises(KeyError):
            preprocess_input(incomplete_data)
    
    def test_invalid_categorical_values(self):
        """Test handling of invalid categorical values"""
        invalid_data = {
            'age': 45.0,
            'bmi': 25.5,
            'sleep_hours': 7.5,
            'family_history': 'Maybe',  # Invalid value
            'diet': 'Healthy',
            'exercise_frequency': 'Daily',
            'smoking_status': 'Never',
            'alcohol_consumption': 'Light'
        }
        
        processed = preprocess_input(invalid_data)
        # Should handle gracefully and return NaN for invalid mappings
        assert pd.isna(processed['family_history'].iloc[0])
    
    def test_numerical_edge_cases(self):
        """Test numerical edge cases"""
        edge_data = {
            'age': 0,  # Edge case
            'bmi': 100,  # Edge case
            'sleep_hours': 0,  # Edge case
            'family_history': 'Yes',
            'diet': 'Healthy',
            'exercise_frequency': 'Daily',
            'smoking_status': 'Never',
            'alcohol_consumption': 'Light'
        }
        
        processed = preprocess_input(edge_data)
        
        # Should handle edge cases without errors
        assert processed['age'].iloc[0] == 0
        assert processed['bmi'].iloc[0] == 100
        assert processed['sleep_hours'].iloc[0] == 0

class TestModelIntegration:
    """Integration tests for the complete model pipeline"""
    
    @pytest.fixture
    def trained_model(self):
        """Mock trained model for testing"""
        # Create a simple mock model
        class MockModel:
            def predict(self, X):
                # Return random predictions for testing
                return np.random.randint(0, 3, len(X))
            
            def predict_proba(self, X):
                # Return random probabilities for testing
                n_samples = len(X)
                probs = np.random.random((n_samples, 3))
                # Normalize to sum to 1
                return probs / probs.sum(axis=1, keepdims=True)
        
        return MockModel()
    
    def test_end_to_end_prediction(self, trained_model):
        """Test end-to-end prediction pipeline"""
        input_data = {
            'age': 45.0,
            'bmi': 25.5,
            'sleep_hours': 7.5,
            'family_history': 'Yes',
            'diet': 'Healthy',
            'exercise_frequency': 'Daily',
            'smoking_status': 'Never',
            'alcohol_consumption': 'Light'
        }
        
        # Preprocess
        processed_data = preprocess_input(input_data)
        
        # Predict
        prediction = trained_model.predict(processed_data)
        probabilities = trained_model.predict_proba(processed_data)
        
        # Validate outputs
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]
        assert probabilities.shape == (1, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

class TestPerformanceAndBounds:
    """Test performance and boundary conditions"""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets"""
        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'id': range(n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'bmi': np.random.uniform(15, 40, n_samples),
            'sleep_hours': np.random.uniform(4, 12, n_samples),
            'family_history': np.random.choice(['Yes', 'No'], n_samples),
            'diet': np.random.choice(['Healthy', 'Moderate', 'Unhealthy'], n_samples),
            'exercise_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], n_samples),
            'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
            'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
            'risk_level': np.random.randint(0, 3, n_samples)
        })
        
        # Should process without errors
        X, y = preprocess_data(large_data)
        
        assert len(X) == n_samples
        assert len(y) == n_samples
        assert X.shape[1] > 0  # Should have features
    
    def test_memory_usage(self):
        """Test memory usage with realistic data sizes"""
        # This is a placeholder for memory usage tests
        # In practice, you might use memory_profiler or similar tools
        pass

if __name__ == "__main__":
    pytest.main([__file__])