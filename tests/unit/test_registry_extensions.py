"""
Test cases for Registry Pattern Extensions.

Following strict TDD approach - tests written first, implementation follows.
Tests cover all 5 registry extensions:
- ModelRegistry for ML model factory
- QualityChecksRegistry for assessment pipelines  
- PreprocessingRegistry for raw processing steps
- TrainingStrategyRegistry for training patterns
- TransferFunctionRegistry for data transformations
"""

import pytest
import torch
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock


class TestModelRegistry:
    """Test cases for ModelRegistry ML model factory."""
    
    def test_model_registry_exists(self):
        """Test that ModelRegistry exists and can be imported."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        assert ModelRegistry is not None
    
    def test_model_registry_is_singleton(self):
        """Test that ModelRegistry follows singleton pattern."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        assert registry1 is registry2
    
    def test_model_registry_has_default_models(self):
        """Test that ModelRegistry contains default ML models."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        models = registry.list_available_models()
        
        # Should have at least these models based on memory
        expected_models = ['utnet2', 'utnet3', 'bm3d', 'learned_denoise']
        for model in expected_models:
            assert model in models
    
    def test_model_registry_create_model(self):
        """Test creating a model instance from registry."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Test creating a mock model (since actual models may not be available)
        model = registry.create_model('utnet2', in_channels=4, out_channels=3)
        assert model is not None
        assert hasattr(model, 'forward')  # Should be a PyTorch module
    
    def test_model_registry_register_custom_model(self):
        """Test registering a custom model in the registry."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Create mock model class
        class MockModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Register custom model
        registry.register_model('custom_model', MockModel)
        
        # Should be able to create instance
        model = registry.create_model('custom_model')
        assert isinstance(model, MockModel)
    
    @pytest.mark.parametrize("in_channels", [3, 4])
    def test_model_registry_create_model_with_params(self, in_channels):
        """Test creating model with specific parameters."""
        from fiddlesticks.registries.model_registry import ModelRegistry

        registry = ModelRegistry()

        # Test parameter passing
        params = {'in_channels': in_channels, 'out_channels': 6, 'hidden_dim': 256}
        model = registry.create_model('utnet2', **params)

        # Should have created model with parameters
        assert model is not None
    
    def test_model_registry_unknown_model_error(self):
        """Test error handling for unknown model names."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Unknown model"):
            registry.create_model('nonexistent_model')


class TestQualityChecksRegistry:
    """Test cases for QualityChecksRegistry assessment pipelines."""
    
    def test_quality_checks_registry_exists(self):
        """Test that QualityChecksRegistry exists and can be imported."""
        from fiddlesticks.registries.quality_checks_registry import QualityChecksRegistry
        assert QualityChecksRegistry is not None
    
    def test_quality_checks_registry_has_default_checks(self):
        """Test that registry contains default quality checks."""
        from fiddlesticks.registries.quality_checks_registry import QualityChecksRegistry
        
        registry = QualityChecksRegistry()
        checks = registry.list_available_checks()
        
        # Should have at least these checks based on memory
        expected_checks = ['overexposure', 'underexposure', 'noise_level', 'blur_detection']
        for check in expected_checks:
            assert check in checks
    
    def test_quality_checks_create_pipeline(self):
        """Test creating quality assessment pipeline."""
        from fiddlesticks.registries.quality_checks_registry import QualityChecksRegistry
        
        registry = QualityChecksRegistry()
        
        config = {
            'overexposure': {'threshold': 0.01},
            'noise_level': {'max_std': 0.1}
        }
        
        pipeline = registry.create_quality_pipeline(config)
        assert pipeline is not None
        assert hasattr(pipeline, '__call__')
    
    def test_quality_checks_pipeline_execution(self):
        """Test executing quality assessment pipeline on image."""
        from fiddlesticks.registries.quality_checks_registry import QualityChecksRegistry
        
        registry = QualityChecksRegistry()
        
        config = {'overexposure': {'threshold': 0.05}}
        pipeline = registry.create_quality_pipeline(config)
        
        # Test with mock image
        mock_image = torch.randn(1, 3, 64, 64)
        results = pipeline(mock_image)
        
        assert isinstance(results, dict)
        assert 'overall_passed' in results
        assert 'overexposure' in results


class TestPreprocessingRegistry:
    """Test cases for PreprocessingRegistry raw processing steps."""
    
    def test_preprocessing_registry_exists(self):
        """Test that PreprocessingRegistry exists and can be imported."""
        from fiddlesticks.registries.preprocessing_registry import PreprocessingRegistry
        assert PreprocessingRegistry is not None
    
    def test_preprocessing_registry_has_default_steps(self):
        """Test that registry contains default preprocessing steps."""
        from fiddlesticks.registries.preprocessing_registry import PreprocessingRegistry
        
        registry = PreprocessingRegistry()
        steps = registry.list_available_steps()
        
        # Should have at least these steps based on memory
        expected_steps = ['normalize', 'gamma_correction', 'white_balance', 'demosaic']
        for step in expected_steps:
            assert step in steps
    
    def test_preprocessing_create_pipeline(self):
        """Test creating preprocessing pipeline."""
        from fiddlesticks.registries.preprocessing_registry import PreprocessingRegistry
        
        registry = PreprocessingRegistry()
        
        config = {
            'normalize': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
            'gamma_correction': {'gamma': 2.2}
        }
        
        pipeline = registry.create_preprocessing_pipeline(config)
        assert pipeline is not None
        assert hasattr(pipeline, '__call__')
    
    def test_preprocessing_pipeline_execution(self):
        """Test executing preprocessing pipeline on image."""
        from fiddlesticks.registries.preprocessing_registry import PreprocessingRegistry
        
        registry = PreprocessingRegistry()
        
        config = {'gamma_correction': {'gamma': 2.2}}
        pipeline = registry.create_preprocessing_pipeline(config)
        
        # Test with mock image
        mock_image = torch.randn(1, 3, 64, 64).abs()  # Ensure positive values
        result = pipeline(mock_image)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == mock_image.shape


class TestTrainingStrategyRegistry:
    """Test cases for TrainingStrategyRegistry training patterns."""
    
    def test_training_strategy_registry_exists(self):
        """Test that TrainingStrategyRegistry exists and can be imported."""
        from fiddlesticks.registries.training_strategy_registry import TrainingStrategyRegistry
        assert TrainingStrategyRegistry is not None
    
    def test_training_strategy_registry_has_default_strategies(self):
        """Test that registry contains default training strategies."""
        from fiddlesticks.registries.training_strategy_registry import TrainingStrategyRegistry
        
        registry = TrainingStrategyRegistry()
        strategies = registry.list_available_strategies()
        
        # Should have at least these strategies based on memory
        expected_strategies = ['supervised', 'self_supervised', 'adversarial', 'multi_task']
        for strategy in expected_strategies:
            assert strategy in strategies
    
    def test_training_strategy_create_trainer(self):
        """Test creating trainer from strategy registry."""
        from fiddlesticks.registries.training_strategy_registry import TrainingStrategyRegistry
        
        registry = TrainingStrategyRegistry()
        
        # Mock operations list
        mock_operations = [Mock(), Mock()]
        
        trainer = registry.create_strategy('supervised', mock_operations)
        assert trainer is not None
        assert hasattr(trainer, 'train')
    
    def test_training_strategy_unknown_strategy_error(self):
        """Test error handling for unknown training strategies."""
        from fiddlesticks.registries.training_strategy_registry import TrainingStrategyRegistry
        
        registry = TrainingStrategyRegistry()
        
        with pytest.raises(ValueError, match="Unknown training strategy"):
            registry.create_strategy('nonexistent_strategy', [])


class TestTransferFunctionRegistry:
    """Test cases for TransferFunctionRegistry data transformations."""
    
    def test_transfer_function_registry_exists(self):
        """Test that TransferFunctionRegistry exists and can be imported."""
        from fiddlesticks.registries.transfer_function_registry import TransferFunctionRegistry
        assert TransferFunctionRegistry is not None
    
    def test_transfer_function_registry_has_default_functions(self):
        """Test that registry contains default transfer functions."""
        from fiddlesticks.registries.transfer_function_registry import TransferFunctionRegistry
        
        registry = TransferFunctionRegistry()
        functions = registry.list_available_functions()
        
        # Should have at least these functions based on memory
        expected_functions = ['identity', 'log', 'gamma', 'sigmoid']
        for func in expected_functions:
            assert func in functions
    
    def test_transfer_function_create_pipeline(self):
        """Test creating transfer function pipeline."""
        from fiddlesticks.registries.transfer_function_registry import TransferFunctionRegistry
        
        registry = TransferFunctionRegistry()
        
        config = {
            'gamma': {'gamma': 2.2},
            'sigmoid': {'scale': 1.0}
        }
        
        pipeline = registry.create_transfer_pipeline(config)
        assert pipeline is not None
        assert hasattr(pipeline, '__call__')
    
    def test_transfer_function_pipeline_execution(self):
        """Test executing transfer function pipeline."""
        from fiddlesticks.registries.transfer_function_registry import TransferFunctionRegistry
        
        registry = TransferFunctionRegistry()
        
        config = {'identity': {}}
        pipeline = registry.create_transfer_pipeline(config)
        
        # Test with mock tensor
        mock_tensor = torch.randn(10, 5)
        result = pipeline(mock_tensor)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == mock_tensor.shape


class TestRegistryExtensionIntegration:
    """Test cases for registry extension integration."""
    
    def test_all_registries_can_be_imported(self):
        """Test that all registry extensions can be imported together."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        from fiddlesticks.registries.quality_checks_registry import QualityChecksRegistry
        from fiddlesticks.registries.preprocessing_registry import PreprocessingRegistry
        from fiddlesticks.registries.training_strategy_registry import TrainingStrategyRegistry
        from fiddlesticks.registries.transfer_function_registry import TransferFunctionRegistry
        
        # All should be importable
        assert ModelRegistry is not None
        assert QualityChecksRegistry is not None
        assert PreprocessingRegistry is not None
        assert TrainingStrategyRegistry is not None
        assert TransferFunctionRegistry is not None
    
    def test_registries_integration_with_main_system(self):
        """Test that registry extensions integrate with main operation system."""
        from fiddlesticks.registries.model_registry import ModelRegistry
        from fiddlesticks.operations.registry import OperationRegistry
        
        model_registry = ModelRegistry()
        operation_registry = OperationRegistry()
        
        # Should be able to use together
        assert model_registry is not None
        assert operation_registry is not None