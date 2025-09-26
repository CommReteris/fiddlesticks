"""
ModelRegistry for ML Model Factory Pattern.

This module provides a unified model registry similar to the augmentations pipeline
approach, enabling configurable model instantiation for various ML models including:
- UTNet2, UTNet3 for denoising
- BM3D for classical denoising  
- Learned denoising networks
- Compression autoencoders (Balle encoder/decoder)
- Custom models registered at runtime

Key Features:
- Singleton pattern for global model registry
- Factory pattern for model creation with parameters
- Runtime model registration capabilities
- Parameter override support for flexible configuration
- Integration with existing Function-Based Composable Pipeline Architecture
"""

import torch
from typing import Dict, Any, Type, List, Optional
from abc import ABC, abstractmethod


class ModelRegistry:
    """
    Singleton registry for ML model factory pattern.
    
    Provides unified model instantiation for various ML models used in the
    Function-Based Composable Pipeline Architecture. Follows the same elegant
    pattern as the augmentations registry.
    
    Key features:
    - Singleton pattern ensures single global registry
    - Factory method for model creation with parameters
    - Runtime registration of custom models
    - Parameter override support for flexible configuration
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ModelRegistry with default models."""
        if not self._initialized:
            self._models = {}
            self._model_configs = {}
            self._register_default_models()
            ModelRegistry._initialized = True
    
    def _register_default_models(self):
        """Register default ML models based on memory patterns."""
        # Mock model classes for testing - real implementations would import actual models
        class MockUTNet2(torch.nn.Module):
            def __init__(self, in_channels=4, out_channels=3, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
            def forward(self, x):
                return self.conv(x)
        
        class MockUTNet3(torch.nn.Module):
            def __init__(self, in_channels=4, out_channels=3, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.norm = torch.nn.BatchNorm2d(out_channels)
            
            def forward(self, x):
                return self.norm(self.conv(x))
        
        class MockBM3DDenoiser(torch.nn.Module):
            def __init__(self, sigma=25.0, **kwargs):
                super().__init__()
                self.sigma = sigma
                self.identity = torch.nn.Identity()
            
            def forward(self, x):
                # Mock BM3D - in reality would apply BM3D algorithm
                return self.identity(x)
        
        class MockLearnedDenoiseNet(torch.nn.Module):
            def __init__(self, num_layers=8, hidden_dim=64, **kwargs):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Conv2d(3, hidden_dim, 3, padding=1),
                    torch.nn.ReLU(),
                    *[torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in range(num_layers-2)],
                    torch.nn.Conv2d(hidden_dim, 3, 3, padding=1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        class MockBalleEncoder(torch.nn.Module):
            def __init__(self, num_filters=192, **kwargs):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, num_filters, 3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        class MockBalleDecoder(torch.nn.Module):
            def __init__(self, num_filters=192, **kwargs):
                super().__init__()
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(num_filters, 3, 3, stride=2, padding=1, output_padding=1)
                )
            
            def forward(self, x):
                return self.decoder(x)
        
        # Register default models from memory
        self._models = {
            'utnet2': MockUTNet2,
            'utnet3': MockUTNet3,
            'bm3d': MockBM3DDenoiser,
            'learned_denoise': MockLearnedDenoiseNet,
            'balle_encoder': MockBalleEncoder,
            'balle_decoder': MockBalleDecoder,
        }
        
        # Default configurations for models
        self._model_configs = {
            'utnet2': {'in_channels': 4, 'out_channels': 3},
            'utnet3': {'in_channels': 4, 'out_channels': 3},
            'bm3d': {'sigma': 25.0},
            'learned_denoise': {'num_layers': 8, 'hidden_dim': 64},
            'balle_encoder': {'num_filters': 192},
            'balle_decoder': {'num_filters': 192},
        }
    
    def register_model(self, model_name: str, model_class: Type[torch.nn.Module], 
                      default_config: Optional[Dict[str, Any]] = None):
        """
        Register a custom model in the registry.
        
        Args:
            model_name: Unique identifier for the model
            model_class: PyTorch module class to register
            default_config: Optional default configuration parameters
        """
        if not issubclass(model_class, torch.nn.Module):
            raise ValueError(f"Model class must be a subclass of torch.nn.Module")
        
        self._models[model_name] = model_class
        if default_config:
            self._model_configs[model_name] = default_config
    
    def create_model(self, model_name: str, **override_params) -> torch.nn.Module:
        """
        Create a model instance from registry with parameter override support.
        
        Args:
            model_name: Name of model to create
            **override_params: Parameters to override defaults
            
        Returns:
            Instantiated PyTorch model
            
        Raises:
            ValueError: If model name is not found in registry
        """
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {self.list_available_models()}")
        
        model_class = self._models[model_name]
        
        # Merge default config with overrides
        params = self._model_configs.get(model_name, {}).copy()
        params.update(override_params)
        
        try:
            return model_class(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to create model '{model_name}' with parameters {params}: {str(e)}")
    
    def list_available_models(self) -> List[str]:
        """
        List all available model names in the registry.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Default configuration dictionary
            
        Raises:
            ValueError: If model name is not found
        """
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self._model_configs.get(model_name, {}).copy()
    
    def create_pipeline(self, model_sequence: List[str], 
                       model_params: Optional[Dict[str, Dict[str, Any]]] = None) -> torch.nn.Sequential:
        """
        Create a sequential pipeline of models.
        
        Args:
            model_sequence: List of model names to chain
            model_params: Optional parameter overrides for each model
            
        Returns:
            Sequential pipeline of models
        """
        if model_params is None:
            model_params = {}
        
        models = []
        for model_name in model_sequence:
            params = model_params.get(model_name, {})
            model = self.create_model(model_name, **params)
            models.append(model)
        
        return torch.nn.Sequential(*models)
    
    def clear_registry(self):
        """Clear all registered models (mainly for testing)."""
        self._models.clear()
        self._model_configs.clear()
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model name exists in registry."""
        return model_name in self._models
    
    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._models)