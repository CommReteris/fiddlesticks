"""
TransferFunctionRegistry for Data Transformation Patterns.

This module provides a unified transfer function registry similar to the augmentations
pipeline approach, enabling systematic data transformation management including:
- Identity transfer for no transformation
- Logarithmic transfer for dynamic range compression
- Gamma transfer for tone mapping and color correction
- Sigmoid transfer for smooth non-linear transformations
- Filmic transfer for cinematic tone mapping
- Reinhard transfer for HDR tone mapping
- Custom transfer functions registered at runtime

Key Features:
- Configurable transfer function factory
- Support for multiple transformation paradigms
- Sequential function application pipeline
- Runtime registration of custom functions
- Integration with existing Function-Based Composable Pipeline Architecture
"""

import torch
import math
from typing import Dict, List, Any, Callable, Optional, Type
from abc import ABC, abstractmethod
import warnings


class TransferFunction(ABC):
    """
    Abstract base class for transfer functions.
    
    Defines the interface that all transfer functions must implement
    to ensure consistency across different transformation paradigms.
    """
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply transfer function to input tensor."""
        pass
    
    @property
    @abstractmethod
    def function_name(self) -> str:
        """Return name of the transfer function."""
        pass
    
    def validate_input(self, x: torch.Tensor) -> bool:
        """Validate input tensor for transfer function."""
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be torch.Tensor, got {type(x)}")
        return True


class IdentityTransferFunction(TransferFunction):
    """
    Identity transfer function.
    
    Passes input through unchanged - useful as a no-op transformation
    or as a baseline in transfer function pipelines.
    """
    
    @property
    def function_name(self) -> str:
        return "identity"
    
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return input tensor unchanged."""
        self.validate_input(x)
        return x


class LogarithmicTransferFunction(TransferFunction):
    """
    Logarithmic transfer function.
    
    Applies logarithmic transformation for dynamic range compression,
    commonly used in HDR processing and low-light enhancement.
    """
    
    @property
    def function_name(self) -> str:
        return "log"
    
    def __call__(self, x: torch.Tensor, epsilon: float = 1e-8, **kwargs) -> torch.Tensor:
        """
        Apply logarithmic transfer function.
        
        Args:
            x: Input tensor
            epsilon: Small value to prevent log(0)
            
        Returns:
            Logarithmically transformed tensor
        """
        self.validate_input(x)
        return torch.log(x + epsilon)


class GammaTransferFunction(TransferFunction):
    """
    Gamma transfer function.
    
    Applies gamma correction for tone mapping and color space conversion,
    fundamental operation in image processing and display systems.
    """
    
    @property
    def function_name(self) -> str:
        return "gamma"
    
    def __call__(self, x: torch.Tensor, gamma: float = 2.2, **kwargs) -> torch.Tensor:
        """
        Apply gamma transfer function.
        
        Args:
            x: Input tensor
            gamma: Gamma correction value
            
        Returns:
            Gamma corrected tensor
        """
        self.validate_input(x)
        return torch.pow(x.clamp(min=1e-8), 1.0 / gamma)


class SigmoidTransferFunction(TransferFunction):
    """
    Sigmoid transfer function.
    
    Applies sigmoid transformation for smooth non-linear mapping,
    useful for contrast enhancement and soft tone mapping.
    """
    
    @property
    def function_name(self) -> str:
        return "sigmoid"
    
    def __call__(self, x: torch.Tensor, scale: float = 1.0, offset: float = 0.0, **kwargs) -> torch.Tensor:
        """
        Apply sigmoid transfer function.
        
        Args:
            x: Input tensor
            scale: Scaling factor for sigmoid steepness
            offset: Offset for sigmoid center point
            
        Returns:
            Sigmoid transformed tensor
        """
        self.validate_input(x)
        return torch.sigmoid((x - offset) * scale)


class FilmicTransferFunction(TransferFunction):
    """
    Filmic transfer function.
    
    Applies filmic tone mapping for cinematic look,
    provides smooth highlight rolloff similar to film response.
    """
    
    @property
    def function_name(self) -> str:
        return "filmic"
    
    def __call__(self, x: torch.Tensor, shoulder_strength: float = 0.22, 
                 linear_strength: float = 0.30, linear_angle: float = 0.10,
                 toe_strength: float = 0.20, toe_numerator: float = 0.01,
                 toe_denominator: float = 0.30, **kwargs) -> torch.Tensor:
        """
        Apply filmic transfer function (Uncharted 2 tone mapping).
        
        Args:
            x: Input tensor
            shoulder_strength: Controls highlight rolloff
            linear_strength: Linear section strength
            linear_angle: Linear section angle
            toe_strength: Shadow region strength
            toe_numerator: Toe curve numerator
            toe_denominator: Toe curve denominator
            
        Returns:
            Filmic tone mapped tensor
        """
        self.validate_input(x)
        
        def filmic_curve(val):
            return ((val * (shoulder_strength * val + linear_angle * linear_strength) + 
                    toe_strength * toe_numerator) / 
                   (val * (shoulder_strength * val + linear_strength) + 
                    toe_strength * toe_denominator)) - toe_numerator / toe_denominator
        
        # Apply filmic curve
        mapped = filmic_curve(x)
        
        # Normalize by white point (filmic curve at maximum input)
        white_point = filmic_curve(torch.tensor(11.2, device=x.device, dtype=x.dtype))
        
        return mapped / white_point


class ReinhardTransferFunction(TransferFunction):
    """
    Reinhard transfer function.
    
    Applies Reinhard tone mapping for HDR to LDR conversion,
    provides global tone mapping with local adaptation capabilities.
    """
    
    @property
    def function_name(self) -> str:
        return "reinhard"
    
    def __call__(self, x: torch.Tensor, key: float = 0.18, white_point: Optional[float] = None, 
                 **kwargs) -> torch.Tensor:
        """
        Apply Reinhard transfer function.
        
        Args:
            x: Input tensor (HDR values)
            key: Key value for tone mapping (middle grey)
            white_point: White point for extended Reinhard (None for basic)
            
        Returns:
            Tone mapped tensor
        """
        self.validate_input(x)
        
        # Calculate luminance average (approximate for RGB)
        if x.dim() >= 3 and x.shape[-3] == 3:  # RGB image
            lum = 0.299 * x[..., 0, :, :] + 0.587 * x[..., 1, :, :] + 0.114 * x[..., 2, :, :]
            avg_lum = torch.mean(lum) + 1e-8
        else:
            avg_lum = torch.mean(x) + 1e-8
        
        # Scale by key value
        scaled = (key / avg_lum) * x
        
        if white_point is None:
            # Basic Reinhard
            return scaled / (1.0 + scaled)
        else:
            # Extended Reinhard with white point
            white_point_tensor = torch.tensor(white_point, device=x.device, dtype=x.dtype)
            numerator = scaled * (1.0 + scaled / (white_point_tensor * white_point_tensor))
            return numerator / (1.0 + scaled)


class PowerTransferFunction(TransferFunction):
    """
    Power transfer function.
    
    Applies power law transformation for contrast adjustment,
    generalizes gamma correction with flexible power values.
    """
    
    @property
    def function_name(self) -> str:
        return "power"
    
    def __call__(self, x: torch.Tensor, power: float = 0.5, **kwargs) -> torch.Tensor:
        """
        Apply power transfer function.
        
        Args:
            x: Input tensor
            power: Power law exponent
            
        Returns:
            Power law transformed tensor
        """
        self.validate_input(x)
        return torch.pow(x.clamp(min=0), power)


class TransferFunctionRegistry:
    """
    Registry for transfer function patterns.
    
    Provides factory pattern for creating different transfer functions
    similar to augmentations pipeline pattern, enabling configurable
    data transformation selection and runtime function registration.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(TransferFunctionRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize TransferFunctionRegistry with default functions."""
        if not self._initialized:
            self._functions = {}
            self._register_default_functions()
            TransferFunctionRegistry._initialized = True
    
    def _register_default_functions(self):
        """Register default transfer functions based on memory patterns."""
        self._functions = {
            'identity': IdentityTransferFunction(),
            'log': LogarithmicTransferFunction(),
            'gamma': GammaTransferFunction(),
            'sigmoid': SigmoidTransferFunction(),
            'filmic': FilmicTransferFunction(),
            'reinhard': ReinhardTransferFunction(),
            'power': PowerTransferFunction(),
        }
    
    def register_function(self, function_name: str, transfer_function: TransferFunction):
        """
        Register a custom transfer function.
        
        Args:
            function_name: Unique identifier for the function
            transfer_function: TransferFunction instance
        """
        if not isinstance(transfer_function, TransferFunction):
            raise ValueError("Transfer function must inherit from TransferFunction")
        
        self._functions[function_name] = transfer_function
    
    def list_available_functions(self) -> List[str]:
        """
        List all available transfer function names.
        
        Returns:
            List of function names
        """
        return list(self._functions.keys())
    
    def get_function(self, function_name: str) -> TransferFunction:
        """
        Get transfer function instance from registry.
        
        Args:
            function_name: Name of function to retrieve
            
        Returns:
            TransferFunction instance
            
        Raises:
            ValueError: If function name is not found
        """
        if function_name not in self._functions:
            available = ', '.join(self.list_available_functions())
            raise ValueError(f"Unknown transfer function: {function_name}. Available: {available}")
        
        return self._functions[function_name]
    
    def create_transfer_pipeline(self, config: Dict[str, Dict[str, Any]]) -> 'TransferFunctionPipeline':
        """
        Create transfer function pipeline from configuration.
        
        Args:
            config: Dictionary mapping function names to parameters
            
        Returns:
            TransferFunctionPipeline instance
        """
        return TransferFunctionPipeline(config, self)
    
    def __contains__(self, function_name: str) -> bool:
        """Check if transfer function exists in registry."""
        return function_name in self._functions
    
    def __len__(self) -> int:
        """Get number of registered transfer functions."""
        return len(self._functions)


class TransferFunctionPipeline:
    """
    Configurable transfer function pipeline.
    
    Applies multiple transfer functions in sequence with configurable
    parameters for each function in the pipeline.
    """
    
    def __init__(self, config: Dict[str, Dict[str, Any]], 
                 registry: Optional[TransferFunctionRegistry] = None):
        """
        Initialize transfer function pipeline.
        
        Args:
            config: Configuration dictionary mapping function names to parameters
            registry: TransferFunctionRegistry instance (creates new if None)
        """
        self.registry = registry or TransferFunctionRegistry()
        self.functions = []
        self.config = config
        
        # Build function pipeline from configuration
        for func_name, params in config.items():
            if func_name not in self.registry:
                available = ', '.join(self.registry.list_available_functions())
                raise ValueError(f"Unknown transfer function: {func_name}. Available: {available}")
            
            transfer_func = self.registry.get_function(func_name)
            self.functions.append((transfer_func, params))
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply configured transfer functions in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor after applying all functions
        """
        current_tensor = x
        
        for func, params in self.functions:
            try:
                current_tensor = func(current_tensor, **params)
            except Exception as e:
                func_name = func.function_name
                raise RuntimeError(f"Error applying transfer function '{func_name}': {e}")
        
        return current_tensor
    
    def apply_single_function(self, x: torch.Tensor, function_name: str, **params) -> torch.Tensor:
        """
        Apply single transfer function with specified parameters.
        
        Args:
            x: Input tensor
            function_name: Name of function to apply
            **params: Function parameters
            
        Returns:
            Transformed tensor
        """
        if function_name not in self.registry:
            raise ValueError(f"Unknown transfer function: {function_name}")
        
        transfer_func = self.registry.get_function(function_name)
        return transfer_func(x, **params)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline configuration.
        
        Returns:
            Dictionary with pipeline configuration details
        """
        return {
            'num_functions': len(self.functions),
            'function_names': [func.function_name for func, _ in self.functions],
            'config': self.config
        }
    
    def validate_pipeline(self, test_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Validate pipeline configuration with optional test tensor.
        
        Args:
            test_tensor: Optional tensor for testing pipeline
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Test with small tensor if none provided
        if test_tensor is None:
            test_tensor = torch.randn(1, 3, 4, 4)
        
        try:
            # Test pipeline execution
            result = self(test_tensor)
            
            # Check for common issues
            if torch.isnan(result).any():
                validation_results['warnings'].append("Pipeline produces NaN values")
            
            if torch.isinf(result).any():
                validation_results['warnings'].append("Pipeline produces infinite values")
            
            # Check value range
            if result.min() < 0 and 'log' in [f.function_name for f, _ in self.functions]:
                validation_results['warnings'].append("Negative values with logarithmic function")
                
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Pipeline execution failed: {e}")
        
        return validation_results