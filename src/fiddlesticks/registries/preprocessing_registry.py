"""
PreprocessingRegistry for Raw Image Processing Steps.

This module provides a unified preprocessing registry similar to the augmentations
pipeline approach, enabling systematic raw image preprocessing including:
- Image normalization with configurable mean/std
- Gamma correction for tone curve adjustment
- White balance with configurable gains
- Demosaicing for Bayer pattern conversion
- Custom preprocessing steps registered at runtime

Key Features:
- Configurable preprocessing pipeline
- Raw image processing specialization
- Sequential step execution with error handling
- Runtime registration of custom preprocessing functions
- Integration with existing Function-Based Composable Pipeline Architecture
"""

import torch
from typing import Dict, List, Any, Callable, Optional, Tuple
import warnings


class PreprocessingRegistry:
    """
    Registry for raw image preprocessing pipelines.
    
    Provides systematic preprocessing similar to augmentations pipeline
    pattern, enabling configurable preprocessing steps with sequential
    execution and comprehensive error handling.
    
    Key features:
    - Configurable preprocessing pipeline
    - Raw image processing specialization
    - Sequential step execution
    - Runtime registration of custom steps
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(PreprocessingRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize PreprocessingRegistry with default preprocessing steps."""
        if not self._initialized:
            self._steps = {}
            self._register_default_steps()
            PreprocessingRegistry._initialized = True
    
    def _register_default_steps(self):
        """Register default preprocessing steps based on memory patterns."""
        
        def normalize_image(image: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> torch.Tensor:
            """Normalize image with given mean and std."""
            if mean is None:
                mean = [0.5, 0.5, 0.5] if image.shape[1] == 3 else [0.5]
            if std is None:
                std = [0.5, 0.5, 0.5] if image.shape[1] == 3 else [0.5]
            
            # Handle different input formats
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
                squeeze_output = True
            else:
                squeeze_output = False
            
            # Convert lists to tensors on same device
            mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
            
            # Ensure broadcasting works correctly
            if mean_tensor.shape[1] != image.shape[1]:
                # Repeat for all channels if single value provided
                if mean_tensor.shape[1] == 1:
                    mean_tensor = mean_tensor.repeat(1, image.shape[1], 1, 1)
                else:
                    raise ValueError(f"Mean channels ({mean_tensor.shape[1]}) don't match image channels ({image.shape[1]})")
            
            if std_tensor.shape[1] != image.shape[1]:
                if std_tensor.shape[1] == 1:
                    std_tensor = std_tensor.repeat(1, image.shape[1], 1, 1)
                else:
                    raise ValueError(f"Std channels ({std_tensor.shape[1]}) don't match image channels ({image.shape[1]})")
            
            normalized = (image - mean_tensor) / (std_tensor + 1e-8)  # Add epsilon to avoid division by zero
            
            return normalized.squeeze(0) if squeeze_output else normalized
        
        def gamma_correction(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
            """Apply gamma correction."""
            # Ensure image values are positive for gamma correction
            image_clamped = torch.clamp(image, min=1e-8)
            return torch.pow(image_clamped, 1.0 / gamma)
        
        def white_balance(image: torch.Tensor, wb_gains: List[float] = None) -> torch.Tensor:
            """Apply white balance gains."""
            if wb_gains is None:
                wb_gains = [1.0, 1.0, 1.0] if image.shape[1] == 3 else [1.0]
            
            # Handle different input formats
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            gains = torch.tensor(wb_gains, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
            
            # Ensure broadcasting works correctly
            if gains.shape[1] != image.shape[1]:
                if gains.shape[1] == 1:
                    gains = gains.repeat(1, image.shape[1], 1, 1)
                elif len(wb_gains) == 3 and image.shape[1] == 4:
                    # Common case: 3-channel gains for 4-channel raw data (extend with 1.0)
                    gains = torch.tensor(wb_gains + [1.0], device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
                else:
                    raise ValueError(f"White balance gains ({gains.shape[1]}) don't match image channels ({image.shape[1]})")
            
            balanced = image * gains
            
            return balanced.squeeze(0) if squeeze_output else balanced
        
        def demosaic_bilinear(bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
            """Simple bilinear demosaicing for Bayer patterns."""
            if len(bayer.shape) == 3:
                bayer = bayer.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            # If already 4-channel (demosaiced raw), convert to 3-channel RGB
            if bayer.shape[1] == 4:
                # Convert 4-channel RGGB to 3-channel RGB
                r = bayer[:, 0:1]  # Red channel
                g = (bayer[:, 1:2] + bayer[:, 2:3]) / 2  # Average of two green channels
                b = bayer[:, 3:4]  # Blue channel
                rgb = torch.cat([r, g, b], dim=1)
            elif bayer.shape[1] == 1:
                # Single channel Bayer pattern - simplified bilinear demosaicing
                # This is a very simplified version - real demosaicing would be much more complex
                rgb = bayer.repeat(1, 3, 1, 1)  # Simple replication for mock implementation
            else:
                # Already RGB or other format
                rgb = bayer
            
            return rgb.squeeze(0) if squeeze_output else rgb
        
        def denormalize(image: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> torch.Tensor:
            """Denormalize image (reverse of normalize)."""
            if mean is None:
                mean = [0.5, 0.5, 0.5] if image.shape[1] == 3 else [0.5]
            if std is None:
                std = [0.5, 0.5, 0.5] if image.shape[1] == 3 else [0.5]
            
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, -1, 1, 1)
            
            # Handle broadcasting
            if mean_tensor.shape[1] != image.shape[1]:
                if mean_tensor.shape[1] == 1:
                    mean_tensor = mean_tensor.repeat(1, image.shape[1], 1, 1)
            
            if std_tensor.shape[1] != image.shape[1]:
                if std_tensor.shape[1] == 1:
                    std_tensor = std_tensor.repeat(1, image.shape[1], 1, 1)
            
            denormalized = image * std_tensor + mean_tensor
            
            return denormalized.squeeze(0) if squeeze_output else denormalized
        
        def clamp_values(image: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
            """Clamp image values to specified range."""
            return torch.clamp(image, min_val, max_val)
        
        # Register default preprocessing steps
        self._steps = {
            'normalize': normalize_image,
            'gamma_correction': gamma_correction,
            'white_balance': white_balance,
            'demosaic': demosaic_bilinear,
            'denormalize': denormalize,
            'clamp_values': clamp_values,
        }
    
    def register_step(self, step_name: str, step_function: Callable[[torch.Tensor], torch.Tensor]):
        """
        Register a custom preprocessing step function.
        
        Args:
            step_name: Unique identifier for the preprocessing step
            step_function: Function that takes tensor and returns processed tensor
        """
        if not callable(step_function):
            raise ValueError("Step function must be callable")
        
        self._steps[step_name] = step_function
    
    def list_available_steps(self) -> List[str]:
        """
        List all available preprocessing step names.
        
        Returns:
            List of preprocessing step names
        """
        return list(self._steps.keys())
    
    def create_preprocessing_pipeline(self, config: Dict[str, Dict[str, Any]]) -> 'PreprocessingPipeline':
        """
        Create preprocessing pipeline from configuration.
        
        Args:
            config: Dictionary mapping step names to their parameters
            
        Returns:
            PreprocessingPipeline instance
            
        Raises:
            ValueError: If unknown preprocessing step is specified
        """
        pipeline_steps = []
        for step_name, params in config.items():
            if step_name not in self._steps:
                available = ', '.join(self.list_available_steps())
                raise ValueError(f"Unknown preprocessing step: {step_name}. Available: {available}")
            
            pipeline_steps.append((self._steps[step_name], params))
        
        return PreprocessingPipeline(pipeline_steps)
    
    def __contains__(self, step_name: str) -> bool:
        """Check if preprocessing step exists in registry."""
        return step_name in self._steps
    
    def __len__(self) -> int:
        """Get number of registered preprocessing steps."""
        return len(self._steps)


class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline.
    
    Executes a sequence of preprocessing steps on images with
    comprehensive error handling and step tracking.
    """
    
    def __init__(self, steps: List[Tuple[Callable, Dict[str, Any]]]):
        """
        Initialize preprocessing pipeline.
        
        Args:
            steps: List of (step_function, parameters) tuples
        """
        self.steps = steps
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply all configured preprocessing steps to image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Processed image tensor
            
        Raises:
            RuntimeError: If any preprocessing step fails
        """
        current_image = image
        
        for i, (step_fn, params) in enumerate(self.steps):
            try:
                current_image = step_fn(current_image, **params)
                
            except Exception as e:
                step_name = getattr(step_fn, '__name__', f'step_{i}')
                raise RuntimeError(f"Preprocessing step '{step_name}' failed: {str(e)}") from e
        
        return current_image
    
    def apply_step(self, image: torch.Tensor, step_index: int) -> torch.Tensor:
        """
        Apply a single preprocessing step by index.
        
        Args:
            image: Input image tensor
            step_index: Index of step to apply
            
        Returns:
            Processed image tensor
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(f"Step index {step_index} out of range [0, {len(self.steps)})")
        
        step_fn, params = self.steps[step_index]
        return step_fn(image, **params)
    
    def get_step_names(self) -> List[str]:
        """
        Get names of all steps in the pipeline.
        
        Returns:
            List of step names
        """
        names = []
        for step_fn, _ in self.steps:
            name = getattr(step_fn, '__name__', 'unknown_step')
            names.append(name)
        return names
    
    def __len__(self) -> int:
        """Get number of steps in pipeline."""
        return len(self.steps)