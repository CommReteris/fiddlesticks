"""
QualityChecksRegistry for Image Quality Assessment Pipelines.

This module provides a unified quality checks registry similar to the augmentations
pipeline approach, enabling systematic quality assessment for images including:
- Overexposure detection and validation
- Underexposure detection and validation  
- Noise level estimation and thresholding
- Blur detection and sharpness measurement
- Dynamic range analysis
- Color accuracy assessment
- Custom quality checks registered at runtime

Key Features:
- Configurable quality assessment pipeline
- Pass/fail validation with configurable thresholds
- Comprehensive quality metrics extraction
- Runtime registration of custom checks
- Integration with existing Function-Based Composable Pipeline Architecture
"""

import torch
from typing import Dict, List, Any, Callable, Optional, Tuple
import warnings


class QualityChecksRegistry:
    """
    Registry for image quality assessment pipelines.
    
    Provides systematic quality assessment similar to augmentations pipeline
    pattern, enabling configurable quality checks with pass/fail validation
    and comprehensive metrics extraction.
    
    Key features:
    - Configurable quality assessment pipeline
    - Pass/fail validation with thresholds
    - Comprehensive metrics extraction
    - Runtime registration of custom checks
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(QualityChecksRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize QualityChecksRegistry with default quality checks."""
        if not self._initialized:
            self._checks = {}
            self._register_default_checks()
            QualityChecksRegistry._initialized = True
    
    def _register_default_checks(self):
        """Register default quality checks based on memory patterns."""
        
        def check_overexposure(image: torch.Tensor, threshold: float = 0.01) -> Dict[str, Any]:
            """Check for overexposed pixels."""
            if image.max() <= 1.0:
                # Normalized image (0-1 range)
                overexposed = (image >= 1.0).float().mean()
            else:
                # Assume 8-bit or similar (scale to 0-1)
                max_val = image.max()
                overexposed = (image >= (max_val * 0.99)).float().mean()
            
            return {
                'overexposure_ratio': overexposed.item(),
                'passed': overexposed.item() <= threshold,
                'threshold_used': threshold,
                'severity': 'high' if overexposed.item() > threshold * 5 else 'medium' if overexposed.item() > threshold * 2 else 'low'
            }
        
        def check_underexposure(image: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
            """Check for underexposed pixels."""
            if image.min() >= 0.0:
                # Normalized image
                underexposed = (image <= 0.05).float().mean()
            else:
                # Handle different ranges
                min_val = image.min()
                underexposed = (image <= (min_val + 0.05 * (image.max() - min_val))).float().mean()
            
            return {
                'underexposure_ratio': underexposed.item(),
                'passed': underexposed.item() <= threshold,
                'threshold_used': threshold,
                'severity': 'high' if underexposed.item() > threshold * 5 else 'medium' if underexposed.item() > threshold * 2 else 'low'
            }
        
        def check_noise_level(image: torch.Tensor, max_std: float = 0.1) -> Dict[str, Any]:
            """Estimate noise level using local variance."""
            # Simple noise estimation using local variance
            if len(image.shape) == 4:  # Batch dimension
                image = image[0]  # Take first image in batch
            
            # Convert to grayscale if RGB
            if image.shape[0] == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0] if len(image.shape) == 3 else image
            
            # Estimate noise using Laplacian variance method
            kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)
            noise_var = laplacian.var()
            noise_std = noise_var.sqrt()
            
            return {
                'estimated_noise_std': noise_std.item(),
                'passed': noise_std.item() <= max_std,
                'threshold_used': max_std,
                'noise_level': 'high' if noise_std.item() > max_std * 2 else 'medium' if noise_std.item() > max_std else 'low'
            }
        
        def check_blur_detection(image: torch.Tensor, min_sharpness: float = 100.0) -> Dict[str, Any]:
            """Detect blur using gradient variance."""
            if len(image.shape) == 4:  # Batch dimension
                image = image[0]  # Take first image in batch
            
            # Convert to grayscale if RGB
            if image.shape[0] == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0] if len(image.shape) == 3 else image
            
            # Sobel gradient magnitude
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
            
            grad_x = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            sharpness = gradient_magnitude.var()
            
            return {
                'sharpness_score': sharpness.item(),
                'passed': sharpness.item() >= min_sharpness,
                'threshold_used': min_sharpness,
                'blur_level': 'high' if sharpness.item() < min_sharpness * 0.3 else 'medium' if sharpness.item() < min_sharpness * 0.7 else 'low'
            }
        
        def check_dynamic_range(image: torch.Tensor, min_range: float = 0.5) -> Dict[str, Any]:
            """Check dynamic range of image."""
            value_range = image.max() - image.min()
            normalized_range = value_range / (image.max() if image.max() > 1.0 else 1.0)
            
            return {
                'dynamic_range': value_range.item(),
                'normalized_range': normalized_range.item(),
                'passed': normalized_range.item() >= min_range,
                'threshold_used': min_range,
                'range_quality': 'good' if normalized_range.item() >= min_range else 'limited'
            }
        
        def check_color_accuracy(image: torch.Tensor, reference: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            """Check color accuracy (simplified without reference)."""
            if len(image.shape) == 4:
                image = image[0]
            
            if image.shape[0] != 3:
                return {
                    'color_deviation': 0.0,
                    'passed': True,
                    'note': 'Non-RGB image, color accuracy not applicable'
                }
            
            # Simple color balance check - measure deviation from neutral
            mean_values = image.view(3, -1).mean(dim=1)
            color_deviation = torch.std(mean_values)
            
            return {
                'color_deviation': color_deviation.item(),
                'channel_means': mean_values.tolist(),
                'passed': color_deviation.item() < 0.1,  # Arbitrary threshold
                'balance_quality': 'good' if color_deviation.item() < 0.05 else 'fair' if color_deviation.item() < 0.1 else 'poor'
            }
        
        # Register default quality checks
        self._checks = {
            'overexposure': check_overexposure,
            'underexposure': check_underexposure,
            'noise_level': check_noise_level,
            'blur_detection': check_blur_detection,
            'dynamic_range': check_dynamic_range,
            'color_accuracy': check_color_accuracy,
        }
    
    def register_check(self, check_name: str, check_function: Callable[[torch.Tensor], Dict[str, Any]]):
        """
        Register a custom quality check function.
        
        Args:
            check_name: Unique identifier for the quality check
            check_function: Function that takes tensor and returns quality metrics dict
        """
        if not callable(check_function):
            raise ValueError("Check function must be callable")
        
        self._checks[check_name] = check_function
    
    def list_available_checks(self) -> List[str]:
        """
        List all available quality check names.
        
        Returns:
            List of quality check names
        """
        return list(self._checks.keys())
    
    def create_quality_pipeline(self, config: Dict[str, Dict[str, Any]]) -> 'QualityChecksPipeline':
        """
        Create quality assessment pipeline from configuration.
        
        Args:
            config: Dictionary mapping check names to their parameters
            
        Returns:
            QualityChecksPipeline instance
            
        Raises:
            ValueError: If unknown quality check is specified
        """
        pipeline_checks = []
        for check_name, params in config.items():
            if check_name not in self._checks:
                available = ', '.join(self.list_available_checks())
                raise ValueError(f"Unknown quality check: {check_name}. Available: {available}")
            
            pipeline_checks.append((self._checks[check_name], params))
        
        return QualityChecksPipeline(pipeline_checks)
    
    def __contains__(self, check_name: str) -> bool:
        """Check if quality check exists in registry."""
        return check_name in self._checks
    
    def __len__(self) -> int:
        """Get number of registered quality checks."""
        return len(self._checks)


class QualityChecksPipeline:
    """
    Configurable image quality assessment pipeline.
    
    Executes a sequence of quality checks on images and provides
    comprehensive quality metrics with pass/fail validation.
    """
    
    def __init__(self, checks: List[Tuple[Callable, Dict[str, Any]]]):
        """
        Initialize quality checks pipeline.
        
        Args:
            checks: List of (check_function, parameters) tuples
        """
        self.checks = checks
    
    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Apply all configured quality checks to image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary containing all quality check results and overall assessment
        """
        results = {}
        all_passed = True
        
        for check_fn, params in self.checks:
            try:
                result = check_fn(image, **params)
                check_name = check_fn.__name__.replace('check_', '')
                results[check_name] = result
                
                # Update overall pass status
                if 'passed' in result:
                    all_passed = all_passed and result['passed']
                    
            except Exception as e:
                check_name = getattr(check_fn, '__name__', 'unknown_check')
                warnings.warn(f"Quality check '{check_name}' failed: {str(e)}")
                results[check_name] = {
                    'error': str(e),
                    'passed': False
                }
                all_passed = False
        
        # Add overall assessment
        results['overall_passed'] = all_passed
        results['total_checks'] = len(self.checks)
        results['checks_passed'] = sum(1 for result in results.values() 
                                     if isinstance(result, dict) and result.get('passed', False))
        
        return results
    
    def get_failed_checks(self, results: Dict[str, Any]) -> List[str]:
        """
        Get list of failed quality check names from results.
        
        Args:
            results: Results dictionary from __call__
            
        Returns:
            List of failed check names
        """
        failed = []
        for check_name, result in results.items():
            if isinstance(result, dict) and result.get('passed', True) == False:
                if check_name not in ['overall_passed', 'total_checks', 'checks_passed']:
                    failed.append(check_name)
        return failed