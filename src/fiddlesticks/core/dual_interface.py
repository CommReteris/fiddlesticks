"""
Dual interface system for Function-Based Composable Pipeline Architecture.

This module provides the dual interface system that enables both simple and advanced
configuration modes:
- Simple interface: ['denoise', 'sharpen', 'tone_map']
- Advanced interface: Full operation specifications with explicit control

Key components:
- OperationResolver: Converts simple operation names to full specifications
- LayeredConfigurationSystem: Validates and suggests configuration improvements
"""

from typing import Dict, List, Any, Tuple, Optional, Union
from .operation_spec import OperationSpec


class OperationResolver:
    """
    Converts simple operation names to advanced configuration specifications.
    
    Provides automatic resolution from simple names like 'denoise' to full
    (category, operation_name, spec) tuples for pipeline construction.
    """
    
    def __init__(self, registry: Dict[str, Dict[str, OperationSpec]], 
                 custom_lookup: Optional[Dict[str, Tuple[str, str]]] = None):
        """
        Initialize OperationResolver with operation registry and lookup table.
        
        Args:
            registry: Complete operation registry organized by categories
            custom_lookup: Optional custom lookup mappings to override defaults
        """
        self.registry = registry
        
        # Default lookup table mapping simple names to (category, operation_name)
        self.operation_lookup = {
            # Input/Output operations
            'load_raw': ('input_output_operations', 'load_raw_file'),
            'load_image': ('input_output_operations', 'load_image'),
            'save_image': ('input_output_operations', 'save_image'),
            'save_raw': ('input_output_operations', 'save_raw'),
            
            # Raw processing operations
            'rawprepare': ('raw_processing_operations', 'rawprepare'),
            'demosaic': ('raw_processing_operations', 'demosaic'),
            'white_balance': ('raw_processing_operations', 'temperature'),
            
            # Denoising operations
            'denoise': ('denoising_operations', 'bilateral'),
            'denoise_ml': ('denoising_operations', 'utnet2'),
            'denoise_classical': ('denoising_operations', 'bm3d'),
            'denoise_temporal': ('burst_processing_operations', 'temporal_denoise'),
            
            # Enhancement operations
            'sharpen': ('enhancement_operations', 'unsharp_mask'),
            'enhance': ('enhancement_operations', 'sharpen'),
            'blur': ('enhancement_operations', 'blurs'),
            
            # Tone mapping operations
            'tone_map': ('tone_mapping_operations', 'sigmoid'),
            'tone_filmic': ('tone_mapping_operations', 'filmicrgb'),
            'exposure': ('tone_mapping_operations', 'exposure'),
            
            # Color processing operations
            'color_balance': ('color_processing_operations', 'colorbalancergb'),
            'color_correct': ('color_processing_operations', 'channelmixerrgb'),
            
            # Geometric operations
            'crop': ('geometric_operations', 'crop'),
            'flip': ('geometric_operations', 'flip'),
            'rotate': ('geometric_operations', 'rotatepixels'),
            
            # Burst processing operations
            'hdr_merge': ('burst_processing_operations', 'hdr_merge'),
            'focus_stack': ('burst_processing_operations', 'focus_stack'),
            'panorama': ('burst_processing_operations', 'panorama_stitch'),
            
            # Quality assessment operations
            'check_quality': ('quality_assessment_operations', 'overexposed'),
            'analyze': ('quality_assessment_operations', 'exposure_analysis'),
        }
        
        # Override with custom lookup if provided
        if custom_lookup:
            self.operation_lookup.update(custom_lookup)
    
    def resolve_operation(self, operation_name: str) -> Tuple[str, str, OperationSpec]:
        """
        Resolve simple operation name to (category, operation_name, spec).
        
        Args:
            operation_name: Simple operation name (e.g., 'denoise')
            
        Returns:
            Tuple of (category, full_operation_name, operation_spec)
            
        Raises:
            ValueError: If operation name is unknown or not found in registry
        """
        if operation_name not in self.operation_lookup:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        category, full_operation_name = self.operation_lookup[operation_name]
        
        if category not in self.registry:
            raise ValueError(f"Operation {full_operation_name} not found in category {category}")
        
        if full_operation_name not in self.registry[category]:
            raise ValueError(f"Operation {full_operation_name} not found in category {category}")
        
        spec = self.registry[category][full_operation_name]
        
        return category, full_operation_name, spec
    
    def resolve_batch(self, operation_names: List[str]) -> List[Tuple[str, str, OperationSpec]]:
        """
        Resolve multiple operation names at once.
        
        Args:
            operation_names: List of simple operation names
            
        Returns:
            List of (category, operation_name, spec) tuples
        """
        return [self.resolve_operation(name) for name in operation_names]


class LayeredConfigurationSystem:
    """
    Configuration system supporting both simple and advanced interface modes.
    
    Provides validation, compatibility checking, and automatic suggestions
    for pipeline configuration improvements.
    """
    
    def __init__(self, registry: Dict[str, Dict[str, OperationSpec]]):
        """
        Initialize LayeredConfigurationSystem with operation registry.
        
        Args:
            registry: Complete operation registry organized by categories
        """
        self.registry = registry
        self.operation_resolver = OperationResolver(registry)
    
    def resolve_simple_config(self, config: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert simple configuration to advanced format.
        
        Args:
            config: Mixed simple/advanced configuration list
            
        Returns:
            List of resolved advanced configuration dictionaries
        """
        resolved = []
        
        for item in config:
            if isinstance(item, str):
                # Pure simple format: 'denoise'
                try:
                    category, operation_name, spec = self.operation_resolver.resolve_operation(item)
                    resolved.append({
                        'operation': operation_name,
                        'category': category,
                        'resolved_spec': spec
                    })
                except ValueError as e:
                    # Keep as-is if resolution fails, let validation handle it
                    resolved.append({
                        'operation': item,
                        'category': None,
                        'error': str(e)
                    })
            
            elif isinstance(item, dict):
                if 'category' not in item:
                    # Simple format with params: {'operation': 'denoise', 'strength': 0.3}
                    operation_name = item.get('operation', '')
                    try:
                        category, full_operation_name, spec = self.operation_resolver.resolve_operation(operation_name)
                        resolved_item = {
                            'operation': full_operation_name,
                            'category': category,
                            'resolved_spec': spec
                        }
                        # Preserve any parameters
                        if 'params' in item:
                            resolved_item['params'] = item['params']
                        else:
                            # Move non-operation keys to params
                            params = {k: v for k, v in item.items() if k != 'operation'}
                            if params:
                                resolved_item['params'] = params
                        
                        resolved.append(resolved_item)
                    except ValueError as e:
                        # Keep as-is if resolution fails
                        resolved_item = item.copy()
                        resolved_item['error'] = str(e)
                        resolved.append(resolved_item)
                else:
                    # Already advanced format, just add spec if missing
                    resolved_item = item.copy()
                    if 'resolved_spec' not in resolved_item:
                        category = item['category']
                        operation = item['operation']
                        if (category in self.registry and 
                            operation in self.registry[category]):
                            resolved_item['resolved_spec'] = self.registry[category][operation]
                    resolved.append(resolved_item)
        
        return resolved
    
    def validate_and_suggest(self, config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate configuration and provide suggestions for improvements.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Dictionary with validation results, warnings, suggestions, and auto-fixes
        """
        # First resolve simple configurations
        resolved_config = self.resolve_simple_config(config)
        
        warnings = []
        suggestions = []
        auto_fixes = []
        
        # Validate type compatibility between consecutive operations
        for i in range(len(resolved_config) - 1):
            current_op = resolved_config[i]
            next_op = resolved_config[i + 1]
            
            # Skip validation if specs are missing
            current_spec = current_op.get('resolved_spec')
            next_spec = next_op.get('resolved_spec')
            
            if not current_spec or not next_spec:
                if 'error' in current_op:
                    warnings.append(f"Step {i}: {current_op['error']}")
                if 'error' in next_op:
                    warnings.append(f"Step {i+1}: {next_op['error']}")
                continue
            
            # Check output -> input type compatibility
            current_outputs = set(current_spec.output_types)
            next_inputs = set(next_spec.input_types)
            
            compatible_types = current_outputs & next_inputs
            if not compatible_types:
                warning = (f"Step {i}: {current_spec.name} outputs {list(current_outputs)} "
                          f"but {next_spec.name} requires {list(next_inputs)} - type mismatch")
                warnings.append(warning)
                
                # Try to suggest conversion operations
                conversion_suggestions = self._suggest_conversion_operations(
                    current_outputs, next_inputs)
                if conversion_suggestions:
                    suggestions.append(f"Consider inserting conversion operation: {conversion_suggestions}")
                    auto_fixes.append({
                        'type': 'insert_conversion',
                        'position': i + 1,
                        'operations': conversion_suggestions
                    })
            
            # Check processing mode compatibility
            current_modes = set(current_spec.supported_modes)
            next_modes = set(next_spec.supported_modes)
            
            if not (current_modes & next_modes):
                warning = (f"Step {i}: Processing mode mismatch between "
                          f"{current_spec.name} and {next_spec.name}")
                warnings.append(warning)
        
        return {
            'resolved_config': resolved_config,
            'warnings': warnings,
            'suggestions': suggestions,
            'auto_fixes': auto_fixes
        }
    
    def _suggest_conversion_operations(self, output_types: set, input_types: set) -> List[str]:
        """
        Suggest conversion operations to bridge type incompatibilities.
        
        Args:
            output_types: Set of output types from current operation
            input_types: Set of input types required by next operation
            
        Returns:
            List of suggested conversion operation names
        """
        suggestions = []
        
        # Common conversion patterns
        conversion_map = {
            # Color space conversions
            ('RGB', 'LAB'): 'rgb_to_lab',
            ('LAB', 'RGB'): 'lab_to_rgb',
            ('RGB', 'GRAYSCALE'): 'rgb_to_grayscale',
            ('GRAYSCALE', 'RGB'): 'grayscale_to_rgb',
            
            # Raw processing conversions
            ('RAW_BAYER', 'RAW_4CH'): 'demosaic',
            ('RAW_4CH', 'RGB'): 'colorin',
            
            # Format conversions
            ('NUMPY_ARRAY', 'RGB'): 'numpy_to_torch',
            ('RGB', 'NUMPY_ARRAY'): 'torch_to_numpy',
        }
        
        # Look for direct conversion paths
        for output_type in output_types:
            for input_type in input_types:
                conversion_key = (output_type.name if hasattr(output_type, 'name') else str(output_type),
                                input_type.name if hasattr(input_type, 'name') else str(input_type))
                if conversion_key in conversion_map:
                    suggestions.append(conversion_map[conversion_key])
        
        return suggestions