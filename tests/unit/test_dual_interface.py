"""
Test cases for dual interface system components.

Following strict TDD approach - tests written first, implementation follows.
Tests cover OperationResolver and LayeredConfigurationSystem that enable
both simple and advanced configuration modes.
"""

import pytest
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch


class TestOperationResolver:
    """Test cases for OperationResolver that converts simple->advanced interface."""
    
    def test_operation_resolver_exists(self):
        """Test that OperationResolver exists and can be imported."""
        from fiddlesticks.core.dual_interface import OperationResolver
        assert OperationResolver is not None
    
    def test_operation_resolver_initialization(self):
        """Test OperationResolver can be initialized with operation registry."""
        from fiddlesticks.core.dual_interface import OperationResolver
        
        # Mock registry for testing
        mock_registry = {
            'denoising_operations': {
                'bilateral': Mock(),
                'utnet2': Mock()
            },
            'enhancement_operations': {
                'sharpen': Mock(),
                'tone_map': Mock()
            }
        }
        
        resolver = OperationResolver(mock_registry)
        assert resolver.registry == mock_registry
        assert hasattr(resolver, 'operation_lookup')
    
    def test_operation_resolver_has_default_lookup_table(self):
        """Test that OperationResolver has default simple->advanced lookup mappings."""
        from fiddlesticks.core.dual_interface import OperationResolver
        
        resolver = OperationResolver({})
        
        # Check default lookup table exists and has expected mappings
        assert hasattr(resolver, 'operation_lookup')
        assert isinstance(resolver.operation_lookup, dict)
        
        # Test some expected default mappings
        expected_mappings = [
            'denoise', 'sharpen', 'tone_map', 'load_raw', 'save_image',
            'crop', 'flip', 'rotate', 'enhance', 'blur'
        ]
        
        for simple_name in expected_mappings:
            assert simple_name in resolver.operation_lookup
            category, operation_name = resolver.operation_lookup[simple_name]
            assert isinstance(category, str)
            assert isinstance(operation_name, str)
    
    def test_resolve_simple_operation_name(self):
        """Test resolving simple operation name to (category, operation_name)."""
        from fiddlesticks.core.dual_interface import OperationResolver
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create mock operation spec
        mock_spec = OperationSpec(
            name="bilateral_filter",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Bilateral filtering"
        )
        
        mock_registry = {
            'denoising_operations': {
                'bilateral': mock_spec
            }
        }
        
        resolver = OperationResolver(mock_registry)
        
        # Test resolving simple name
        category, operation_name, spec = resolver.resolve_operation('denoise')
        
        assert category == 'denoising_operations'
        assert operation_name == 'bilateral'  # Default mapping
        assert spec == mock_spec
    
    def test_resolve_unknown_operation(self):
        """Test error handling for unknown operation names."""
        from fiddlesticks.core.dual_interface import OperationResolver
        
        resolver = OperationResolver({})
        
        with pytest.raises(ValueError, match="Unknown operation: unknown_op"):
            resolver.resolve_operation('unknown_op')
    
    def test_resolve_operation_missing_from_registry(self):
        """Test error handling when operation exists in lookup but not registry."""
        from fiddlesticks.core.dual_interface import OperationResolver
        
        # Empty registry but lookup table has mappings
        resolver = OperationResolver({})
        
        with pytest.raises(ValueError, match="Operation bilateral not found in category denoising_operations"):
            resolver.resolve_operation('denoise')  # Maps to bilateral but registry is empty
    
    def test_custom_operation_lookup_override(self):
        """Test that operation lookup can be customized/overridden."""
        from fiddlesticks.core.dual_interface import OperationResolver
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create custom spec
        custom_spec = OperationSpec(
            name="custom_denoise",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Custom denoising"
        )
        
        registry = {
            'custom_operations': {
                'custom_denoise': custom_spec
            }
        }
        
        # Custom lookup mapping
        custom_lookup = {
            'denoise': ('custom_operations', 'custom_denoise')
        }
        
        resolver = OperationResolver(registry, custom_lookup)
        
        category, operation_name, spec = resolver.resolve_operation('denoise')
        
        assert category == 'custom_operations'
        assert operation_name == 'custom_denoise'
        assert spec == custom_spec
    
    def test_resolve_batch_operations(self):
        """Test resolving multiple operations at once."""
        from fiddlesticks.core.dual_interface import OperationResolver
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create mock specs
        denoise_spec = OperationSpec(
            name="bilateral", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Denoise"
        )
        
        sharpen_spec = OperationSpec(
            name="unsharp_mask", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Sharpen"
        )
        
        registry = {
            'denoising_operations': {'bilateral': denoise_spec},
            'enhancement_operations': {'unsharp_mask': sharpen_spec}
        }
        
        resolver = OperationResolver(registry)
        
        # Test batch resolution
        simple_operations = ['denoise', 'sharpen']
        resolved = resolver.resolve_batch(simple_operations)
        
        assert len(resolved) == 2
        assert resolved[0] == ('denoising_operations', 'bilateral', denoise_spec)
        assert resolved[1] == ('enhancement_operations', 'unsharp_mask', sharpen_spec)


class TestLayeredConfigurationSystem:
    """Test cases for LayeredConfigurationSystem that validates configurations."""
    
    def test_layered_configuration_system_exists(self):
        """Test that LayeredConfigurationSystem exists and can be imported."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        assert LayeredConfigurationSystem is not None
    
    def test_layered_configuration_initialization(self):
        """Test LayeredConfigurationSystem initialization."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.dual_interface import OperationResolver
        
        mock_registry = {}
        resolver = OperationResolver(mock_registry)
        
        config_system = LayeredConfigurationSystem(mock_registry)
        
        assert hasattr(config_system, 'registry')
        assert hasattr(config_system, 'operation_resolver')
        assert config_system.registry == mock_registry
    
    def test_resolve_simple_config_to_advanced(self):
        """Test converting simple configuration to advanced format."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create mock specs
        denoise_spec = OperationSpec(
            name="bilateral", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Denoise"
        )
        
        registry = {
            'denoising_operations': {'bilateral': denoise_spec}
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Test simple string list
        simple_config = ['denoise']
        resolved = config_system.resolve_simple_config(simple_config)
        
        assert len(resolved) == 1
        assert resolved[0]['operation'] == 'bilateral'
        assert resolved[0]['category'] == 'denoising_operations'
        assert resolved[0]['resolved_spec'] == denoise_spec
    
    def test_resolve_simple_config_with_params(self):
        """Test converting simple config with parameters to advanced format."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        denoise_spec = OperationSpec(
            name="bilateral", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Denoise"
        )
        
        registry = {
            'denoising_operations': {'bilateral': denoise_spec}
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Test simple config with parameters
        simple_config = [
            {'operation': 'denoise', 'strength': 0.3}
        ]
        resolved = config_system.resolve_simple_config(simple_config)
        
        assert len(resolved) == 1
        assert resolved[0]['operation'] == 'bilateral'
        assert resolved[0]['category'] == 'denoising_operations'
        assert resolved[0]['params']['strength'] == 0.3
        assert resolved[0]['resolved_spec'] == denoise_spec
    
    def test_validate_and_suggest_valid_config(self):
        """Test validation of valid configuration."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create compatible operations
        denoise_spec = OperationSpec(
            name="bilateral", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Denoise"
        )
        
        sharpen_spec = OperationSpec(
            name="unsharp_mask", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Sharpen"
        )
        
        registry = {
            'denoising_operations': {'bilateral': denoise_spec},
            'enhancement_operations': {'unsharp_mask': sharpen_spec}
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Test validation of compatible operations
        config = [
            {'operation': 'bilateral', 'category': 'denoising_operations'},
            {'operation': 'unsharp_mask', 'category': 'enhancement_operations'}
        ]
        
        result = config_system.validate_and_suggest(config)
        
        assert result['warnings'] == []  # No warnings for compatible operations
        assert len(result['resolved_config']) == 2
        assert result['suggestions'] == []
        assert result['auto_fixes'] == []
    
    def test_validate_and_suggest_incompatible_config(self):
        """Test validation with incompatible operations generates warnings."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create incompatible operations (RGB -> GRAYSCALE -> RGB)
        rgb_to_gray_spec = OperationSpec(
            name="rgb_to_gray", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.GRAYSCALE],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="RGB to grayscale"
        )
        
        enhance_rgb_spec = OperationSpec(
            name="enhance", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],  # Requires RGB input
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Enhance RGB"
        )
        
        registry = {
            'color_operations': {'rgb_to_gray': rgb_to_gray_spec},
            'enhancement_operations': {'enhance': enhance_rgb_spec}
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Test incompatible chain
        config = [
            {'operation': 'rgb_to_gray', 'category': 'color_operations'},
            {'operation': 'enhance', 'category': 'enhancement_operations'}  # Can't enhance grayscale as RGB
        ]
        
        result = config_system.validate_and_suggest(config)
        
        assert len(result['warnings']) > 0
        assert 'type mismatch' in result['warnings'][0].lower()
        assert len(result['suggestions']) > 0  # Should suggest fixes
    
    def test_mixed_simple_advanced_config(self):
        """Test handling configuration that mixes simple and advanced formats."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        denoise_spec = OperationSpec(
            name="bilateral", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="Denoise"
        )
        
        registry = {
            'denoising_operations': {'bilateral': denoise_spec}
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Mixed format: simple string + advanced dict
        mixed_config = [
            'denoise',  # Simple format
            {  # Advanced format
                'operation': 'bilateral',
                'category': 'denoising_operations',
                'params': {'sigma_color': 0.1}
            }
        ]
        
        resolved = config_system.resolve_simple_config(mixed_config)
        
        assert len(resolved) == 2
        # First should be resolved from simple
        assert resolved[0]['operation'] == 'bilateral'
        assert resolved[0]['category'] == 'denoising_operations'
        # Second should remain as advanced
        assert resolved[1]['operation'] == 'bilateral'
        assert resolved[1]['params']['sigma_color'] == 0.1
    
    def test_auto_fix_generation(self):
        """Test automatic fix generation for common configuration issues."""
        from fiddlesticks.core.dual_interface import LayeredConfigurationSystem
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create operations with conversion path
        rgb_spec = OperationSpec(
            name="rgb_op", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.RGB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="RGB operation"
        )
        
        lab_spec = OperationSpec(
            name="lab_op", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.LAB], output_types=[InputOutputType.LAB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="LAB operation"
        )
        
        rgb_to_lab_spec = OperationSpec(
            name="rgb_to_lab", supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB], output_types=[InputOutputType.LAB],
            input_count=(1, 1), output_count=1, requires_metadata=[],
            produces_metadata=[], constraints={}, description="RGB to LAB conversion"
        )
        
        registry = {
            'color_operations': {
                'rgb_op': rgb_spec,
                'lab_op': lab_spec,
                'rgb_to_lab': rgb_to_lab_spec
            }
        }
        
        config_system = LayeredConfigurationSystem(registry)
        
        # Incompatible chain that can be auto-fixed
        config = [
            {'operation': 'rgb_op', 'category': 'color_operations'},
            {'operation': 'lab_op', 'category': 'color_operations'}  # Needs RGB->LAB conversion
        ]
        
        result = config_system.validate_and_suggest(config)
        
        assert len(result['warnings']) > 0
        assert len(result['auto_fixes']) > 0
        # Should suggest inserting rgb_to_lab conversion
        assert any('rgb_to_lab' in str(fix) for fix in result['auto_fixes'])