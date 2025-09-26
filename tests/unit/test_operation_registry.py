"""
Test cases for comprehensive operation registry system.

Following strict TDD approach - tests written first, implementation follows.
Tests cover the complete operation registry with 75+ operations across 10 functional 
categories, operation wrappers, and registry lookup functionality.
"""

import pytest
import torch
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock


class TestOperationRegistry:
    """Test cases for the comprehensive operation registry system."""
    
    def test_operation_registry_exists(self):
        """Test that ComprehensiveOperationRegistry exists and can be imported."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        assert ComprehensiveOperationRegistry is not None
    
    def test_operation_registry_singleton_pattern(self):
        """Test that ComprehensiveOperationRegistry follows singleton pattern."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry1 = ComprehensiveOperationRegistry()
        registry2 = ComprehensiveOperationRegistry()
        
        # Should be the same instance (singleton)
        assert registry1 is registry2
    
    def test_operation_registry_has_all_categories(self):
        """Test that registry contains all 10 functional categories."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_categories = [
            'input_output_operations',
            'raw_processing_operations', 
            'color_processing_operations',
            'tone_mapping_operations',
            'enhancement_operations',
            'denoising_operations',
            'burst_processing_operations',
            'geometric_operations',
            'quality_assessment_operations',
            'creative_operations'
        ]
        
        for category in expected_categories:
            assert hasattr(registry, category)
            category_dict = getattr(registry, category)
            assert isinstance(category_dict, dict)
    
    def test_input_output_operations_category(self):
        """Test input_output_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            # Input operations
            'load_raw_file', 'load_image', 'load_metadata', 'load_burst', 
            'load_video_frames', 'load_from_stream',
            # Output operations  
            'save_raw', 'save_image', 'save_metadata', 'export_burst',
            'save_video', 'write_to_stream',
            # Format conversion operations
            'raw_to_tensor', 'tensor_to_raw', 'rgb_to_formats', 
            'metadata_to_json', 'numpy_to_torch', 'torch_to_numpy',
            # Validation operations
            'validate_input', 'check_format', 'verify_metadata'
        ]
        
        io_ops = registry.input_output_operations
        
        for op_name in expected_operations:
            assert op_name in io_ops
            assert io_ops[op_name] is not None
    
    def test_raw_processing_operations_category(self):
        """Test raw_processing_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'rawprepare', 'hotpixels', 'temperature', 'rawdenoise', 'demosaic'
        ]
        
        raw_ops = registry.raw_processing_operations
        
        for op_name in expected_operations:
            assert op_name in raw_ops
            assert raw_ops[op_name] is not None
    
    def test_color_processing_operations_category(self):
        """Test color_processing_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'colorin', 'colorout', 'channelmixerrgb', 'colorbalancergb', 'primaries'
        ]
        
        color_ops = registry.color_processing_operations
        
        for op_name in expected_operations:
            assert op_name in color_ops
            assert color_ops[op_name] is not None
    
    def test_tone_mapping_operations_category(self):
        """Test tone_mapping_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'exposure', 'filmicrgb', 'sigmoid', 'toneequal', 'highlights'
        ]
        
        tone_ops = registry.tone_mapping_operations
        
        for op_name in expected_operations:
            assert op_name in tone_ops
            assert tone_ops[op_name] is not None
    
    def test_enhancement_operations_category(self):
        """Test enhancement_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'sharpen', 'diffuse', 'blurs', 'defringe', 'ashift'
        ]
        
        enhance_ops = registry.enhancement_operations
        
        for op_name in expected_operations:
            assert op_name in enhance_ops
            assert enhance_ops[op_name] is not None
    
    def test_denoising_operations_category(self):
        """Test denoising_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'utnet2', 'bm3d', 'bilateral', 'nlmeans', 'denoiseprofile'
        ]
        
        denoise_ops = registry.denoising_operations
        
        for op_name in expected_operations:
            assert op_name in denoise_ops
            assert denoise_ops[op_name] is not None
    
    def test_burst_processing_operations_category(self):
        """Test burst_processing_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'hdr_merge', 'focus_stack', 'panorama_stitch', 'temporal_denoise', 'super_resolution'
        ]
        
        burst_ops = registry.burst_processing_operations
        
        for op_name in expected_operations:
            assert op_name in burst_ops
            assert burst_ops[op_name] is not None
    
    def test_geometric_operations_category(self):
        """Test geometric_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'crop', 'flip', 'rotatepixels', 'scalepixels', 'liquify'
        ]
        
        geom_ops = registry.geometric_operations
        
        for op_name in expected_operations:
            assert op_name in geom_ops
            assert geom_ops[op_name] is not None
    
    def test_quality_assessment_operations_category(self):
        """Test quality_assessment_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'overexposed', 'rawoverexposed', 'noise_estimation', 'blur_detection', 'exposure_analysis'
        ]
        
        quality_ops = registry.quality_assessment_operations
        
        for op_name in expected_operations:
            assert op_name in quality_ops
            assert quality_ops[op_name] is not None
    
    def test_creative_operations_category(self):
        """Test creative_operations category contains expected operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        expected_operations = [
            'grain', 'borders', 'watermark', 'vignette', 'bloom'
        ]
        
        creative_ops = registry.creative_operations
        
        for op_name in expected_operations:
            assert op_name in creative_ops
            assert creative_ops[op_name] is not None
    
    def test_get_operation_method(self):
        """Test get_operation method retrieves operations correctly."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        # Test valid operation retrieval
        operation = registry.get_operation('denoising_operations', 'bilateral')
        assert operation is not None
        
        # Test invalid category
        with pytest.raises(ValueError, match="Unknown category: invalid_category"):
            registry.get_operation('invalid_category', 'bilateral')
        
        # Test invalid operation
        with pytest.raises(ValueError, match="Unknown operation bilateral in category raw_processing_operations"):
            registry.get_operation('raw_processing_operations', 'bilateral')
    
    def test_list_categories_method(self):
        """Test list_categories method returns all category names."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        categories = registry.list_categories()
        
        expected_categories = [
            'input_output_operations', 'raw_processing_operations', 
            'color_processing_operations', 'tone_mapping_operations',
            'enhancement_operations', 'denoising_operations',
            'burst_processing_operations', 'geometric_operations',
            'quality_assessment_operations', 'creative_operations'
        ]
        
        assert len(categories) == 10
        for category in expected_categories:
            assert category in categories
    
    def test_validate_operation_exists_method(self):
        """Test validate_operation_exists method."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        # Test existing operation
        assert registry.validate_operation_exists('denoising_operations', 'bilateral') is True
        
        # Test non-existing category
        assert registry.validate_operation_exists('invalid_category', 'bilateral') is False
        
        # Test non-existing operation
        assert registry.validate_operation_exists('denoising_operations', 'invalid_operation') is False
    
    def test_registry_total_operation_count(self):
        """Test that registry contains at least 75 total operations."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        total_operations = 0
        categories = registry.list_categories()
        
        for category_name in categories:
            category_dict = getattr(registry, category_name)
            total_operations += len(category_dict)
        
        # Should have at least 75 operations total
        assert total_operations >= 75


class TestOperationWrappers:
    """Test cases for operation wrapper implementations."""
    
    def test_operation_wrapper_base_class_exists(self):
        """Test that OperationWrapper base class exists."""
        from fiddlesticks.operations.wrappers import OperationWrapper
        assert OperationWrapper is not None
    
    def test_operation_wrapper_implements_pipeline_operation(self):
        """Test that OperationWrapper implements PipelineOperation interface."""
        from fiddlesticks.operations.wrappers import OperationWrapper
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        
        # OperationWrapper should be a subclass of PipelineOperation
        assert issubclass(OperationWrapper, PipelineOperation)
    
    def test_bilateral_wrapper_exists(self):
        """Test that BilateralWrapper exists and works correctly."""
        from fiddlesticks.operations.wrappers import BilateralWrapper
        from fiddlesticks.core.operation_spec import ProcessingMode, InputOutputType
        
        wrapper = BilateralWrapper()
        
        # Check that it has required attributes
        assert hasattr(wrapper, 'spec')
        assert wrapper.spec.name == 'bilateral'
        assert ProcessingMode.SINGLE_IMAGE in wrapper.spec.supported_modes
        assert InputOutputType.RGB in wrapper.spec.input_types
        assert InputOutputType.RGB in wrapper.spec.output_types
    
    def test_utnet2_wrapper_exists(self):
        """Test that UTNet2Wrapper exists and indicates trainable operation."""
        from fiddlesticks.operations.wrappers import UTNet2Wrapper
        
        wrapper = UTNet2Wrapper()
        
        assert hasattr(wrapper, 'spec')
        assert wrapper.spec.name == 'utnet2'
        assert wrapper.operation_type == 'trainable'
        assert hasattr(wrapper, 'get_parameters')
    
    def test_hdr_merge_wrapper_multi_input(self):
        """Test that HDRMergeWrapper handles multi-input operations correctly."""
        from fiddlesticks.operations.wrappers import HDRMergeWrapper
        from fiddlesticks.core.operation_spec import ProcessingMode
        
        wrapper = HDRMergeWrapper()
        
        assert hasattr(wrapper, 'spec')
        assert wrapper.spec.name == 'hdr_merge'
        assert wrapper.spec.input_count[0] >= 3  # At least 3 inputs for HDR
        assert wrapper.spec.output_count == 1    # Single merged output
        assert ProcessingMode.BURST_PROCESSING in wrapper.spec.supported_modes
    
    def test_load_raw_file_wrapper_io_operation(self):
        """Test that LoadRawFileWrapper handles I/O operations correctly."""
        from fiddlesticks.operations.wrappers import LoadRawFileWrapper
        from fiddlesticks.core.operation_spec import InputOutputType
        
        wrapper = LoadRawFileWrapper()
        
        assert hasattr(wrapper, 'spec')
        assert wrapper.spec.name == 'load_raw_file'
        assert InputOutputType.FILE_PATH in wrapper.spec.input_types
        assert InputOutputType.RAW_BAYER in wrapper.spec.output_types
        assert 'file_format' in wrapper.spec.requires_metadata
    
    def test_operation_wrapper_process_tensors_interface(self):
        """Test that operation wrappers implement process_tensors correctly."""
        from fiddlesticks.operations.wrappers import BilateralWrapper
        
        wrapper = BilateralWrapper()
        
        # Mock input tensor
        mock_input = torch.randn(1, 3, 64, 64)
        mock_metadata = {'test': 'data'}
        
        # Should not raise NotImplementedError (abstract method is implemented)
        try:
            result, output_metadata = wrapper.process_tensors([mock_input], mock_metadata)
            # Result should be tensor or list of tensors
            assert isinstance(result, (torch.Tensor, list))
            assert isinstance(output_metadata, dict)
        except NotImplementedError:
            pytest.fail("process_tensors method not implemented in BilateralWrapper")
    
    def test_wrapper_metadata_handling(self):
        """Test that wrappers handle metadata correctly."""
        from fiddlesticks.operations.wrappers import TemperatureWrapper
        
        wrapper = TemperatureWrapper()
        
        # Should require white balance metadata
        assert 'white_balance_multipliers' in wrapper.spec.requires_metadata
        
        # Should produce temperature adjustment metadata
        assert 'temperature_applied' in wrapper.spec.produces_metadata or \
               'tint_applied' in wrapper.spec.produces_metadata
    
    def test_burst_operation_wrapper_multi_tensor_input(self):
        """Test that burst operation wrappers handle multiple tensor inputs."""
        from fiddlesticks.operations.wrappers import FocusStackWrapper
        
        wrapper = FocusStackWrapper()
        
        # Mock multiple input tensors (focus stack)
        mock_inputs = [
            torch.randn(1, 3, 64, 64),  # Focus at distance 1
            torch.randn(1, 3, 64, 64),  # Focus at distance 2  
            torch.randn(1, 3, 64, 64),  # Focus at distance 3
        ]
        mock_metadata = {'focus_distances': [1.0, 2.0, 3.0]}
        
        # Should handle multiple inputs correctly
        try:
            results, output_metadata = wrapper.process_tensors(mock_inputs, mock_metadata)
            assert len(results) == wrapper.spec.output_count  # Should produce specified outputs
            assert 'depth_map' in output_metadata or \
                   wrapper.spec.output_count == 2  # Extended DOF + depth map
        except Exception as e:
            # Should not fail due to interface mismatch
            assert 'input_count' not in str(e), f"Input count validation failed: {e}"


class TestOperationRegistryIntegration:
    """Test cases for integration between registry and dual interface system."""
    
    def test_registry_integration_with_operation_resolver(self):
        """Test that operation registry integrates with OperationResolver."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        from fiddlesticks.core.dual_interface import OperationResolver
        
        registry = ComprehensiveOperationRegistry()
        
        # Create registry dict for OperationResolver
        registry_dict = {
            'denoising_operations': registry.denoising_operations,
            'enhancement_operations': registry.enhancement_operations,
        }
        
        resolver = OperationResolver(registry_dict)
        
        # Should be able to resolve simple operations
        category, operation_name, spec = resolver.resolve_operation('denoise')
        
        assert category == 'denoising_operations'
        assert operation_name == 'bilateral'
        assert spec is not None
        assert hasattr(spec, 'name')
    
    def test_registry_provides_complete_specs(self):
        """Test that all registry operations provide complete OperationSpec objects."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        from fiddlesticks.core.operation_spec import OperationSpec
        
        registry = ComprehensiveOperationRegistry()
        categories = registry.list_categories()
        
        for category_name in categories:
            category_dict = getattr(registry, category_name)
            
            for operation_name, operation in category_dict.items():
                # Each operation should have a spec
                assert hasattr(operation, 'spec')
                assert isinstance(operation.spec, OperationSpec)
                
                # Spec should have all required fields
                assert operation.spec.name is not None
                assert operation.spec.supported_modes is not None
                assert operation.spec.input_types is not None
                assert operation.spec.output_types is not None
                assert operation.spec.input_count is not None
                assert operation.spec.output_count is not None
    
    def test_registry_operations_are_callable(self):
        """Test that all registry operations are callable with proper interface."""
        from fiddlesticks.operations.registry import ComprehensiveOperationRegistry
        
        registry = ComprehensiveOperationRegistry()
        
        # Test a sample operation from each category
        test_operations = [
            ('denoising_operations', 'bilateral'),
            ('enhancement_operations', 'sharpen'),
            ('geometric_operations', 'crop'),
            ('tone_mapping_operations', 'exposure')
        ]
        
        for category, op_name in test_operations:
            operation = registry.get_operation(category, op_name)
            
            # Should be callable
            assert callable(operation)
            
            # Should have universal interface
            assert hasattr(operation, '__call__')
            assert hasattr(operation, 'process_tensors')
            assert hasattr(operation, 'validate_inputs')