"""
Test cases for ProcessingMode, InputOutputType enums and OperationSpec dataclass.

Following strict TDD approach - tests written first, implementation follows.
"""

import pytest
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


class TestProcessingMode:
    """Test cases for ProcessingMode enum."""
    
    def test_processing_mode_exists(self):
        """Test that ProcessingMode enum exists and can be imported."""
        from fiddlesticks.core.operation_spec import ProcessingMode
        assert ProcessingMode is not None
    
    def test_processing_mode_has_all_required_values(self):
        """Test that ProcessingMode enum has all required processing modes."""
        from fiddlesticks.core.operation_spec import ProcessingMode
        
        # Check all required processing modes exist
        assert hasattr(ProcessingMode, 'SINGLE_IMAGE')
        assert hasattr(ProcessingMode, 'BURST_PROCESSING')
        assert hasattr(ProcessingMode, 'BATCH_PROCESSING')
        assert hasattr(ProcessingMode, 'GROUP_PROCESSING')
    
    def test_processing_mode_values(self):
        """Test that ProcessingMode enum values are correct strings."""
        from fiddlesticks.core.operation_spec import ProcessingMode
        
        assert ProcessingMode.SINGLE_IMAGE.value == "single_image"
        assert ProcessingMode.BURST_PROCESSING.value == "burst_processing"
        assert ProcessingMode.BATCH_PROCESSING.value == "batch_processing"
        assert ProcessingMode.GROUP_PROCESSING.value == "group_processing"
    
    def test_processing_mode_is_enum(self):
        """Test that ProcessingMode is a proper enum."""
        from fiddlesticks.core.operation_spec import ProcessingMode
        
        assert issubclass(ProcessingMode, Enum)
        assert len(ProcessingMode) == 4
    
    def test_processing_mode_string_representation(self):
        """Test string representations of ProcessingMode values."""
        from fiddlesticks.core.operation_spec import ProcessingMode
        
        assert str(ProcessingMode.SINGLE_IMAGE) == "ProcessingMode.SINGLE_IMAGE"
        assert repr(ProcessingMode.BURST_PROCESSING) == "<ProcessingMode.BURST_PROCESSING: 'burst_processing'>"


class TestInputOutputType:
    """Test cases for InputOutputType enum."""
    
    def test_input_output_type_exists(self):
        """Test that InputOutputType enum exists and can be imported."""
        from fiddlesticks.core.operation_spec import InputOutputType
        assert InputOutputType is not None
    
    def test_input_output_type_has_all_required_values(self):
        """Test that InputOutputType enum has all required data types."""
        from fiddlesticks.core.operation_spec import InputOutputType
        
        # Core image types
        assert hasattr(InputOutputType, 'RAW_BAYER')
        assert hasattr(InputOutputType, 'RAW_4CH')
        assert hasattr(InputOutputType, 'RGB')
        assert hasattr(InputOutputType, 'LAB')
        assert hasattr(InputOutputType, 'GRAYSCALE')
        assert hasattr(InputOutputType, 'MULTI_EXPOSURE')
        assert hasattr(InputOutputType, 'MASK')
        
        # I/O specific types
        assert hasattr(InputOutputType, 'FILE_PATH')
        assert hasattr(InputOutputType, 'STREAM')
        assert hasattr(InputOutputType, 'NUMPY_ARRAY')
        assert hasattr(InputOutputType, 'JSON_STRING')
        assert hasattr(InputOutputType, 'METADATA')
    
    def test_input_output_type_values(self):
        """Test that InputOutputType enum values are correct strings."""
        from fiddlesticks.core.operation_spec import InputOutputType
        
        # Core image types
        assert InputOutputType.RAW_BAYER.value == "raw_bayer"
        assert InputOutputType.RAW_4CH.value == "raw_4ch"
        assert InputOutputType.RGB.value == "rgb"
        assert InputOutputType.LAB.value == "lab"
        assert InputOutputType.GRAYSCALE.value == "grayscale"
        assert InputOutputType.MULTI_EXPOSURE.value == "multi_exposure"
        assert InputOutputType.MASK.value == "mask"
        
        # I/O specific types
        assert InputOutputType.FILE_PATH.value == "file_path"
        assert InputOutputType.STREAM.value == "stream"
        assert InputOutputType.NUMPY_ARRAY.value == "numpy_array"
        assert InputOutputType.JSON_STRING.value == "json_string"
        assert InputOutputType.METADATA.value == "metadata"
    
    def test_input_output_type_is_enum(self):
        """Test that InputOutputType is a proper enum."""
        from fiddlesticks.core.operation_spec import InputOutputType
        
        assert issubclass(InputOutputType, Enum)
        assert len(InputOutputType) == 12  # 7 core + 5 I/O types
    
    def test_input_output_type_categorization(self):
        """Test that InputOutputType values can be categorized correctly."""
        from fiddlesticks.core.operation_spec import InputOutputType
        
        # Image data types
        image_types = {
            InputOutputType.RAW_BAYER,
            InputOutputType.RAW_4CH,
            InputOutputType.RGB,
            InputOutputType.LAB,
            InputOutputType.GRAYSCALE,
            InputOutputType.MULTI_EXPOSURE,
            InputOutputType.MASK
        }
        
        # I/O and metadata types
        io_types = {
            InputOutputType.FILE_PATH,
            InputOutputType.STREAM,
            InputOutputType.NUMPY_ARRAY,
            InputOutputType.JSON_STRING,
            InputOutputType.METADATA
        }
        
        # Verify no overlap and complete coverage
        all_types = set(InputOutputType)
        assert image_types.union(io_types) == all_types
        assert image_types.intersection(io_types) == set()


class TestOperationSpec:
    """Test cases for OperationSpec dataclass."""
    
    def test_operation_spec_exists(self):
        """Test that OperationSpec dataclass exists and can be imported."""
        from fiddlesticks.core.operation_spec import OperationSpec
        assert OperationSpec is not None
    
    def test_operation_spec_is_dataclass(self):
        """Test that OperationSpec is a proper dataclass."""
        from fiddlesticks.core.operation_spec import OperationSpec
        from dataclasses import is_dataclass
        
        assert is_dataclass(OperationSpec)
    
    def test_operation_spec_has_all_required_fields(self):
        """Test that OperationSpec has all required fields with correct types."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Test field existence through annotation inspection
        annotations = OperationSpec.__annotations__
        
        assert 'name' in annotations
        assert 'supported_modes' in annotations
        assert 'input_types' in annotations
        assert 'output_types' in annotations
        assert 'input_count' in annotations
        assert 'output_count' in annotations
        assert 'requires_metadata' in annotations
        assert 'produces_metadata' in annotations
        assert 'constraints' in annotations
        assert 'description' in annotations
    
    def test_operation_spec_field_types(self):
        """Test that OperationSpec fields have correct type annotations."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        import typing
        
        annotations = OperationSpec.__annotations__
        
        # Check basic types
        assert annotations['name'] == str
        assert annotations['output_count'] == int
        assert annotations['description'] == str
        
        # Check complex types (these will be checked more thoroughly in integration)
        assert 'List' in str(annotations['supported_modes'])
        assert 'List' in str(annotations['input_types'])
        assert 'List' in str(annotations['output_types'])
        assert 'Tuple' in str(annotations['input_count'])
        assert 'List' in str(annotations['requires_metadata'])
        assert 'List' in str(annotations['produces_metadata'])
        assert 'Dict' in str(annotations['constraints'])
    
    def test_operation_spec_can_be_instantiated(self):
        """Test that OperationSpec can be instantiated with valid data."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create a minimal valid OperationSpec
        spec = OperationSpec(
            name="test_operation",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Test operation"
        )
        
        assert spec.name == "test_operation"
        assert spec.supported_modes == [ProcessingMode.SINGLE_IMAGE]
        assert spec.input_types == [InputOutputType.RGB]
        assert spec.output_types == [InputOutputType.RGB]
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1
        assert spec.requires_metadata == []
        assert spec.produces_metadata == []
        assert spec.constraints == {}
        assert spec.description == "Test operation"
    
    def test_operation_spec_complex_instantiation(self):
        """Test OperationSpec with complex multi-image operation."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create HDR merge operation spec
        hdr_spec = OperationSpec(
            name="hdr_merge",
            supported_modes=[ProcessingMode.BURST_PROCESSING, ProcessingMode.GROUP_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(3, None),  # At least 3 images, no upper limit
            output_count=1,
            requires_metadata=['exposure_values', 'alignment_data'],
            produces_metadata=['hdr_merge_method', 'merged_exposure_range'],
            constraints={
                'requires_exposure_bracketing': True,
                'requires_alignment': True,
                'gpu_memory_requirements': '4GB',
                'computational_cost': 'high'
            },
            description='HDR bracketed exposure merging'
        )
        
        assert hdr_spec.name == "hdr_merge"
        assert len(hdr_spec.supported_modes) == 2
        assert hdr_spec.input_count == (3, None)
        assert 'requires_exposure_bracketing' in hdr_spec.constraints
        assert 'exposure_values' in hdr_spec.requires_metadata
    
    def test_operation_spec_validation_constraints(self):
        """Test OperationSpec validates constraints on field values."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Test that input_count must be valid tuple
        with pytest.raises((ValueError, TypeError)):
            OperationSpec(
                name="invalid_spec",
                supported_modes=[ProcessingMode.SINGLE_IMAGE],
                input_types=[InputOutputType.RGB],
                output_types=[InputOutputType.RGB],
                input_count=(-1, 1),  # Invalid: negative minimum
                output_count=1,
                requires_metadata=[],
                produces_metadata=[],
                constraints={},
                description="Invalid operation"
            )
    
    def test_operation_spec_equality(self):
        """Test OperationSpec equality comparison."""
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        spec1 = OperationSpec(
            name="test_op",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Test"
        )
        
        spec2 = OperationSpec(
            name="test_op",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Test"
        )
        
        spec3 = OperationSpec(
            name="different_op",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=[],
            constraints={},
            description="Different"
        )
        
        assert spec1 == spec2
        assert spec1 != spec3