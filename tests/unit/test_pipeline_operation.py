"""
Test cases for PipelineOperation abstract base class.

Following strict TDD approach - tests written first, implementation follows.
Tests cover the universal interface that enables implementation-agnostic operation composition.
"""

import pytest
import torch
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch


class TestPipelineOperation:
    """Test cases for PipelineOperation abstract base class."""
    
    def test_pipeline_operation_exists(self):
        """Test that PipelineOperation abstract base class exists and can be imported."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        assert PipelineOperation is not None
    
    def test_pipeline_operation_is_abstract(self):
        """Test that PipelineOperation is a proper abstract base class."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        
        assert issubclass(PipelineOperation, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            PipelineOperation()
    
    def test_pipeline_operation_has_required_attributes(self):
        """Test that PipelineOperation instances have all required attributes."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create a concrete implementation to test instance attributes
        class TestOperation(PipelineOperation):
            def process_tensors(self, data, metadata, **kwargs):
                return data, {}
        
        spec = OperationSpec(
            name="test",
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
        
        instance = TestOperation(spec)
        
        # Check required instance attributes exist
        assert hasattr(instance, 'spec')
        assert hasattr(instance, 'metadata_cache')
        assert hasattr(instance, 'process_tensors')
        assert hasattr(instance, 'validate_inputs')
        assert hasattr(instance, '__call__')
        assert hasattr(instance, 'operation_type')
        assert hasattr(instance, 'get_parameters')
    
    def test_pipeline_operation_has_abstract_methods(self):
        """Test that PipelineOperation has required abstract methods."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        
        # process_tensors should be abstract
        assert hasattr(PipelineOperation, 'process_tensors')
        assert getattr(PipelineOperation.process_tensors, '__isabstractmethod__', False)
    
    def test_concrete_operation_implementation(self):
        """Test that concrete implementations can be created and used."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        # Create a concrete implementation for testing
        class TestOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                """Simple test implementation that doubles the input."""
                if isinstance(data, list):
                    result = [tensor * 2 for tensor in data]
                else:
                    result = data * 2
                
                output_metadata = {'operation_applied': 'double', 'input_shape': str(data[0].shape if isinstance(data, list) else data.shape)}
                return result, output_metadata
        
        # Create test spec
        spec = OperationSpec(
            name="test_double",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'input_shape'],
            constraints={},
            description="Test doubling operation"
        )
        
        # Create and test the operation
        op = TestOperation(spec)
        assert op.spec == spec
        assert hasattr(op, 'metadata_cache')
        assert op.operation_type == "non_trainable"  # Default value
        assert op.get_parameters() is None  # Default implementation
    
    def test_operation_call_interface_single_tensor(self):
        """Test the __call__ interface with single tensor input."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class TestOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                # Expect list input, return list output
                if not isinstance(data, list):
                    raise ValueError("Expected list input")
                result = [tensor + 1 for tensor in data]
                return result, {'processed': True}
        
        spec = OperationSpec(
            name="test_add_one",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['processed'],
            constraints={},
            description="Test add one operation"
        )
        
        op = TestOperation(spec)
        
        # Test with single tensor (should be converted to list internally)
        input_tensor = torch.randn(3, 64, 64)
        result, result_metadata = op(input_tensor)
        
        # Should get back single tensor (converted from list)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, input_tensor + 1)
        assert result_metadata['processed'] is True
    
    def test_operation_call_interface_multi_tensor(self):
        """Test the __call__ interface with multi-tensor input."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class HDRMergeOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                # Expect list of tensors, return single tensor (average)
                if not isinstance(data, list) or len(data) < 3:
                    raise ValueError("HDR merge requires at least 3 input tensors")
                
                result = torch.stack(data).mean(dim=0)  # Simple average for test
                return [result], {'merged_count': len(data)}
        
        hdr_spec = OperationSpec(
            name="hdr_merge",
            supported_modes=[ProcessingMode.BURST_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(3, None),  # At least 3 inputs
            output_count=1,
            requires_metadata=[],
            produces_metadata=['merged_count'],
            constraints={},
            description="HDR merge operation"
        )
        
        op = HDRMergeOperation(hdr_spec)
        
        # Test with multiple tensors
        input_tensors = [torch.randn(3, 64, 64) for _ in range(5)]
        result, result_metadata = op(input_tensors)
        
        # Should get back single tensor
        assert isinstance(result, torch.Tensor)
        expected = torch.stack(input_tensors).mean(dim=0)
        assert torch.allclose(result, expected)
        assert result_metadata['merged_count'] == 5
    
    def test_operation_validation(self):
        """Test input validation functionality."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class ValidatingOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                return data, {}
        
        # Create operation that requires exactly 2 inputs
        spec = OperationSpec(
            name="dual_input_op",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(2, 2),  # Exactly 2 inputs required
            output_count=1,
            requires_metadata=['required_field'],
            produces_metadata=[],
            constraints={},
            description="Operation requiring exactly 2 inputs"
        )
        
        op = ValidatingOperation(spec)
        
        # Test validation failure with wrong input count - updated regex to match actual error
        with pytest.raises(ValueError, match=r"requires \(2, 2\) inputs"):
            op(torch.randn(3, 64, 64))  # Single tensor when 2 required
        
        # Test validation failure with missing metadata
        with pytest.raises(ValueError, match="required_field"):
            op([torch.randn(3, 64, 64), torch.randn(3, 64, 64)])
    
    def test_trainable_operation_interface(self):
        """Test interface for trainable operations (ML models)."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class TrainableOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
                self.model = torch.nn.Conv2d(3, 3, 3, padding=1)  # Simple conv layer
            
            @property
            def operation_type(self) -> str:
                return "trainable"
            
            def get_parameters(self) -> Optional[torch.nn.Module]:
                return self.model
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                # Apply the neural network
                if isinstance(data, list):
                    result = [self.model(tensor.unsqueeze(0)).squeeze(0) for tensor in data]
                else:
                    result = self.model(data.unsqueeze(0)).squeeze(0)
                return result, {'model_applied': True}
        
        spec = OperationSpec(
            name="learned_filter",
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['model_applied'],
            constraints={},
            description="Learned filtering operation"
        )
        
        op = TrainableOperation(spec)
        
        # Test trainable properties
        assert op.operation_type == "trainable"
        assert op.get_parameters() is not None
        assert isinstance(op.get_parameters(), torch.nn.Module)
        
        # Test operation functionality
        input_tensor = torch.randn(3, 64, 64)
        result, metadata = op(input_tensor)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape
        assert metadata['model_applied'] is True
    
    def test_metadata_handling(self):
        """Test metadata caching and propagation."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class MetadataOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                # Use metadata_cache to store information
                self.metadata_cache['last_processed_shape'] = str(data[0].shape if isinstance(data, list) else data.shape)
                
                # Add to output metadata
                output_metadata = {
                    'operation_id': self.spec.name,
                    'input_metadata_keys': list(metadata.keys()),
                    'cache_size': len(self.metadata_cache)
                }
                
                return data, output_metadata
        
        spec = OperationSpec(
            name="metadata_test",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_id', 'input_metadata_keys', 'cache_size'],
            constraints={},
            description="Metadata handling test operation"
        )
        
        op = MetadataOperation(spec)
        
        # Test with input metadata
        input_tensor = torch.randn(3, 32, 32)
        input_metadata = {'source': 'test', 'timestamp': 12345}
        
        result, output_metadata = op(input_tensor, metadata=input_metadata)
        
        # Check metadata handling
        assert output_metadata['operation_id'] == 'metadata_test'
        assert 'source' in output_metadata['input_metadata_keys']
        assert 'timestamp' in output_metadata['input_metadata_keys']
        assert output_metadata['cache_size'] >= 1
        
        # Check metadata_cache was updated
        assert 'last_processed_shape' in op.metadata_cache
        assert op.metadata_cache['last_processed_shape'] == 'torch.Size([3, 32, 32])'
    
    def test_operation_error_handling(self):
        """Test error handling in operations."""
        from fiddlesticks.core.pipeline_operation import PipelineOperation
        from fiddlesticks.core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
        
        class ErrorProneOperation(PipelineOperation):
            def __init__(self, spec: OperationSpec):
                super().__init__(spec)
            
            def process_tensors(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                              metadata: Dict[str, Any], **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
                # Simulate operation that fails under certain conditions
                if 'cause_error' in kwargs:
                    raise RuntimeError("Simulated operation failure")
                
                return data, {'success': True}
        
        spec = OperationSpec(
            name="error_test",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['success'],
            constraints={},
            description="Error handling test operation"
        )
        
        op = ErrorProneOperation(spec)
        input_tensor = torch.randn(3, 16, 16)
        
        # Test successful operation
        result, metadata = op(input_tensor)
        assert metadata['success'] is True
        
        # Test error propagation
        with pytest.raises(RuntimeError, match="Simulated operation failure"):
            op(input_tensor, cause_error=True)