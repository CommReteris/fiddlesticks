"""
PipelineOperation abstract base class for Function-Based Composable Pipeline Architecture.

This module defines the universal interface that all operations must implement,
enabling seamless composition of classical algorithms, ML models, and GPU operations
while maintaining complete implementation agnosticism.

The universal contract: (tensor|List[tensor], metadata, **kwargs) → (result, metadata)
"""

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Any, Optional
import torch

from .operation_spec import OperationSpec


class PipelineOperation(ABC):
    """
    Universal abstract base class for all pipeline operations.
    
    This class defines the common interface that enables implementation-agnostic
    operation composition. All operations, regardless of implementation (classical
    algorithms, ML models, GPU operations), must inherit from this class.
    
    Key principles:
    - Universal interface: All operations follow the same calling convention
    - Flexible input/output: Supports both single tensors and lists of tensors
    - Metadata propagation: Operations can consume and produce metadata
    - Type safety: Input/output validation based on operation specifications
    - Implementation hiding: Pipeline doesn't care about implementation details
    
    Attributes:
        spec: OperationSpec defining operation characteristics
        metadata_cache: Dictionary for storing operation-specific metadata
    """
    
    def __init__(self, spec: OperationSpec):
        """
        Initialize PipelineOperation with specification.
        
        Args:
            spec: OperationSpec defining operation characteristics
        """
        self.spec = spec
        self.metadata_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def process_tensors(
        self, 
        data: Union[torch.Tensor, List[torch.Tensor]], 
        metadata: Dict[str, Any], 
        **kwargs
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """
        Process input data and return processed data with metadata.
        
        This is the core method that implementations must override. It defines
        the actual processing logic while maintaining the universal interface.
        
        Args:
            data: Input tensor(s) to process
            metadata: Input metadata dictionary
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Tuple of (processed_data, output_metadata)
            
        Note:
            Implementations should handle both single tensor and list inputs
            appropriately based on their operation specification.
        """
        pass
    
    def validate_inputs(self, inputs: List[torch.Tensor], metadata: Dict[str, Any]) -> bool:
        """
        Validate input data against operation specification.
        
        Args:
            inputs: List of input tensors
            metadata: Input metadata dictionary
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs don't meet operation requirements
        """
        # Validate input count
        min_inputs, max_inputs = self.spec.input_count
        if len(inputs) < min_inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least {min_inputs} inputs, got {len(inputs)}")
        if max_inputs is not None and len(inputs) > max_inputs:
            raise ValueError(f"Operation {self.spec.name} accepts at most {max_inputs} inputs, got {len(inputs)}")
        
        # Validate required metadata
        for required_field in self.spec.requires_metadata:
            if required_field not in metadata:
                raise ValueError(f"Operation {self.spec.name} requires metadata field: {required_field}")
        
        return True
    
    def __call__(
        self, 
        data: Union[torch.Tensor, List[torch.Tensor]], 
        metadata: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """
        Main entry point for operation execution with universal interface.
        
        This method provides the universal interface that all operations expose
        to the pipeline. It handles input validation, data format conversion,
        and output format standardization.
        
        Args:
            data: Input tensor(s) - single tensor or list of tensors
            metadata: Optional input metadata dictionary
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Tuple of (processed_data, output_metadata)
            
        The method automatically:
        - Converts single tensor to list for multi-input operations
        - Validates inputs against operation specification
        - Calls process_tensors with proper format
        - Converts output back to single tensor for single-output operations
        """
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Handle input format conversion
        if isinstance(data, torch.Tensor):
            # Check if operation requires multiple inputs but got single tensor
            min_inputs, _ = self.spec.input_count
            if min_inputs > 1:
                raise ValueError(f"Operation {self.spec.name} requires {self.spec.input_count} inputs, got single tensor")
            input_data = [data]
        else:
            input_data = data
        
        # Validate inputs
        self.validate_inputs(input_data, metadata)
        
        # Process data
        outputs, output_metadata = self.process_tensors(input_data, metadata, **kwargs)
        
        # Handle output format conversion
        if self.spec.output_count == 1 and isinstance(outputs, list):
            # Convert single-item list back to single tensor for single-output operations
            outputs = outputs[0]
        
        return outputs, output_metadata
    
    @property
    def operation_type(self) -> str:
        """
        Return the type of operation for training/inference handling.
        
        Returns:
            "trainable" for ML models, "non_trainable" for classical algorithms
            
        Note:
            Subclasses should override this property if they contain trainable parameters.
        """
        return "non_trainable"
    
    def get_parameters(self) -> Optional[torch.nn.Module]:
        """
        Return trainable parameters if this is a learnable operation.
        
        Returns:
            PyTorch module containing trainable parameters, or None for non-trainable operations
            
        Note:
            Trainable operations should override this method to return their neural network components.
        """
        return None