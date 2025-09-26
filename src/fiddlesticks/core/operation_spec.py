"""
Core operation specification types for Function-Based Composable Pipeline Architecture.

This module defines the fundamental types used to specify operations in the pipeline:
- ProcessingMode: Enum defining how operations handle input data
- InputOutputType: Enum defining data types that flow through operations
- OperationSpec: Dataclass specifying complete operation characteristics
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


class ProcessingMode(Enum):
    """
    Enum defining different processing modes for operations.
    
    Each mode specifies how an operation handles input data:
    - SINGLE_IMAGE: Process one image at a time
    - BURST_PROCESSING: Process related images (HDR brackets, focus stacks)
    - BATCH_PROCESSING: Process independent images identically 
    - GROUP_PROCESSING: Process images with relationships (panorama tiles)
    """
    
    SINGLE_IMAGE = "single_image"
    BURST_PROCESSING = "burst_processing"
    BATCH_PROCESSING = "batch_processing"
    GROUP_PROCESSING = "group_processing"


class InputOutputType(Enum):
    """
    Enum defining data types that can flow through pipeline operations.
    
    Core image types:
    - RAW_BAYER: Single-channel Bayer pattern raw sensor data
    - RAW_4CH: Four-channel demosaiced raw data
    - RGB: Three-channel RGB image data
    - LAB: Three-channel LAB color space data
    - GRAYSCALE: Single-channel grayscale image
    - MULTI_EXPOSURE: Multiple exposure images for HDR
    - MASK: Binary/alpha mask data
    
    I/O and metadata types:
    - FILE_PATH: File system path as string
    - STREAM: Real-time data stream
    - NUMPY_ARRAY: NumPy array data
    - JSON_STRING: JSON-formatted metadata
    - METADATA: Structured metadata dictionary
    """
    
    # Core image types
    RAW_BAYER = "raw_bayer"
    RAW_4CH = "raw_4ch"
    RGB = "rgb"
    LAB = "lab"
    GRAYSCALE = "grayscale"
    MULTI_EXPOSURE = "multi_exposure"
    MASK = "mask"
    
    # I/O specific types
    FILE_PATH = "file_path"
    STREAM = "stream"
    NUMPY_ARRAY = "numpy_array"
    JSON_STRING = "json_string"
    METADATA = "metadata"


@dataclass
class OperationSpec:
    """
    Complete specification for a pipeline operation.
    
    This dataclass defines all characteristics of an operation including:
    - Input/output types and counts
    - Processing modes supported
    - Metadata requirements and production
    - Constraints and requirements
    
    Attributes:
        name: Unique identifier for the operation
        supported_modes: List of ProcessingMode values this operation supports
        input_types: List of InputOutputType values this operation accepts
        output_types: List of InputOutputType values this operation produces
        input_count: Tuple of (minimum, maximum) input count, None for unlimited
        output_count: Number of outputs this operation produces
        requires_metadata: List of metadata fields required for operation
        produces_metadata: List of metadata fields produced by operation
        constraints: Dictionary of additional constraints and requirements
        description: Human-readable description of operation functionality
    """
    
    name: str
    supported_modes: List[ProcessingMode]
    input_types: List[InputOutputType]
    output_types: List[InputOutputType]
    input_count: Tuple[int, Optional[int]]
    output_count: int
    requires_metadata: List[str]
    produces_metadata: List[str]
    constraints: Dict[str, Any]
    description: str
    
    def __post_init__(self):
        """Validate OperationSpec fields after initialization."""
        # Validate input_count tuple
        min_inputs, max_inputs = self.input_count
        if min_inputs < 0:
            raise ValueError(f"Minimum input count cannot be negative: {min_inputs}")
        if max_inputs is not None and max_inputs < min_inputs:
            raise ValueError(f"Maximum input count ({max_inputs}) cannot be less than minimum ({min_inputs})")
        
        # Validate output_count
        if self.output_count < 0:
            raise ValueError(f"Output count cannot be negative: {self.output_count}")
        
        # Validate non-empty required fields
        if not self.name:
            raise ValueError("Operation name cannot be empty")
        if not self.supported_modes:
            raise ValueError("At least one processing mode must be supported")
        if not self.input_types:
            raise ValueError("At least one input type must be specified")
        if not self.output_types:
            raise ValueError("At least one output type must be specified")