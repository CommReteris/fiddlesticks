"""
I/O Operations for Function-Based Composable Pipeline Architecture.

This module provides comprehensive Input/Output operations that form the critical
pipeline boundaries, enabling data loading, saving, streaming, and format conversion:

File Operations:
- LoadRawOperation: Load raw sensor data from files
- LoadRgbOperation: Load RGB images from standard formats
- LoadMetadataOperation: Load metadata from JSON files
- SaveRawOperation: Save raw tensor data to files
- SaveRgbOperation: Save RGB images to standard formats
- SaveMetadataOperation: Save metadata to JSON files

Stream Operations:
- StreamInputOperation: Read data from streams (cameras, network, etc.)
- StreamOutputOperation: Write data to streams (displays, network, etc.)

Format Conversion Operations:
- RawToRgbOperation: Convert raw sensor data to RGB
- RgbToLabOperation: Convert RGB to LAB color space
- LabToRgbOperation: Convert LAB back to RGB color space
- RgbToGrayscaleOperation: Convert RGB to single-channel grayscale

All operations follow the universal PipelineOperation interface and maintain
proper metadata flow throughout the processing pipeline.
"""

# File Operations
from .file_operations import (
    LoadRawOperation,
    LoadRgbOperation,
    LoadMetadataOperation,
    SaveRawOperation,
    SaveRgbOperation,
    SaveMetadataOperation,
)
# Format Conversion Operations
from .format_conversion import (
    RawToRgbOperation,
    RgbToLabOperation,
    LabToRgbOperation,
    RgbToGrayscaleOperation,
)
# Stream Operations
from .stream_operations import StreamInputOperation, StreamOutputOperation

# Convenience imports
__all__ = [
    # File Operations
    "LoadRawOperation",
    "LoadRgbOperation",
    "LoadMetadataOperation",
    "SaveRawOperation",
    "SaveRgbOperation",
    "SaveMetadataOperation",
    # Stream Operations
    "StreamInputOperation",
    "StreamOutputOperation",
    # Format Conversion Operations
    "RawToRgbOperation",
    "RgbToLabOperation",
    "LabToRgbOperation",
    "RgbToGrayscaleOperation",
]
