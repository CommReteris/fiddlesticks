"""
Fiddlesticks: Function-Based Composable Pipeline Architecture for Computer Vision

A comprehensive, production-ready system that handles everything from basic image
processing to advanced computational photography workflows while maintaining the
core principle: the pipeline doesn't care about implementation details, only
functional intent.

Key Features:
- Universal operation interface supporting classical algorithms, ML models, and GPU operations
- Dual interface system (simple for beginners, advanced for power users)
- 75+ operations across 10 functional categories
- 65+ Kornia GPU-accelerated computer vision operations
- Registry pattern extensions for models, quality checks, preprocessing, training
- PyTorch Lightning and Hydra integration
- Production features: A/B testing, containerization, monitoring
"""

__version__ = "0.1.0"
__author__ = "FiddleSticksTeam"
__email__ = "team@fiddlesticks.dev"

# Core imports for easy access - only import existing modules
from .core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
from .core.pipeline_operation import PipelineOperation

# TODO: Add these imports as modules are implemented
# from .core.dual_interface import OperationResolver, LayeredConfigurationSystem
# from .execution.pipeline import OperationPipeline
# from .execution.assembler import SmartPipelineAssembler

__all__ = [
    "OperationSpec",
    "ProcessingMode", 
    "InputOutputType",
    "PipelineOperation",
    # TODO: Add to __all__ as modules are implemented
    # "OperationResolver",
    # "LayeredConfigurationSystem",
    # "OperationPipeline",
    # "SmartPipelineAssembler"
]