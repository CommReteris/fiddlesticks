"""
Framework integration patterns for Function-Based Composable Pipeline Architecture.

This package provides integration patterns with popular ML frameworks and tools:
- PyTorch Lightning: Professional ML workflow integration
- Hydra: Configuration management system integration
- Task-agnostic training patterns for different ML paradigms

Key Features:
- ImageProcessingTask: Lightning module wrapper for pipelines
- PipelineVisualizationCallback: Training visualization and monitoring
- QualityMetricsCallback: Quality metrics logging integration
- Configuration-driven framework integration patterns
"""

from .lightning_integration import (
    ImageProcessingTask,
    PipelineVisualizationCallback,
    QualityMetricsCallback,
    create_lightning_task,
    get_default_callbacks,
    create_trainer_with_pipeline,
)

__all__ = [
    "ImageProcessingTask",
    "PipelineVisualizationCallback",
    "QualityMetricsCallback",
    "create_lightning_task",
    "get_default_callbacks",
    "create_trainer_with_pipeline",
]
