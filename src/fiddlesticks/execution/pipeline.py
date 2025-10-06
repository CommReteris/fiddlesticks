"""
OperationPipeline main execution engine.

This module provides the core pipeline execution engine that chains operations
following the universal operation interface. It supports:
- Sequential operation execution with metadata propagation
- Trainable operation identification for ML training
- Error handling and validation
- Empty pipeline handling
- Configuration-driven pipeline construction

Key Features:
- Universal operation interface compatibility
- Metadata history tracking
- Automatic operation instantiation from registry
- Trainable operation extraction for PyTorch Lightning integration
"""

from typing import Dict, List, Any, Tuple, Optional, Union

import torch

from ..operations.registry import OperationRegistry


class OperationPipeline:
    """
    Main execution engine for chaining pipeline operations.

    Provides sequential execution of operations with metadata propagation,
    automatic operation instantiation from registry, and support for both
    simple and advanced configuration formats.
    """

    def __init__(self, config: List[Dict[str, Any]]):
        """
        Initialize OperationPipeline with operation configuration.

        Args:
            config: List of operation configuration dictionaries
                   Each dict should contain 'category' and 'operation' keys
                   Optional 'params' key for operation parameters

        Raises:
            ValueError: If configuration contains invalid operations
        """
        self.config = config
        self.operations = []
        self.metadata_history = []
        self.registry = OperationRegistry()

        # Build operations from configuration
        self._build_operations()

    def _build_operations(self):
        """Build operation instances from configuration."""
        self.operations = []

        for op_config in self.config:
            try:
                # Extract operation details
                category = op_config.get("category")
                operation_name = op_config.get("operation")
                params = op_config.get("params", {})

                if not category or not operation_name:
                    raise ValueError(
                        "Operation config must contain 'category' and 'operation' keys"
                    )

                # Get operation from registry
                operation = self.registry.get_operation(category, operation_name)

                # Store operation with parameters
                self.operations.append((operation, params))

            except Exception as e:
                raise ValueError(f"Failed to create operation {op_config}: {e}")

    def __call__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        initial_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], List[Dict[str, Any]]]:
        """
        Execute pipeline on input data.

        Args:
            data: Input data tensor or list of tensors
            initial_metadata: Optional initial metadata dictionary

        Returns:
            Tuple of (processed_data, metadata_history)
        """
        # Handle empty pipeline case
        if not self.operations:
            return data, []

        # Initialize processing
        current_data = data
        metadata_history = []
        current_metadata = initial_metadata or {}

        # Execute operations sequentially
        for operation, params in self.operations:
            try:
                # Execute operation with current data and metadata
                current_data, output_metadata = operation(
                    current_data, current_metadata, **params
                )

                # Update metadata for next operation
                current_metadata.update(output_metadata)

                # Store metadata in history
                metadata_history.append(output_metadata.copy())

            except Exception as e:
                raise RuntimeError(
                    f"Error executing operation {operation.spec.name if hasattr(operation, 'spec') else 'unknown'}: {e}"
                )

        return current_data, metadata_history

    def get_trainable_operations(self) -> List[Tuple[str, torch.nn.Module]]:
        """
        Get all trainable operations in the pipeline.

        Returns:
            List of (operation_name, module) tuples for trainable operations
        """
        trainable_ops = []

        for i, (operation, params) in enumerate(self.operations):
            # Check if operation is trainable
            if (
                hasattr(operation, "operation_type")
                and operation.operation_type == "trainable"
            ):
                # Get trainable parameters
                if hasattr(operation, "get_parameters"):
                    model = operation.get_parameters()
                    if model is not None:
                        # Use operation name or fallback to index
                        op_name = (
                            operation.spec.name
                            if hasattr(operation, "spec")
                            else f"operation_{i}"
                        )
                        trainable_ops.append((op_name, model))

        return trainable_ops

    def execute_optimized(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """
        Execute pipeline with basic optimizations.

        This is a simplified version that will be enhanced by OptimizedPipelineExecutor.
        Currently just calls the standard execution path.

        Args:
            data: Input data tensor or list of tensors
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (processed_data, combined_metadata)
        """
        result, metadata_history = self(data, metadata)

        # Combine metadata from all operations
        combined_metadata = {}
        for meta in metadata_history:
            combined_metadata.update(meta)

        return result, combined_metadata

    def get_operation_count(self) -> int:
        """Get total number of operations in pipeline."""
        return len(self.operations)

    def get_operation_names(self) -> List[str]:
        """Get names of all operations in pipeline."""
        names = []
        for operation, _ in self.operations:
            if hasattr(operation, "spec") and hasattr(operation.spec, "name"):
                names.append(operation.spec.name)
            else:
                names.append(operation.__class__.__name__)
        return names

    def validate_configuration(self) -> List[str]:
        """
        Validate pipeline configuration.

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        if not self.config:
            warnings.append("Empty pipeline configuration")

        for i, op_config in enumerate(self.config):
            if "category" not in op_config:
                warnings.append(f"Operation {i}: Missing 'category' field")
            if "operation" not in op_config:
                warnings.append(f"Operation {i}: Missing 'operation' field")

        return warnings

    def __len__(self) -> int:
        """Return number of operations in pipeline."""
        return len(self.operations)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        op_names = self.get_operation_names()
        return f"OperationPipeline({len(self.operations)} operations: {' → '.join(op_names)})"
