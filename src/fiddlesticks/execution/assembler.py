"""
SmartPipelineAssembler with compatibility validation.

This module provides intelligent pipeline assembly with comprehensive validation,
compatibility checking, and automatic fix suggestions. It supports:
- Input/output type compatibility validation between operations
- Processing mode compatibility checking
- Missing operation suggestions for type conversion
- Automatic fix generation for common pipeline issues
- Integration with the comprehensive operation registry

Key Features:
- Type-aware pipeline validation
- Intelligent operation suggestion system
- Automatic fix generation for type mismatches
- Support for complex multi-input/output operations
- Integration with dual interface system
"""

from typing import Dict, List, Any, Tuple, Optional

from ..core.operation_spec import OperationSpec, InputOutputType, ProcessingMode
from ..operations.registry import OperationRegistry


class SmartPipelineAssembler:
    """
    Intelligent pipeline assembler with compatibility validation.

    Provides comprehensive validation of pipeline configurations including
    type compatibility, processing mode checking, and automatic suggestions
    for fixing common pipeline issues.
    """

    def __init__(self, registry: Optional[OperationRegistry] = None):
        """
        Initialize SmartPipelineAssembler with operation registry.

        Args:
            registry: Operation registry for validation (creates new if None)
        """
        self.registry = registry or OperationRegistry()

        # Build registry lookup for validation
        self._build_registry_lookup()

    def _build_registry_lookup(self):
        """Build internal lookup structures for efficient validation."""
        self.registry_lookup = {}

        # Build category -> operations mapping
        for category in self.registry.list_categories():
            category_dict = getattr(self.registry, category)
            self.registry_lookup[category] = {}

            for operation_name, operation in category_dict.items():
                if hasattr(operation, "spec"):
                    self.registry_lookup[category][operation_name] = operation.spec
                else:
                    # Create basic spec for operations without explicit spec
                    from ..core.operation_spec import (
                        OperationSpec,
                        ProcessingMode,
                        InputOutputType,
                    )

                    basic_spec = OperationSpec(
                        name=operation_name,
                        supported_modes=[ProcessingMode.SINGLE_IMAGE],
                        input_types=[InputOutputType.RGB],
                        output_types=[InputOutputType.RGB],
                        input_count=(1, 1),
                        output_count=1,
                        requires_metadata=[],
                        produces_metadata=[],
                        constraints={},
                        description=f"Basic spec for {operation_name}",
                    )
                    self.registry_lookup[category][operation_name] = basic_spec

    def validate_pipeline_compatibility(
        self, config: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Validate pipeline configuration for compatibility issues.

        Args:
            config: List of operation configuration dictionaries

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        if not config:
            return warnings  # Empty pipeline is valid

        # Validate each operation exists
        for i, op_config in enumerate(config):
            category = op_config.get("category")
            operation = op_config.get("operation")

            if not category or not operation:
                warnings.append(
                    f"Step {i}: Missing category or operation specification"
                )
                continue

            if category not in self.registry_lookup:
                warnings.append(f"Step {i}: Unknown category '{category}'")
                continue

            if operation not in self.registry_lookup[category]:
                warnings.append(
                    f"Step {i}: Unknown operation '{operation}' in category '{category}'"
                )
                continue

        # Validate type compatibility between consecutive operations
        for i in range(len(config) - 1):
            current_op = config[i]
            next_op = config[i + 1]

            # Get operation specs
            current_spec = self._get_operation_spec(current_op)
            next_spec = self._get_operation_spec(next_op)

            if current_spec is None or next_spec is None:
                continue  # Skip if specs couldn't be retrieved

            # Check output -> input type compatibility
            current_outputs = set(current_spec.output_types)
            next_inputs = set(next_spec.input_types)

            compatible_types = current_outputs & next_inputs
            if not compatible_types:
                current_output_names = [t.value for t in current_outputs]
                next_input_names = [t.value for t in next_inputs]

                warnings.append(
                    f"Step {i}: {current_spec.name} outputs {current_output_names} "
                    f"but {next_spec.name} requires {next_input_names} - type mismatch"
                )

            # Check processing mode compatibility
            current_modes = set(current_spec.supported_modes)
            next_modes = set(next_spec.supported_modes)

            if not (current_modes & next_modes):
                warnings.append(
                    f"Step {i}: Processing mode incompatible between "
                    f"{current_spec.name} and {next_spec.name}"
                )

        return warnings

    def _get_operation_spec(self, op_config: Dict[str, Any]) -> Optional[OperationSpec]:
        """Get operation specification from configuration."""
        category = op_config.get("category")
        operation = op_config.get("operation")

        if not category or not operation:
            return None

        if category not in self.registry_lookup:
            return None

        if operation not in self.registry_lookup[category]:
            return None

        return self.registry_lookup[category][operation]

    def suggest_missing_operations(
        self, from_type: InputOutputType, to_type: InputOutputType
    ) -> List[str]:
        """
        Suggest operations that can convert from one type to another.

        Args:
            from_type: Source data type
            to_type: Target data type

        Returns:
            List of suggested operation names
        """
        suggestions = []

        # Common conversion patterns
        conversion_map = {
            (InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH): ["demosaic"],
            (InputOutputType.RAW_BAYER, InputOutputType.RGB): ["demosaic", "colorin"],
            (InputOutputType.RAW_4CH, InputOutputType.RGB): ["colorin"],
            (InputOutputType.RGB, InputOutputType.LAB): ["rgb_to_lab"],
            (InputOutputType.LAB, InputOutputType.RGB): ["lab_to_rgb"],
            (InputOutputType.RGB, InputOutputType.GRAYSCALE): ["rgb_to_grayscale"],
            (InputOutputType.GRAYSCALE, InputOutputType.RGB): ["grayscale_to_rgb"],
            (InputOutputType.NUMPY_ARRAY, InputOutputType.RGB): ["numpy_to_torch"],
            (InputOutputType.RGB, InputOutputType.NUMPY_ARRAY): ["torch_to_numpy"],
            (InputOutputType.FILE_PATH, InputOutputType.RAW_BAYER): ["load_raw_file"],
            (InputOutputType.FILE_PATH, InputOutputType.RGB): ["load_image"],
            (InputOutputType.RGB, InputOutputType.FILE_PATH): ["save_image"],
        }

        # Direct conversion lookup
        conversion_key = (from_type, to_type)
        if conversion_key in conversion_map:
            suggestions.extend(conversion_map[conversion_key])

        # Search registry for operations that can perform conversion
        for category, operations in self.registry_lookup.items():
            for op_name, spec in operations.items():
                if from_type in spec.input_types and to_type in spec.output_types:
                    if op_name not in suggestions:
                        suggestions.append(op_name)

        return suggestions

    def generate_auto_fixes(
        self, warnings: List[str], config: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate automatic fixes for pipeline warnings.

        Args:
            warnings: List of warning messages from validation
            config: Original pipeline configuration

        Returns:
            List of auto-fix suggestion dictionaries
        """
        auto_fixes = []

        for warning in warnings:
            if "type mismatch" in warning.lower():
                # Parse warning to extract operation details
                if "Step" in warning:
                    try:
                        # Extract step number
                        step_part = warning.split("Step")[1].split(":")[0].strip()
                        step_num = int(step_part)

                        if step_num < len(config) - 1:
                            current_op = config[step_num]
                            next_op = config[step_num + 1]

                            current_spec = self._get_operation_spec(current_op)
                            next_spec = self._get_operation_spec(next_op)

                            if current_spec and next_spec:
                                # Find conversion operations
                                for out_type in current_spec.output_types:
                                    for in_type in next_spec.input_types:
                                        conversion_ops = (
                                            self.suggest_missing_operations(
                                                out_type, in_type
                                            )
                                        )
                                        if conversion_ops:
                                            auto_fixes.append(
                                                {
                                                    "type": "insert_conversion",
                                                    "position": step_num + 1,
                                                    "operations": conversion_ops,
                                                    "reason": f"Convert {out_type.value} to {in_type.value}",
                                                }
                                            )
                                            break
                    except (ValueError, IndexError):
                        continue

            elif "unknown category" in warning.lower():
                auto_fixes.append(
                    {
                        "type": "fix_category",
                        "operation": "check_category_name",
                        "reason": "Verify category name spelling and availability",
                    }
                )

            elif "unknown operation" in warning.lower():
                auto_fixes.append(
                    {
                        "type": "fix_operation",
                        "operation": "check_operation_name",
                        "reason": "Verify operation name spelling and availability",
                    }
                )

        return auto_fixes


class MetadataDependencyResolver:
    """
    Dependency analysis system for metadata requirements across pipeline operations.

    Analyzes metadata dependencies between operations, validates that all required
    metadata can be satisfied, and provides suggestions for missing providers.

    Key features:
    - Dependency graph building from operation specifications
    - Pipeline validation for metadata requirements
    - Intelligent suggestions for missing metadata providers
    - Comprehensive error reporting for unmet dependencies
    """

    def __init__(self, operations: List[OperationSpec]):
        """
        Initialize MetadataDependencyResolver with operation specifications.

        Args:
            operations: List of OperationSpec objects to analyze
        """
        self.operations = operations
        self.dependency_graph = self._build_dependency_graph()

    def validate_pipeline(self, initial_metadata: Dict[str, Any]) -> List[str]:
        """
        Validate that all metadata requirements can be satisfied throughout pipeline.

        Args:
            initial_metadata: Initial metadata available at pipeline start

        Returns:
            List of warning messages for unmet dependencies
        """
        warnings = []
        available_metadata = set(initial_metadata.keys())

        for op in self.operations:
            # Check if all required metadata is available
            missing_metadata = set(op.requires_metadata) - available_metadata

            if missing_metadata:
                # Try to suggest providers for missing metadata
                suggestions = self._suggest_providers(missing_metadata)
                warning = f"{op.name} missing: {list(missing_metadata)}"
                if suggestions:
                    warning += f". Consider: {suggestions}"
                warnings.append(warning)

            # Add metadata that this operation produces
            available_metadata.update(op.produces_metadata)

        return warnings

    def _suggest_providers(self, missing_metadata: set) -> List[str]:
        """
        Suggest operations that could provide missing metadata.

        Args:
            missing_metadata: Set of missing metadata field names

        Returns:
            List of operation names that could provide the missing metadata
        """
        suggestions = []

        for op in self.operations:
            # Check if this operation produces any of the missing metadata
            produced_metadata = set(op.produces_metadata)
            if produced_metadata & missing_metadata:  # Intersection
                suggestions.append(op.name)

        return suggestions

    def _build_dependency_graph(self) -> Dict[str, Any]:
        """
        Build dependency graph analyzing metadata flow between operations.

        Returns:
            Dictionary representing the dependency graph structure
        """
        graph = {
            "producers": {},  # metadata_field -> list of operations that produce it
            "consumers": {},  # metadata_field -> list of operations that require it
            "dependencies": {},  # operation_name -> list of required metadata fields
        }

        for op in self.operations:
            # Map metadata producers
            for metadata_field in op.produces_metadata:
                if metadata_field not in graph["producers"]:
                    graph["producers"][metadata_field] = []
                graph["producers"][metadata_field].append(op.name)

            # Map metadata consumers
            for metadata_field in op.requires_metadata:
                if metadata_field not in graph["consumers"]:
                    graph["consumers"][metadata_field] = []
                graph["consumers"][metadata_field].append(op.name)

            # Map operation dependencies
            graph["dependencies"][op.name] = op.requires_metadata

        return graph

    def generate_auto_fixes(
        self, warnings: List[str], config: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate automatic fixes for pipeline warnings.

        Args:
            warnings: List of warning messages from validation
            config: Original pipeline configuration

        Returns:
            List of auto-fix suggestion dictionaries
        """
        auto_fixes = []

        for warning in warnings:
            if "type mismatch" in warning.lower():
                # Parse warning to extract operation details
                if "Step" in warning:
                    try:
                        # Extract step number
                        step_part = warning.split("Step")[1].split(":")[0].strip()
                        step_num = int(step_part)

                        if step_num < len(config) - 1:
                            current_op = config[step_num]
                            next_op = config[step_num + 1]

                            current_spec = self._get_operation_spec(current_op)
                            next_spec = self._get_operation_spec(next_op)

                            if current_spec and next_spec:
                                # Find conversion operations
                                for out_type in current_spec.output_types:
                                    for in_type in next_spec.input_types:
                                        conversion_ops = (
                                            self.suggest_missing_operations(
                                                out_type, in_type
                                            )
                                        )
                                        if conversion_ops:
                                            auto_fixes.append(
                                                {
                                                    "type": "insert_conversion",
                                                    "position": step_num + 1,
                                                    "operations": conversion_ops,
                                                    "reason": f"Convert {out_type.value} to {in_type.value}",
                                                }
                                            )
                                            break
                    except (ValueError, IndexError):
                        continue

            elif "unknown category" in warning.lower():
                auto_fixes.append(
                    {
                        "type": "fix_category",
                        "operation": "check_category_name",
                        "reason": "Verify category name spelling and availability",
                    }
                )

            elif "unknown operation" in warning.lower():
                auto_fixes.append(
                    {
                        "type": "fix_operation",
                        "operation": "check_operation_name",
                        "reason": "Verify operation name spelling and availability",
                    }
                )

        return auto_fixes

    def get_compatible_operations(
        self, input_type: InputOutputType, output_type: Optional[InputOutputType] = None
    ) -> List[Tuple[str, str]]:
        """
        Get operations compatible with given input and optional output types.

        Args:
            input_type: Required input type
            output_type: Optional required output type

        Returns:
            List of (category, operation_name) tuples
        """
        compatible_ops = []

        for category, operations in self.registry_lookup.items():
            for op_name, spec in operations.items():
                # Check input compatibility
                if input_type in spec.input_types:
                    # Check output compatibility if specified
                    if output_type is None or output_type in spec.output_types:
                        compatible_ops.append((category, op_name))

        return compatible_ops

    def estimate_pipeline_performance(
        self, config: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Estimate performance characteristics of pipeline configuration.

        Args:
            config: Pipeline configuration

        Returns:
            Dictionary with performance estimates
        """
        performance_info = {
            "estimated_operations": len(config),
            "memory_intensive_operations": 0,
            "gpu_required_operations": 0,
            "trainable_operations": 0,
            "batch_friendly_operations": 0,
        }

        for op_config in config:
            spec = self._get_operation_spec(op_config)
            if spec:
                # Check for memory intensive operations
                if spec.input_count[0] > 1:  # Multi-input operations
                    performance_info["memory_intensive_operations"] += 1

                # Check for GPU requirements
                if spec.constraints.get("requires_gpu", False):
                    performance_info["gpu_required_operations"] += 1

                # Check for batch-friendly operations
                if ProcessingMode.BATCH_PROCESSING in spec.supported_modes:
                    performance_info["batch_friendly_operations"] += 1

        return performance_info

    def __len__(self) -> int:
        """Return number of categories in registry."""
        return len(self.registry_lookup)

    def __repr__(self) -> str:
        """String representation of assembler."""
        total_ops = sum(len(ops) for ops in self.registry_lookup.values())
        return f"SmartPipelineAssembler({len(self.registry_lookup)} categories, {total_ops} operations)"
