"""
Test cases for pipeline execution system components.

Following strict TDD approach - tests written first, implementation follows.
Tests cover the complete pipeline execution system including:
- OperationPipeline main execution engine
- SmartPipelineAssembler with compatibility validation
- MetadataDependencyResolver for dependency analysis
- OptimizedPipelineExecutor with performance optimizations
- PipelineDebugger for introspection tools
"""

import pytest
import torch


class TestOperationPipeline:
    """Test cases for the main OperationPipeline execution engine."""

    def test_operation_pipeline_exists(self):
        """Test that OperationPipeline exists and can be imported."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        assert OperationPipeline is not None

    def test_operation_pipeline_initialization(self):
        """Test OperationPipeline initialization with operation config."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [
            {
                "category": "denoising_operations",
                "operation": "bilateral",
                "params": {"sigma": 1.0},
            },
            {
                "category": "enhancement_operations",
                "operation": "sharpen",
                "params": {"amount": 1.2},
            },
        ]

        pipeline = OperationPipeline(config)

        assert pipeline is not None
        assert hasattr(pipeline, "operations")
        assert hasattr(pipeline, "metadata_history")
        assert len(pipeline.operations) == 2

    def test_operation_pipeline_call_interface(self):
        """Test OperationPipeline call interface with data processing."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)

        # Mock input data
        input_data = torch.randn(1, 3, 64, 64)

        # Should be callable
        assert callable(pipeline)

        # Call should return processed data and metadata history
        result, metadata = pipeline(input_data)

        assert isinstance(result, torch.Tensor)
        assert isinstance(metadata, list)
        assert len(metadata) == 1  # One operation, one metadata entry

    def test_operation_pipeline_metadata_propagation(self):
        """Test that metadata is properly propagated through pipeline."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [
            {"category": "denoising_operations", "operation": "bilateral"},
            {"category": "enhancement_operations", "operation": "sharpen"},
        ]
        pipeline = OperationPipeline(config)

        input_data = torch.randn(1, 3, 64, 64)
        initial_metadata = {"test_key": "test_value"}

        result, metadata_history = pipeline(input_data, initial_metadata)

        # Metadata history should contain entries for each operation
        assert len(metadata_history) == 2

        # Each metadata entry should be a dictionary
        for metadata_entry in metadata_history:
            assert isinstance(metadata_entry, dict)

    def test_operation_pipeline_get_trainable_operations(self):
        """Test getting trainable operations from pipeline."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [
            {"category": "denoising_operations", "operation": "utnet2"},  # Trainable
            {
                "category": "enhancement_operations",
                "operation": "sharpen",
            },  # Non-trainable
        ]
        pipeline = OperationPipeline(config)

        trainable_ops = pipeline.get_trainable_operations()

        assert isinstance(trainable_ops, list)
        # Should contain at least one trainable operation (utnet2)
        assert len(trainable_ops) >= 1

        # Each entry should be (name, module) tuple
        for name, module in trainable_ops:
            assert isinstance(name, str)
            assert hasattr(module, "parameters")  # Should be a torch.nn.Module

    def test_operation_pipeline_error_handling(self):
        """Test pipeline error handling for invalid configurations."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        # Invalid operation configuration
        invalid_config = [{"category": "invalid_category", "operation": "invalid_op"}]

        with pytest.raises(ValueError, match="Unknown category|Unknown operation"):
            OperationPipeline(invalid_config)

    def test_operation_pipeline_empty_config(self):
        """Test pipeline with empty configuration."""
        from fiddlesticks.execution.pipeline import OperationPipeline

        empty_config = []
        pipeline = OperationPipeline(empty_config)

        # Should handle empty pipeline gracefully
        input_data = torch.randn(1, 3, 64, 64)
        result, metadata = pipeline(input_data)

        # Should return input unchanged
        assert torch.equal(result, input_data)
        assert metadata == []


class TestSmartPipelineAssembler:
    """Test cases for SmartPipelineAssembler with compatibility validation."""

    def test_smart_pipeline_assembler_exists(self):
        """Test that SmartPipelineAssembler exists and can be imported."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler

        assert SmartPipelineAssembler is not None

    def test_smart_pipeline_assembler_initialization(self):
        """Test SmartPipelineAssembler initialization with registry."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        assert assembler is not None
        assert hasattr(assembler, "registry")
        assert hasattr(assembler, "validate_pipeline_compatibility")

    def test_validate_pipeline_compatibility_valid(self):
        """Test pipeline compatibility validation with valid configuration."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        # Valid configuration with compatible operations
        config = [
            {
                "category": "raw_processing_operations",
                "operation": "demosaic",
            },  # RAW_BAYER → RAW_4CH
            {
                "category": "color_processing_operations",
                "operation": "colorin",
            },  # RAW_4CH → RGB
            {"category": "enhancement_operations", "operation": "sharpen"},  # RGB → RGB
        ]

        warnings = assembler.validate_pipeline_compatibility(config)

        assert isinstance(warnings, list)
        # Should have no warnings for valid pipeline
        assert len(warnings) == 0

    def test_validate_pipeline_compatibility_invalid(self):
        """Test pipeline compatibility validation with incompatible operations."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        # Invalid configuration with incompatible types
        config = [
            {"category": "enhancement_operations", "operation": "sharpen"},  # RGB → RGB
            {
                "category": "raw_processing_operations",
                "operation": "demosaic",
            },  # RAW_BAYER → RAW_4CH (incompatible)
        ]

        warnings = assembler.validate_pipeline_compatibility(config)

        assert isinstance(warnings, list)
        # Should have warnings for incompatible pipeline
        assert len(warnings) > 0

        # Check warning content
        warning_text = " ".join(warnings)
        assert (
            "type mismatch" in warning_text.lower()
            or "incompatible" in warning_text.lower()
        )

    def test_suggest_missing_operations(self):
        """Test suggestion of missing operations for type conversion."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.operations.registry import OperationRegistry
        from fiddlesticks.core.operation_spec import InputOutputType

        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        # Request conversion from RAW_BAYER to RGB
        suggestions = assembler.suggest_missing_operations(
            InputOutputType.RAW_BAYER, InputOutputType.RGB
        )

        assert isinstance(suggestions, list)
        # Should suggest appropriate conversion operations
        assert len(suggestions) > 0

    def test_generate_auto_fixes(self):
        """Test automatic generation of pipeline fixes."""
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        # Configuration with type mismatch
        config = [
            {"category": "enhancement_operations", "operation": "sharpen"},
            {"category": "raw_processing_operations", "operation": "demosaic"},
        ]

        warnings = assembler.validate_pipeline_compatibility(config)
        auto_fixes = assembler.generate_auto_fixes(warnings, config)

        assert isinstance(auto_fixes, list)
        # Should provide auto-fix suggestions
        for fix in auto_fixes:
            assert isinstance(fix, dict)
            assert "type" in fix
            assert "position" in fix or "operation" in fix


class TestMetadataDependencyResolver:
    """Test cases for MetadataDependencyResolver dependency analysis."""

    def test_metadata_dependency_resolver_exists(self):
        """Test that MetadataDependencyResolver exists and can be imported."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver

        assert MetadataDependencyResolver is not None

    def test_metadata_dependency_resolver_initialization(self):
        """Test MetadataDependencyResolver initialization with operations."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.core.operation_spec import (
            OperationSpec,
            ProcessingMode,
            InputOutputType,
        )

        # Create mock operation specs
        spec1 = OperationSpec(
            name="op1",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=["input_meta"],
            produces_metadata=["output_meta"],
            constraints={},
            description="Test operation 1",
        )

        resolver = MetadataDependencyResolver([spec1])

        assert resolver is not None
        assert hasattr(resolver, "operations")
        assert hasattr(resolver, "dependency_graph")

    def test_validate_pipeline_metadata_valid(self):
        """Test metadata validation with satisfied dependencies."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.core.operation_spec import (
            OperationSpec,
            ProcessingMode,
            InputOutputType,
        )

        # Operation that requires 'sensor_info' metadata
        spec = OperationSpec(
            name="rawprepare",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=["sensor_info"],
            produces_metadata=["raw_prepared"],
            constraints={},
            description="Raw preparation",
        )

        resolver = MetadataDependencyResolver([spec])

        # Initial metadata contains required 'sensor_info'
        initial_metadata = {"sensor_info": {"model": "Canon EOS R5"}}
        warnings = resolver.validate_pipeline(initial_metadata)

        assert isinstance(warnings, list)
        # Should have no warnings when dependencies are satisfied
        assert len(warnings) == 0

    def test_validate_pipeline_metadata_missing(self):
        """Test metadata validation with missing dependencies."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.core.operation_spec import (
            OperationSpec,
            ProcessingMode,
            InputOutputType,
        )

        # Operation that requires 'sensor_info' metadata
        spec = OperationSpec(
            name="rawprepare",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=["sensor_info"],
            produces_metadata=["raw_prepared"],
            constraints={},
            description="Raw preparation",
        )

        resolver = MetadataDependencyResolver([spec])

        # Initial metadata missing required 'sensor_info'
        initial_metadata = {}
        warnings = resolver.validate_pipeline(initial_metadata)

        assert isinstance(warnings, list)
        # Should have warnings for missing dependencies
        assert len(warnings) > 0

        # Check warning content
        warning_text = " ".join(warnings)
        assert "sensor_info" in warning_text

    def test_suggest_metadata_providers(self):
        """Test suggesting operations that can provide missing metadata."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver

        resolver = MetadataDependencyResolver([])

        # Request providers for missing metadata
        providers = resolver.suggest_providers({"exposure_values"})

        assert isinstance(providers, list)
        # Should suggest operations that produce the required metadata


class TestOptimizedPipelineExecutor:
    """Test cases for OptimizedPipelineExecutor with performance optimizations."""

    def test_optimized_pipeline_executor_exists(self):
        """Test that OptimizedPipelineExecutor exists and can be imported."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor

        assert OptimizedPipelineExecutor is not None

    def test_optimized_executor_initialization(self):
        """Test OptimizedPipelineExecutor initialization with operations."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.operations.wrappers import BilateralWrapper, SharpenWrapper

        operations = [BilateralWrapper(), SharpenWrapper()]
        executor = OptimizedPipelineExecutor(operations)

        assert executor is not None
        assert hasattr(executor, "operations")
        assert hasattr(executor, "execution_plan")
        assert len(executor.operations) == 2

    def test_create_execution_plan(self):
        """Test creation of optimized execution plan."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.operations.wrappers import (
            BilateralWrapper,
            SharpenWrapper,
            UTNet2Wrapper,
        )

        operations = [BilateralWrapper(), SharpenWrapper(), UTNet2Wrapper()]
        executor = OptimizedPipelineExecutor(operations)

        execution_plan = executor.execution_plan

        assert isinstance(execution_plan, dict)
        assert "batching_groups" in execution_plan
        assert "memory_checkpoints" in execution_plan
        assert "device_assignments" in execution_plan
        assert "parallelization_opportunities" in execution_plan

    def test_execute_optimized(self):
        """Test optimized execution with performance enhancements."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.operations.wrappers import BilateralWrapper

        operations = [BilateralWrapper()]
        executor = OptimizedPipelineExecutor(operations)

        input_data = torch.randn(1, 3, 64, 64)

        result, metadata = executor.execute_optimized(input_data)

        assert isinstance(result, torch.Tensor)
        assert isinstance(metadata, dict)

    def test_batching_optimization(self):
        """Test batching optimization for consecutive operations."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.operations.wrappers import BilateralWrapper, SharpenWrapper

        # Two consecutive single-image operations that can be batched
        operations = [BilateralWrapper(), SharpenWrapper()]
        executor = OptimizedPipelineExecutor(operations)

        # Execution plan should identify batching opportunities
        batching_groups = executor.execution_plan["batching_groups"]
        assert len(batching_groups) > 0

    def test_memory_management(self):
        """Test memory management for memory-intensive operations."""
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.operations.wrappers import HDRMergeWrapper

        # Memory-intensive multi-input operation
        operations = [HDRMergeWrapper()]
        executor = OptimizedPipelineExecutor(operations)

        # Should identify memory checkpoints
        memory_checkpoints = executor.execution_plan["memory_checkpoints"]
        assert isinstance(memory_checkpoints, list)


class TestMetadataDependencyResolver:
    """Test cases for MetadataDependencyResolver dependency analysis."""

    def test_metadata_dependency_resolver_exists(self):
        """Test that MetadataDependencyResolver exists and can be imported."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver

        assert MetadataDependencyResolver is not None

    def test_metadata_dependency_resolver_initialization(self):
        """Test MetadataDependencyResolver initialization with operation specs."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()

        # Get operation specs for resolver
        operations = [
            registry.get_operation("raw_processing_operations", "demosaic").spec,
            registry.get_operation("color_processing_operations", "colorin").spec,
        ]

        resolver = MetadataDependencyResolver(operations)

        assert resolver is not None
        assert resolver.operations == operations
        assert hasattr(resolver, "dependency_graph")

    def test_validate_pipeline_metadata_dependencies_valid(self):
        """Test metadata dependency validation with sufficient initial metadata."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()

        # Operations with metadata requirements
        operations = [
            registry.get_operation("raw_processing_operations", "demosaic").spec,
            registry.get_operation("color_processing_operations", "colorin").spec,
        ]

        resolver = MetadataDependencyResolver(operations)

        # Provide all required metadata
        initial_metadata = {
            "bayer_pattern": "RGGB",
            "color_profile": "sRGB",
            "white_point": "D65",
        }

        warnings = resolver.validate_pipeline(initial_metadata)

        assert isinstance(warnings, list)
        assert len(warnings) == 0  # Should have no warnings

    def test_validate_pipeline_metadata_dependencies_missing(self):
        """Test metadata dependency validation with missing metadata."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()

        # Operations with metadata requirements
        operations = [
            registry.get_operation("raw_processing_operations", "demosaic").spec,
            registry.get_operation("color_processing_operations", "colorin").spec,
        ]

        resolver = MetadataDependencyResolver(operations)

        # Provide incomplete metadata
        initial_metadata = {
            "bayer_pattern": "RGGB"
            # Missing: color_profile, white_point
        }

        warnings = resolver.validate_pipeline(initial_metadata)

        assert isinstance(warnings, list)
        assert len(warnings) > 0  # Should have warnings for missing metadata
        # Check that warnings mention missing metadata
        warning_text = " ".join(warnings)
        assert "color_profile" in warning_text or "white_point" in warning_text

    def test_suggest_metadata_providers(self):
        """Test metadata provider suggestions for missing requirements."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()

        # Operations that produce metadata
        operations = [
            registry.get_operation("raw_processing_operations", "demosaic").spec,
            registry.get_operation("color_processing_operations", "colorin").spec,
        ]

        resolver = MetadataDependencyResolver(operations)

        # Test suggestion for missing metadata
        missing_metadata = {"missing_field"}
        suggestions = resolver._suggest_providers(missing_metadata)

        assert isinstance(suggestions, list)
        # Should be a list (may be empty if no providers found)

    def test_build_dependency_graph(self):
        """Test dependency graph building from operations."""
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.operations.registry import OperationRegistry

        registry = OperationRegistry()

        operations = [
            registry.get_operation("raw_processing_operations", "demosaic").spec,
            registry.get_operation("color_processing_operations", "colorin").spec,
        ]

        resolver = MetadataDependencyResolver(operations)

        # Should have built dependency graph during initialization
        assert hasattr(resolver, "dependency_graph")
        assert isinstance(resolver.dependency_graph, dict)


class TestPipelineDebugger:
    """Test cases for PipelineDebugger introspection tools."""

    def test_pipeline_debugger_exists(self):
        """Test that PipelineDebugger exists and can be imported."""
        from fiddlesticks.execution.debugger import PipelineDebugger

        assert PipelineDebugger is not None

    def test_pipeline_debugger_initialization(self):
        """Test PipelineDebugger initialization."""
        from fiddlesticks.execution.debugger import PipelineDebugger

        debugger = PipelineDebugger()

        assert debugger is not None
        assert hasattr(debugger, "trace_execution")
        assert hasattr(debugger, "analyze_performance")

    def test_trace_execution(self):
        """Test execution tracing with intermediate results."""
        from fiddlesticks.execution.debugger import PipelineDebugger
        from fiddlesticks.execution.pipeline import OperationPipeline

        debugger = PipelineDebugger()

        config = [
            {"category": "denoising_operations", "operation": "bilateral"},
            {"category": "enhancement_operations", "operation": "sharpen"},
        ]
        pipeline = OperationPipeline(config)

        input_data = torch.randn(1, 3, 64, 64)

        trace = debugger.trace_execution(pipeline, input_data)

        assert isinstance(trace, dict)
        assert "inputs" in trace
        assert "steps" in trace
        assert len(trace["steps"]) == 2  # Two operations

        # Each step should have operation details
        for step in trace["steps"]:
            assert "operation" in step
            assert "input_shape" in step
            assert "output_shape" in step
            assert "metadata" in step

    def test_analyze_performance(self):
        """Test performance analysis of pipeline execution."""
        from fiddlesticks.execution.debugger import PipelineDebugger
        from fiddlesticks.execution.pipeline import OperationPipeline

        debugger = PipelineDebugger()

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)

        input_data = torch.randn(1, 3, 64, 64)

        performance_report = debugger.analyze_performance(pipeline, input_data)

        assert isinstance(performance_report, dict)
        assert "total_time" in performance_report
        assert "operation_times" in performance_report
        assert "memory_usage" in performance_report

    def test_save_intermediate_results(self):
        """Test saving intermediate results for inspection."""
        from fiddlesticks.execution.debugger import PipelineDebugger
        from fiddlesticks.execution.pipeline import OperationPipeline

        debugger = PipelineDebugger()

        config = [
            {"category": "denoising_operations", "operation": "bilateral"},
            {"category": "enhancement_operations", "operation": "sharpen"},
        ]
        pipeline = OperationPipeline(config)

        input_data = torch.randn(1, 3, 64, 64)

        # Enable intermediate result saving
        debugger.enable_intermediate_saving(True)
        trace = debugger.trace_execution(pipeline, input_data)

        # Should have saved intermediate results
        assert debugger.has_intermediate_results()
        intermediate_results = debugger.get_intermediate_results()

        assert isinstance(intermediate_results, list)
        assert len(intermediate_results) == 2  # One per operation

    def test_pipeline_visualization(self):
        """Test pipeline visualization generation."""
        from fiddlesticks.execution.debugger import PipelineDebugger
        from fiddlesticks.execution.pipeline import OperationPipeline

        debugger = PipelineDebugger()

        config = [
            {"category": "raw_processing_operations", "operation": "demosaic"},
            {"category": "denoising_operations", "operation": "bilateral"},
            {"category": "enhancement_operations", "operation": "sharpen"},
        ]
        pipeline = OperationPipeline(config)

        # Generate pipeline visualization
        visualization = debugger.generate_pipeline_visualization(pipeline)

        assert isinstance(visualization, dict)
        assert "nodes" in visualization
        assert "edges" in visualization
        assert "data_flow" in visualization

        # Should have nodes for each operation plus input and output nodes
        assert len(visualization["nodes"]) == 5  # 3 operations + input + output


class TestPipelineExecutionIntegration:
    """Test cases for integration between pipeline execution components."""

    def test_all_execution_components_can_be_imported(self):
        """Test that all pipeline execution components can be imported together."""
        from fiddlesticks.execution.pipeline import OperationPipeline
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.execution.assembler import MetadataDependencyResolver
        from fiddlesticks.execution.optimizer import OptimizedPipelineExecutor
        from fiddlesticks.execution.debugger import PipelineDebugger

        assert all(
            [
                OperationPipeline,
                SmartPipelineAssembler,
                MetadataDependencyResolver,
                OptimizedPipelineExecutor,
                PipelineDebugger,
            ]
        )

    def test_pipeline_execution_complete_workflow(self):
        """Test complete pipeline execution workflow with all components."""
        from fiddlesticks.execution.pipeline import OperationPipeline
        from fiddlesticks.execution.assembler import SmartPipelineAssembler
        from fiddlesticks.execution.debugger import PipelineDebugger
        from fiddlesticks.operations.registry import OperationRegistry

        # 1. Validate pipeline configuration
        registry = OperationRegistry()
        assembler = SmartPipelineAssembler(registry)

        config = [
            {"category": "denoising_operations", "operation": "bilateral"},
            {"category": "enhancement_operations", "operation": "sharpen"},
        ]

        warnings = assembler.validate_pipeline_compatibility(config)
        assert len(warnings) == 0  # Should be valid

        # 2. Create and execute pipeline
        pipeline = OperationPipeline(config)
        input_data = torch.randn(1, 3, 64, 64)

        result, metadata = pipeline(input_data)
        assert isinstance(result, torch.Tensor)
        assert isinstance(metadata, list)

        # 3. Debug and analyze execution
        debugger = PipelineDebugger()
        trace = debugger.trace_execution(pipeline, input_data)

        assert "inputs" in trace
        assert "steps" in trace
        assert len(trace["steps"]) == 2
