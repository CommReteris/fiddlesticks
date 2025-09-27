"""
Pipeline Debugger for execution tracing and performance analysis.

This module provides debugging and introspection tools for pipeline execution,
including tracing, performance analysis, and visualization capabilities.
"""

import time
from typing import Dict, List, Any

import torch

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PipelineDebugger:
    """
    Pipeline debugging and introspection tool.

    Provides capabilities for:
    - Execution tracing with intermediate results
    - Performance analysis and profiling
    - Pipeline visualization
    - Memory usage monitoring
    """

    def __init__(self):
        """Initialize the pipeline debugger."""
        self.intermediate_saving_enabled = False
        self.intermediate_results: List[torch.Tensor] = []
        self.execution_traces: List[Dict[str, Any]] = []

    def trace_execution(self, pipeline, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Trace pipeline execution with detailed step information.

        Args:
            pipeline: The operation pipeline to trace
            input_data: Input tensor data

        Returns:
            Dict containing execution trace with inputs and steps
        """
        trace = {
            "inputs": {
                "shape": list(input_data.shape),
                "dtype": str(input_data.dtype),
                "device": str(input_data.device),
            },
            "steps": [],
        }

        current_data = input_data

        # Execute each operation in the pipeline
        for i, (operation, params) in enumerate(pipeline.operations):
            step_start_time = time.time()

            # Get operation info from operation spec
            if hasattr(operation, "spec") and hasattr(operation.spec, "name"):
                operation_name = operation.spec.name
            else:
                operation_name = f"operation_{i}"

            if hasattr(operation, "spec") and hasattr(operation.spec, "category"):
                category = operation.spec.category
            else:
                # Try to infer category from operation class name or default
                category = getattr(operation, "category", "unknown")

            input_shape = list(current_data.shape)

            try:
                # Execute the operation directly
                result = operation(current_data, {}, **params)

                if isinstance(result, tuple):
                    current_data, metadata = result
                else:
                    current_data, metadata = result, {}

            except Exception as e:
                # Handle execution errors gracefully
                metadata = {"error": str(e)}
                current_data = input_data  # Keep original data

            step_end_time = time.time()
            execution_time = step_end_time - step_start_time

            # Save intermediate result if enabled
            if self.intermediate_saving_enabled:
                self.intermediate_results.append(
                    current_data.clone()
                    if isinstance(current_data, torch.Tensor)
                    else current_data
                )

            # Record step information
            step_info = {
                "operation": operation_name,
                "category": category,
                "input_shape": input_shape,
                "output_shape": (
                    list(current_data.shape)
                    if isinstance(current_data, torch.Tensor)
                    else "unknown"
                ),
                "execution_time": execution_time,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }

            trace["steps"].append(step_info)

        self.execution_traces.append(trace)
        return trace

    def analyze_performance(self, pipeline, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze pipeline performance including timing and memory usage.

        Args:
            pipeline: The operation pipeline to analyze
            input_data: Input tensor data

        Returns:
            Dict containing performance analysis
        """
        # Get initial memory usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        else:
            initial_memory = 0.0  # Fallback when psutil not available

        start_time = time.time()

        # Execute pipeline with tracing
        trace = self.trace_execution(pipeline, input_data)

        end_time = time.time()
        total_time = end_time - start_time

        # Get final memory usage
        if PSUTIL_AVAILABLE:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
        else:
            final_memory = 0.0  # Fallback when psutil not available
            memory_delta = 0.0

        # Extract operation times from trace
        operation_times = {}
        for step in trace["steps"]:
            operation_name = f"{step['category']}.{step['operation']}"
            operation_times[operation_name] = step["execution_time"]

        performance_report = {
            "total_time": total_time,
            "operation_times": operation_times,
            "memory_usage": {
                "initial_mb": initial_memory,
                "final_mb": final_memory,
                "delta_mb": memory_delta,
            },
            "input_info": trace["inputs"],
            "num_operations": len(trace["steps"]),
        }

        return performance_report

    def enable_intermediate_saving(self, enabled: bool):
        """Enable or disable saving of intermediate results."""
        self.intermediate_saving_enabled = enabled
        if not enabled:
            self.intermediate_results.clear()

    def has_intermediate_results(self) -> bool:
        """Check if intermediate results are available."""
        return len(self.intermediate_results) > 0

    def get_intermediate_results(self) -> List[torch.Tensor]:
        """Get saved intermediate results."""
        return self.intermediate_results.copy()

    def generate_pipeline_visualization(self, pipeline) -> Dict[str, Any]:
        """
        Generate pipeline visualization data.

        Args:
            pipeline: The operation pipeline to visualize

        Returns:
            Dict containing visualization data with nodes, edges, and data flow
        """
        nodes = []
        edges = []
        data_flow = []

        # Create nodes for each operation
        for i, (operation, params) in enumerate(pipeline.operations):
            # Get operation info from operation spec
            if hasattr(operation, "spec") and hasattr(operation.spec, "name"):
                operation_name = operation.spec.name
            else:
                operation_name = f"operation_{i}"

            if hasattr(operation, "spec") and hasattr(operation.spec, "category"):
                category = operation.spec.category
            else:
                # Try to infer category from operation class name or default
                category = getattr(operation, "category", "unknown")

            node = {
                "id": f"op_{i}",
                "name": operation_name,
                "category": category,
                "type": "operation",
                "position": i,
            }
            nodes.append(node)

            # Create edge to next operation
            if i < len(pipeline.operations) - 1:
                edge = {"from": f"op_{i}", "to": f"op_{i+1}", "type": "data_flow"}
                edges.append(edge)

                # Data flow information
                next_operation, next_params = pipeline.operations[i + 1]
                if hasattr(next_operation, "spec") and hasattr(
                    next_operation.spec, "name"
                ):
                    next_operation_name = next_operation.spec.name
                else:
                    next_operation_name = f"operation_{i+1}"

                flow = {
                    "from_operation": operation_name,
                    "to_operation": next_operation_name,
                    "step": i,
                }
                data_flow.append(flow)

        # Add input and output nodes
        input_node = {
            "id": "input",
            "name": "Input",
            "category": "input",
            "type": "data",
            "position": -1,
        }
        output_node = {
            "id": "output",
            "name": "Output",
            "category": "output",
            "type": "data",
            "position": len(pipeline.operations),
        }

        nodes.insert(0, input_node)
        nodes.append(output_node)

        # Connect input to first operation
        if len(pipeline.operations) > 0:
            edges.insert(0, {"from": "input", "to": "op_0", "type": "data_flow"})

            # Connect last operation to output
            edges.append(
                {
                    "from": f"op_{len(pipeline.operations)-1}",
                    "to": "output",
                    "type": "data_flow",
                }
            )

        visualization = {
            "nodes": nodes,
            "edges": edges,
            "data_flow": data_flow,
            "pipeline_info": {
                "num_operations": len(pipeline.operations),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

        return visualization

    def save_trace_to_file(self, filepath: str):
        """Save execution traces to file."""
        import json

        # Convert traces to JSON-serializable format
        serializable_traces = []
        for trace in self.execution_traces:
            serializable_trace = {"inputs": trace["inputs"], "steps": []}

            for step in trace["steps"]:
                serializable_step = {
                    "operation": step["operation"],
                    "category": step["category"],
                    "input_shape": step["input_shape"],
                    "output_shape": step["output_shape"],
                    "execution_time": step["execution_time"],
                    "metadata": step["metadata"],
                }
                serializable_trace["steps"].append(serializable_step)

            serializable_traces.append(serializable_trace)

        with open(filepath, "w") as f:
            json.dump(serializable_traces, f, indent=2)

    def clear_traces(self):
        """Clear all stored traces and intermediate results."""
        self.execution_traces.clear()
        self.intermediate_results.clear()
