"""
Test cases for PyTorch Lightning integration.

Following strict TDD approach - tests written first, implementation follows.
Tests cover Lightning integration components including:
- ImageProcessingTask Lightning module wrapper
- PipelineVisualizationCallback for monitoring
- QualityMetricsCallback for quality tracking
- Multi-optimizer configuration support
- Adaptive loss computation for different operation types
"""

from unittest.mock import Mock

import pytest
import torch

# Skip tests if lightning not available
try:
    import pytorch_lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="PyTorch Lightning not available")
class TestImageProcessingTask:
    """Test cases for ImageProcessingTask Lightning module."""

    def test_image_processing_task_exists(self):
        """Test that ImageProcessingTask exists and can be imported."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask

        assert ImageProcessingTask is not None

    def test_image_processing_task_inherits_lightning_module(self):
        """Test that ImageProcessingTask inherits from Lightning module."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask

        # Should be a subclass of L.LightningModule
        assert issubclass(ImageProcessingTask, L.LightningModule)

    def test_image_processing_task_initialization(self):
        """Test ImageProcessingTask initialization with pipeline and loss config."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        # Create mock pipeline
        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)

        # Loss configuration
        loss_config = {"mse_loss": 1.0, "ssim_loss": 0.5}

        task = ImageProcessingTask(pipeline, loss_config)

        assert task is not None
        assert hasattr(task, "pipeline")
        assert hasattr(task, "loss_functions")
        assert task.pipeline == pipeline

    def test_training_step_implementation(self):
        """Test training_step method implementation."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = ImageProcessingTask(pipeline, loss_config)

        # Mock batch data
        inputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        batch = (inputs, targets)

        # Training step should return loss tensor
        loss = task.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_validation_step_implementation(self):
        """Test validation_step method implementation."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = ImageProcessingTask(pipeline, loss_config)

        inputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        batch = (inputs, targets)

        # Validation step should return metrics dictionary
        metrics = task.validation_step(batch, 0)

        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_configure_optimizers_single_operation(self):
        """Test optimizer configuration for single trainable operation."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        # Pipeline with trainable operation
        config = [{"category": "denoising_operations", "operation": "utnet2"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = ImageProcessingTask(pipeline, loss_config)

        # Should configure optimizers for trainable operations
        optimizers = task.configure_optimizers()

        # Should return optimizer(s)
        assert optimizers is not None
        # Could be single optimizer or list of optimizers
        if isinstance(optimizers, list):
            assert len(optimizers) >= 1
            for opt in optimizers:
                assert hasattr(opt, "step")  # Should be optimizer
        else:
            assert hasattr(optimizers, "step")  # Should be optimizer

    def test_configure_optimizers_multiple_operations(self):
        """Test optimizer configuration for multiple trainable operations."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        # Pipeline with multiple trainable operations
        config = [
            {"category": "denoising_operations", "operation": "utnet2"},
            {
                "category": "enhancement_operations",
                "operation": "sharpen",
            },  # Non-trainable
        ]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = ImageProcessingTask(pipeline, loss_config)

        optimizers = task.configure_optimizers()

        # Should handle mixed trainable/non-trainable operations
        assert optimizers is not None

    def test_adaptive_loss_computation(self):
        """Test adaptive loss computation based on operation metadata."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 0.7, "l1_loss": 0.3}

        task = ImageProcessingTask(pipeline, loss_config)

        # Mock outputs and targets
        outputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        metadata = {"operation_applied": "bilateral", "some_metric": 0.5}

        # Test adaptive loss computation
        loss = task._compute_adaptive_loss(outputs, targets, metadata)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_metadata_logging(self):
        """Test logging of pipeline metadata during training."""
        from fiddlesticks.integrations.lightning_integration import ImageProcessingTask
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = ImageProcessingTask(pipeline, loss_config)

        # Mock the log method
        task.log = Mock()

        inputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        batch = (inputs, targets)

        # Training step should log metadata
        task.training_step(batch, 0)

        # Should have called log method
        assert task.log.called


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="PyTorch Lightning not available")
class TestPipelineVisualizationCallback:
    """Test cases for PipelineVisualizationCallback."""

    def test_pipeline_visualization_callback_exists(self):
        """Test that PipelineVisualizationCallback exists."""
        from fiddlesticks.integrations.lightning_integration import (
            PipelineVisualizationCallback,
        )

        assert PipelineVisualizationCallback is not None

    def test_callback_inherits_lightning_callback(self):
        """Test that callback inherits from Lightning Callback."""
        from fiddlesticks.integrations.lightning_integration import (
            PipelineVisualizationCallback,
        )

        assert issubclass(PipelineVisualizationCallback, L.Callback)

    def test_callback_initialization(self):
        """Test callback initialization with configuration."""
        from fiddlesticks.integrations.lightning_integration import (
            PipelineVisualizationCallback,
        )

        callback = PipelineVisualizationCallback(save_every_n_epochs=5)

        assert callback is not None
        assert hasattr(callback, "save_every_n_epochs")
        assert callback.save_every_n_epochs == 5

    def test_on_validation_epoch_end_hook(self):
        """Test on_validation_epoch_end hook for saving visualizations."""
        from fiddlesticks.integrations.lightning_integration import (
            PipelineVisualizationCallback,
        )

        callback = PipelineVisualizationCallback()

        # Mock trainer and pl_module
        trainer = Mock()
        trainer.current_epoch = 0

        pl_module = Mock()
        pl_module.pipeline = Mock()
        pl_module.pipeline.operations = [Mock(), Mock()]

        # Should not raise error
        callback.on_validation_epoch_end(trainer, pl_module)

    def test_save_visualization_method(self):
        """Test visualization saving functionality."""
        from fiddlesticks.integrations.lightning_integration import (
            PipelineVisualizationCallback,
        )

        callback = PipelineVisualizationCallback()

        # Mock tensor output
        mock_output = torch.randn(1, 3, 64, 64)
        operation_name = "bilateral"

        # Should handle saving (even if mock implementation)
        try:
            callback._save_visualization(operation_name, mock_output)
        except Exception as e:
            # Expected for mock implementation
            assert "not implemented" in str(e).lower() or "mock" in str(e).lower()


@pytest.mark.skipif(not LIGHTNING_AVAILABLE, reason="PyTorch Lightning not available")
class TestQualityMetricsCallback:
    """Test cases for QualityMetricsCallback."""

    def test_quality_metrics_callback_exists(self):
        """Test that QualityMetricsCallback exists."""
        from fiddlesticks.integrations.lightning_integration import (
            QualityMetricsCallback,
        )

        assert QualityMetricsCallback is not None

    def test_callback_inherits_lightning_callback(self):
        """Test that callback inherits from Lightning Callback."""
        from fiddlesticks.integrations.lightning_integration import (
            QualityMetricsCallback,
        )

        assert issubclass(QualityMetricsCallback, L.Callback)

    def test_on_train_batch_end_hook(self):
        """Test on_train_batch_end hook for quality metrics logging."""
        from fiddlesticks.integrations.lightning_integration import (
            QualityMetricsCallback,
        )

        callback = QualityMetricsCallback()

        # Mock components
        trainer = Mock()
        pl_module = Mock()
        pl_module.log = Mock()

        outputs = {"quality_metrics": {"psnr": 25.0, "ssim": 0.85}}
        batch = Mock()

        # Should extract and log quality metrics
        callback.on_train_batch_end(trainer, pl_module, outputs, batch, 0)

        # Should have called log for quality metrics
        assert pl_module.log.called

    def test_extract_metrics_from_metadata(self):
        """Test extraction of quality metrics from pipeline metadata."""
        from fiddlesticks.integrations.lightning_integration import (
            QualityMetricsCallback,
        )

        callback = QualityMetricsCallback()

        metadata = {
            "operation_applied": "bilateral",
            "quality_metrics": {"psnr": 25.0, "ssim": 0.85, "noise_level": 0.02},
            "other_data": "ignored",
        }

        metrics = callback._extract_metrics_from_metadata(metadata)

        assert isinstance(metrics, dict)
        assert "quality/psnr" in metrics
        assert "quality/ssim" in metrics
        assert "quality/noise_level" in metrics
        assert metrics["quality/psnr"] == 25.0


class TestLightningIntegrationUtils:
    """Test cases for Lightning integration utility functions."""

    def test_create_lightning_task_factory(self):
        """Test factory function for creating Lightning tasks from pipelines."""
        from fiddlesticks.integrations.lightning_integration import (
            create_lightning_task,
        )
        from fiddlesticks.execution.pipeline import OperationPipeline

        config = [{"category": "denoising_operations", "operation": "bilateral"}]
        pipeline = OperationPipeline(config)
        loss_config = {"mse_loss": 1.0}

        task = create_lightning_task(pipeline, loss_config)

        if LIGHTNING_AVAILABLE:
            assert hasattr(task, "training_step")
            assert hasattr(task, "configure_optimizers")
        else:
            # Should handle gracefully when Lightning not available
            assert task is None or hasattr(task, "warning")

    def test_get_default_callbacks(self):
        """Test getting default Lightning callbacks for pipeline tasks."""
        from fiddlesticks.integrations.lightning_integration import (
            get_default_callbacks,
        )

        callbacks = get_default_callbacks()

        assert isinstance(callbacks, list)
        if LIGHTNING_AVAILABLE:
            # Should include default callbacks
            assert len(callbacks) >= 1
        else:
            # Should return empty list or warning when Lightning not available
            assert len(callbacks) == 0
