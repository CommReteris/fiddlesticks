"""
PyTorch Lightning integration for Function-Based Composable Pipeline Architecture.

This module provides professional ML workflow integration through PyTorch Lightning,
enabling scalable training, validation, and monitoring of pipeline-based models.

Key Components:
- ImageProcessingTask: Lightning module wrapper for operation pipelines
- PipelineVisualizationCallback: Visualization and monitoring during training
- QualityMetricsCallback: Quality metrics logging and tracking
- Multi-optimizer support for complex training scenarios

Features:
- Automatic optimizer configuration for trainable operations
- Adaptive loss computation based on operation metadata
- Comprehensive logging and monitoring integration
- Multi-GPU and distributed training support
- Professional callback system for pipeline introspection
"""

import warnings
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn.functional as F

# Handle PyTorch Lightning availability
try:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

    # Create mock classes for when Lightning is not available
    class L:
        class LightningModule:
            def __init__(self):
                pass

            def log(self, *args, **kwargs):
                pass

    class Callback:
        pass


from ..execution.pipeline import OperationPipeline


class ImageProcessingTask(L.LightningModule):
    """
    PyTorch Lightning module wrapper for operation pipelines.

    Provides professional ML training workflows with automatic optimizer
    configuration, adaptive loss computation, and comprehensive logging
    for pipeline-based image processing tasks.

    Features:
    - Automatic optimizer configuration for trainable operations
    - Adaptive loss computation based on operation metadata
    - Multi-optimizer support for complex pipelines
    - Comprehensive logging and monitoring
    - Integration with Lightning's training ecosystem
    """

    def __init__(self, pipeline: OperationPipeline, loss_config: Dict[str, float]):
        """
        Initialize ImageProcessingTask with pipeline and loss configuration.

        Args:
            pipeline: OperationPipeline instance to wrap
            loss_config: Dictionary mapping loss function names to weights
                        e.g., {'mse_loss': 1.0, 'ssim_loss': 0.5}
        """
        super().__init__()

        if not LIGHTNING_AVAILABLE:
            warnings.warn("PyTorch Lightning not available. Limited functionality.")

        self.pipeline = pipeline
        self.loss_config = loss_config
        self.loss_functions = self._create_loss_functions(loss_config)

        # Store trainable operations for optimizer configuration
        self.trainable_operations = pipeline.get_trainable_operations()

        # Initialize automatic optimization if Lightning is available
        if LIGHTNING_AVAILABLE and hasattr(self, "automatic_optimization"):
            self.automatic_optimization = len(self.trainable_operations) <= 1

    def _create_loss_functions(
        self, loss_config: Dict[str, float]
    ) -> List[Tuple[callable, float]]:
        """Create loss functions from configuration."""
        loss_registry = {
            "mse_loss": F.mse_loss,
            "l1_loss": F.l1_loss,
            "smooth_l1_loss": F.smooth_l1_loss,
            "huber_loss": F.huber_loss,
        }

        # Try to import additional loss functions if available
        try:
            import kornia.losses as KL

            kornia_losses = {}

            # Add available loss functions with proper error handling
            try:
                kornia_losses["ssim_loss"] = KL.SSIMLoss(window_size=11)
            except (AttributeError, TypeError):
                pass

            try:
                kornia_losses["ms_ssim_loss"] = KL.MS_SSIMLoss()
            except (AttributeError, TypeError):
                pass

            try:
                kornia_losses["total_variation"] = KL.TotalVariation()
            except (AttributeError, TypeError):
                pass

            # LPIPSLoss may not be available in all Kornia versions
            try:
                kornia_losses["lpips_loss"] = KL.LPIPSLoss(net_type="alex")
            except (AttributeError, TypeError):
                pass

            loss_registry.update(kornia_losses)
        except ImportError:
            pass

        loss_functions = []
        for loss_name, weight in loss_config.items():
            if loss_name in loss_registry:
                loss_functions.append((loss_registry[loss_name], weight))
            else:
                warnings.warn(f"Unknown loss function: {loss_name}")

        if not loss_functions:
            # Default to MSE loss
            loss_functions.append((F.mse_loss, 1.0))

        return loss_functions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Forward pass through the pipeline."""
        return self.pipeline(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step with adaptive loss computation.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index

        Returns:
            Training loss tensor
        """
        inputs, targets = batch

        # Forward pass through pipeline
        outputs, metadata_history = self.pipeline(inputs)

        # Combine metadata from all operations
        combined_metadata = {}
        for metadata in metadata_history:
            combined_metadata.update(metadata)

        # Compute adaptive loss
        loss = self._compute_adaptive_loss(outputs, targets, combined_metadata)

        # Log training loss and metadata
        if LIGHTNING_AVAILABLE:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

            # Log operation metadata
            for key, value in combined_metadata.items():
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    self.log(f"train/{key}", value, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """
        Validation step with comprehensive metrics.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        inputs, targets = batch

        # Forward pass through pipeline
        outputs, metadata_history = self.pipeline(inputs)

        # Combine metadata
        combined_metadata = {}
        for metadata in metadata_history:
            combined_metadata.update(metadata)

        # Compute validation loss
        val_loss = self._compute_adaptive_loss(outputs, targets, combined_metadata)

        # Create metrics dictionary
        metrics = {"val_loss": val_loss}

        # Add metadata to metrics
        for key, value in combined_metadata.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                metrics[f"val_{key}"] = value

        # Log validation metrics
        if LIGHTNING_AVAILABLE:
            for metric_name, metric_value in metrics.items():
                self.log(
                    metric_name,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        return metrics

    def configure_optimizers(self):
        """
        Configure optimizers for trainable operations.

        Returns:
            Optimizer or list of optimizers for trainable operations
        """
        if not self.trainable_operations:
            # No trainable operations - return dummy optimizer
            return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

        optimizers = []
        for op_name, module in self.trainable_operations:
            # Create optimizer for each trainable operation
            optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
            optimizers.append(optimizer)

        # Return single optimizer if only one, list otherwise
        return optimizers[0] if len(optimizers) == 1 else optimizers

    def _compute_adaptive_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute adaptive loss based on operation metadata.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            metadata: Combined metadata from pipeline operations

        Returns:
            Computed loss tensor
        """
        total_loss = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

        for loss_fn, weight in self.loss_functions:
            try:
                if callable(loss_fn):
                    if hasattr(loss_fn, "__call__") and hasattr(loss_fn, "forward"):
                        # Kornia loss function
                        loss_value = loss_fn(outputs, targets)
                    else:
                        # PyTorch functional loss
                        loss_value = loss_fn(outputs, targets)

                    total_loss += weight * loss_value

            except Exception as e:
                warnings.warn(f"Error computing loss with {loss_fn}: {e}")

        # Apply metadata-based loss adaptation if available
        if "quality_metrics" in metadata:
            quality_factor = self._compute_quality_factor(metadata["quality_metrics"])
            total_loss *= quality_factor

        return total_loss

    def _compute_quality_factor(self, quality_metrics: Dict[str, float]) -> float:
        """Compute quality-based loss scaling factor."""
        # Simple quality factor based on available metrics
        factor = 1.0

        if "psnr" in quality_metrics:
            # Scale loss inversely with PSNR quality
            psnr = quality_metrics["psnr"]
            factor *= max(0.5, min(2.0, 30.0 / max(psnr, 10.0)))

        return factor


class PipelineVisualizationCallback(Callback):
    """
    Callback for visualizing intermediate pipeline results during training.

    Saves visualization images of intermediate operation outputs to help
    monitor pipeline behavior and debug training issues.

    Features:
    - Configurable visualization saving frequency
    - Automatic intermediate result capture
    - Support for different image formats
    - Integration with Lightning's logging system
    """

    def __init__(
        self, save_every_n_epochs: int = 1, output_dir: str = "visualizations"
    ):
        """
        Initialize PipelineVisualizationCallback.

        Args:
            save_every_n_epochs: Save visualizations every N epochs
            output_dir: Directory to save visualization images
        """
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir

        # Create output directory
        import os

        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save visualizations at the end of validation epochs."""
        if not LIGHTNING_AVAILABLE:
            return

        current_epoch = trainer.current_epoch

        # Save visualizations every N epochs
        if current_epoch % self.save_every_n_epochs == 0:
            self._save_pipeline_visualizations(pl_module, current_epoch)

    def _save_pipeline_visualizations(self, pl_module, epoch: int):
        """Save visualizations of pipeline intermediate results."""
        try:
            # Get pipeline operations
            if hasattr(pl_module, "pipeline"):
                operations = pl_module.pipeline.operations

                for i, (operation, params) in enumerate(operations):
                    # Check if operation has cached output
                    if hasattr(operation, "last_output"):
                        output = operation.last_output
                        op_name = (
                            operation.spec.name
                            if hasattr(operation, "spec")
                            else f"op_{i}"
                        )
                        self._save_visualization(f"epoch_{epoch}_{op_name}", output)

        except Exception as e:
            warnings.warn(f"Failed to save pipeline visualizations: {e}")

    def _save_visualization(self, name: str, tensor: torch.Tensor):
        """Save tensor visualization as image."""
        try:
            import torchvision.utils as vutils
            import os

            # Normalize tensor to [0, 1] range
            if tensor.min() < 0 or tensor.max() > 1:
                tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

            # Save as image grid
            filepath = os.path.join(self.output_dir, f"{name}.png")
            vutils.save_image(tensor[:8], filepath, nrow=4, normalize=False)

        except ImportError:
            raise NotImplementedError("torchvision required for visualization saving")
        except Exception as e:
            warnings.warn(f"Failed to save visualization {name}: {e}")


class QualityMetricsCallback(Callback):
    """
    Callback for logging quality metrics during training.

    Extracts and logs quality metrics from pipeline operation metadata,
    providing detailed monitoring of processing quality throughout training.

    Features:
    - Automatic quality metrics extraction from metadata
    - Comprehensive logging integration
    - Configurable metric filtering and aggregation
    - Real-time quality monitoring
    """

    def __init__(self, log_prefix: str = "quality"):
        """
        Initialize QualityMetricsCallback.

        Args:
            log_prefix: Prefix for logged metric names
        """
        super().__init__()
        self.log_prefix = log_prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log quality metrics at the end of training batches."""
        if not LIGHTNING_AVAILABLE:
            return

        # Extract quality metrics from outputs
        if isinstance(outputs, dict) and "quality_metrics" in outputs:
            metrics = self._extract_metrics_from_metadata(outputs)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                pl_module.log(metric_name, metric_value, on_step=True, on_epoch=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log quality metrics at the end of validation batches."""
        if not LIGHTNING_AVAILABLE:
            return

        if isinstance(outputs, dict) and "quality_metrics" in outputs:
            metrics = self._extract_metrics_from_metadata(outputs)

            # Log validation metrics
            for metric_name, metric_value in metrics.items():
                val_metric_name = metric_name.replace(
                    self.log_prefix, f"val_{self.log_prefix}"
                )
                pl_module.log(
                    val_metric_name, metric_value, on_step=False, on_epoch=True
                )

    def _extract_metrics_from_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract quality metrics from pipeline metadata.

        Args:
            metadata: Pipeline metadata dictionary

        Returns:
            Dictionary of extracted quality metrics
        """
        extracted_metrics = {}

        # Look for quality metrics in metadata
        if "quality_metrics" in metadata:
            quality_data = metadata["quality_metrics"]

            if isinstance(quality_data, dict):
                for metric_name, metric_value in quality_data.items():
                    if isinstance(metric_value, (int, float)):
                        extracted_metrics[f"{self.log_prefix}/{metric_name}"] = float(
                            metric_value
                        )

        # Look for individual quality-related fields
        quality_fields = [
            "psnr",
            "ssim",
            "lpips",
            "noise_level",
            "sharpness",
            "contrast",
        ]
        for field in quality_fields:
            if field in metadata and isinstance(metadata[field], (int, float)):
                extracted_metrics[f"{self.log_prefix}/{field}"] = float(metadata[field])

        return extracted_metrics


# Utility functions for Lightning integration
def create_lightning_task(
    pipeline: OperationPipeline, loss_config: Dict[str, float]
) -> Optional[ImageProcessingTask]:
    """
    Factory function for creating Lightning tasks from pipelines.

    Args:
        pipeline: OperationPipeline to wrap
        loss_config: Loss function configuration

    Returns:
        ImageProcessingTask instance or None if Lightning unavailable
    """
    if not LIGHTNING_AVAILABLE:
        warnings.warn("PyTorch Lightning not available. Cannot create Lightning task.")
        return None

    return ImageProcessingTask(pipeline, loss_config)


def get_default_callbacks() -> List[Callback]:
    """
    Get default Lightning callbacks for pipeline tasks.

    Returns:
        List of default callbacks
    """
    if not LIGHTNING_AVAILABLE:
        return []

    callbacks = [
        PipelineVisualizationCallback(save_every_n_epochs=5),
        QualityMetricsCallback(),
    ]

    # Add Lightning's built-in callbacks if available
    try:
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

        callbacks.extend(
            [
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath="checkpoints",
                    filename="pipeline-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=3,
                    mode="min",
                ),
                EarlyStopping(
                    monitor="val_loss", min_delta=0.001, patience=10, mode="min"
                ),
            ]
        )
    except ImportError:
        pass

    return callbacks


def create_trainer_with_pipeline(
    pipeline: OperationPipeline, loss_config: Dict[str, float], **trainer_kwargs
) -> Optional[Any]:
    """
    Create complete Lightning trainer with pipeline task and callbacks.

    Args:
        pipeline: OperationPipeline to train
        loss_config: Loss function configuration
        **trainer_kwargs: Additional arguments for Lightning Trainer

    Returns:
        Configured Lightning Trainer or None if unavailable
    """
    if not LIGHTNING_AVAILABLE:
        warnings.warn("PyTorch Lightning not available. Cannot create trainer.")
        return None

    # Create task and callbacks
    task = ImageProcessingTask(pipeline, loss_config)
    callbacks = get_default_callbacks()

    # Default trainer configuration
    default_config = {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": callbacks,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }

    # Update with user-provided arguments
    default_config.update(trainer_kwargs)

    # Create trainer
    trainer = L.Trainer(**default_config)

    return trainer, task
