"""
Model wrappers for integrating ML models with the fiddlesticks pipeline system.

This module provides PipelineOperation wrappers for machine learning models,
enabling them to be used seamlessly within the composable pipeline architecture.
Each wrapper handles the interface between the universal pipeline contract
and the specific model's requirements.
"""

from typing import List, Dict, Any, Tuple, Optional

import torch

from ..core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
from ..core.pipeline_operation import PipelineOperation
from ..registries.model_registry import ModelRegistry


class UTNet2Wrapper(PipelineOperation):
    """
    Pipeline wrapper for UTNet2 deep learning denoiser.

    This wrapper integrates the UTNet2 model into the fiddlesticks pipeline system,
    providing the universal operation interface while handling UTNet2-specific
    functionality like model loading, inference, and metadata management.

    Features:
    - Automatic model instantiation from registry
    - Proper tensor handling for 4-channel Bayer inputs
    - Metadata propagation with denoising information
    - Trainable parameter exposure for training pipelines
    - GPU/CPU device management
    """

    def __init__(self, spec: OperationSpec, **model_kwargs):
        """
        Initialize UTNet2Wrapper with operation specification.

        Args:
            spec: OperationSpec defining the operation characteristics
            **model_kwargs: Additional parameters for UTNet2 model creation
                            (e.g., funit, activation, preupsample)
        """
        super().__init__(spec)

        # Get model registry and create UTNet2 instance
        registry = ModelRegistry()

        # Default UTNet2 parameters
        default_params = {
            "in_channels": 4,
            "funit": 32,
            "activation": "LeakyReLU",
            "preupsample": False,
        }
        default_params.update(model_kwargs)

        # Create UTNet2 model
        self.model = registry.create_model("utnet2", **default_params)
        self.model_params = default_params

        # Track device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def operation_type(self) -> str:
        """Return operation type for training/inference handling."""
        return "trainable"

    def get_parameters(self) -> Optional[torch.nn.Module]:
        """Return the UTNet2 model for parameter access."""
        return self.model

    def process_tensors(
        self, inputs: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process input tensors through UTNet2 denoising model.

        Args:
            inputs: List containing single tensor of shape [B, C, H, W]
                   where C=4 for Bayer pattern or C=3 for RGB
            metadata: Input metadata dictionary
            **kwargs: Additional processing parameters

        Returns:
            Tuple of (processed_tensors, updated_metadata)
            - processed_tensors: List with denoised RGB tensor
            - updated_metadata: Metadata with denoising information
        """
        if len(inputs) != 1:
            raise ValueError(
                f"UTNet2Wrapper expects exactly 1 input tensor, got {len(inputs)}"
            )

        input_tensor = inputs[0]

        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        # Store original metadata and add processing info
        output_metadata = metadata.copy()

        # Track input characteristics
        input_shape = input_tensor.shape
        input_channels = input_shape[1] if len(input_shape) >= 2 else 1

        # Validate input channels
        expected_channels = self.model_params["in_channels"]
        if input_channels != expected_channels:
            raise ValueError(
                f"UTNet2 expects {expected_channels}-channel input, got {input_channels}-channel tensor"
            )

        # Process through UTNet2
        self.model.eval()  # Set to evaluation mode for inference
        with torch.no_grad():
            denoised_output = self.model(input_tensor)

        # Update metadata with processing information
        output_metadata.update(
            {
                "denoised": True,
                "denoiser_model": "utnet2",
                "denoiser_params": self.model_params.copy(),
                "input_shape": list(input_shape),
                "output_shape": list(denoised_output.shape),
                "input_channels": input_channels,
                "output_channels": denoised_output.shape[1],
                "device": str(self.device),
                "upsampled": input_channels == 4,  # Bayer inputs are upsampled 2x
            }
        )

        return [denoised_output], output_metadata


class ModelWrapperFactory:
    """
    Factory class for creating model wrappers with proper operation specifications.

    This factory simplifies the creation of model wrappers by providing pre-configured
    operation specifications for common models and use cases.
    """

    @staticmethod
    def create_utnet2_denoiser(
        funit: int = 32, activation: str = "LeakyReLU", **kwargs
    ) -> UTNet2Wrapper:
        """
        Create a UTNet2Wrapper configured for Bayer pattern denoising.

        Args:
            funit: Network capacity parameter (default: 32)
            activation: Activation function name (default: 'LeakyReLU')
            **kwargs: Additional model parameters

        Returns:
            Configured UTNet2Wrapper instance
        """
        spec = OperationSpec(
            name="utnet2_bayer_denoiser",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RAW_4CH],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["denoised", "denoiser_model", "upsampled"],
            constraints={"requires_gpu": torch.cuda.is_available()},
            description="UTNet2 deep learning denoiser for 4-channel Bayer raw images",
        )

        return UTNet2Wrapper(spec, funit=funit, activation=activation, **kwargs)

    @staticmethod
    def create_utnet2_rgb_denoiser(
        funit: int = 32, activation: str = "LeakyReLU", **kwargs
    ) -> UTNet2Wrapper:
        """
        Create a UTNet2Wrapper configured for RGB image denoising.

        Args:
            funit: Network capacity parameter (default: 32)
            activation: Activation function name (default: 'LeakyReLU')
            **kwargs: Additional model parameters

        Returns:
            Configured UTNet2Wrapper instance for RGB inputs
        """
        spec = OperationSpec(
            name="utnet2_rgb_denoiser",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["denoised", "denoiser_model"],
            constraints={"requires_gpu": torch.cuda.is_available()},
            description="UTNet2 deep learning denoiser for 3-channel RGB images",
        )

        return UTNet2Wrapper(
            spec, in_channels=3, funit=funit, activation=activation, **kwargs
        )


# Convenience functions for common use cases
def create_utnet2_denoiser(**kwargs) -> UTNet2Wrapper:
    """Create a standard UTNet2 denoiser for Bayer inputs."""
    return ModelWrapperFactory.create_utnet2_denoiser(**kwargs)


def create_utnet2_rgb_denoiser(**kwargs) -> UTNet2Wrapper:
    """Create a UTNet2 denoiser for RGB inputs."""
    return ModelWrapperFactory.create_utnet2_rgb_denoiser(**kwargs)
