"""
Format Conversion Operations for Function-Based Composable Pipeline Architecture.

This module implements format conversion operations that enable seamless data type
transitions within the pipeline:
- RawToRgbOperation: Convert raw sensor data to RGB
- RgbToLabOperation: Convert RGB to LAB color space
- LabToRgbOperation: Convert LAB back to RGB color space
- RgbToGrayscaleOperation: Convert RGB to single-channel grayscale

These operations handle the mathematical transformations while maintaining proper
metadata flow and the universal PipelineOperation interface.
"""

import time
from typing import Dict, Any, List, Tuple, Optional

import torch

try:
    import kornia

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

from ..core.pipeline_operation import PipelineOperation
from ..core.operation_spec import OperationSpec, ProcessingMode, InputOutputType


class RawToRgbOperation(PipelineOperation):
    """
    Convert raw sensor data to RGB.

    This operation performs demosaicing and color correction to convert
    4-channel raw sensor data to 3-channel RGB data. It handles:
    - Bayer pattern demosaicing
    - White balance correction
    - Color space conversion
    - Basic tone mapping

    Input: 4-channel raw tensor data
    Output: 3-channel RGB tensor data
    """

    def __init__(
        self, white_balance: Optional[List[float]] = None, gamma: float = 2.2, **kwargs
    ):
        """Initialize RawToRgbOperation with specification."""
        spec = OperationSpec(
            name="raw_to_rgb",
            supported_modes=[
                ProcessingMode.SINGLE_IMAGE,
                ProcessingMode.BATCH_PROCESSING,
            ],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["conversion_info", "conversion_time"],
            constraints={"white_balance": white_balance, "gamma": gamma},
            description="Convert raw sensor data to RGB",
        )
        super().__init__(spec)

        self.white_balance = white_balance or [1.0, 1.0, 1.0]
        self.gamma = gamma

    def _simple_demosaic(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simple demosaicing for 4-channel raw data.

        This is a simplified implementation. Real-world demosaicing would
        use more sophisticated algorithms like bilinear, bicubic, or
        learned demosaicing methods.
        """
        # Assume 4-channel raw is arranged as [R, G1, G2, B]
        r_channel = raw_tensor[:, 0:1, :, :]
        g1_channel = raw_tensor[:, 1:2, :, :]
        g2_channel = raw_tensor[:, 2:3, :, :]
        b_channel = raw_tensor[:, 3:4, :, :]

        # Average the two green channels
        g_channel = (g1_channel + g2_channel) / 2.0

        # Combine into RGB
        rgb_tensor = torch.cat([r_channel, g_channel, b_channel], dim=1)

        return rgb_tensor

    def _apply_white_balance(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Apply white balance gains to RGB channels."""
        wb_gains = torch.tensor(
            self.white_balance, device=rgb_tensor.device, dtype=rgb_tensor.dtype
        )
        wb_gains = wb_gains.view(1, 3, 1, 1)  # Reshape for broadcasting

        return rgb_tensor * wb_gains

    def _apply_gamma_correction(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction for tone mapping."""
        # Clamp to prevent negative values
        rgb_tensor = torch.clamp(rgb_tensor, min=0.0)

        # Apply gamma correction
        return torch.pow(rgb_tensor, 1.0 / self.gamma)

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Convert raw data to RGB.

        Args:
            data: List containing raw tensor
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (RGB tensor list, updated metadata)
        """
        start_time = time.time()

        raw_tensor = data[0]

        # Validate input format
        if raw_tensor.shape[1] != 4:
            raise ValueError(
                f"Expected 4-channel raw data, got {raw_tensor.shape[1]} channels"
            )

        # Perform demosaicing
        rgb_tensor = self._simple_demosaic(raw_tensor)

        # Apply white balance
        rgb_tensor = self._apply_white_balance(rgb_tensor)

        # Apply gamma correction
        rgb_tensor = self._apply_gamma_correction(rgb_tensor)

        # Clamp to valid range [0, 1]
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)

        # Update metadata
        conversion_info = {
            "input_format": "raw_4ch",
            "output_format": "rgb",
            "white_balance": self.white_balance,
            "gamma": self.gamma,
            "demosaic_method": "simple",
        }

        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "conversion_info": conversion_info,
                "conversion_time": time.time() - start_time,
                "data_type": "rgb",
            }
        )

        return [rgb_tensor], output_metadata


class RgbToLabOperation(PipelineOperation):
    """
    Convert RGB to LAB color space.

    This operation converts RGB color data to the perceptually uniform
    LAB color space, which is useful for:
    - Perceptual color analysis
    - Color correction and grading
    - Advanced image processing algorithms

    Input: 3-channel RGB tensor data
    Output: 3-channel LAB tensor data
    """

    def __init__(self, **kwargs):
        """Initialize RgbToLabOperation with specification."""
        spec = OperationSpec(
            name="rgb_to_lab",
            supported_modes=[
                ProcessingMode.SINGLE_IMAGE,
                ProcessingMode.BATCH_PROCESSING,
            ],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.LAB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["conversion_info", "conversion_time"],
            constraints={},
            description="Convert RGB to LAB color space",
        )
        super().__init__(spec)

    def _rgb_to_xyz(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Convert RGB to XYZ color space (intermediate step)."""
        # sRGB to XYZ conversion matrix
        rgb_to_xyz_matrix = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            device=rgb_tensor.device,
            dtype=rgb_tensor.dtype,
        )

        # Apply gamma correction (sRGB to linear RGB)
        linear_rgb = torch.where(
            rgb_tensor <= 0.04045,
            rgb_tensor / 12.92,
            torch.pow((rgb_tensor + 0.055) / 1.055, 2.4),
        )

        # Matrix multiplication for color space conversion
        # Reshape for matrix multiplication: [B, C, H, W] -> [B, H*W, C]
        batch_size, channels, height, width = linear_rgb.shape
        linear_rgb_flat = linear_rgb.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        # Apply conversion matrix
        xyz_flat = torch.matmul(linear_rgb_flat, rgb_to_xyz_matrix.T)

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        xyz_tensor = xyz_flat.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)

        return xyz_tensor

    def _xyz_to_lab(self, xyz_tensor: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to LAB color space."""
        # D65 illuminant reference white point
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        reference_white = torch.tensor(
            [xn, yn, zn], device=xyz_tensor.device, dtype=xyz_tensor.dtype
        )
        reference_white = reference_white.view(1, 3, 1, 1)

        # Normalize by reference white
        xyz_norm = xyz_tensor / reference_white

        # Apply f(t) function for LAB conversion
        delta = 6.0 / 29.0
        delta_cubed = delta**3

        f_xyz = torch.where(
            xyz_norm > delta_cubed,
            torch.pow(xyz_norm, 1.0 / 3.0),
            xyz_norm / (3.0 * delta**2) + 4.0 / 29.0,
        )

        fx, fy, fz = f_xyz[:, 0:1], f_xyz[:, 1:2], f_xyz[:, 2:3]

        # Calculate LAB values
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        lab_tensor = torch.cat([L, a, b], dim=1)

        return lab_tensor

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Convert RGB to LAB.

        Args:
            data: List containing RGB tensor
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (LAB tensor list, updated metadata)
        """
        start_time = time.time()

        rgb_tensor = data[0]

        # Validate input format
        if rgb_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3-channel RGB data, got {rgb_tensor.shape[1]} channels"
            )

        # Use Kornia if available for more accurate conversion
        if KORNIA_AVAILABLE:
            lab_tensor = kornia.color.rgb_to_lab(rgb_tensor)
        else:
            # Manual conversion
            xyz_tensor = self._rgb_to_xyz(rgb_tensor)
            lab_tensor = self._xyz_to_lab(xyz_tensor)

        # Update metadata
        conversion_info = {
            "input_format": "rgb",
            "output_format": "lab",
            "method": "kornia" if KORNIA_AVAILABLE else "manual",
        }

        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "conversion_info": conversion_info,
                "conversion_time": time.time() - start_time,
                "data_type": "lab",
            }
        )

        return [lab_tensor], output_metadata


class LabToRgbOperation(PipelineOperation):
    """
    Convert LAB back to RGB color space.

    This operation converts LAB color data back to RGB, which is useful
    for displaying processed images or further RGB-based processing.

    Input: 3-channel LAB tensor data
    Output: 3-channel RGB tensor data
    """

    def __init__(self, **kwargs):
        """Initialize LabToRgbOperation with specification."""
        spec = OperationSpec(
            name="lab_to_rgb",
            supported_modes=[
                ProcessingMode.SINGLE_IMAGE,
                ProcessingMode.BATCH_PROCESSING,
            ],
            input_types=[InputOutputType.LAB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["conversion_info", "conversion_time"],
            constraints={},
            description="Convert LAB to RGB color space",
        )
        super().__init__(spec)

    def _lab_to_xyz(self, lab_tensor: torch.Tensor) -> torch.Tensor:
        """Convert LAB to XYZ color space (intermediate step)."""
        L, a, b = lab_tensor[:, 0:1], lab_tensor[:, 1:2], lab_tensor[:, 2:3]

        # Calculate intermediate values
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        # Apply inverse f(t) function
        delta = 6.0 / 29.0
        delta_cubed = delta**3

        xyz_norm = torch.zeros_like(lab_tensor)

        # X component
        xyz_norm[:, 0:1] = torch.where(
            fx > delta, torch.pow(fx, 3.0), 3.0 * delta**2 * (fx - 4.0 / 29.0)
        )

        # Y component
        xyz_norm[:, 1:2] = torch.where(
            fy > delta, torch.pow(fy, 3.0), 3.0 * delta**2 * (fy - 4.0 / 29.0)
        )

        # Z component
        xyz_norm[:, 2:3] = torch.where(
            fz > delta, torch.pow(fz, 3.0), 3.0 * delta**2 * (fz - 4.0 / 29.0)
        )

        # D65 illuminant reference white point
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        reference_white = torch.tensor(
            [xn, yn, zn], device=lab_tensor.device, dtype=lab_tensor.dtype
        )
        reference_white = reference_white.view(1, 3, 1, 1)

        xyz_tensor = xyz_norm * reference_white

        return xyz_tensor

    def _xyz_to_rgb(self, xyz_tensor: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to RGB color space."""
        # XYZ to sRGB conversion matrix
        xyz_to_rgb_matrix = torch.tensor(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            device=xyz_tensor.device,
            dtype=xyz_tensor.dtype,
        )

        # Matrix multiplication for color space conversion
        batch_size, channels, height, width = xyz_tensor.shape
        xyz_flat = xyz_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        # Apply conversion matrix
        linear_rgb_flat = torch.matmul(xyz_flat, xyz_to_rgb_matrix.T)

        # Reshape back
        linear_rgb = linear_rgb_flat.reshape(batch_size, height, width, 3).permute(
            0, 3, 1, 2
        )

        # Apply gamma correction (linear RGB to sRGB)
        rgb_tensor = torch.where(
            linear_rgb <= 0.0031308,
            12.92 * linear_rgb,
            1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055,
        )

        # Clamp to valid range
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)

        return rgb_tensor

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Convert LAB to RGB.

        Args:
            data: List containing LAB tensor
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (RGB tensor list, updated metadata)
        """
        start_time = time.time()

        lab_tensor = data[0]

        # Validate input format
        if lab_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3-channel LAB data, got {lab_tensor.shape[1]} channels"
            )

        # Use Kornia if available for more accurate conversion
        if KORNIA_AVAILABLE:
            rgb_tensor = kornia.color.lab_to_rgb(lab_tensor)
        else:
            # Manual conversion
            xyz_tensor = self._lab_to_xyz(lab_tensor)
            rgb_tensor = self._xyz_to_rgb(xyz_tensor)

        # Update metadata
        conversion_info = {
            "input_format": "lab",
            "output_format": "rgb",
            "method": "kornia" if KORNIA_AVAILABLE else "manual",
        }

        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "conversion_info": conversion_info,
                "conversion_time": time.time() - start_time,
                "data_type": "rgb",
            }
        )

        return [rgb_tensor], output_metadata


class RgbToGrayscaleOperation(PipelineOperation):
    """
    Convert RGB to single-channel grayscale.

    This operation converts 3-channel RGB data to single-channel grayscale
    using standard luminance weights for perceptually accurate conversion.

    Input: 3-channel RGB tensor data
    Output: 1-channel grayscale tensor data
    """

    def __init__(self, weights: Optional[List[float]] = None, **kwargs):
        """Initialize RgbToGrayscaleOperation with specification."""
        spec = OperationSpec(
            name="rgb_to_grayscale",
            supported_modes=[
                ProcessingMode.SINGLE_IMAGE,
                ProcessingMode.BATCH_PROCESSING,
            ],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["conversion_info", "conversion_time"],
            constraints={"weights": weights},
            description="Convert RGB to single-channel grayscale",
        )
        super().__init__(spec)

        # Standard luminance weights (ITU-R BT.709)
        self.weights = weights or [0.2989, 0.5870, 0.1140]

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Convert RGB to grayscale.

        Args:
            data: List containing RGB tensor
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (grayscale tensor list, updated metadata)
        """
        start_time = time.time()

        rgb_tensor = data[0]

        # Validate input format
        if rgb_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3-channel RGB data, got {rgb_tensor.shape[1]} channels"
            )

        # Use Kornia if available for optimized conversion
        if KORNIA_AVAILABLE:
            grayscale_tensor = kornia.color.rgb_to_grayscale(rgb_tensor)
        else:
            # Manual conversion using luminance weights
            weights = torch.tensor(
                self.weights, device=rgb_tensor.device, dtype=rgb_tensor.dtype
            )
            weights = weights.view(1, 3, 1, 1)  # Reshape for broadcasting

            # Apply weighted sum across channels
            grayscale_tensor = torch.sum(rgb_tensor * weights, dim=1, keepdim=True)

        # Update metadata
        conversion_info = {
            "input_format": "rgb",
            "output_format": "grayscale",
            "weights": self.weights,
            "method": "kornia" if KORNIA_AVAILABLE else "manual",
        }

        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "conversion_info": conversion_info,
                "conversion_time": time.time() - start_time,
                "data_type": "grayscale",
            }
        )

        return [grayscale_tensor], output_metadata
