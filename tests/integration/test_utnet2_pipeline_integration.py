"""
Integration test for UTNet2 model in fiddlesticks pipeline system.

This test demonstrates end-to-end functionality of the real UTNet2 implementation
integrated with the fiddlesticks composable pipeline architecture. It validates
the complete flow from raw Bayer input through UTNet2 processing to RGB output.
"""

import pytest
import torch
from fiddlesticks.core.operation_spec import (
    OperationSpec,
    ProcessingMode,
    InputOutputType,
)
from fiddlesticks.operations.model_wrappers import UTNet2Wrapper, ModelWrapperFactory


class TestUTNet2PipelineIntegration:
    """Integration tests for UTNet2 in complete pipeline workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.bayer_channels = 4
        self.rgb_channels = 3

        # Generate synthetic Bayer pattern data
        self.bayer_input = torch.randn(
            self.batch_size, self.bayer_channels, self.height, self.width
        )

        # Generate RGB test data
        self.rgb_input = torch.randn(
            self.batch_size, self.rgb_channels, self.height, self.width
        )

        # Basic metadata
        self.initial_metadata = {
            "source": "test_camera",
            "iso": 1600,
            "exposure_time": 0.02,
            "white_balance": "auto",
        }

    def test_end_to_end_bayer_denoising_pipeline(self):
        """Test complete pipeline: Bayer input → UTNet2 → RGB output."""
        # Create UTNet2 wrapper using factory
        wrapper = ModelWrapperFactory.create_utnet2_denoiser(
            funit=16
        )  # Smaller for test speed

        # Verify wrapper properties
        assert wrapper.operation_type == "trainable"
        assert wrapper.get_parameters() is not None
        assert isinstance(wrapper.get_parameters(), torch.nn.Module)

        # Process through pipeline
        output_tensors, output_metadata = wrapper.process_tensors(
            [self.bayer_input], self.initial_metadata
        )

        # Validate output structure
        assert isinstance(output_tensors, list)
        assert len(output_tensors) == 1

        output_tensor = output_tensors[0]
        assert isinstance(output_tensor, torch.Tensor)

        # Validate output shape (should be upsampled 2x due to PixelShuffle)
        expected_shape = (self.batch_size, 3, self.height * 2, self.width * 2)
        assert output_tensor.shape == expected_shape

        # Validate output tensor properties
        assert torch.isfinite(
            output_tensor
        ).all(), "Output should contain finite values"
        assert not torch.isnan(output_tensor).any(), "Output should not contain NaN"

        # Validate metadata updates
        assert "denoised" in output_metadata
        assert output_metadata["denoised"] is True
        assert output_metadata["denoiser_model"] == "utnet2"
        assert output_metadata["upsampled"] is True
        assert output_metadata["input_channels"] == 4
        assert output_metadata["output_channels"] == 3

        # Original metadata should be preserved
        assert output_metadata["source"] == "test_camera"
        assert output_metadata["iso"] == 1600

    def test_end_to_end_rgb_denoising_pipeline(self):
        """Test complete pipeline: RGB input → UTNet2 → RGB output (same resolution)."""
        # Create RGB UTNet2 wrapper
        wrapper = ModelWrapperFactory.create_utnet2_rgb_denoiser(funit=16)

        # Process through pipeline
        output_tensors, output_metadata = wrapper.process_tensors(
            [self.rgb_input], self.initial_metadata
        )

        # Validate output
        assert len(output_tensors) == 1
        output_tensor = output_tensors[0]

        # RGB input should maintain same resolution (no PixelShuffle)
        expected_shape = (self.batch_size, 3, self.height, self.width)
        assert output_tensor.shape == expected_shape

        # Validate metadata
        assert output_metadata["denoised"] is True
        assert output_metadata["upsampled"] is False  # No upsampling for RGB
        assert output_metadata["input_channels"] == 3
        assert output_metadata["output_channels"] == 3

    def test_pipeline_with_different_model_configurations(self):
        """Test pipeline with different UTNet2 configurations."""
        configs = [
            {"funit": 8, "activation": "LeakyReLU"},
            {"funit": 16, "activation": "PReLU"},
            {"funit": 32, "activation": "ELU"},
        ]

        for config in configs:
            wrapper = ModelWrapperFactory.create_utnet2_denoiser(**config)

            output_tensors, output_metadata = wrapper.process_tensors(
                [self.bayer_input], self.initial_metadata
            )

            # Should work with all configurations
            assert len(output_tensors) == 1
            assert output_tensors[0].shape == (
                self.batch_size,
                3,
                self.height * 2,
                self.width * 2,
            )
            assert output_metadata["denoised"] is True

            # Configuration should be stored in metadata
            assert "denoiser_params" in output_metadata
            stored_config = output_metadata["denoiser_params"]
            assert stored_config["funit"] == config["funit"]
            assert stored_config["activation"] == config["activation"]

    def test_error_handling_wrong_input_channels(self):
        """Test proper error handling for incorrect input channels."""
        # Create Bayer wrapper (expects 4 channels)
        wrapper = ModelWrapperFactory.create_utnet2_denoiser()

        # Try to process RGB input (3 channels) - should fail
        with pytest.raises(
            ValueError, match="UTNet2 expects 4-channel input, got 3-channel tensor"
        ):
            wrapper.process_tensors([self.rgb_input], self.initial_metadata)

    def test_gradient_computation_in_training_mode(self):
        """Test that gradients can be computed for training."""
        wrapper = ModelWrapperFactory.create_utnet2_denoiser(funit=8)  # Small for speed

        # Get model parameters
        model = wrapper.get_parameters()
        assert model is not None

        # Create input that requires gradients
        input_tensor = self.bayer_input.clone().requires_grad_(True)

        # Create target (random RGB)
        target = torch.randn(self.batch_size, 3, self.height * 2, self.width * 2)

        # Forward pass
        model.train()  # Set to training mode
        output = model(input_tensor)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Verify gradients were computed
        assert input_tensor.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(
                param.grad
            ).any(), f"Parameter {name} gradients should not be NaN"

    def test_metadata_preservation_and_enhancement(self):
        """Test that metadata is properly preserved and enhanced."""
        wrapper = ModelWrapperFactory.create_utnet2_denoiser()

        # Rich input metadata
        rich_metadata = {
            "camera_model": "Test Camera Pro",
            "lens": "50mm f/1.4",
            "focal_length": 50,
            "aperture": 1.4,
            "iso": 3200,
            "exposure_time": 1 / 100,
            "timestamp": "2024-01-01T12:00:00Z",
            "location": {"lat": 37.7749, "lon": -122.4194},
            "processing_history": ["raw_load", "black_level_correction"],
        }

        output_tensors, output_metadata = wrapper.process_tensors(
            [self.bayer_input], rich_metadata
        )

        # All original metadata should be preserved
        for key, value in rich_metadata.items():
            assert key in output_metadata
            assert output_metadata[key] == value

        # New processing metadata should be added
        processing_keys = [
            "denoised",
            "denoiser_model",
            "denoiser_params",
            "input_shape",
            "output_shape",
            "input_channels",
            "output_channels",
            "device",
            "upsampled",
        ]

        for key in processing_keys:
            assert key in output_metadata, f"Missing processing metadata: {key}"

    def test_device_handling(self):
        """Test proper device handling (CPU/GPU)."""
        wrapper = ModelWrapperFactory.create_utnet2_denoiser(funit=8)

        # Test with CPU tensor
        cpu_input = self.bayer_input.to("cpu")
        output_tensors, output_metadata = wrapper.process_tensors(
            [cpu_input], self.initial_metadata
        )

        # Output should be valid regardless of device
        assert len(output_tensors) == 1
        assert torch.isfinite(output_tensors[0]).all()
        assert "device" in output_metadata

        # If CUDA is available, test GPU processing
        if torch.cuda.is_available():
            cuda_input = self.bayer_input.to("cuda")
            gpu_output_tensors, gpu_output_metadata = wrapper.process_tensors(
                [cuda_input], self.initial_metadata
            )

            assert len(gpu_output_tensors) == 1
            assert torch.isfinite(gpu_output_tensors[0]).all()
            assert "cuda" in gpu_output_metadata["device"].lower()


class TestUTNet2ModelComparison:
    """Compare real UTNet2 with mock version to validate improvement."""

    def test_real_vs_mock_parameter_count(self):
        """Verify real UTNet2 has significantly more parameters than mock."""
        # Create real UTNet2
        from fiddlesticks.registries.model_registry import ModelRegistry

        registry = ModelRegistry()
        real_model = registry.create_model("utnet2", funit=32)

        # Count parameters
        real_params = sum(p.numel() for p in real_model.parameters())

        # Real model should have substantial parameters (>100K)
        assert (
            real_params > 100000
        ), f"Real UTNet2 should have >100K parameters, got {real_params}"

        # Verify it's a proper U-Net architecture
        assert hasattr(real_model, "convs1")
        assert hasattr(real_model, "up1")
        assert hasattr(real_model, "output_module")

    def test_real_model_output_quality(self):
        """Test that real model produces reasonable output."""
        wrapper = ModelWrapperFactory.create_utnet2_denoiser(funit=16)

        # Create noisy input
        clean_signal = torch.randn(1, 4, 32, 32)
        noise = torch.randn_like(clean_signal) * 0.1
        noisy_input = clean_signal + noise

        # Process through real model
        output_tensors, _ = wrapper.process_tensors([noisy_input], {})
        denoised_output = output_tensors[0]

        # Output should be different from input (processing occurred)
        assert denoised_output.shape == (1, 3, 64, 64)  # Upsampled
        assert not torch.allclose(
            noisy_input[:, :3], denoised_output[:, :, ::2, ::2], atol=0.1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
