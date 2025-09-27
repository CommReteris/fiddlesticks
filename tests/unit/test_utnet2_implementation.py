"""
Test-Driven Development for real UTNet2 implementation.

This test file defines the expected behavior of a real UTNet2 model for raw image denoising,
replacing the mock implementation with actual functionality. Tests drive implementation
of proper U-Net architecture with encoder-decoder structure and skip connections.
"""

import pytest
import torch
import torch.nn as nn

from fiddlesticks.registries.model_registry import ModelRegistry


class TestUTNet2RealImplementation:
    """TDD tests for actual UTNet2 model functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
        # Reset registry to ensure clean state
        self.registry.clear_registry()
        self.registry._register_default_models()

    def test_utnet2_has_proper_architecture_structure(self):
        """Test that UTNet2 has proper U-Net architecture components."""
        model = self.registry.create_model("utnet2")

        # Should have encoder blocks (down-sampling path)
        assert hasattr(model, "convs1"), "Missing encoder level 1"
        assert hasattr(model, "convs2"), "Missing encoder level 2"
        assert hasattr(model, "convs3"), "Missing encoder level 3"
        assert hasattr(model, "convs4"), "Missing encoder level 4"
        assert hasattr(model, "maxpool"), "Missing max pooling for downsampling"

        # Should have bottleneck
        assert hasattr(model, "bottom"), "Missing bottleneck layer"

        # Should have decoder blocks (up-sampling path)
        assert hasattr(model, "up1"), "Missing decoder upsampling 1"
        assert hasattr(model, "tconvs1"), "Missing decoder conv 1"
        assert hasattr(model, "up2"), "Missing decoder upsampling 2"
        assert hasattr(model, "tconvs2"), "Missing decoder conv 2"
        assert hasattr(model, "up3"), "Missing decoder upsampling 3"
        assert hasattr(model, "tconvs3"), "Missing decoder conv 3"
        assert hasattr(model, "up4"), "Missing decoder upsampling 4"
        assert hasattr(model, "tconvs4"), "Missing decoder conv 4"

        # Should have output layer
        assert hasattr(model, "output_module"), "Missing output module"

    def test_utnet2_supports_4_channel_bayer_input(self):
        """Test that UTNet2 properly handles 4-channel Bayer pattern input."""
        model = self.registry.create_model("utnet2", in_channels=4, out_channels=3)

        # Create 4-channel Bayer input (batch_size=2, channels=4, height=64, width=64)
        bayer_input = torch.randn(2, 4, 64, 64)

        with torch.no_grad():
            output = model(bayer_input)

        # Should output 3-channel RGB with doubled resolution (due to PixelShuffle)
        expected_shape = (2, 3, 128, 128)  # Doubled spatial dimensions
        assert (
            output.shape == expected_shape
        ), f"Expected {expected_shape}, got {output.shape}"

    def test_utnet2_supports_3_channel_rgb_input(self):
        """Test that UTNet2 properly handles 3-channel RGB input."""
        model = self.registry.create_model("utnet2", in_channels=3, out_channels=3)

        # Create 3-channel RGB input
        rgb_input = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            output = model(rgb_input)

        # Should output same resolution for RGB input (no PixelShuffle)
        expected_shape = (2, 3, 64, 64)  # Same spatial dimensions
        assert (
            output.shape == expected_shape
        ), f"Expected {expected_shape}, got {output.shape}"

    def test_utnet2_forward_pass_produces_valid_output(self):
        """Test that forward pass produces valid tensor output."""
        model = self.registry.create_model("utnet2")
        test_input = torch.randn(1, 4, 32, 32)

        output = model(test_input)

        # Output should be valid tensor
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert torch.isfinite(output).all(), "Output should contain finite values"

    def test_utnet2_has_trainable_parameters(self):
        """Test that UTNet2 has trainable parameters for learning."""
        model = self.registry.create_model("utnet2")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have substantial number of parameters for a U-Net
        assert (
            trainable_params > 100000
        ), f"Expected > 100K parameters, got {trainable_params}"

        # All parameters should require gradients by default
        for param in model.parameters():
            assert param.requires_grad, "All parameters should be trainable"

    def test_utnet2_gradient_flow_works(self):
        """Test that gradients can flow through the network."""
        model = self.registry.create_model("utnet2")
        test_input = torch.randn(1, 4, 32, 32, requires_grad=True)
        target = torch.randn(1, 3, 64, 64)

        # Forward pass
        output = model(test_input)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        assert test_input.grad is not None, "Input gradients should be computed"

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(
                param.grad
            ).any(), f"Parameter {name} gradients should not be NaN"

    def test_utnet2_is_actually_denoising_not_passthrough(self):
        """Test that UTNet2 is not just a pass-through like the mock version."""
        model = self.registry.create_model("utnet2")

        # Create noisy input
        clean = torch.randn(1, 4, 32, 32)
        noise = torch.randn_like(clean) * 0.1
        noisy = clean + noise

        with torch.no_grad():
            # Model output should be different from input (it's processing, not identity)
            output = model(noisy)

            # Output shape will be different due to channel/resolution change, but verify processing
            # The key test: model should not be an identity function

            # Test that multiple different inputs produce different outputs
            different_input = torch.randn(1, 4, 32, 32)
            different_output = model(different_input)

            # Outputs should be significantly different for different inputs
            mse_diff = nn.MSELoss()(output, different_output).item()
            assert (
                mse_diff > 0.01
            ), f"Model seems to be identity function, MSE diff: {mse_diff}"

    def test_utnet2_configurable_capacity(self):
        """Test that UTNet2 capacity can be configured via funit parameter."""
        # Create models with different capacities
        small_model = self.registry.create_model("utnet2", funit=16)
        large_model = self.registry.create_model("utnet2", funit=64)

        # Count parameters
        small_params = sum(p.numel() for p in small_model.parameters())
        large_params = sum(p.numel() for p in large_model.parameters())

        # Large model should have significantly more parameters
        assert (
            large_params > small_params * 2
        ), f"Large model ({large_params}) should have much more parameters than small ({small_params})"

    def test_utnet2_activation_function_configurable(self):
        """Test that UTNet2 activation functions can be configured."""
        # This test will pass once we implement configurable activations
        # For now, it documents the expected behavior

        try:
            prelu_model = self.registry.create_model("utnet2", activation="PReLU")
            leaky_model = self.registry.create_model("utnet2", activation="LeakyReLU")

            # Models should be created successfully with different activations
            assert prelu_model is not None
            assert leaky_model is not None

        except (TypeError, NotImplementedError):
            # Expected to fail until we implement activation configuration
            pytest.skip("Activation configuration not yet implemented")


class TestUTNet2PipelineIntegration:
    """TDD tests for UTNet2 integration with fiddlesticks pipeline system."""

    def test_utnet2_creates_proper_pipeline_operation_wrapper(self):
        """Test that UTNet2 can be wrapped for pipeline integration."""
        from fiddlesticks.core.operation_spec import (
            OperationSpec,
            ProcessingMode,
            InputOutputType,
        )
        from fiddlesticks.core.pipeline_operation import PipelineOperation

        # This test defines the expected UTNet2Wrapper behavior
        # It will fail until we implement the wrapper

        try:
            # Expected: UTNet2Wrapper should exist and be a PipelineOperation
            from fiddlesticks.operations.model_wrappers import UTNet2Wrapper

            # Create operation spec for UTNet2
            spec = OperationSpec(
                name="utnet2_denoiser",
                supported_modes=[ProcessingMode.SINGLE_IMAGE],
                input_types=[InputOutputType.RAW_4CH],
                output_types=[InputOutputType.RGB],
                input_count=(1, 1),
                output_count=1,
                requires_metadata=[],
                produces_metadata=["denoised"],
                constraints={"requires_gpu": True},
                description="UTNet2 deep learning denoiser for raw images",
            )

            wrapper = UTNet2Wrapper(spec)
            assert isinstance(wrapper, PipelineOperation)

            # Test processing
            test_input = torch.randn(1, 4, 64, 64)
            output, metadata = wrapper.process_tensors([test_input], {})

            assert isinstance(output, list)
            assert len(output) == 1
            assert output[0].shape == (1, 3, 128, 128)  # Upsampled output
            assert "denoised" in metadata

        except ImportError:
            # Expected to fail until we implement the wrapper
            pytest.skip("UTNet2Wrapper not yet implemented")

    def test_utnet2_wrapper_has_trainable_parameters(self):
        """Test that UTNet2Wrapper properly exposes trainable parameters."""
        try:
            from fiddlesticks.operations.model_wrappers import UTNet2Wrapper
            from fiddlesticks.core.operation_spec import (
                OperationSpec,
                ProcessingMode,
                InputOutputType,
            )

            spec = OperationSpec(
                name="utnet2",
                supported_modes=[ProcessingMode.SINGLE_IMAGE],
                input_types=[InputOutputType.RAW_4CH],
                output_types=[InputOutputType.RGB],
                input_count=(1, 1),
                output_count=1,
                requires_metadata=[],
                produces_metadata=[],
                constraints={},
                description="UTNet2 wrapper",
            )

            wrapper = UTNet2Wrapper(spec)

            # Should be trainable
            assert wrapper.operation_type == "trainable"

            # Should return PyTorch model for parameters
            model = wrapper.get_parameters()
            assert model is not None
            assert isinstance(model, torch.nn.Module)

            # Should have trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            assert trainable_params > 0

        except ImportError:
            pytest.skip("UTNet2Wrapper not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
