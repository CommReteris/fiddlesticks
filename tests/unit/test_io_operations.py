"""
Test suite for I/O Operations in Function-Based Composable Pipeline Architecture.

This module tests the Input/Output operations that form the critical pipeline boundaries:
- File loading operations (load_raw, load_rgb, load_metadata)
- File saving operations (save_raw, save_rgb, save_metadata)
- Stream operations (stream_input, stream_output)
- Format conversion operations (raw_to_rgb, rgb_to_lab, etc.)

Following TDD approach - these tests define the expected interface before implementation.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
from fiddlesticks.core.operation_spec import (
    OperationSpec,
    ProcessingMode,
    InputOutputType,
)
# Test imports - will fail until we implement the I/O operations
from fiddlesticks.io.file_operations import (
    LoadRawOperation,
    LoadRgbOperation,
    LoadMetadataOperation,
    SaveRawOperation,
    SaveRgbOperation,
    SaveMetadataOperation,
)
from fiddlesticks.io.format_conversion import (
    RawToRgbOperation,
    RgbToLabOperation,
    LabToRgbOperation,
    RgbToGrayscaleOperation,
)
from fiddlesticks.io.stream_operations import (
    StreamInputOperation,
    StreamOutputOperation,
)


class TestFileLoadingOperations:
    """Test file loading operations for various data types."""

    @pytest.fixture
    def sample_raw_tensor(self):
        """Create sample 4-channel raw tensor."""
        return torch.rand(1, 4, 64, 64)

    @pytest.fixture
    def sample_rgb_tensor(self):
        """Create sample RGB tensor."""
        return torch.rand(1, 3, 64, 64)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata dictionary."""
        return {
            "iso": 100,
            "exposure_time": 1 / 60,
            "focal_length": 50.0,
            "aperture": 2.8,
            "white_balance": [1.0, 1.0, 1.0],
        }

    def test_load_raw_operation_spec(self):
        """Test LoadRawOperation has correct specification."""
        operation = LoadRawOperation()
        spec = operation.spec

        assert spec.name == "load_raw"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert (
            InputOutputType.NUMPY_ARRAY in spec.input_types
        )  # Changed to trigger input
        assert (
            InputOutputType.RAW_BAYER in spec.output_types
            or InputOutputType.RAW_4CH in spec.output_types
        )
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_load_raw_operation_execution(self, sample_raw_tensor):
        """Test LoadRawOperation can load raw file and return tensor."""
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Create operation with file path
            operation = LoadRawOperation(file_path=temp_path)

            # Mock file loading to return our sample tensor
            with patch("torch.load", return_value=sample_raw_tensor):
                # Use dummy trigger tensor
                trigger = torch.tensor([1.0])
                result, metadata = operation(trigger)

                assert result.shape == sample_raw_tensor.shape
                assert result.dtype == torch.float32
                assert isinstance(metadata, dict)
                assert "file_path" in metadata

        finally:
            os.unlink(temp_path)

    def test_load_rgb_operation_spec(self):
        """Test LoadRgbOperation has correct specification."""
        operation = LoadRgbOperation()
        spec = operation.spec

        assert spec.name == "load_rgb"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert (
            InputOutputType.NUMPY_ARRAY in spec.input_types
        )  # Changed to trigger input
        assert InputOutputType.RGB in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_load_rgb_operation_execution(self, sample_rgb_tensor):
        """Test LoadRgbOperation can load RGB image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Create operation with file path
            operation = LoadRgbOperation(file_path=temp_path)

            # Mock image loading
            with patch(
                "torchvision.io.read_image", return_value=sample_rgb_tensor.squeeze(0)
            ):
                # Use dummy trigger tensor
                trigger = torch.tensor([1.0])
                result, metadata = operation(trigger)

                assert result.shape[0] == 1  # batch dimension
                assert result.shape[1] == 3  # RGB channels
                assert isinstance(metadata, dict)
                assert "file_path" in metadata

        finally:
            os.unlink(temp_path)

    def test_load_metadata_operation_spec(self):
        """Test LoadMetadataOperation has correct specification."""
        operation = LoadMetadataOperation()
        spec = operation.spec

        assert spec.name == "load_metadata"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert (
            InputOutputType.NUMPY_ARRAY in spec.input_types
        )  # Changed to trigger input
        assert InputOutputType.METADATA in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_load_metadata_operation_execution(self, sample_metadata):
        """Test LoadMetadataOperation can load metadata from file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Create operation with file path
            operation = LoadMetadataOperation(file_path=temp_path)

            # Mock JSON loading
            with patch("json.load", return_value=sample_metadata):
                # Use dummy trigger tensor
                trigger = torch.tensor([1.0])
                result, metadata = operation(trigger)

                # Result should be metadata tensor representation
                assert isinstance(result, torch.Tensor)
                assert "loaded_metadata" in metadata
                assert metadata["loaded_metadata"] == sample_metadata

        finally:
            os.unlink(temp_path)


class TestFileSavingOperations:
    """Test file saving operations for various data types."""

    @pytest.fixture
    def sample_raw_tensor(self):
        """Create sample 4-channel raw tensor."""
        return torch.rand(1, 4, 64, 64)

    @pytest.fixture
    def sample_rgb_tensor(self):
        """Create sample RGB tensor."""
        return torch.rand(1, 3, 64, 64)

    def test_save_raw_operation_spec(self):
        """Test SaveRawOperation has correct specification."""
        operation = SaveRawOperation()
        spec = operation.spec

        assert spec.name == "save_raw"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert (
            InputOutputType.RAW_BAYER in spec.input_types
            or InputOutputType.RAW_4CH in spec.input_types
        )
        assert InputOutputType.FILE_PATH in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_save_raw_operation_execution(self, sample_raw_tensor):
        """Test SaveRawOperation can save raw tensor to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            operation = SaveRawOperation(
                output_path=temp_dir, filename_template="test_raw_{timestamp}_{uuid}.pt"
            )

            result, metadata = operation(sample_raw_tensor)

            # Should return success indicator tensor
            assert isinstance(result, torch.Tensor)
            assert result.item() == 1.0  # Success indicator
            assert "output_path" in metadata

            # Verify file was created
            saved_path = metadata["output_path"]
            assert os.path.exists(saved_path)

    def test_save_rgb_operation_spec(self):
        """Test SaveRgbOperation has correct specification."""
        operation = SaveRgbOperation()
        spec = operation.spec

        assert spec.name == "save_rgb"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert InputOutputType.RGB in spec.input_types
        assert InputOutputType.FILE_PATH in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_save_rgb_operation_execution(self, sample_rgb_tensor):
        """Test SaveRgbOperation can save RGB tensor as image file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            operation = SaveRgbOperation(output_path=temp_dir, format="png")

            result, metadata = operation(sample_rgb_tensor)

            # Should return success indicator tensor
            assert isinstance(result, torch.Tensor)
            assert result.item() == 1.0  # Success indicator
            assert "output_path" in metadata

            # Verify file was created
            saved_path = metadata["output_path"]
            assert os.path.exists(saved_path)
            assert saved_path.endswith(".png")


class TestStreamOperations:
    """Test stream operations for real-time processing."""

    def test_stream_input_operation_spec(self):
        """Test StreamInputOperation has correct specification."""
        operation = StreamInputOperation()
        spec = operation.spec

        assert spec.name == "stream_input"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert InputOutputType.STREAM in spec.input_types
        assert (
            InputOutputType.RGB in spec.output_types
            or InputOutputType.RAW_4CH in spec.output_types
        )
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_stream_input_operation_execution(self):
        """Test StreamInputOperation can read from stream."""
        mock_stream = Mock()
        mock_frame = torch.rand(3, 64, 64)  # RGB frame
        mock_stream.read.return_value = (True, mock_frame)

        operation = StreamInputOperation(stream_source=mock_stream)

        # Stream input takes empty tensor as trigger
        trigger = torch.empty(1)
        result, metadata = operation(trigger)

        assert result.shape == torch.Size([1, 3, 64, 64])  # Added batch dim
        assert "stream_info" in metadata
        assert metadata["stream_info"]["success"] is True

    def test_stream_output_operation_spec(self):
        """Test StreamOutputOperation has correct specification."""
        operation = StreamOutputOperation()
        spec = operation.spec

        assert spec.name == "stream_output"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert InputOutputType.RGB in spec.input_types
        assert InputOutputType.STREAM in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1


class TestFormatConversionOperations:
    """Test format conversion operations between different data types."""

    @pytest.fixture
    def sample_raw_tensor(self):
        """Create sample 4-channel raw tensor."""
        return torch.rand(1, 4, 64, 64)

    @pytest.fixture
    def sample_rgb_tensor(self):
        """Create sample RGB tensor."""
        return torch.rand(1, 3, 64, 64)

    def test_raw_to_rgb_operation_spec(self):
        """Test RawToRgbOperation has correct specification."""
        operation = RawToRgbOperation()
        spec = operation.spec

        assert spec.name == "raw_to_rgb"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert (
            InputOutputType.RAW_4CH in spec.input_types
            or InputOutputType.RAW_BAYER in spec.input_types
        )
        assert InputOutputType.RGB in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_raw_to_rgb_operation_execution(self, sample_raw_tensor):
        """Test RawToRgbOperation converts raw to RGB."""
        operation = RawToRgbOperation()

        result, metadata = operation(sample_raw_tensor)

        assert result.shape[0] == 1  # batch dimension preserved
        assert result.shape[1] == 3  # converted to RGB
        assert result.dtype == torch.float32
        assert "conversion_info" in metadata
        assert metadata["conversion_info"]["input_format"] == "raw_4ch"
        assert metadata["conversion_info"]["output_format"] == "rgb"

    def test_rgb_to_lab_operation_spec(self):
        """Test RgbToLabOperation has correct specification."""
        operation = RgbToLabOperation()
        spec = operation.spec

        assert spec.name == "rgb_to_lab"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert InputOutputType.RGB in spec.input_types
        assert InputOutputType.LAB in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_rgb_to_lab_operation_execution(self, sample_rgb_tensor):
        """Test RgbToLabOperation converts RGB to LAB."""
        operation = RgbToLabOperation()

        result, metadata = operation(sample_rgb_tensor)

        assert result.shape == sample_rgb_tensor.shape  # Same dimensions
        assert result.dtype == torch.float32
        assert "conversion_info" in metadata
        assert metadata["conversion_info"]["input_format"] == "rgb"
        assert metadata["conversion_info"]["output_format"] == "lab"

    def test_rgb_to_grayscale_operation_spec(self):
        """Test RgbToGrayscaleOperation has correct specification."""
        operation = RgbToGrayscaleOperation()
        spec = operation.spec

        assert spec.name == "rgb_to_grayscale"
        assert ProcessingMode.SINGLE_IMAGE in spec.supported_modes
        assert InputOutputType.RGB in spec.input_types
        assert InputOutputType.GRAYSCALE in spec.output_types
        assert spec.input_count == (1, 1)
        assert spec.output_count == 1

    def test_rgb_to_grayscale_operation_execution(self, sample_rgb_tensor):
        """Test RgbToGrayscaleOperation converts RGB to grayscale."""
        operation = RgbToGrayscaleOperation()

        result, metadata = operation(sample_rgb_tensor)

        assert result.shape[0] == 1  # batch dimension preserved
        assert result.shape[1] == 1  # converted to single channel
        assert result.shape[2:] == sample_rgb_tensor.shape[2:]  # spatial dims preserved
        assert result.dtype == torch.float32
        assert "conversion_info" in metadata


class TestIOIntegration:
    """Test integration of I/O operations with pipeline system."""

    def test_io_operations_pipeline_compatibility(self):
        """Test that I/O operations have compatible specifications for pipeline use."""
        # Test that I/O operations have proper specs for pipeline compatibility
        load_op = LoadRgbOperation()
        convert_op = RgbToGrayscaleOperation()
        save_op = SaveRgbOperation()

        # Operations should have compatible input/output types
        assert load_op.spec.name == "load_rgb"
        assert convert_op.spec.name == "rgb_to_grayscale"
        assert save_op.spec.name == "save_rgb"

        # Verify basic compatibility: RGB output from load compatible with RGB input to convert
        assert InputOutputType.RGB in load_op.spec.output_types
        assert InputOutputType.RGB in convert_op.spec.input_types
        assert InputOutputType.GRAYSCALE in convert_op.spec.output_types

    def test_io_operations_metadata_flow(self):
        """Test that metadata flows properly through I/O operations."""
        # This test ensures I/O operations preserve and enhance metadata
        load_op = LoadMetadataOperation()

        # Mock metadata loading
        sample_metadata = {"iso": 100, "exposure_time": 1 / 60}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with patch("json.load", return_value=sample_metadata):
                # Create operation with file path instead of using tensor
                load_op = LoadMetadataOperation(file_path=temp_path)
                trigger = torch.tensor([1.0])
                initial_metadata = {"processing_stage": "loading"}

                result, output_metadata = load_op(trigger, initial_metadata)

                # Should preserve initial metadata and add loaded metadata
                assert "processing_stage" in output_metadata
                assert "loaded_metadata" in output_metadata
                assert output_metadata["loaded_metadata"] == sample_metadata

        finally:
            os.unlink(temp_path)
