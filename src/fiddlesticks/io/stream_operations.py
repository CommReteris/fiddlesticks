"""
Stream I/O Operations for Function-Based Composable Pipeline Architecture.

This module implements streaming operations for real-time data processing:
- StreamInputOperation: Read data from streams (cameras, network, etc.)
- StreamOutputOperation: Write data to streams (displays, network, etc.)

These operations enable real-time pipeline processing with live data sources
and destinations, maintaining the universal PipelineOperation interface.
"""

import queue
import threading
import time
from typing import Dict, Any, List, Tuple

import torch

from ..core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
from ..core.pipeline_operation import PipelineOperation


class StreamInputOperation(PipelineOperation):
    """
    Read data from input streams.

    This operation reads data from various stream sources including:
    - Camera feeds (via OpenCV or similar)
    - Network streams (RTMP, WebRTC, etc.)
    - File streams (video files)
    - Custom stream sources

    Input: Stream trigger tensor (empty tensor to trigger read)
    Output: Data tensor read from stream (RGB or RAW)
    """

    def __init__(
        self,
        stream_source=None,
        output_format: str = "rgb",
        buffer_size: int = 10,
        timeout: float = 1.0,
        **kwargs,
    ):
        """Initialize StreamInputOperation with specification."""

        # Determine output type based on format
        output_type = (
            InputOutputType.RGB if output_format == "rgb" else InputOutputType.RAW_4CH
        )

        spec = OperationSpec(
            name="stream_input",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[InputOutputType.STREAM],
            output_types=[output_type],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["stream_info", "read_time", "frame_number", "timestamp"],
            constraints={
                "real_time": True,
                "buffer_size": buffer_size,
                "timeout": timeout,
            },
            description="Read data from input streams",
        )
        super().__init__(spec)

        self.stream_source = stream_source
        self.output_format = output_format
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.frame_number = 0

        # Stream buffer for handling async reads
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.is_streaming = False
        self.stream_thread = None

    def start_streaming(self):
        """Start the streaming thread for continuous frame capture."""
        if self.stream_source is None:
            raise ValueError("No stream source provided")

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def stop_streaming(self):
        """Stop the streaming thread."""
        self.is_streaming = False
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=2.0)

    def _stream_loop(self):
        """Continuous streaming loop running in separate thread."""
        while self.is_streaming and self.stream_source is not None:
            try:
                # Read from stream source
                if hasattr(self.stream_source, "read"):
                    success, frame = self.stream_source.read()
                    if success:
                        timestamp = time.time()
                        frame_data = {
                            "frame": frame,
                            "timestamp": timestamp,
                            "frame_number": self.frame_number,
                        }

                        # Add to buffer (non-blocking)
                        try:
                            self.frame_buffer.put_nowait(frame_data)
                            self.frame_number += 1
                        except queue.Full:
                            # Buffer full, drop oldest frame
                            try:
                                self.frame_buffer.get_nowait()
                                self.frame_buffer.put_nowait(frame_data)
                            except queue.Empty:
                                pass
                else:
                    # Fallback for mock sources
                    if callable(self.stream_source):
                        frame = self.stream_source()
                        timestamp = time.time()
                        frame_data = {
                            "frame": frame,
                            "timestamp": timestamp,
                            "frame_number": self.frame_number,
                        }

                        try:
                            self.frame_buffer.put_nowait(frame_data)
                            self.frame_number += 1
                        except queue.Full:
                            try:
                                self.frame_buffer.get_nowait()
                                self.frame_buffer.put_nowait(frame_data)
                            except queue.Empty:
                                pass

                    time.sleep(0.033)  # ~30 FPS for mock sources

            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(0.1)

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Read data from stream.

        Args:
            data: List containing stream trigger tensor
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (stream data tensor list, updated metadata)
        """
        start_time = time.time()

        # Handle mock stream sources for testing
        if self.stream_source is not None:
            if hasattr(self.stream_source, "read"):
                # Direct stream read (like OpenCV VideoCapture)
                success, frame_data = self.stream_source.read()
                if not success:
                    raise RuntimeError("Failed to read from stream")

                # Convert frame_data to tensor
                if isinstance(frame_data, torch.Tensor):
                    frame_tensor = frame_data
                else:
                    # Assume numpy array from OpenCV
                    frame_tensor = torch.from_numpy(frame_data).float()

                frame_info = {
                    "success": True,
                    "timestamp": time.time(),
                    "frame_number": self.frame_number,
                }
                self.frame_number += 1

            elif hasattr(self.frame_buffer, "get"):
                # Use buffered streaming
                try:
                    frame_data = self.frame_buffer.get(timeout=self.timeout)
                    frame_tensor = frame_data["frame"]
                    frame_info = {
                        "success": True,
                        "timestamp": frame_data["timestamp"],
                        "frame_number": frame_data["frame_number"],
                    }
                except queue.Empty:
                    raise RuntimeError(f"Stream timeout after {self.timeout}s")
            else:
                # Mock source
                if isinstance(self.stream_source, torch.Tensor):
                    frame_tensor = self.stream_source
                else:
                    # Generate mock data based on output format
                    if self.output_format == "rgb":
                        frame_tensor = torch.rand(3, 64, 64)
                    else:
                        frame_tensor = torch.rand(4, 64, 64)

                frame_info = {
                    "success": True,
                    "timestamp": time.time(),
                    "frame_number": self.frame_number,
                }
                self.frame_number += 1
        else:
            # No source - generate mock data for testing
            if self.output_format == "rgb":
                frame_tensor = torch.rand(3, 64, 64)
            else:
                frame_tensor = torch.rand(4, 64, 64)

            frame_info = {
                "success": True,
                "timestamp": time.time(),
                "frame_number": self.frame_number,
            }
            self.frame_number += 1

        # Ensure batch dimension
        if frame_tensor.dim() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)

        # Validate output format
        expected_channels = 3 if self.output_format == "rgb" else 4
        if frame_tensor.shape[1] != expected_channels:
            # Try to convert if needed
            if self.output_format == "rgb" and frame_tensor.shape[1] == 4:
                frame_tensor = frame_tensor[:, :3, :, :]  # Drop alpha channel
            elif self.output_format == "rgb" and frame_tensor.shape[1] == 1:
                frame_tensor = frame_tensor.repeat(1, 3, 1, 1)  # Replicate grayscale
            else:
                raise ValueError(
                    f"Expected {expected_channels} channels for {self.output_format}, got {frame_tensor.shape[1]}"
                )

        # Update metadata
        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "stream_info": frame_info,
                "read_time": time.time() - start_time,
                "frame_number": frame_info["frame_number"],
                "timestamp": frame_info["timestamp"],
                "data_type": self.output_format,
            }
        )

        return [frame_tensor], output_metadata


class StreamOutputOperation(PipelineOperation):
    """
    Write data to output streams.

    This operation writes processed data to various stream destinations:
    - Display windows (via OpenCV or similar)
    - Network streams (RTMP push, WebRTC, etc.)
    - File streams (video recording)
    - Custom stream destinations

    Input: Data tensor to stream out (RGB primarily)
    Output: Stream status tensor
    """

    def __init__(
        self,
        stream_destination=None,
        input_format: str = "rgb",
        fps: float = 30.0,
        **kwargs,
    ):
        """Initialize StreamOutputOperation with specification."""

        # Determine input type based on format
        input_type = (
            InputOutputType.RGB if input_format == "rgb" else InputOutputType.RAW_4CH
        )

        spec = OperationSpec(
            name="stream_output",
            supported_modes=[ProcessingMode.SINGLE_IMAGE],
            input_types=[input_type],
            output_types=[InputOutputType.STREAM],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=["stream_info", "write_time", "frame_number"],
            constraints={"real_time": True, "fps": fps},
            description="Write data to output streams",
        )
        super().__init__(spec)

        self.stream_destination = stream_destination
        self.input_format = input_format
        self.fps = fps
        self.frame_number = 0
        self.last_frame_time = 0

    def process_tensors(
        self, data: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Write data to stream.

        Args:
            data: List containing data tensor to stream
            metadata: Input metadata dictionary
            **kwargs: Additional parameters

        Returns:
            Tuple of (stream status tensor list, updated metadata)
        """
        start_time = time.time()

        input_tensor = data[0]

        # Validate input format
        expected_channels = 3 if self.input_format == "rgb" else 4
        if input_tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels for {self.input_format}, got {input_tensor.shape[1]}"
            )

        # Frame rate control
        current_time = time.time()
        frame_interval = 1.0 / self.fps
        time_since_last_frame = current_time - self.last_frame_time

        if time_since_last_frame < frame_interval:
            sleep_time = frame_interval - time_since_last_frame
            time.sleep(sleep_time)

        self.last_frame_time = time.time()

        # Write to stream destination
        success = False
        error_message = None

        try:
            if self.stream_destination is not None:
                if hasattr(self.stream_destination, "write"):
                    # Direct stream write (like OpenCV VideoWriter)
                    # Remove batch dimension and convert to appropriate format
                    output_tensor = input_tensor.squeeze(0)

                    # Convert to numpy array if needed
                    if hasattr(self.stream_destination, "write"):
                        # Assume it expects numpy array
                        output_array = output_tensor.detach().cpu().numpy()
                        # Convert from [C, H, W] to [H, W, C] for OpenCV
                        output_array = output_array.transpose(1, 2, 0)
                        success = self.stream_destination.write(output_array)
                elif callable(self.stream_destination):
                    # Custom stream function
                    self.stream_destination(input_tensor)
                    success = True
                else:
                    # Mock destination - just indicate success
                    success = True
            else:
                # No destination - mock success for testing
                success = True

        except Exception as e:
            error_message = str(e)
            success = False

        # Create status tensor
        status_tensor = torch.tensor([1.0 if success else 0.0])

        # Update frame number
        if success:
            self.frame_number += 1

        # Update metadata
        stream_info = {
            "success": success,
            "frame_number": self.frame_number,
            "fps": self.fps,
            "timestamp": self.last_frame_time,
        }

        if error_message:
            stream_info["error"] = error_message

        output_metadata = metadata.copy()
        output_metadata.update(
            {
                "stream_info": stream_info,
                "write_time": time.time() - start_time,
                "frame_number": self.frame_number,
                "data_type": "stream",
            }
        )

        return [status_tensor], output_metadata

    def __del__(self):
        """Cleanup stream destination on deletion."""
        if self.stream_destination is not None and hasattr(
            self.stream_destination, "release"
        ):
            try:
                self.stream_destination.release()
            except:
                pass  # Ignore cleanup errors
