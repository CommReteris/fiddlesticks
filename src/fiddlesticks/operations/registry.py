"""
Comprehensive Operation Registry System.

Contains the complete operation registry with 75+ operations across 10 functional
categories, organized by functional intent rather than implementation details.

The registry follows the singleton pattern to ensure consistent operation
access across the entire pipeline system.
"""

from typing import Dict, List, Any, Optional
from ..core.operation_spec import OperationSpec, ProcessingMode, InputOutputType
from ..core.pipeline_operation import PipelineOperation


class ComprehensiveOperationRegistry:
    """
    Comprehensive registry containing 75+ operations across 10 functional categories.
    
    Operations are grouped by functional intent (what they do) rather than 
    implementation details (how they do it). This enables the pipeline to be
    completely implementation-agnostic.
    
    Categories:
    - input_output_operations: Load/save data, format conversion, validation
    - raw_processing_operations: Raw sensor data processing
    - color_processing_operations: Color space and color management
    - tone_mapping_operations: Tone mapping and exposure adjustment
    - enhancement_operations: Image enhancement and correction
    - denoising_operations: Noise reduction algorithms
    - burst_processing_operations: Multi-image processing
    - geometric_operations: Geometric transformations
    - quality_assessment_operations: Quality analysis and validation
    - creative_operations: Creative effects and artistic processing
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - ensure only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the comprehensive operation registry."""
        if self._initialized:
            return
        
        # Import operation wrappers - will be implemented next
        from .wrappers import (
            # Input/Output operations
            LoadRawFileWrapper, LoadImageWrapper, LoadMetadataWrapper, LoadBurstWrapper,
            LoadVideoFramesWrapper, LoadFromStreamWrapper, LoadNumpyWrapper, LoadJsonWrapper,
            LoadCsvWrapper, SaveRawWrapper, SaveImageWrapper, SaveMetadataWrapper, 
            ExportBurstWrapper, SaveVideoWrapper, WriteToStreamWrapper, SaveNumpyWrapper,
            SaveJsonWrapper, SaveCsvWrapper, RawToTensorWrapper, TensorToRawWrapper, 
            RGBFormatsWrapper, MetadataToJsonWrapper, NumpyToTorchWrapper, TorchToNumpyWrapper, 
            ValidateInputWrapper, CheckFormatWrapper, VerifyMetadataWrapper,
            
            # Raw processing operations
            RawPrepareWrapper, HotPixelWrapper, TemperatureWrapper, RawDenoiseWrapper, DemosaicWrapper,
            
            # Color processing operations
            ColorInWrapper, ColorOutWrapper, ChannelMixerWrapper, ColorBalanceWrapper, PrimariesWrapper,
            
            # Tone mapping operations
            ExposureWrapper, FilmicRGBWrapper, SigmoidWrapper, ToneEqualWrapper, HighlightsWrapper,
            
            # Enhancement operations
            SharpenWrapper, DiffuseWrapper, BlurWrapper, DefringeWrapper, AshiftWrapper,
            
            # Denoising operations
            UTNet2Wrapper, BM3DWrapper, BilateralWrapper, NLMeansWrapper, DenoiseProfileWrapper,
            
            # Burst processing operations
            HDRMergeWrapper, FocusStackWrapper, PanoramaWrapper, TemporalDenoiseWrapper, SuperResWrapper,
            
            # Geometric operations
            CropWrapper, FlipWrapper, RotatePixelsWrapper, ScalePixelsWrapper, LiquifyWrapper,
            
            # Quality assessment operations
            OverexposedWrapper, RawOverexposedWrapper, NoiseEstimationWrapper, BlurDetectionWrapper, ExposureAnalysisWrapper,
            
            # Creative operations
            GrainWrapper, BordersWrapper, WatermarkWrapper, VignetteWrapper, BloomWrapper,
            SepiaWrapper, VintageWrapper, CrossProcessWrapper
        )
        
        # Input/Output Operations (28 operations)
        self.input_output_operations = {
            # Input operations
            'load_raw_file': LoadRawFileWrapper(),
            'load_image': LoadImageWrapper(),
            'load_metadata': LoadMetadataWrapper(),
            'load_burst': LoadBurstWrapper(),
            'load_video_frames': LoadVideoFramesWrapper(),
            'load_from_stream': LoadFromStreamWrapper(),
            'load_numpy': LoadNumpyWrapper(),
            'load_json': LoadJsonWrapper(),
            'load_csv': LoadCsvWrapper(),
            
            # Output operations  
            'save_raw': SaveRawWrapper(),
            'save_image': SaveImageWrapper(),
            'save_metadata': SaveMetadataWrapper(),
            'export_burst': ExportBurstWrapper(),
            'save_video': SaveVideoWrapper(),
            'write_to_stream': WriteToStreamWrapper(),
            'save_numpy': SaveNumpyWrapper(),
            'save_json': SaveJsonWrapper(),
            'save_csv': SaveCsvWrapper(),
            
            # Format conversion operations
            'raw_to_tensor': RawToTensorWrapper(),
            'tensor_to_raw': TensorToRawWrapper(),
            'rgb_to_formats': RGBFormatsWrapper(),
            'metadata_to_json': MetadataToJsonWrapper(),
            'numpy_to_torch': NumpyToTorchWrapper(),
            'torch_to_numpy': TorchToNumpyWrapper(),
            
            # Validation operations
            'validate_input': ValidateInputWrapper(),
            'check_format': CheckFormatWrapper(),
            'verify_metadata': VerifyMetadataWrapper(),
        }
        
        # Raw Processing Operations (5 operations)
        self.raw_processing_operations = {
            'rawprepare': RawPrepareWrapper(),
            'hotpixels': HotPixelWrapper(),
            'temperature': TemperatureWrapper(),
            'rawdenoise': RawDenoiseWrapper(),
            'demosaic': DemosaicWrapper(),
        }
        
        # Color Processing Operations (5 operations)
        self.color_processing_operations = {
            'colorin': ColorInWrapper(),
            'colorout': ColorOutWrapper(),
            'channelmixerrgb': ChannelMixerWrapper(),
            'colorbalancergb': ColorBalanceWrapper(),
            'primaries': PrimariesWrapper(),
        }
        
        # Tone Mapping Operations (5 operations)
        self.tone_mapping_operations = {
            'exposure': ExposureWrapper(),
            'filmicrgb': FilmicRGBWrapper(),
            'sigmoid': SigmoidWrapper(),
            'toneequal': ToneEqualWrapper(),
            'highlights': HighlightsWrapper(),
        }
        
        # Enhancement Operations (5 operations)
        self.enhancement_operations = {
            'sharpen': SharpenWrapper(),
            'diffuse': DiffuseWrapper(),
            'blurs': BlurWrapper(),
            'defringe': DefringeWrapper(),
            'ashift': AshiftWrapper(),
        }
        
        # Denoising Operations (5 operations)
        self.denoising_operations = {
            'utnet2': UTNet2Wrapper(),
            'bm3d': BM3DWrapper(),
            'bilateral': BilateralWrapper(),
            'nlmeans': NLMeansWrapper(),
            'denoiseprofile': DenoiseProfileWrapper(),
        }
        
        # Burst Processing Operations (5 operations)
        self.burst_processing_operations = {
            'hdr_merge': HDRMergeWrapper(),
            'focus_stack': FocusStackWrapper(),
            'panorama_stitch': PanoramaWrapper(),
            'temporal_denoise': TemporalDenoiseWrapper(),
            'super_resolution': SuperResWrapper(),
        }
        
        # Geometric Operations (5 operations)
        self.geometric_operations = {
            'crop': CropWrapper(),
            'flip': FlipWrapper(),
            'rotatepixels': RotatePixelsWrapper(),
            'scalepixels': ScalePixelsWrapper(),
            'liquify': LiquifyWrapper(),
        }
        
        # Quality Assessment Operations (5 operations)
        self.quality_assessment_operations = {
            'overexposed': OverexposedWrapper(),
            'rawoverexposed': RawOverexposedWrapper(),
            'noise_estimation': NoiseEstimationWrapper(),
            'blur_detection': BlurDetectionWrapper(),
            'exposure_analysis': ExposureAnalysisWrapper(),
        }
        
        # Creative Operations (8 operations)
        self.creative_operations = {
            'grain': GrainWrapper(),
            'borders': BordersWrapper(),
            'watermark': WatermarkWrapper(),
            'vignette': VignetteWrapper(),
            'bloom': BloomWrapper(),
            'sepia': SepiaWrapper(),
            'vintage': VintageWrapper(),
            'cross_process': CrossProcessWrapper(),
        }
        
        self._initialized = True
    
    def get_operation(self, category: str, operation_name: str) -> PipelineOperation:
        """
        Retrieve an operation from the registry.
        
        Args:
            category: The functional category name
            operation_name: The specific operation name
            
        Returns:
            The operation instance
            
        Raises:
            ValueError: If category or operation doesn't exist
        """
        if not hasattr(self, category):
            raise ValueError(f"Unknown category: {category}")
        
        category_dict = getattr(self, category)
        if operation_name not in category_dict:
            raise ValueError(f"Unknown operation {operation_name} in category {category}")
        
        return category_dict[operation_name]
    
    def list_categories(self) -> List[str]:
        """
        List all available operation categories.
        
        Returns:
            List of category names
        """
        return [
            'input_output_operations',
            'raw_processing_operations', 
            'color_processing_operations',
            'tone_mapping_operations',
            'enhancement_operations',
            'denoising_operations',
            'burst_processing_operations',
            'geometric_operations',
            'quality_assessment_operations',
            'creative_operations'
        ]
    
    def validate_operation_exists(self, category: str, operation_name: str) -> bool:
        """
        Validate that an operation exists in the registry.
        
        Args:
            category: The functional category name
            operation_name: The specific operation name
            
        Returns:
            True if operation exists, False otherwise
        """
        try:
            self.get_operation(category, operation_name)
            return True
        except ValueError:
            return False
    
    def get_total_operation_count(self) -> int:
        """
        Get the total number of operations across all categories.
        
        Returns:
            Total operation count
        """
        total = 0
        for category_name in self.list_categories():
            category_dict = getattr(self, category_name)
            total += len(category_dict)
        return total