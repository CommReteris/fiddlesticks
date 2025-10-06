"""
Kornia operation wrappers for GPU-accelerated computer vision operations.

This module provides Kornia operation wrappers across 9 categories:
- kornia_filter_operations: Bilateral, Gaussian, Sobel, Laplacian, etc.
- kornia_color_operations: RGB/HSV/LAB conversions, sepia effects
- kornia_enhance_operations: Brightness, contrast, gamma adjustments
- kornia_geometry_operations: Rotation, scaling, warping, elastic transforms
- kornia_camera_operations: Camera projections and 3D operations
- kornia_feature_operations: Harris, corner detection, local features
- kornia_augmentation_operations: Geometric and photometric augmentations
- kornia_loss_operations: SSIM, MS-SSIM, LPIPS losses
- kornia_metrics_operations: PSNR, SSIM, accuracy measurements

Key Features:
- Universal PipelineOperation interface compliance
- GPU-accelerated processing with automatic device handling
- Comprehensive error handling and validation
- Complete metadata propagation
- Support for both single tensors and tensor lists
"""

from typing import Dict, List, Any, Tuple, Optional

import torch

try:
    import kornia
    import kornia.filters as KF
    import kornia.color as KC
    import kornia.enhance as KE
    import kornia.geometry as KG
    import kornia.augmentation as KA
    import kornia.losses as KL
    import kornia.metrics as KM
    import kornia.feature as KFeat
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    # TODO: need warning at least
    # Create mock modules for testing without Kornia
    class MockKornia:
        def __getattr__(self, name):
            return lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))

    class MockKorniaColor:
        def rgb_to_grayscale(self, x, *args, **kwargs):
            # Convert RGB (B, 3, H, W) to grayscale (B, 1, H, W)
            if len(x.shape) == 4 and x.shape[1] == 3:
                return torch.mean(x, dim=1, keepdim=True)
            return x

        def __getattr__(self, name):
            return lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))

    class MockKorniaGeometry:
        def __init__(self):
            # Create nested camera and depth modules
            self.camera = MockKornia()
            self.camera.project_points = lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))
            self.camera.unproject_points = lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))

            self.depth = MockKornia()
            self.depth.depth_to_3d = lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))

        def __getattr__(self, name):
            return lambda *args, **kwargs: torch.zeros_like(args[0] if args else torch.zeros(1))

    kornia = MockKornia()
    KF = KE = KA = KL = KM = KFeat = MockKornia()
    KC = MockKorniaColor()
    KG = MockKorniaGeometry()

from ..core.pipeline_operation import PipelineOperation
from ..core.operation_spec import OperationSpec, ProcessingMode, InputOutputType


class KorniaOperationWrapper(PipelineOperation):
    """
    Base wrapper class for Kornia operations.
    
    Provides universal interface for all Kornia operations while maintaining
    GPU acceleration and proper tensor handling. All Kornia operations inherit
    from this base class to ensure consistency.
    
    Key features:
    - Automatic device handling (CPU/GPU)
    - Tensor format validation and conversion
    - Error handling for missing Kornia dependency
    - Metadata propagation with operation tracking
    """
    
    def __init__(self, kornia_function: callable, spec: OperationSpec):
        """
        Initialize KorniaOperationWrapper with Kornia function and spec.
        
        Args:
            kornia_function: The Kornia function to wrap
            spec: OperationSpec defining operation characteristics
        """
        super().__init__(spec)
        self.kornia_function = kornia_function
        
        if not KORNIA_AVAILABLE:
            import warnings
            warnings.warn(f"Kornia not available. Operation '{spec.name}' will use mock implementation.")
    
    @property
    def operation_type(self) -> str:
        """Kornia operations are typically non-trainable."""
        return "non_trainable"
    
    def get_parameters(self) -> Optional[torch.nn.Module]:
        """Kornia operations typically don't have trainable parameters."""
        return None
    
    def process_tensors(
        self, 
        inputs: List[torch.Tensor], 
        metadata: Dict[str, Any], 
        **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process tensors using wrapped Kornia function.
        
        Args:
            inputs: List of input tensors
            metadata: Input metadata dictionary
            **kwargs: Operation-specific parameters
            
        Returns:
            Tuple of (output_tensors, updated_metadata)
        """
        if not inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least one input")
        
        # Get primary input tensor
        input_tensor = inputs[0]
        device = input_tensor.device
        
        try:
            # Apply Kornia function with parameters
            if len(inputs) == 1:
                output_tensor = self.kornia_function(input_tensor, **kwargs)
            else:
                # Some Kornia functions take multiple inputs
                output_tensor = self.kornia_function(*inputs, **kwargs)
            
            # Ensure output is on same device
            if hasattr(output_tensor, 'to'):
                output_tensor = output_tensor.to(device)
            
            # Update metadata
            output_metadata = metadata.copy()
            output_metadata.update({
                'operation_applied': self.spec.name,
                'kornia_function': self.kornia_function.__name__ if hasattr(self.kornia_function, '__name__') else str(self.kornia_function),
                'operation_parameters': kwargs,
                'input_shape': list(input_tensor.shape),
                'output_shape': list(output_tensor.shape) if hasattr(output_tensor, 'shape') else None,
                'device': str(device)
            })
            
            return [output_tensor], output_metadata
            
        except Exception as e:
            raise RuntimeError(f"Kornia operation {self.spec.name} failed: {str(e)}")


# =============================================================================
# KORNIA FILTER OPERATIONS
# =============================================================================

class KorniaBilateralFilterWrapper(KorniaOperationWrapper):
    """Wrapper for Kornia bilateral filter operation."""
    
    def __init__(self):
        spec = OperationSpec(
            name='bilateral_filter',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'operation_parameters'],
            constraints={'requires_kornia': True},
            description='Edge-preserving bilateral filtering'
        )
        super().__init__(KF.bilateral_blur, spec)


class KorniaGaussianBlur2DWrapper(KorniaOperationWrapper):
    """Wrapper for Kornia 2D Gaussian blur operation."""

    def __init__(self):
        spec = OperationSpec(
            name='gaussian_blur2d',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'operation_parameters'],
            constraints={'requires_kornia': True},
            description='2D Gaussian blur filtering'
        )
        super().__init__(KF.gaussian_blur2d, spec)

    def process_tensors(
        self, inputs: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process tensors using Kornia gaussian_blur2d with default parameters.

        Args:
            inputs: List of input tensors
            metadata: Input metadata dictionary
            **kwargs: Additional parameters (kernel_size, sigma)

        Returns:
            Tuple of (output_tensors, updated_metadata)
        """
        if not inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least one input")

        input_tensor = inputs[0]
        device = input_tensor.device

        # Set default parameters if not provided
        kernel_size = kwargs.get("kernel_size", (5, 5))  # Default 5x5 kernel
        sigma = kwargs.get("sigma", (1.0, 1.0))  # Default sigma of 1.0

        # Ensure kernel_size and sigma are tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))

        try:
            # Call kornia gaussian_blur2d with parameters
            output_tensor = self.kornia_function(input_tensor, kernel_size, sigma)

            # Ensure output is on same device
            if hasattr(output_tensor, "to"):
                output_tensor = output_tensor.to(device)

            # Update metadata
            output_metadata = metadata.copy()
            output_metadata.update(
                {
                    "operation_applied": self.spec.name,
                    "kernel_size": kernel_size,
                    "sigma": sigma,
                    "kornia_function": (
                        self.kornia_function.__name__
                        if hasattr(self.kornia_function, "__name__")
                        else str(self.kornia_function)
                    ),
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "device": str(device),
                }
            )

            return [output_tensor], output_metadata

        except Exception as e:
            raise RuntimeError(f"Kornia operation {self.spec.name} failed: {str(e)}")


# =============================================================================
# KORNIA COLOR OPERATIONS
# =============================================================================

class KorniaRGBToGrayscaleWrapper(KorniaOperationWrapper):
    """Wrapper for Kornia RGB to grayscale conversion."""
    
    def __init__(self):
        spec = OperationSpec(
            name='rgb_to_grayscale',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'color_conversion'],
            constraints={'requires_kornia': True},
            description='Convert RGB image to grayscale'
        )
        super().__init__(KC.rgb_to_grayscale, spec)


# =============================================================================
# KORNIA ENHANCEMENT OPERATIONS
# =============================================================================

class KorniaAdjustBrightnessWrapper(KorniaOperationWrapper):
    """Wrapper for Kornia brightness adjustment."""

    def __init__(self):
        spec = OperationSpec(
            name='adjust_brightness',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'brightness_factor'],
            constraints={'requires_kornia': True},
            description='Adjust image brightness'
        )
        super().__init__(KE.adjust_brightness, spec)

    def process_tensors(
        self, inputs: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process tensors using Kornia adjust_brightness with factor parameter.

        Args:
            inputs: List of input tensors
            metadata: Input metadata dictionary (may contain 'brightness_factor')
            **kwargs: Additional parameters (may contain 'factor')

        Returns:
            Tuple of (output_tensors, updated_metadata)
        """
        if not inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least one input")

        input_tensor = inputs[0]
        device = input_tensor.device

        # Extract brightness factor from metadata or kwargs, with default
        factor = metadata.get("brightness_factor") or kwargs.get("factor", 1.0)

        try:
            # Call kornia adjust_brightness with factor
            output_tensor = self.kornia_function(input_tensor, factor)

            # Ensure output is on same device
            if hasattr(output_tensor, "to"):
                output_tensor = output_tensor.to(device)

            # Update metadata
            output_metadata = metadata.copy()
            output_metadata.update(
                {
                    "operation_applied": self.spec.name,
                    "brightness_factor": factor,
                    "kornia_function": (
                        self.kornia_function.__name__
                        if hasattr(self.kornia_function, "__name__")
                        else str(self.kornia_function)
                    ),
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "device": str(device),
                }
            )

            return [output_tensor], output_metadata

        except Exception as e:
            raise RuntimeError(f"Kornia operation {self.spec.name} failed: {str(e)}")


# =============================================================================
# KORNIA GEOMETRY OPERATIONS
# =============================================================================

class KorniaRotateWrapper(KorniaOperationWrapper):
    """Wrapper for Kornia rotation operation."""

    def __init__(self):
        spec = OperationSpec(
            name='rotate',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['operation_applied', 'rotation_angle'],
            constraints={'requires_kornia': True},
            description='Rotate image by specified angle'
        )
        super().__init__(KG.rotate, spec)

    def process_tensors(
        self, inputs: List[torch.Tensor], metadata: Dict[str, Any], **kwargs
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Process tensors using Kornia rotate function with proper angle handling.

        Args:
            inputs: List of input tensors
            metadata: Input metadata dictionary (should contain 'angle')
            **kwargs: Additional parameters

        Returns:
            Tuple of (output_tensors, updated_metadata)
        """
        if not inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least one input")

        input_tensor = inputs[0]
        device = input_tensor.device

        # Extract angle from metadata or kwargs
        angle = metadata.get("angle") or kwargs.get("angle")
        if angle is None:
            raise ValueError(
                "Rotate operation requires 'angle' parameter in metadata or kwargs"
            )

        try:
            # Convert angle to tensor if it's not already
            if not isinstance(angle, torch.Tensor):
                angle_tensor = torch.tensor(angle, dtype=torch.float32, device=device)
            else:
                angle_tensor = angle.to(device)

            # Call kornia rotate with angle as tensor
            output_tensor = self.kornia_function(input_tensor, angle_tensor)

            # Ensure output is on same device
            if hasattr(output_tensor, "to"):
                output_tensor = output_tensor.to(device)

            # Update metadata
            output_metadata = metadata.copy()
            output_metadata.update(
                {
                    "operation_applied": self.spec.name,
                    "rotation_angle": angle,
                    "kornia_function": self.kornia_function.__name__,
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "device": str(device),
                }
            )

            return [output_tensor], output_metadata

        except Exception as e:
            raise RuntimeError(f"Kornia operation {self.spec.name} failed: {str(e)}")


# =============================================================================
# KORNIA OPERATIONS REGISTRY
# =============================================================================

def get_kornia_operations_registry() -> Dict[str, Dict[str, KorniaOperationWrapper]]:
    """
    Get complete registry of Kornia operation wrappers.
    
    Returns:
        Dictionary organized by categories containing 65+ Kornia operations
    """
    return {
        'kornia_filter_operations': {
            'bilateral_filter': KorniaBilateralFilterWrapper(),
            'gaussian_blur2d': KorniaGaussianBlur2DWrapper(),
            'sobel': _create_filter_wrapper('sobel', KF.sobel, 'Sobel edge detection'),
            'laplacian': _create_filter_wrapper('laplacian', KF.laplacian, 'Laplacian edge detection'),
            'box_blur': _create_filter_wrapper('box_blur', KF.box_blur, 'Box blur filtering'),
            'median_blur': _create_filter_wrapper('median_blur', KF.median_blur, 'Median blur filtering'),
            'motion_blur': _create_filter_wrapper('motion_blur', KF.motion_blur, 'Motion blur filtering'),
            'unsharp_mask': _create_filter_wrapper('unsharp_mask', KF.unsharp_mask, 'Unsharp mask filtering'),
            'canny': _create_filter_wrapper('canny', KF.canny, 'Canny edge detection'),
            'spatial_gradient': _create_filter_wrapper('spatial_gradient', KF.spatial_gradient, 'Spatial gradient computation'),
        },
        'kornia_color_operations': {
            'rgb_to_grayscale': KorniaRGBToGrayscaleWrapper(),
            'rgb_to_hsv': _create_color_wrapper('rgb_to_hsv', KC.rgb_to_hsv, 'RGB to HSV conversion'),
            'hsv_to_rgb': _create_color_wrapper('hsv_to_rgb', KC.hsv_to_rgb, 'HSV to RGB conversion'),
            'rgb_to_lab': _create_color_wrapper('rgb_to_lab', KC.rgb_to_lab, 'RGB to LAB conversion'),
            'lab_to_rgb': _create_color_wrapper('lab_to_rgb', KC.lab_to_rgb, 'LAB to RGB conversion'),
            'rgb_to_yuv': _create_color_wrapper('rgb_to_yuv', KC.rgb_to_yuv, 'RGB to YUV conversion'),
            'yuv_to_rgb': _create_color_wrapper('yuv_to_rgb', KC.yuv_to_rgb, 'YUV to RGB conversion'),
            'rgb_to_xyz': _create_color_wrapper('rgb_to_xyz', KC.rgb_to_xyz, 'RGB to XYZ conversion'),
            'xyz_to_rgb': _create_color_wrapper('xyz_to_rgb', KC.xyz_to_rgb, 'XYZ to RGB conversion'),
            'sepia': _create_color_wrapper('sepia', KC.sepia, 'Sepia color effect'),
        },
        'kornia_enhance_operations': {
            'adjust_brightness': KorniaAdjustBrightnessWrapper(),
            'adjust_contrast': _create_enhance_wrapper('adjust_contrast', KE.adjust_contrast, 'Adjust image contrast'),
            'adjust_gamma': _create_enhance_wrapper('adjust_gamma', KE.adjust_gamma, 'Adjust image gamma'),
            'adjust_hue': _create_enhance_wrapper('adjust_hue', KE.adjust_hue, 'Adjust image hue'),
            'adjust_saturation': _create_enhance_wrapper('adjust_saturation', KE.adjust_saturation, 'Adjust image saturation'),
            'normalize': _create_enhance_wrapper('normalize', KE.normalize, 'Normalize image'),
            'denormalize': _create_enhance_wrapper('denormalize', KE.denormalize, 'Denormalize image'),
            'equalize_hist': _create_enhance_wrapper('equalize_hist', KE.equalize_clahe, 'Histogram equalization'),
            'invert': _create_enhance_wrapper('invert', KE.invert, 'Invert image colors'),
            'posterize': _create_enhance_wrapper('posterize', KE.posterize, 'Posterize image'),
            'sharpness': _create_enhance_wrapper('sharpness', KE.sharpness, 'Adjust image sharpness'),
            'solarize': _create_enhance_wrapper('solarize', KE.solarize, 'Solarize image'),
        },
        'kornia_geometry_operations': {
            'rotate': KorniaRotateWrapper(),
            'translate': _create_geometry_wrapper('translate', KG.translate, 'Translate image'),
            'scale': _create_geometry_wrapper('scale', KG.scale, 'Scale image'),
            'shear': _create_geometry_wrapper('shear', KG.shear, 'Shear image'),
            'resize': _create_geometry_wrapper('resize', KG.resize, 'Resize image'),
            'crop_by_boxes': _create_geometry_wrapper('crop_by_boxes', KG.crop_by_boxes, 'Crop by bounding boxes'),
            'center_crop': _create_geometry_wrapper('center_crop', KG.center_crop, 'Center crop image'),
            'crop_and_resize': _create_geometry_wrapper('crop_and_resize', KG.crop_and_resize, 'Crop and resize'),
            'hflip': _create_geometry_wrapper('hflip', KG.hflip, 'Horizontal flip'),
            'vflip': _create_geometry_wrapper('vflip', KG.vflip, 'Vertical flip'),
            'warp_perspective': _create_geometry_wrapper('warp_perspective', KG.warp_perspective, 'Perspective warping'),
            'warp_affine': _create_geometry_wrapper('warp_affine', KG.warp_affine, 'Affine warping'),
            'elastic_transform2d': _create_geometry_wrapper('elastic_transform2d', KG.elastic_transform2d, 'Elastic transformation'),
            'thin_plate_spline': _create_geometry_wrapper('thin_plate_spline', KG.thin_plate_spline, 'Thin plate spline warping'),
        },
        'kornia_camera_operations': {
            'project_points': _create_camera_wrapper('project_points', KG.camera.project_points, '3D to 2D point projection'),
            'unproject_points': _create_camera_wrapper('unproject_points', KG.camera.unproject_points, '2D to 3D point unprojection'),
            'depth_to_3d': _create_camera_wrapper('depth_to_3d', KG.depth.depth_to_3d, 'Depth map to 3D points'),
        },
        'kornia_feature_operations': {
            'harris_response': _create_feature_wrapper('harris_response', KFeat.harris_response, 'Harris corner response'),
            'gftt_response': _create_feature_wrapper('gftt_response', KFeat.gftt_response, 'Good features to track response'),
            'hessian_response': _create_feature_wrapper('hessian_response', KFeat.hessian_response, 'Hessian response'),
            'dog_response': _create_feature_wrapper('dog_response', KFeat.dog_response, 'Difference of Gaussians response'),
        },
        'kornia_augmentation_operations': {
            'random_crop': _create_augmentation_wrapper('random_crop', KA.RandomCrop, 'Random crop'),
            'random_resized_crop': _create_augmentation_wrapper('random_resized_crop', KA.RandomResizedCrop, 'Random resized crop'),
            'center_crop_aug': _create_augmentation_wrapper('center_crop_aug', KA.CenterCrop, 'Center crop augmentation'),
            'random_rotation': _create_augmentation_wrapper('random_rotation', KA.RandomRotation, 'Random rotation'),
            'random_affine': _create_augmentation_wrapper('random_affine', KA.RandomAffine, 'Random affine transformation'),
            'random_perspective': _create_augmentation_wrapper('random_perspective', KA.RandomPerspective, 'Random perspective'),
            'random_elastic_transform': _create_augmentation_wrapper('random_elastic_transform', KA.RandomElasticTransform, 'Random elastic transform'),
            'random_thin_plate_spline': _create_augmentation_wrapper('random_thin_plate_spline', KA.RandomThinPlateSpline, 'Random thin plate spline'),
            'random_horizontal_flip': _create_augmentation_wrapper('random_horizontal_flip', KA.RandomHorizontalFlip, 'Random horizontal flip'),
            'random_vertical_flip': _create_augmentation_wrapper('random_vertical_flip', KA.RandomVerticalFlip, 'Random vertical flip'),
            'color_jitter': _create_augmentation_wrapper('color_jitter', KA.ColorJitter, 'Color jitter augmentation'),
            'random_brightness': _create_augmentation_wrapper('random_brightness', KA.RandomBrightness, 'Random brightness'),
            'random_contrast': _create_augmentation_wrapper('random_contrast', KA.RandomContrast, 'Random contrast'),
            'random_gamma': _create_augmentation_wrapper('random_gamma', KA.RandomGamma, 'Random gamma'),
            'random_hue': _create_augmentation_wrapper('random_hue', KA.RandomHue, 'Random hue'),
            'random_saturation': _create_augmentation_wrapper('random_saturation', KA.RandomSaturation, 'Random saturation'),
            'random_gaussian_noise': _create_augmentation_wrapper('random_gaussian_noise', KA.RandomGaussianNoise, 'Random Gaussian noise'),
            'random_gaussian_blur': _create_augmentation_wrapper('random_gaussian_blur', KA.RandomGaussianBlur, 'Random Gaussian blur'),
            'random_motion_blur': _create_augmentation_wrapper('random_motion_blur', KA.RandomMotionBlur, 'Random motion blur'),
            'random_solarize': _create_augmentation_wrapper('random_solarize', KA.RandomSolarize, 'Random solarize'),
            'random_posterize': _create_augmentation_wrapper('random_posterize', KA.RandomPosterize, 'Random posterize'),
            'random_erasing': _create_augmentation_wrapper('random_erasing', KA.RandomErasing, 'Random erasing'),
        },
        'kornia_loss_operations': {
            'ssim_loss': _create_loss_wrapper('ssim_loss', KL.SSIMLoss, 'SSIM loss function'),
            'ms_ssim_loss': _create_loss_wrapper('ms_ssim_loss', KL.MS_SSIMLoss, 'Multi-scale SSIM loss'),
            'psnr_loss': _create_loss_wrapper('psnr_loss', KL.PSNRLoss, 'PSNR loss function'),
            'total_variation': _create_loss_wrapper('total_variation', KL.TotalVariation, 'Total variation loss'),
            'focal_loss': _create_loss_wrapper('focal_loss', KL.FocalLoss, 'Focal loss function'),
            'dice_loss': _create_loss_wrapper('dice_loss', KL.DiceLoss, 'Dice loss function'),
            'tversky_loss': _create_loss_wrapper('tversky_loss', KL.TverskyLoss, 'Tversky loss function'),
            'lovasz_hinge_loss': _create_loss_wrapper('lovasz_hinge_loss', KL.LovaszHingeLoss, 'Lovasz hinge loss'),
            'lovasz_softmax_loss': _create_loss_wrapper('lovasz_softmax_loss', KL.LovaszSoftmaxLoss, 'Lovasz softmax loss'),
        },
        'kornia_metrics_operations': {
            'psnr': _create_metric_wrapper('psnr', KM.psnr, 'PSNR metric'),
            'ssim': _create_metric_wrapper('ssim', KM.ssim, 'SSIM metric'),
            'mean_iou': _create_metric_wrapper('mean_iou', KM.mean_iou, 'Mean IoU metric'),
            'accuracy': _create_metric_wrapper('accuracy', KM.accuracy, 'Accuracy metric'),
        }
    }


# =============================================================================
# HELPER FUNCTIONS FOR CREATING OPERATION WRAPPERS
# =============================================================================

def _create_filter_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a filter operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'operation_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_color_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a color operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.LAB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.RGB, InputOutputType.LAB, InputOutputType.GRAYSCALE],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'color_conversion'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_enhance_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create an enhancement operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'enhancement_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_geometry_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a geometry operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'geometric_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_camera_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a camera operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.METADATA],
        output_types=[InputOutputType.RGB, InputOutputType.METADATA],
        input_count=(1, 2),
        output_count=1,
        requires_metadata=['camera_intrinsics'],
        produces_metadata=['operation_applied', 'camera_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_feature_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a feature operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.GRAYSCALE, InputOutputType.METADATA],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'feature_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)


def _create_augmentation_wrapper(name: str, kornia_class: type, description: str) -> KorniaOperationWrapper:
    """Create an augmentation operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        input_count=(1, 1),
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'augmentation_parameters'],
        constraints={'requires_kornia': True},
        description=description
    )

    # Handle different augmentation initialization requirements
    if name == "random_crop":
        kornia_func = (
            kornia_class(size=(32, 32))  # Default crop size
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_resized_crop":
        kornia_func = (
            kornia_class(size=(32, 32))  # Default crop size
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "center_crop_aug":
        kornia_func = (
            kornia_class(size=(32, 32))  # Default crop size
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_rotation":
        kornia_func = (
            kornia_class(degrees=30.0)  # Default rotation range ±30 degrees
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_affine":
        kornia_func = (
            kornia_class(degrees=15.0)  # Default rotation range ±15 degrees
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_perspective":
        kornia_func = (
            kornia_class(distortion_scale=0.1)  # Default perspective distortion
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_gaussian_blur":
        kornia_func = (
            kornia_class(
                kernel_size=(3, 3), sigma=(0.1, 2.0)
            )  # Default kernel and sigma range
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "random_motion_blur":
        kornia_func = (
            kornia_class(
                kernel_size=3, angle=15.0, direction=0.5
            )  # Default motion blur parameters
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    else:
        kornia_func = (
            kornia_class() if hasattr(kornia_class, "__call__") else kornia_class
        )

    return KorniaOperationWrapper(kornia_func, spec)


def _create_loss_wrapper(name: str, kornia_class: type, description: str) -> KorniaOperationWrapper:
    """Create a loss operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.METADATA],  # Loss returns scalar
        input_count=(2, 2),  # Prediction and target
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'loss_value'],
        constraints={'requires_kornia': True, 'operation_type': 'loss'},
        description=description
    )

    # Handle different loss function initialization requirements
    if name == "ssim_loss":
        kornia_func = (
            kornia_class(window_size=11)
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "ms_ssim_loss":
        kornia_func = (
            kornia_class() if hasattr(kornia_class, "__call__") else kornia_class
        )
    elif name == "psnr_loss":
        kornia_func = (
            kornia_class(max_val=1.0)
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "focal_loss":
        kornia_func = (
            kornia_class(alpha=1.0, gamma=2.0)  # Default focal loss parameters
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    elif name == "tversky_loss":
        kornia_func = (
            kornia_class(alpha=0.5, beta=0.5)  # Default Tversky loss parameters
            if hasattr(kornia_class, "__call__")
            else kornia_class
        )
    else:
        kornia_func = (
            kornia_class() if hasattr(kornia_class, "__call__") else kornia_class
        )

    return KorniaOperationWrapper(kornia_func, spec)


def _create_metric_wrapper(name: str, kornia_func: callable, description: str) -> KorniaOperationWrapper:
    """Create a metric operation wrapper."""
    spec = OperationSpec(
        name=name,
        supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
        input_types=[InputOutputType.RGB, InputOutputType.GRAYSCALE],
        output_types=[InputOutputType.METADATA],  # Metric returns scalar
        input_count=(2, 2),  # Prediction and target
        output_count=1,
        requires_metadata=[],
        produces_metadata=['operation_applied', 'metric_value'],
        constraints={'requires_kornia': True, 'operation_type': 'metric'},
        description=description
    )
    return KorniaOperationWrapper(kornia_func, spec)
