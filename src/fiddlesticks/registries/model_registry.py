"""
ModelRegistry for ML Model Factory Pattern.

This module provides a unified model registry similar to the augmentations pipeline
approach, enabling configurable model instantiation for various ML models including:
- UTNet2, UTNet3 for denoising
- BM3D for classical denoising  
- Learned denoising networks
- Compression autoencoders (Balle encoder/decoder)
- Custom models registered at runtime

Key Features:
- Singleton pattern for global model registry
- Factory pattern for model creation with parameters
- Runtime model registration capabilities
- Parameter override support for flexible configuration
- Integration with existing Function-Based Composable Pipeline Architecture
"""

from typing import Dict, Any, Type, List, Optional

import torch


class ModelRegistry:
    """
    Singleton registry for ML model factory pattern.
    
    Provides unified model instantiation for various ML models used in the
    Function-Based Composable Pipeline Architecture. Follows the same elegant
    pattern as the augmentations registry.
    
    Key features:
    - Singleton pattern ensures single global registry
    - Factory method for model creation with parameters
    - Runtime registration of custom models
    - Parameter override support for flexible configuration
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize ModelRegistry with default models."""
        if not self._initialized:
            self._models = {}
            self._model_configs = {}
            self._register_default_models()
            ModelRegistry._initialized = True

    def _register_default_models(self):
        """Register default ML models based on memory patterns."""

        def get_activation_class_params(activation: str) -> tuple:
            """Get the PyTorch activation class and parameters for a given activation name."""
            if activation == "PReLU":
                return torch.nn.PReLU, {}
            elif activation == "ELU":
                return torch.nn.ELU, {"inplace": True}
            elif activation == "Hardswish":
                return torch.nn.Hardswish, {"inplace": True}
            elif activation == "LeakyReLU":
                return torch.nn.LeakyReLU, {"inplace": True, "negative_slope": 0.2}
            else:
                raise ValueError(f"Unknown activation function: {activation}")

        class Denoiser(torch.nn.Module):
            """Base class for all image denoising models."""

            def __init__(self, in_channels: int):
                super().__init__()
                assert (
                    in_channels == 3 or in_channels == 4
                ), f"{in_channels=} should be 3 or 4"

        class UTNet2(Denoiser):
            """Real U-Net architecture for image denoising with transposed convolutions."""

            def __init__(
                self,
                in_channels: int = 4,
                funit: int = 32,
                activation: str = "LeakyReLU",
                preupsample: bool = False,
                **kwargs,  # Accept additional kwargs for compatibility
            ):
                super().__init__(in_channels=in_channels)
                assert (in_channels == 3 and not preupsample) or in_channels == 4
                activation_fun, activation_params = get_activation_class_params(
                    activation
                )

                # Optional upsampling of input (for 4-channel Bayer only)
                if preupsample:
                    self.preprocess = torch.nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    )
                else:
                    self.preprocess = torch.nn.Identity()

                # Encoder path - level 1 (highest resolution)
                self.convs1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, funit, 3, padding=1),
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(funit, funit, 3, padding=1),
                    activation_fun(**activation_params),
                )
                self.maxpool = torch.nn.MaxPool2d(2)

                # Encoder path - level 2
                self.convs2 = torch.nn.Sequential(
                    torch.nn.Conv2d(funit, 2 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Encoder path - level 3
                self.convs3 = torch.nn.Sequential(
                    torch.nn.Conv2d(2 * funit, 4 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Encoder path - level 4 (lowest resolution before bottleneck)
                self.convs4 = torch.nn.Sequential(
                    torch.nn.Conv2d(4 * funit, 8 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Bottleneck at lowest resolution
                self.bottom = torch.nn.Sequential(
                    torch.nn.Conv2d(8 * funit, 16 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(16 * funit, 16 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Decoder path - level 1 (lowest resolution after bottleneck)
                self.up1 = torch.nn.ConvTranspose2d(16 * funit, 8 * funit, 2, stride=2)
                self.tconvs1 = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        16 * funit, 8 * funit, 3, padding=1
                    ),  # 16 = 8 (from up1) + 8 (from skip)
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Decoder path - level 2
                self.up2 = torch.nn.ConvTranspose2d(8 * funit, 4 * funit, 2, stride=2)
                self.tconvs2 = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        8 * funit, 4 * funit, 3, padding=1
                    ),  # 8 = 4 (from up2) + 4 (from skip)
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Decoder path - level 3
                self.up3 = torch.nn.ConvTranspose2d(4 * funit, 2 * funit, 2, stride=2)
                self.tconvs3 = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        4 * funit, 2 * funit, 3, padding=1
                    ),  # 4 = 2 (from up3) + 2 (from skip)
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Decoder path - level 4 (highest resolution)
                self.up4 = torch.nn.ConvTranspose2d(2 * funit, funit, 2, stride=2)
                self.tconvs4 = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        2 * funit, funit, 3, padding=1
                    ),  # 2 = 1 (from up4) + 1 (from skip)
                    activation_fun(**activation_params),
                    torch.nn.Conv2d(funit, funit, 3, padding=1),
                    activation_fun(**activation_params),
                )

                # Output layer - depends on input type
                if in_channels == 3 or preupsample:
                    # For RGB input, direct mapping to RGB output (same resolution)
                    self.output_module = torch.nn.Sequential(
                        torch.nn.Conv2d(funit, 3, 1)
                    )
                elif in_channels == 4:
                    # For Bayer input, map to RGB while doubling resolution with PixelShuffle
                    self.output_module = torch.nn.Sequential(
                        torch.nn.Conv2d(funit, 4 * 3, 1), torch.nn.PixelShuffle(2)
                    )
                else:
                    raise NotImplementedError(f"{in_channels=}")

                # Initialize weights for better convergence
                for m in self.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                    if isinstance(m, torch.nn.ConvTranspose2d):
                        torch.nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )

            def forward(self, l):
                """Process input through U-Net for denoising."""
                # Preprocessing (identity or upsampling)
                l1 = self.preprocess(l)

                # Encoder path with skip connection storage
                l1 = self.convs1(l1)  # Level 1 features (stored for skip connection)
                l2 = self.convs2(self.maxpool(l1))  # Level 2 features
                l3 = self.convs3(self.maxpool(l2))  # Level 3 features
                l4 = self.convs4(self.maxpool(l3))  # Level 4 features

                # Bottleneck and decoder path with skip connections
                l = torch.cat(
                    [self.up1(self.bottom(self.maxpool(l4))), l4], dim=1
                )  # Skip connection 1
                l = torch.cat(
                    [self.up2(self.tconvs1(l)), l3], dim=1
                )  # Skip connection 2
                l = torch.cat(
                    [self.up3(self.tconvs2(l)), l2], dim=1
                )  # Skip connection 3
                l = torch.cat(
                    [self.up4(self.tconvs3(l)), l1], dim=1
                )  # Skip connection 4

                # Final convolutions and output
                l = self.tconvs4(l)
                return self.output_module(l)

        class MockUTNet3(torch.nn.Module):
            def __init__(self, in_channels=4, out_channels=3, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.norm = torch.nn.BatchNorm2d(out_channels)

            def forward(self, x):
                return self.norm(self.conv(x))

        class MockBM3DDenoiser(torch.nn.Module):
            def __init__(self, sigma=25.0, **kwargs):
                super().__init__()
                self.sigma = sigma
                self.identity = torch.nn.Identity()

            def forward(self, x):
                # Mock BM3D - in reality would apply BM3D algorithm
                return self.identity(x)

        class MockLearnedDenoiseNet(torch.nn.Module):
            def __init__(self, num_layers=8, hidden_dim=64, **kwargs):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Conv2d(3, hidden_dim, 3, padding=1),
                    torch.nn.ReLU(),
                    *[torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in range(num_layers-2)],
                    torch.nn.Conv2d(hidden_dim, 3, 3, padding=1)
                )

            def forward(self, x):
                return self.layers(x)

        class MockBalleEncoder(torch.nn.Module):
            def __init__(self, num_filters=192, **kwargs):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, num_filters, 3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
                )

            def forward(self, x):
                return self.encoder(x)

        class MockBalleDecoder(torch.nn.Module):
            def __init__(self, num_filters=192, **kwargs):
                super().__init__()
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(num_filters, 3, 3, stride=2, padding=1, output_padding=1)
                )

            def forward(self, x):
                return self.decoder(x)

        # Register default models from memory
        self._models = {
            "utnet2": UTNet2,
            "utnet3": MockUTNet3,
            "bm3d": MockBM3DDenoiser,
            "learned_denoise": MockLearnedDenoiseNet,
            "balle_encoder": MockBalleEncoder,
            "balle_decoder": MockBalleDecoder,
        }

        # Default configurations for models
        self._model_configs = {
            'utnet2': {'in_channels': 4, 'out_channels': 3},
            'utnet3': {'in_channels': 4, 'out_channels': 3},
            'bm3d': {'sigma': 25.0},
            'learned_denoise': {'num_layers': 8, 'hidden_dim': 64},
            'balle_encoder': {'num_filters': 192},
            'balle_decoder': {'num_filters': 192},
        }

    def register_model(self, model_name: str, model_class: Type[torch.nn.Module], 
                      default_config: Optional[Dict[str, Any]] = None):
        """
        Register a custom model in the registry.
        
        Args:
            model_name: Unique identifier for the model
            model_class: PyTorch module class to register
            default_config: Optional default configuration parameters
        """
        if not issubclass(model_class, torch.nn.Module):
            raise ValueError(f"Model class must be a subclass of torch.nn.Module")

        self._models[model_name] = model_class
        if default_config:
            self._model_configs[model_name] = default_config

    def create_model(self, model_name: str, **override_params) -> torch.nn.Module:
        """
        Create a model instance from registry with parameter override support.
        
        Args:
            model_name: Name of model to create
            **override_params: Parameters to override defaults
            
        Returns:
            Instantiated PyTorch model
            
        Raises:
            ValueError: If model name is not found in registry
        """
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {self.list_available_models()}")

        model_class = self._models[model_name]

        # Merge default config with overrides
        params = self._model_configs.get(model_name, {}).copy()
        params.update(override_params)

        try:
            return model_class(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to create model '{model_name}' with parameters {params}: {str(e)}")

    def list_available_models(self) -> List[str]:
        """
        List all available model names in the registry.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Default configuration dictionary
            
        Raises:
            ValueError: If model name is not found
        """
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")

        return self._model_configs.get(model_name, {}).copy()

    def create_pipeline(self, model_sequence: List[str], 
                       model_params: Optional[Dict[str, Dict[str, Any]]] = None) -> torch.nn.Sequential:
        """
        Create a sequential pipeline of models.
        
        Args:
            model_sequence: List of model names to chain
            model_params: Optional parameter overrides for each model
            
        Returns:
            Sequential pipeline of models
        """
        if model_params is None:
            model_params = {}

        models = []
        for model_name in model_sequence:
            params = model_params.get(model_name, {})
            model = self.create_model(model_name, **params)
            models.append(model)

        return torch.nn.Sequential(*models)

    def clear_registry(self):
        """Clear all registered models (mainly for testing)."""
        self._models.clear()
        self._model_configs.clear()

    def __contains__(self, model_name: str) -> bool:
        """Check if model name exists in registry."""
        return model_name in self._models

    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._models)
