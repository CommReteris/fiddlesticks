"""
TrainingStrategyRegistry for ML Training Patterns.

This module provides a unified training strategy registry similar to the augmentations
pipeline approach, enabling systematic training pattern management including:
- Supervised training for standard input/output pairs
- Self-supervised training for representation learning
- Adversarial training for robust model training
- Multi-task training for joint objective optimization
- Few-shot training for limited data scenarios
- Custom training strategies registered at runtime

Key Features:
- Configurable training strategy factory
- Support for multiple training paradigms
- Task-agnostic training infrastructure
- Runtime registration of custom strategies
- Integration with existing Function-Based Composable Pipeline Architecture
"""

import torch
from typing import Dict, List, Any, Callable, Optional, Type
from abc import ABC, abstractmethod
import warnings


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    
    Defines the interface that all training strategies must implement
    to ensure consistency across different training paradigms.
    """
    
    @abstractmethod
    def train(self, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute training with specified number of epochs."""
        pass
    
    @abstractmethod
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizers for the training strategy."""
        pass
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute single training step and return loss."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return name of the training strategy."""
        pass


class SupervisedTrainingStrategy(TrainingStrategy):
    """
    Standard supervised training strategy.
    
    Implements traditional supervised learning with input/output pairs
    and standard loss computation.
    """
    
    def __init__(self, operations: List[Any], loss_fn: Optional[Callable] = None, 
                 learning_rate: float = 1e-3):
        """
        Initialize supervised training strategy.
        
        Args:
            operations: List of pipeline operations to train
            loss_fn: Loss function for supervised training
            learning_rate: Learning rate for optimizer
        """
        self.operations = operations
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizers = None
        self.metrics = {}
    
    @property
    def strategy_name(self) -> str:
        return "supervised"
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure Adam optimizer for supervised training."""
        if self.optimizers is None:
            self.optimizers = []
            for operation in self.operations:
                if hasattr(operation, 'get_parameters') and operation.get_parameters() is not None:
                    model = operation.get_parameters()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    self.optimizers.append(optimizer)
        return self.optimizers
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute supervised training step."""
        inputs, targets = batch
        
        # Forward pass through operations
        current_data = inputs
        for operation in self.operations:
            if hasattr(operation, '__call__'):
                current_data, _ = operation(current_data)
        
        # Compute supervised loss
        loss = self.loss_fn(current_data, targets)
        return loss
    
    def train(self, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute supervised training loop."""
        dataloader = kwargs.get('dataloader', None)
        if dataloader is None:
            raise ValueError("Supervised training requires 'dataloader' in kwargs")
        
        optimizers = self.configure_optimizers()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Zero gradients
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                # Training step
                loss = self.training_step(batch, batch_idx)
                epoch_losses.append(loss.item())
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                for optimizer in optimizers:
                    optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.metrics[f'epoch_{epoch}_loss'] = avg_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
        
        return self.metrics


class SelfSupervisedTrainingStrategy(TrainingStrategy):
    """
    Self-supervised training strategy.
    
    Implements self-supervised learning paradigms where the model
    learns representations from the data itself without external labels.
    """
    
    def __init__(self, operations: List[Any], pretext_task: str = 'reconstruction',
                 learning_rate: float = 1e-3):
        """
        Initialize self-supervised training strategy.
        
        Args:
            operations: List of pipeline operations to train
            pretext_task: Type of pretext task (reconstruction, contrastive, etc.)
            learning_rate: Learning rate for optimizer
        """
        self.operations = operations
        self.pretext_task = pretext_task
        self.learning_rate = learning_rate
        self.optimizers = None
        self.metrics = {}
    
    @property
    def strategy_name(self) -> str:
        return "self_supervised"
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizer for self-supervised training."""
        if self.optimizers is None:
            self.optimizers = []
            for operation in self.operations:
                if hasattr(operation, 'get_parameters') and operation.get_parameters() is not None:
                    model = operation.get_parameters()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    self.optimizers.append(optimizer)
        return self.optimizers
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute self-supervised training step."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, _ = batch  # Ignore labels in self-supervised learning
        else:
            inputs = batch
        
        # Apply pretext task
        if self.pretext_task == 'reconstruction':
            # Reconstruction task: input -> corrupted -> reconstructed
            targets = inputs
            # Add noise or corruption (simplified)
            corrupted = inputs + torch.randn_like(inputs) * 0.1
            
            # Forward pass
            current_data = corrupted
            for operation in self.operations:
                if hasattr(operation, '__call__'):
                    current_data, _ = operation(current_data)
            
            # Reconstruction loss
            loss = torch.nn.functional.mse_loss(current_data, targets)
            
        else:
            # Default to reconstruction
            targets = inputs
            current_data = inputs
            for operation in self.operations:
                if hasattr(operation, '__call__'):
                    current_data, _ = operation(current_data)
            loss = torch.nn.functional.mse_loss(current_data, targets)
        
        return loss
    
    def train(self, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute self-supervised training loop."""
        dataloader = kwargs.get('dataloader', None)
        if dataloader is None:
            raise ValueError("Self-supervised training requires 'dataloader' in kwargs")
        
        optimizers = self.configure_optimizers()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                loss = self.training_step(batch, batch_idx)
                epoch_losses.append(loss.item())
                
                loss.backward()
                
                for optimizer in optimizers:
                    optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.metrics[f'epoch_{epoch}_loss'] = avg_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Self-supervised Loss: {avg_loss:.6f}")
        
        return self.metrics


class AdversarialTrainingStrategy(TrainingStrategy):
    """
    Adversarial training strategy.
    
    Implements adversarial training for robust model training,
    which can improve model generalization and robustness.
    """
    
    def __init__(self, operations: List[Any], epsilon: float = 0.1,
                 learning_rate: float = 1e-3):
        """
        Initialize adversarial training strategy.
        
        Args:
            operations: List of pipeline operations to train
            epsilon: Perturbation strength for adversarial examples
            learning_rate: Learning rate for optimizer
        """
        self.operations = operations
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.optimizers = None
        self.metrics = {}
    
    @property
    def strategy_name(self) -> str:
        return "adversarial"
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizer for adversarial training."""
        if self.optimizers is None:
            self.optimizers = []
            for operation in self.operations:
                if hasattr(operation, 'get_parameters') and operation.get_parameters() is not None:
                    model = operation.get_parameters()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    self.optimizers.append(optimizer)
        return self.optimizers
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute adversarial training step."""
        inputs, targets = batch
        
        # Generate adversarial examples (simplified FGSM-style)
        inputs.requires_grad_(True)
        
        # Forward pass for adversarial example generation
        current_data = inputs
        for operation in self.operations:
            if hasattr(operation, '__call__'):
                current_data, _ = operation(current_data)
        
        # Compute loss for gradient
        loss = torch.nn.functional.mse_loss(current_data, targets)
        
        # Compute gradient
        grad = torch.autograd.grad(loss, inputs, create_graph=False, retain_graph=False)[0]
        
        # Create adversarial example
        adversarial_inputs = inputs + self.epsilon * grad.sign()
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        # Forward pass with adversarial examples
        current_data = adversarial_inputs
        for operation in self.operations:
            if hasattr(operation, '__call__'):
                current_data, _ = operation(current_data)
        
        # Adversarial loss
        adversarial_loss = torch.nn.functional.mse_loss(current_data, targets)
        
        return adversarial_loss
    
    def train(self, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute adversarial training loop."""
        dataloader = kwargs.get('dataloader', None)
        if dataloader is None:
            raise ValueError("Adversarial training requires 'dataloader' in kwargs")
        
        optimizers = self.configure_optimizers()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                loss = self.training_step(batch, batch_idx)
                epoch_losses.append(loss.item())
                
                loss.backward()
                
                for optimizer in optimizers:
                    optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.metrics[f'epoch_{epoch}_loss'] = avg_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Adversarial Loss: {avg_loss:.6f}")
        
        return self.metrics


class MultiTaskTrainingStrategy(TrainingStrategy):
    """
    Multi-task training strategy.
    
    Implements multi-task learning where multiple related tasks
    are trained jointly to improve generalization.
    """
    
    def __init__(self, operations: List[Any], task_weights: Optional[Dict[str, float]] = None,
                 learning_rate: float = 1e-3):
        """
        Initialize multi-task training strategy.
        
        Args:
            operations: List of pipeline operations to train
            task_weights: Weights for different tasks
            learning_rate: Learning rate for optimizer
        """
        self.operations = operations
        self.task_weights = task_weights or {'task_1': 1.0, 'task_2': 1.0}
        self.learning_rate = learning_rate
        self.optimizers = None
        self.metrics = {}
    
    @property
    def strategy_name(self) -> str:
        return "multi_task"
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizer for multi-task training."""
        if self.optimizers is None:
            self.optimizers = []
            for operation in self.operations:
                if hasattr(operation, 'get_parameters') and operation.get_parameters() is not None:
                    model = operation.get_parameters()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    self.optimizers.append(optimizer)
        return self.optimizers
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute multi-task training step."""
        # Assume batch contains multiple task targets
        if isinstance(batch, dict):
            inputs = batch['inputs']
            task_targets = {k: v for k, v in batch.items() if k != 'inputs'}
        else:
            # Fallback: treat as single task
            inputs, targets = batch
            task_targets = {'task_1': targets}
        
        # Forward pass
        current_data = inputs
        for operation in self.operations:
            if hasattr(operation, '__call__'):
                current_data, _ = operation(current_data)
        
        # Compute multi-task loss
        total_loss = 0.0
        for task_name, targets in task_targets.items():
            task_loss = torch.nn.functional.mse_loss(current_data, targets)
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * task_loss
        
        return total_loss
    
    def train(self, epochs: int = 100, **kwargs) -> Dict[str, Any]:
        """Execute multi-task training loop."""
        dataloader = kwargs.get('dataloader', None)
        if dataloader is None:
            raise ValueError("Multi-task training requires 'dataloader' in kwargs")
        
        optimizers = self.configure_optimizers()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                loss = self.training_step(batch, batch_idx)
                epoch_losses.append(loss.item())
                
                loss.backward()
                
                for optimizer in optimizers:
                    optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.metrics[f'epoch_{epoch}_loss'] = avg_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Multi-task Loss: {avg_loss:.6f}")
        
        return self.metrics


class TrainingStrategyRegistry:
    """
    Registry for training strategy patterns.
    
    Provides factory pattern for creating different training strategies
    similar to augmentations pipeline pattern, enabling configurable
    training paradigm selection and runtime strategy registration.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(TrainingStrategyRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize TrainingStrategyRegistry with default strategies."""
        if not self._initialized:
            self._strategies = {}
            self._register_default_strategies()
            TrainingStrategyRegistry._initialized = True
    
    def _register_default_strategies(self):
        """Register default training strategies based on memory patterns."""
        self._strategies = {
            'supervised': SupervisedTrainingStrategy,
            'self_supervised': SelfSupervisedTrainingStrategy,
            'adversarial': AdversarialTrainingStrategy,
            'multi_task': MultiTaskTrainingStrategy,
        }
    
    def register_strategy(self, strategy_name: str, strategy_class: Type[TrainingStrategy]):
        """
        Register a custom training strategy.
        
        Args:
            strategy_name: Unique identifier for the strategy
            strategy_class: TrainingStrategy subclass
        """
        if not issubclass(strategy_class, TrainingStrategy):
            raise ValueError("Strategy class must inherit from TrainingStrategy")
        
        self._strategies[strategy_name] = strategy_class
    
    def list_available_strategies(self) -> List[str]:
        """
        List all available training strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def create_strategy(self, strategy_name: str, operations: List[Any], **kwargs) -> TrainingStrategy:
        """
        Create training strategy instance from registry.
        
        Args:
            strategy_name: Name of strategy to create
            operations: List of pipeline operations to train
            **kwargs: Strategy-specific parameters
            
        Returns:
            TrainingStrategy instance
            
        Raises:
            ValueError: If strategy name is not found
        """
        if strategy_name not in self._strategies:
            available = ', '.join(self.list_available_strategies())
            raise ValueError(f"Unknown training strategy: {strategy_name}. Available: {available}")
        
        strategy_class = self._strategies[strategy_name]
        return strategy_class(operations, **kwargs)
    
    def __contains__(self, strategy_name: str) -> bool:
        """Check if training strategy exists in registry."""
        return strategy_name in self._strategies
    
    def __len__(self) -> int:
        """Get number of registered training strategies."""
        return len(self._strategies)