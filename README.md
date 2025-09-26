# Fiddlesticks

**Function-Based Composable Pipeline Architecture for Computer Vision**

A comprehensive, production-ready system that handles everything from basic image processing to advanced computational photography workflows while maintaining the core principle: **the pipeline doesn't care about implementation details, only functional intent**.

## 🚀 Key Features

- **Universal Operation Interface**: Classical algorithms, ML models, and GPU operations work identically
- **Dual Interface System**: Simple for beginners (`['denoise', 'sharpen']`), advanced for power users
- **75+ Operations**: Across 10 functional categories (raw processing, enhancement, denoising, etc.)
- **65+ Kornia Operations**: GPU-accelerated computer vision operations
- **Registry Pattern Extensions**: Models, quality checks, preprocessing, training strategies
- **Framework Integration**: PyTorch Lightning and Hydra out-of-the-box
- **Production Ready**: A/B testing, containerization, monitoring, debugging tools

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FIDDLESTICKS ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│  Universal Pipeline Operation Interface                     │
│  ┌─────────────────────────────────────────────────────────┤
│  │ (tensor|List[tensor], metadata, **kwargs)              │
│  │ → (result, metadata)                                    │
│  └─────────────────────────────────────────────────────────┤
│                                                             │
│  Comprehensive Operation Registry (10 Categories)          │
│  ┌─────────────────────────────────────────────────────────┤
│  │ • Raw Processing    • Enhancement    • Denoising       │
│  │ • Color Processing  • Tone Mapping   • Geometric       │
│  │ • Burst Processing  • Quality Assess • Creative        │
│  │ • I/O Operations                                        │
│  └─────────────────────────────────────────────────────────┤
│                                                             │
│  Pipeline Execution System                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │ Smart Assembly → Validation → Optimization → Execution │
│  └─────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
pip install fiddlesticks

# Or for development
git clone https://github.com/fiddlesticks-team/fiddlesticks
cd fiddlesticks
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Simple Interface (Beginner-Friendly)

```python
from fiddlesticks import OperationPipeline

# Ultra-simple configuration
pipeline = OperationPipeline(['load_raw', 'denoise', 'sharpen', 'save_image'])
result = pipeline(input_data)

# Simple with parameters
pipeline = OperationPipeline([
    {'operation': 'load_raw', 'format': 'dng'},
    {'operation': 'denoise', 'strength': 0.3},
    {'operation': 'sharpen', 'amount': 1.2},
    {'operation': 'save_image', 'format': 'tiff'}
])
```

### Advanced Interface (Power Users)

```python
# Full specification with explicit control
advanced_pipeline = OperationPipeline([
    {
        'operation': 'load_raw_file',
        'category': 'input_output_operations',
        'params': {'file_path': '/data/raw/IMG_001.dng'},
        'metadata_requirements': ['file_format']
    },
    {
        'operation': 'utnet2',
        'category': 'denoising_operations',
        'params': {'checkpoint': 'models/utnet2_best.pth'},
        'constraints': {'gpu_memory': '8GB'}
    },
    {
        'operation': 'filmicrgb',
        'category': 'tone_mapping_operations',
        'params': {'white_point': 4.0}
    }
])
```

### Multi-Image Processing

```python
# HDR Processing
hdr_pipeline = OperationPipeline([
    {'operation': 'load_burst', 'category': 'input_output_operations'},
    {'operation': 'hdr_merge', 'category': 'burst_processing_operations'},
    {'operation': 'tone_mapping', 'category': 'tone_mapping_operations'},
    {'operation': 'save_image', 'category': 'input_output_operations'}
])

# Focus Stacking
focus_pipeline = OperationPipeline([
    {'operation': 'load_burst', 'category': 'input_output_operations'}, 
    {'operation': 'focus_stack', 'category': 'burst_processing_operations'},
    {'operation': 'save_image', 'category': 'input_output_operations'}
])
```

## 🧪 PyTorch Lightning Integration

```python
import lightning as L
from fiddlesticks.framework.lightning import ImageProcessingTask

# Create Lightning module from pipeline
task = ImageProcessingTask(
    pipeline=your_pipeline,
    loss_config={'ssim': 0.5, 'lpips': 0.3}
)

# Standard Lightning training
trainer = L.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(task, datamodule=your_datamodule)
```

## ⚙️ Hydra Configuration

```yaml
# conf/pipeline/denoising_workflow.yaml
operations:
  - operation: rawprepare
    category: raw_processing_operations
    params: {sensor_calibration: true}
  
  - operation: utnet2
    category: denoising_operations
    params: {checkpoint: "${model.checkpoint_path}"}
  
  - operation: save_image
    category: input_output_operations
    params: {format: tiff, bit_depth: 16}

# Usage: python train.py pipeline=denoising_workflow
```

## 🔧 Available Operations

### Raw Processing Operations (12 operations)
- `rawprepare`, `hotpixels`, `temperature`, `rawdenoise`, `demosaic`

### Color Processing Operations (8 operations)  
- `colorin`, `colorout`, `channelmixerrgb`, `colorbalancergb`, `primaries`

### Tone Mapping Operations (7 operations)
- `exposure`, `filmicrgb`, `sigmoid`, `toneequal`, `highlights`

### Enhancement Operations (8 operations)
- `sharpen`, `diffuse`, `blurs`, `defringe`, `ashift`

### Denoising Operations (10 operations)
- `utnet2`, `bm3d`, `bilateral`, `nlmeans`, `denoiseprofile`

### Burst Processing Operations (6 operations)
- `hdr_merge`, `focus_stack`, `panorama_stitch`, `temporal_denoise`

### And 4 more categories with 65+ additional operations...

## 🚀 Production Features

### A/B Testing
```python
from fiddlesticks.production.ab_testing import ABTestingPipeline

ab_pipeline = ABTestingPipeline(pipeline_a, pipeline_b)
result, metadata = ab_pipeline.process_with_split_testing(data, test_ratio=0.3)
```

### Performance Monitoring
```python
from fiddlesticks.production.monitoring import PipelineMonitor

monitor = PipelineMonitor(pipeline)
metrics = monitor.profile_execution(data)
print(f"Processing time: {metrics['total_time']:.2f}s")
```

### Pipeline Debugging
```python
from fiddlesticks.execution.debugger import PipelineDebugger

debugger = PipelineDebugger()
trace = debugger.trace_execution(pipeline, data)
debugger.visualize_intermediate_results(trace)
```

## 🧠 Architecture Benefits

1. **Universal Abstraction**: Every operation follows the same interface regardless of implementation
2. **Function-Based Organization**: Operations grouped by what they do, not how they do it  
3. **Implementation Agnostic**: Classical algorithms, ML models, GPU operations work identically
4. **Progressive Disclosure**: Start simple, add complexity gradually
5. **Production Ready**: Comprehensive validation, optimization, monitoring, debugging

## 📚 Documentation

- [API Reference](https://fiddlesticks.readthedocs.io/api/)
- [User Guide](https://fiddlesticks.readthedocs.io/guide/)
- [Examples](https://github.com/fiddlesticks-team/fiddlesticks/tree/main/examples)
- [Contributing](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This architecture was inspired by the elegant augmentations pipeline approach and evolved through comprehensive design discussions focusing on implementation-agnostic, function-based composition principles.