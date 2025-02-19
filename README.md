# Vision Project - Clothing Classification

A deep learning project comparing EfficientNet-B2 and Vision Transformer (ViT) models for clothing classification tasks.

## Project Goals

1. Achieve accuracy above 90% on training data
2. Obtain acceptable results on real-world images
3. Deploy model on mobile devices

## Project Overview

This project implements and compares two state-of-the-art deep learning architectures for clothing classification:
- EfficientNet-B2
- Vision Transformer (ViT-B/16)

The comparison focuses on:
- Model accuracy
- Loss metrics
- Model size
- Inference time
- Practical deployment considerations

## Dataset

The project uses the `clothing-dataset-small` which is organized as follows:
    clothing-dataset-small/
    ├── train/
    ├── test/
    └── validation/

## Key Findings

### Model Comparison

#### Model Size
- ViT-B/16: 327.38 MB
- EfficientNet-B2: 29.87 MB

#### Performance Metrics
- EfficientNet-B2 achieves comparable accuracy to ViT (80-100% relative accuracy)
- EfficientNet-B2 shows higher loss values (1.7x to 5.1x relative to ViT)

### Final Recommendation

EfficientNet-B2 is selected as the better model for this application due to:
- Significantly smaller model size (~11x smaller)
- Better runtime performance
- Comparable accuracy to ViT
- Better suited for mobile deployment

## Technical Stack

- PyTorch 2.0.1
- TorchVision 0.15.2
- CUDA 11.7
- Python 3.10.12

### Key Libraries
- torch
- torchvision
- torchinfo
- pandas
- numpy
- matplotlib
- sklearn

## Project Structure

Vision_Project/
├── clothing-dataset-small/
│   ├── train/
│   ├── test/
│   └── validation/
├── Models/
│   ├── efficientnet_b2.pth
│   └── vit_b_16.pth
└── Vision.ipynb
```

## Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision torchinfo pandas numpy matplotlib sklearn
```
3. Ensure you have CUDA support if using GPU acceleration
4. Open and run the Vision.ipynb notebook

## Model Deployment

The EfficientNet-B2 model is recommended for deployment due to its smaller size (29.87 MB) and efficient runtime performance, making it suitable for mobile devices while maintaining good accuracy.

## License

[Specify your license here]

## Contributors

[Add contributor information here]

## Acknowledgments

- PyTorch team
- TorchVision contributors
- [Add any other acknowledgments]

