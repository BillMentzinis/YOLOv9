# 🚀 YOLOv9 Training Wrapper

A complete wrapper for training YOLOv9 models with the official repository. This project automatically sets up YOLOv9, downloads weights, and provides an easy interface for training with your custom dataset.

## ✨ Features

- 🔧 **Automatic Setup**: Clones and configures YOLOv9 repository
- ⬇️ **Weight Management**: Downloads and manages pretrained weights
- 🎯 **Easy Training**: Simple commands for fine-tuning or training from scratch
- 📊 **Custom Configs**: Generates model configs for your specific dataset
- 🔄 **GitHub Ready**: Perfect for cloud training setups

## 📋 Prerequisites

```bash
# Install wrapper dependencies
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.7+
- Git (for repository cloning)
- CUDA (optional, for GPU training)

## 🚀 Quick Start

### 1. Setup Your Dataset
```bash
# Create dataset structure
python train.py --data ./dataset --classes person vehicle building --setup-only
```

This creates the YOLO dataset structure:
```
dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt, YOLO format)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── test/
    ├── images/     # Test images (optional)
    └── labels/     # Test labels (optional)
```

### 2. Label Format (YOLO Format)
Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```
All coordinates are normalized (0-1):
```
0 0.5 0.3 0.1 0.15    # person at center-top
1 0.2 0.7 0.05 0.08   # vehicle at left-bottom
```

### 3. Train Your Model

**Fine-tuning (Recommended):**
```bash
python train.py \
    --data ./dataset \
    --classes person vehicle building \
    --epochs 100 \
    --batch-size 16 \
    --model yolov9c
```

**Training from Scratch:**
```bash
python train.py \
    --data ./dataset \
    --classes person vehicle building \
    --epochs 300 \
    --batch-size 8 \
    --from-scratch \
    --model yolov9c
```

## 🎯 Advanced Usage

### Setup Repository Only
```bash
# Just clone and setup YOLOv9 repository
python train.py --setup-repo
```

### Download Specific Weights
```bash
# Download specific model weights
python train.py --download-weights yolov9e.pt
```

### Custom Training Parameters
```bash
python train.py \
    --data ./dataset \
    --classes person vehicle drone aircraft \
    --epochs 200 \
    --batch-size 32 \
    --img-size 1280 \
    --model yolov9e \
    --weights ./weights/yolov9e.pt
```

### Force Repository Update
```bash
# Re-clone repository (useful for updates)
python train.py --force-update --setup-repo
```

## 🔧 Available Models

| Model | Size | Speed | mAP | Parameters |
|-------|------|-------|-----|------------|
| YOLOv9t | 640 | ⚡⚡⚡ | 38.3 | 2.0M |
| YOLOv9s | 640 | ⚡⚡ | 46.8 | 7.2M |
| YOLOv9m | 640 | ⚡ | 51.4 | 20.1M |
| YOLOv9c | 640 | ⚡ | 53.0 | 25.5M |
| YOLOv9e | 640 | ⚡ | 55.6 | 58.1M |

## 📊 Training Results

After training completes, find your models in:
```
runs/train/yolov9_custom/
├── weights/
│   ├── best.pt      # Best validation model
│   └── last.pt      # Latest checkpoint
├── results.png      # Training curves
└── confusion_matrix.png
```

## 🔍 Using Trained Models

Once trained, use your model with the inference script:

```bash
# Single image
python inference.py \
    --weights runs/train/yolov9_custom/weights/best.pt \
    --source test_image.jpg \
    --output result.jpg \
    --show

# Video processing
python inference.py \
    --weights runs/train/yolov9_custom/weights/best.pt \
    --source drone_video.mp4 \
    --output detected_video.mp4

# Batch process images
python inference.py \
    --weights runs/train/yolov9_custom/weights/best.pt \
    --source ./test_images/ \
    --batch \
    --output ./results/
```

## 🛠️ Customization & Advanced Features

### Custom Model Architecture
The wrapper automatically creates custom model configs based on your dataset. You can also:

1. **Modify Generated Configs**: Edit `custom_yolov9c.yaml` after generation
2. **Access YOLOv9 Source**: Full access to `yolov9_official/` directory
3. **Custom Loss Functions**: Modify training in `yolov9_official/train.py`
4. **Data Augmentation**: Configure augmentations in model configs

### Directory Structure After Setup
```
your_project/
├── train.py              # Main training wrapper
├── inference.py          # Inference script  
├── requirements.txt      # Dependencies
├── weights/              # Downloaded pretrained weights
├── yolov9_official/      # Official YOLOv9 repository
├── dataset/              # Your training data
├── data.yaml            # Generated data config
├── custom_yolov9c.yaml  # Generated model config
└── runs/                # Training outputs
    └── train/yolov9_custom/weights/best.pt
```

## ⚙️ Key Parameters

### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--img-size`: Input image size (640, 1280 for small objects)
- `--model`: Model variant (yolov9t/s/m/c/e)

### Model Selection Guide
- **YOLOv9t**: Fastest, mobile deployment
- **YOLOv9s**: Balanced speed/accuracy
- **YOLOv9c**: **Recommended** for most use cases
- **YOLOv9e**: Highest accuracy, slower inference

## 🐛 Troubleshooting

### Common Issues

**Git Clone Fails:**
```bash
# Check git installation
git --version

# If behind corporate firewall, try HTTPS
# (automatically handled by the script)
```

**Out of Memory:**
```bash
# Reduce batch size
--batch-size 4

# Or reduce image size
--img-size 640
```

**No GPU Detected:**
```bash
# Check CUDA/PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Poor Detection Results:**
- Ensure labels are in correct YOLO format
- Check if class names match your dataset
- Try training for more epochs (300+ for from-scratch)
- Use larger image size for small objects (--img-size 1280)
- Consider data augmentation adjustments

### Verification Steps
```bash
# 1. Check repository setup
python train.py --setup-repo

# 2. Verify dataset structure
python train.py --data ./dataset --classes person car --setup-only

# 3. Test with minimal training
python train.py --data ./dataset --classes person --epochs 1 --batch-size 1
```

## 🌟 GitHub Deployment

This project is designed to be GitHub-ready:

1. **Push to GitHub**: All dependencies auto-install
2. **Cloud Training**: Works on Google Colab, Kaggle, etc.
3. **No Manual Setup**: Everything downloads automatically
4. **Version Control**: YOLOv9 repo isolated in `yolov9_official/`

### Example Colab Usage
```python
# In Google Colab
!git clone https://github.com/your-username/your-yolov9-project.git
%cd your-yolov9-project
!pip install -r requirements.txt

# Train model (automatic GPU detection)
!python train.py --data ./dataset --classes person vehicle --epochs 50
```

## 📈 Performance Tips

1. **Use pretrained weights** for faster convergence
2. **Start with yolov9c** - best balance of speed/accuracy  
3. **Train for 100+ epochs** for fine-tuning
4. **Use 300+ epochs** when training from scratch
5. **Adjust batch size** based on your GPU memory
6. **Use larger image sizes** (1280) for small object detection

## 🎨 Next Steps

Once you have a working model:
1. **Evaluate on test set** using YOLOv9's built-in metrics
2. **Export to ONNX** for deployment optimization
3. **Implement real-time inference** with your trained model
4. **Fine-tune hyperparameters** for your specific use case
5. **Scale up training** with more diverse data

Perfect for drone detection, autonomous vehicles, security systems, and any custom object detection task!