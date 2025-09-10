# train.py
"""
YOLOv9 Training Wrapper
Integrates with official YOLOv9 repository for training and fine-tuning
"""

import os
import sys
import yaml
import torch
import argparse
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile
from urllib.parse import urlparse

# Constants
YOLOV9_REPO_URL = "https://github.com/WongKinYiu/yolov9.git"
YOLOV9_WEIGHTS_BASE_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/"
YOLOV9_DIR = "yolov9_official"

class YOLOv9Integration:
    """Handles YOLOv9 repository setup and training integration"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.yolov9_dir = self.base_dir / YOLOV9_DIR
        self.weights_dir = self.base_dir / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        
    def setup_yolov9_repo(self, force_update=False):
        """
        Setup YOLOv9 repository
        
        Args:
            force_update: Force re-clone even if directory exists
        """
        if self.yolov9_dir.exists() and not force_update:
            print(f"YOLOv9 repository already exists at: {self.yolov9_dir}")
            return True
            
        if self.yolov9_dir.exists() and force_update:
            print(f"Removing existing YOLOv9 directory...")
            shutil.rmtree(self.yolov9_dir)
            
        print(f"Cloning YOLOv9 repository to: {self.yolov9_dir}")
        try:
            result = subprocess.run([
                "git", "clone", YOLOV9_REPO_URL, str(self.yolov9_dir)
            ], check=True, capture_output=True, text=True)
            
            print("YOLOv9 repository cloned successfully!")
            
            # Install YOLOv9 requirements
            req_file = self.yolov9_dir / "requirements.txt"
            if req_file.exists():
                print("Installing YOLOv9 requirements...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], check=True)
                print("Requirements installed successfully!")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print("Git not found. Please install Git first.")
            return False
            
    def download_weights(self, model_name="yolov9c.pt", force_download=False):
        """
        Download YOLOv9 pretrained weights
        
        Args:
            model_name: Name of the model weights to download
            force_download: Force re-download even if file exists
        """
        weights_path = self.weights_dir / model_name
        
        if weights_path.exists() and not force_download:
            print(f"Weights already exist: {weights_path}")
            return str(weights_path)
            
        weights_url = YOLOV9_WEIGHTS_BASE_URL + model_name
        print(f"Downloading {model_name} from: {weights_url}")
        
        try:
            response = requests.get(weights_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(weights_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
                            
            print(f"\nWeights downloaded successfully: {weights_path}")
            return str(weights_path)
            
        except Exception as e:
            print(f"Error downloading weights: {e}")
            if weights_path.exists():
                weights_path.unlink()
            return None
            
    def create_custom_model_config(self, num_classes, model_base="yolov9c"):
        """
        Create custom model configuration for your dataset
        
        Args:
            num_classes: Number of classes in your dataset
            model_base: Base model configuration to use
        """
        # Map model names to config files
        model_configs = {
            "yolov9t": "yolov9t.yaml",
            "yolov9s": "yolov9s.yaml", 
            "yolov9m": "yolov9m.yaml",
            "yolov9c": "yolov9c.yaml",
            "yolov9e": "yolov9e.yaml"
        }
        
        base_config = model_configs.get(model_base, "yolov9c.yaml")
        custom_config_path = self.base_dir / f"custom_{base_config}"
        
        # Read base config from YOLOv9 repo
        base_config_path = self.yolov9_dir / "models" / "detect" / base_config
        
        if not base_config_path.exists():
            print(f"Warning: Base config not found: {base_config_path}")
            print("Using default configuration template")
            # Create a minimal config
            config_content = f"""# Custom YOLOv9 configuration
# Based on {base_config}

# Parameters
nc: {num_classes}  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
max_channels: 512
activation: SiLU

# Anchors (you may need to adjust these for your dataset)
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# Backbone and Head will be loaded from YOLOv9's architecture
# This config mainly sets the number of classes
"""
        else:
            with open(base_config_path, 'r') as f:
                config_content = f.read()
                
            # Update number of classes
            config_content = config_content.replace(f"nc: 80", f"nc: {num_classes}")
            
        with open(custom_config_path, 'w') as f:
            f.write(config_content)
            
        print(f"Custom model config created: {custom_config_path}")
        return str(custom_config_path)
        
    def train_model(self, data_config, epochs=100, batch_size=16, img_size=640, 
                   weights=None, from_scratch=False, model_config=None, 
                   custom_args=None):
        """
        Train YOLOv9 model using official repository
        
        Args:
            data_config: Path to data.yaml config file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            weights: Path to pretrained weights (optional)
            from_scratch: Whether to train from scratch
            model_config: Custom model configuration file
            custom_args: Additional arguments for training
        """
        if not self.yolov9_dir.exists():
            print("YOLOv9 repository not found. Setting up...")
            if not self.setup_yolov9_repo():
                raise RuntimeError("Failed to setup YOLOv9 repository")
                
        # Prepare training command
        train_script = self.yolov9_dir / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")
            
        cmd = [
            sys.executable, str(train_script),
            "--data", str(data_config),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size), 
            "--img", str(img_size),
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--project", "runs/train",
            "--name", "yolov9_custom"
        ]
        
        # Add model configuration
        if model_config:
            cmd.extend(["--cfg", str(model_config)])
            
        # Add weights for fine-tuning
        if weights and not from_scratch:
            cmd.extend(["--weights", str(weights)])
            print(f"Fine-tuning with weights: {weights}")
        elif from_scratch:
            print("Training from scratch (no pretrained weights)")
        else:
            print("No weights specified - will train from scratch")
            
        # Add custom arguments
        if custom_args:
            cmd.extend(custom_args)
            
        print(f"Starting YOLOv9 training...")
        print(f"Command: {' '.join(cmd)}")
        
        # Change to YOLOv9 directory for training
        original_dir = os.getcwd()
        try:
            os.chdir(self.yolov9_dir)
            result = subprocess.run(cmd, check=True)
            print("Training completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")
            raise
        finally:
            os.chdir(original_dir)

def create_data_config(dataset_path, class_names, output_path="data.yaml"):
    """
    Create data configuration file for training
    
    Args:
        dataset_path: Path to dataset directory
        class_names: List of class names
        output_path: Output path for config file
    """
    # Validate inputs
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not class_names or not all(isinstance(name, str) and name.strip() for name in class_names):
        raise ValueError("Class names must be a non-empty list of non-empty strings")
    
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created data config: {output_path}")
        return output_path
    except (OSError, PermissionError, yaml.YAMLError) as e:
        print(f"Error creating config file {output_path}: {e}")
        raise

def setup_dataset_structure(base_path):
    """
    Create proper dataset directory structure
    
    Args:
        base_path: Base directory for dataset
    """
    
    directories = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_path in directories:
        full_path = Path(base_path) / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    print(f"\nDataset structure created at: {base_path}")
    print("Please place your images and labels in the appropriate directories:")
    print("- Images: .jpg, .png files")
    print("- Labels: .txt files in YOLO format (class x_center y_center width height)")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv9 Model with Official Repository')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--classes', type=str, nargs='+', required=True, 
                       help='Class names (e.g., --classes person car truck)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, help='Pretrained weights path')
    parser.add_argument('--model', type=str, default='yolov9c', help='Model variant (yolov9t/s/m/c/e)')
    parser.add_argument('--setup-only', action='store_true', help='Only setup dataset structure')
    parser.add_argument('--from-scratch', action='store_true', help='Train from scratch (no pretrained weights)')
    parser.add_argument('--setup-repo', action='store_true', help='Setup YOLOv9 repository only')
    parser.add_argument('--download-weights', type=str, help='Download specific weights (e.g., yolov9c.pt)')
    parser.add_argument('--force-update', action='store_true', help='Force update/re-clone repository')
    
    args = parser.parse_args()
    
    # Initialize YOLOv9 integration
    yolo_integration = YOLOv9Integration()
    
    # Setup repository if requested
    if args.setup_repo or args.force_update:
        yolo_integration.setup_yolov9_repo(force_update=args.force_update)
        if args.setup_repo:
            return
    
    # Download weights if requested
    if args.download_weights:
        yolo_integration.download_weights(args.download_weights, force_download=True)
        return
    
    # Setup dataset structure
    if args.setup_only:
        setup_dataset_structure(args.data)
        return
    
    # Validate training parameters
    if args.from_scratch:
        print("⚠️  Training from scratch - make sure you have:")
        print("   • 5000+ diverse training images") 
        print("   • Proper validation set")
        print("   • Patience for 300+ epochs")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Setup YOLOv9 repository
    if not yolo_integration.yolov9_dir.exists():
        print("Setting up YOLOv9 repository...")
        if not yolo_integration.setup_yolov9_repo():
            print("Failed to setup YOLOv9 repository")
            return
    
    # Download weights if needed and not training from scratch
    weights_path = args.weights
    if not args.from_scratch and not weights_path:
        model_weights = f"{args.model}.pt"
        print(f"Downloading default weights: {model_weights}")
        weights_path = yolo_integration.download_weights(model_weights)
        if not weights_path:
            print("Failed to download weights. Training from scratch.")
            args.from_scratch = True
    
    # Create data config
    data_config = create_data_config(args.data, args.classes)
    
    # Create custom model config
    model_config = yolo_integration.create_custom_model_config(
        num_classes=len(args.classes),
        model_base=args.model
    )
    
    # Start training
    try:
        yolo_integration.train_model(
            data_config=data_config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            weights=weights_path,
            from_scratch=args.from_scratch,
            model_config=model_config
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Models saved in: runs/train/yolov9_custom/weights/")
        print(f"Best model: runs/train/yolov9_custom/weights/best.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()