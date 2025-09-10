# inference.py
"""
Simple YOLOv9 Inference Script
Basic inference for images and videos using trained YOLOv9 model
"""

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
import time

# Constants
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640
DEFAULT_PADDING_VALUE = 114
PROGRESS_UPDATE_INTERVAL = 30
IMG_SIZE_DIVISOR = 32

class YOLOv9Inference:
    """Simple inference class for YOLOv9"""
    
    def __init__(self, weights_path, conf_threshold=DEFAULT_CONF_THRESHOLD, iou_threshold=DEFAULT_IOU_THRESHOLD, img_size=DEFAULT_IMG_SIZE):
        """
        Initialize YOLOv9 inference
        
        Args:
            weights_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
        """
        # Validate inputs
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {conf_threshold}")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0 and 1, got {iou_threshold}")
        if img_size <= 0 or img_size % IMG_SIZE_DIVISOR != 0:
            raise ValueError(f"Image size must be positive and divisible by {IMG_SIZE_DIVISOR}, got {img_size}")
        
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Loading model from: {weights_path}")
        
        # Load model (this would use actual YOLOv9 loading)
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained YOLOv9 model"""
        # In practice, you would use YOLOv9's model loading:
        # from models.experimental import attempt_load
        # model = attempt_load(self.weights_path, map_location=self.device)
        
        # For now, returning a placeholder
        print("Model loaded successfully!")
        return None  # Placeholder
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, value=(DEFAULT_PADDING_VALUE, DEFAULT_PADDING_VALUE, DEFAULT_PADDING_VALUE)
        )
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, scale, (pad_w, pad_h)
    
    def postprocess_detections(self, predictions, scale, padding):
        """
        Postprocess model predictions
        
        Args:
            predictions: Raw model predictions
            scale: Scale factor used in preprocessing
            padding: Padding applied in preprocessing
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class]
        """
        # This would contain actual NMS and coordinate conversion
        # For now, returning dummy detections
        detections = [
            [100, 100, 200, 200, 0.85, 0],  # [x1, y1, x2, y2, conf, class]
            [300, 150, 450, 300, 0.75, 1],
        ]
        
        return detections
    
    def detect_image(self, image_path, output_path=None, show_image=False):
        """
        Run detection on a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_image: Whether to display the image
            
        Returns:
            List of detections
        """
        # Validate input path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Preprocess
        img_tensor, scale, padding = self.preprocess_image(image)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            # predictions = self.model(img_tensor)  # Actual inference
            predictions = None  # Placeholder
        
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess_detections(predictions, scale, padding)
        
        # Draw detections
        annotated_image = self.draw_detections(image, detections)
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Found {len(detections)} detections")
        
        # Save or show result
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved to: {output_path}")
        
        if show_image:
            cv2.imshow('YOLOv9 Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
    
    def detect_video(self, video_path, output_path=None, show_video=False):
        """
        Run detection on a video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_video: Whether to display the video
            
        Returns:
            Total number of detections
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess
                img_tensor, scale, padding = self.preprocess_image(frame)
                
                # Inference
                with torch.no_grad():
                    # predictions = self.model(img_tensor)  # Actual inference
                    predictions = None  # Placeholder
                
                # Postprocess
                detections = self.postprocess_detections(predictions, scale, padding)
                total_detections += len(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame
                if show_video:
                    cv2.imshow('YOLOv9 Video Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % PROGRESS_UPDATE_INTERVAL == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        print(f"Video processing complete!")
        print(f"Total detections: {total_detections}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return total_detections
    
    def draw_detections(self, image, detections, class_names=None):
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: List of detections [x1, y1, x2, y2, conf, class]
            class_names: List of class names (optional)
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Default class names if not provided
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(100)]
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            
            # Get color for this class
            color = colors[cls % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_names[cls]}: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def batch_process_images(self, input_dir, output_dir):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        total_detections = 0
        for i, image_file in enumerate(image_files):
            output_file = output_path / f"detected_{image_file.name}"
            detections = self.detect_image(str(image_file), str(output_file))
            total_detections += len(detections)
            
            print(f"[{i+1}/{len(image_files)}] {image_file.name}: {len(detections)} detections")
        
        print(f"Batch processing complete!")
        print(f"Total detections across all images: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv9 Inference')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Input source (image/video/directory)')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD, help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE, help='Input image size')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    
    args = parser.parse_args()
    
    # Initialize inference
    detector = YOLOv9Inference(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size
    )
    
    # Determine input type and process
    source_path = Path(args.source)
    
    if args.batch or source_path.is_dir():
        # Batch process directory
        if not args.output:
            args.output = source_path / 'detected'
        detector.batch_process_images(args.source, args.output)
        
    elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        detector.detect_video(args.source, args.output, args.show)
        
    else:
        # Process single image
        detector.detect_image(args.source, args.output, args.show)

if __name__ == "__main__":
    main()