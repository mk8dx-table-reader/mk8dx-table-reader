"""
ONNX Runtime version of yoloUseModelPlayers.py
This file uses ONNX Runtime instead of Ultralytics YOLO for production deployment.
No Ultralytics dependencies required!

The model detects two types of objects:
- Type 0: Player names
- Type 1: Player scores
"""
import cv2
import numpy as np
import onnxruntime as ort
import PIL

try:
    from .base_player_detector import BasePlayerDetector
except ImportError:
    from base_player_detector import BasePlayerDetector


class OnnxPlayerDetector(BasePlayerDetector):
    def __init__(self, model_session):
        self.model_session = model_session

    def preprocess_image(self, image, target_size=640):
        """
        Preprocess image for YOLO ONNX model inference.
        Handles dynamic image sizes while maintaining aspect ratio.
        
        Args:
            image: Input image (str path, PIL Image, or numpy array)
            target_size: Target size for the longer edge (default: 640)
        
        Returns:
            tuple: (preprocessed_image, original_shape, scale_factor, pad)
        """
        # Convert input to numpy array
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, PIL.Image.Image):
            img = np.array(image)
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = image.copy()
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        original_shape = img.shape[:2]  # (height, width)
        
        # Calculate scale to fit target size while maintaining aspect ratio
        scale = min(target_size / original_shape[0], target_size / original_shape[1])
        new_shape = (int(original_shape[1] * scale), int(original_shape[0] * scale))
        
        # Resize image
        img_resized = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (pad to square for YOLO)
        pad_h = target_size - img_resized.shape[0]
        pad_w = target_size - img_resized.shape[1]
        
        # Pad with gray (114, 114, 114) - YOLO default
        img_padded = cv2.copyMakeBorder(
            img_resized, 
            0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)
        )
        
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_padded.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format (channels first)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, original_shape, scale, (0, pad_h, 0, pad_w)


    def postprocess_detections(self, predictions, original_shape, scale, pad, confidence_threshold=0.6):
        """
        Post-process YOLO ONNX model predictions.
        
        Args:
            predictions: Raw model output
            original_shape: Original image shape (height, width)
            scale: Scale factor used in preprocessing
            pad: Padding values (top, bottom, left, right)
            confidence_threshold: Confidence threshold for filtering detections
        
        Returns:
            dict: Detections organized by class type {class_id: [(box, confidence), ...]}
        """
        # YOLO ONNX output format: [batch, num_detections, 6]
        # where 6 values are: [x1, y1, x2, y2, confidence, class_id]
        
        detections = predictions[0]  # Remove batch dimension
        
        # Initialize storage for each type
        type_detections = {0: [], 1: []}
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            class_id = int(cls)
            confidence = float(conf)
            
            # Skip detections below threshold
            if confidence < confidence_threshold:
                continue
            
            # Convert coordinates back to original image space
            # Remove padding
            x1 = x1 - pad[2]
            y1 = y1 - pad[0]
            x2 = x2 - pad[2]
            y2 = y2 - pad[0]
            
            # Unscale coordinates
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
            
            # Clip to image boundaries
            x1 = max(0, min(x1, original_shape[1]))
            y1 = max(0, min(y1, original_shape[0]))
            x2 = max(0, min(x2, original_shape[1]))
            y2 = max(0, min(y2, original_shape[0]))
            
            box = [x1, y1, x2, y2]
            
            if class_id in type_detections:
                type_detections[class_id].append((box, confidence))
        
        return type_detections


    def process_detections(self, image, confidence_threshold=0.6, **kwargs):
        """
        Process detections using ONNX Runtime session.
        Returns bounding boxes for player names (type 0) and player scores (type 1).
        
        Args:
            image: Input image (str path, PIL Image, or numpy array)
            confidence_threshold: Confidence threshold for high-confidence detections
            **kwargs: Additional arguments (unused, for compatibility)
        
        Returns:
            tuple: (first_type_found, second_type_boxes)
                - first_type_found: Bounding box for player name [x1, y1, x2, y2]
                - second_type_boxes: Bounding box for player score [x1, y1, x2, y2]
        """
        # Get original image dimensions
        if isinstance(image, str):
            img = cv2.imread(image)
            height, width = img.shape[:2]
        elif hasattr(image, 'size'):  # PIL Image
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        # Preprocess image
        input_tensor, original_shape, scale, pad = self.preprocess_image(image)
        
        # Get input name from the model
        input_name = self.model_session.get_inputs()[0].name
        
        # Run inference with high confidence
        predictions = self.model_session.run(None, {input_name: input_tensor})
        
        # Post-process detections
        type_detections = self.postprocess_detections(
            predictions[0], 
            original_shape, 
            scale, 
            pad, 
            confidence_threshold
        )
        
        # Check if we have enough high-confidence detections
        detected = len(type_detections[0]) > 0 and len(type_detections[1]) > 0
        
        # If not enough high-confidence detections, try with lower threshold
        if not detected:
            type_detections_low = self.postprocess_detections(
                predictions[0], 
                original_shape, 
                scale, 
                pad, 
                confidence_threshold=0.01  # Very low confidence
            )
            
            # Merge low confidence detections
            for class_id in type_detections:
                type_detections[class_id].extend(type_detections_low[class_id])
        
        # Sort detections by confidence for each type
        for class_id in type_detections:
            type_detections[class_id].sort(key=lambda x: x[1], reverse=True)
        
        # Best guess strategy: ensure every type has at least one bounding box
        first_type_found = []
        second_type_boxes = []
        
        # Get the best detection for each type, or use fallback
        if type_detections[0]:
            first_type_found = type_detections[0][0][0]  # Best detection for type 0 (names)
        else:
            # Fallback: create a default bounding box in upper-left area
            first_type_found = [0, 0, width * 0.25, height * 0.25]
        
        if type_detections[1]:
            second_type_boxes = type_detections[1][0][0]  # Best detection for type 1 (scores)
        else:
            # Fallback: create a default bounding box in upper-right area
            second_type_boxes = [width * 0.75, 0, width, height * 0.25]
        
        return first_type_found, second_type_boxes

# Keep standalone functions for backward compatibility if needed, or just point to the class
def process_detections(image, model_session, confidence_threshold=0.6, **kwargs):
    detector = OnnxPlayerDetector(model_session)
    return detector.process_detections(image, confidence_threshold, **kwargs)



# Example usage
if __name__ == "__main__":
    import onnxruntime as ort
    
    image = "dataset/20241112_093434_image.png"
    model_path = "mk8dx_table_reader/models/detectPlayers.onnx"
    
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model_session = ort.InferenceSession(model_path, providers=providers)
    
    first_found, second_boxes = process_detections(image, model_session)
    
    print(f"Type 0 (names) bounding box: {first_found}")
    print(f"Type 1 (scores) bounding box: {second_boxes}")
    
    # With best guess strategy, all types will always have bounding boxes
    print("All object types now have guaranteed bounding boxes!")
