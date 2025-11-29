from abc import ABC, abstractmethod

class BaseTableDetector(ABC):
    """
    Abstract base class for table detection models.
    """
    
    @abstractmethod
    def process_detections(self, image, confidence_threshold=0.8):
        """
        Process detections on the given image.
        
        Args:
            image: Input image (str path, PIL Image, or numpy array)
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            tuple: (first_type_found, second_type_boxes)
                - first_type_found: Bounding box for the table [x1, y1, x2, y2]
                - second_type_boxes: List of bounding boxes for player rows [[x1, y1, x2, y2], ...]
        """
        pass
