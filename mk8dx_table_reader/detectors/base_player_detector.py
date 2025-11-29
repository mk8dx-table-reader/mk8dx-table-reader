from abc import ABC, abstractmethod

class BasePlayerDetector(ABC):
    """
    Abstract base class for player detection models.
    """
    
    @abstractmethod
    def process_detections(self, image, confidence_threshold=0.6):
        """
        Process detections on the given image.
        
        Args:
            image: Input image (str path, PIL Image, or numpy array)
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            tuple: (first_type_found, second_type_boxes)
                - first_type_found: Bounding box for player name [x1, y1, x2, y2]
                - second_type_boxes: Bounding box for player score [x1, y1, x2, y2]
        """
        pass
