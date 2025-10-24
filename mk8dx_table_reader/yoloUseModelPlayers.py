import cv2
from ultralytics import YOLO
import PIL


def process_detections(image, model, confidence_threshold = 0.6, **kwargs):
    # Initialize YOLO model

    # Get image dimensions
    if isinstance(image, str):
        import cv2
        img = cv2.imread(image)
        height, width = img.shape[:2]
    elif hasattr(image, 'size'):  # PIL Image
        width, height = image.size
    else:
        height, width = image.shape[:2]

    # Run YOLO detection with high confidence first
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )
    detected = len(results[0].boxes) > 3    

    # Run YOLO detection with very low confidence for best guess strategy
    results_low_conf = model.predict(
        source=image,
        conf=0.01,  # Very low confidence to catch everything
        verbose=False
    )
    
    # Initialize storage for each type
    type_detections = {0: [], 1: []}

    # Process high confidence detections first
    for r in results[0].boxes:
        class_id = int(r.cls[0])
        box = r.xyxy[0].cpu().numpy()
        confidence = float(r.conf[0])
        if class_id in type_detections:
            type_detections[class_id].append((box, confidence))
    
    # Process low confidence detections for missing types
    for r in results_low_conf[0].boxes:
        class_id = int(r.cls[0])
        box = r.xyxy[0].cpu().numpy()
        confidence = float(r.conf[0])
        if class_id in type_detections:
            type_detections[class_id].append((box, confidence))
    
    # Sort detections by confidence for each type
    for class_id in type_detections:
        type_detections[class_id].sort(key=lambda x: x[1], reverse=True)
    
    # Best guess strategy: ensure every type has at least one bounding box
    first_type_found = []
    second_type_boxes = []

    
    # Get the best detection for each type, or use fallback
    if type_detections[0]:
        first_type_found = type_detections[0][0][0]  # Best detection for type 0
    else:
        # Fallback: create a default bounding box in upper-left area
        first_type_found = [0, 0, width*0.25, height*0.25]
    
    if type_detections[1]:
        second_type_boxes = type_detections[1][0][0]  # Best detection for type 1
    else:
        # Fallback: create a default bounding box in upper-right area
        second_type_boxes = [width*0.75, 0, width, height*0.25]
    
    
    return first_type_found, second_type_boxes

# Example usage
if __name__ == "__main__":
    image = "dataset/20241112_093434_image.png"
    model_path = "runs/detect/good3Gray/weights/best.pt"
    model = YOLO(model_path)
    first_found, second_boxes = process_detections(image, model)

    print(f"Type 0 (first) bounding box: {first_found}")
    print(f"Type 1 (second) bounding box: {second_boxes}")

    
    # With best guess strategy, all types will always have bounding boxes
    print("All object types now have guaranteed bounding boxes!")