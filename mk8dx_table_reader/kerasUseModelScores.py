"""
ONNX Runtime version of kerasUseModelScores.py
This file replaces TensorFlow/Keras inference with ONNX Runtime for production use.
No TensorFlow dependencies required!
"""
import numpy as np
import cv2
import os
import PIL
import onnxruntime as ort

# Define the character mapping (same as during training)
CHARACTERS = '0123456789'
char_to_idx = {char: idx for idx, char in enumerate(CHARACTERS)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS) + 1  # 10 characters + 1 for CTC blank = 11


def preprocess_image(img, target_height=64):
    """Preprocess an image file for prediction - updated for CTC model"""
    
    # Calculate new width to maintain aspect ratio
    h, w = img.shape
    target_width = int(w * (target_height / h))
    
    # Resize image
    img_resized = cv2.resize(img, (target_width, target_height))

    # Normalize pixel values
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Pad to fixed width if necessary (use same max_width as in training)
    max_width = 80  # Must match training
    if img_normalized.shape[1] > max_width:
        img_normalized = img_normalized[:, :max_width]
    else:
        padded_img = np.zeros((target_height, max_width), dtype=np.float32)
        padded_img[:, :img_normalized.shape[1]] = img_normalized
        img_normalized = padded_img
    
    # Add batch and channel dimensions for model input (must match training exactly!)
    img_normalized = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)
    
    return img_normalized


def decode_predictions_numpy(pred):
    """
    Convert model output to text predictions using CTC decode (NumPy implementation).
    This replaces TensorFlow's CTC decoder with a pure NumPy greedy decoder.
    
    Args:
        pred: Model predictions of shape (batch_size, time_steps, num_classes)
    
    Returns:
        List of predicted text strings
    """
    batch_size = pred.shape[0]
    predicted_labels = []
    
    for i in range(batch_size):
        # Get the most likely class for each timestep (greedy decoding)
        sequence = np.argmax(pred[i], axis=-1)  # Shape: (time_steps,)
        
        # CTC decoding: remove consecutive duplicates and blank tokens
        decoded = []
        previous_token = None
        
        for token in sequence:
            # Skip blank token (index 10)
            if token == 10:  # CTC blank
                previous_token = None
                continue
            
            # Skip consecutive duplicates
            if token != previous_token:
                if 0 <= token < len(CHARACTERS):  # Valid character
                    decoded.append(CHARACTERS[token])
                previous_token = token
        
        predicted_labels.append(''.join(decoded))
    
    return predicted_labels


def load_model_for_inference(model_path):
    """
    Load the ONNX model for inference using ONNX Runtime.
    
    Args:
        model_path: Path to the .onnx model file
    
    Returns:
        ONNX Runtime inference session
    """
    try:
        # Create ONNX Runtime session
        # Use CPUExecutionProvider for CPU inference or CUDAExecutionProvider for GPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        print(f"ONNX model loaded successfully from: {model_path}")
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Using providers: {session.get_providers()}")
        
        # Print input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"Input name: {input_info.name}, shape: {input_info.shape}, dtype: {input_info.type}")
        print(f"Output name: {output_info.name}, shape: {output_info.shape}, dtype: {output_info.type}")
        
        return session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        raise


def recognize_number_from_image(model_session, image):
    """
    Recognize number in a given image using ONNX Runtime.
    
    Args:
        model_session: ONNX Runtime inference session
        image: PIL Image or OpenCV image (grayscale)
    
    Returns:
        Predicted number as string
    """
    # Convert PIL image to OpenCV format if needed
    if isinstance(image, PIL.Image.Image):
        # Convert PIL to numpy array first
        numpy_image = np.array(image)
        # Handle different image modes
        if len(numpy_image.shape) == 3:
            if numpy_image.shape[2] == 3:  # RGB
                opencvImage = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            elif numpy_image.shape[2] == 4:  # RGBA
                opencvImage = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2GRAY)
            else:
                opencvImage = numpy_image[:, :, 0]  # Take first channel
        else:
            opencvImage = numpy_image  # Already grayscale
    elif isinstance(image, str):
        # Load from file path
        opencvImage = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if opencvImage is None:
            raise ValueError(f"Could not load image from path: {image}")
    else:
        # Assume it's already an OpenCV image
        opencvImage = image
        if len(opencvImage.shape) == 3:
            opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image
    preprocessed = preprocess_image(opencvImage)
    
    # Run inference
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    # ONNX Runtime expects numpy arrays
    predictions = model_session.run([output_name], {input_name: preprocessed})[0]
    
    # Decode predictions using numpy-based CTC decoder
    decoded_texts = decode_predictions_numpy(predictions)
    
    return decoded_texts[0] if decoded_texts else ""


# Maintain backward compatibility - these functions can be called from existing code
def build_inference_model():
    """Deprecated: No longer needed with ONNX Runtime"""
    raise NotImplementedError(
        "This function is deprecated when using ONNX Runtime. "
        "Use load_model_for_inference() instead with an ONNX model path."
    )


if __name__ == "__main__":
    # Example usage
    print("ONNX Runtime Score Recognition Module")
    print("=" * 50)
    
    # Check if ONNX model exists
    onnx_model_path = "mk8dx_table_reader/models/number_recognition_model.onnx"
    
    if not os.path.exists(onnx_model_path):
        print(f"ERROR: ONNX model not found at: {onnx_model_path}")
        print("Please run the conversion script first to create the ONNX model.")
    else:
        # Load model
        session = load_model_for_inference(onnx_model_path)
        
        # Test with a sample image if available
        test_image_path = "dataset/croppedScores/1.png"
        if os.path.exists(test_image_path):
            print(f"\nTesting with image: {test_image_path}")
            result = recognize_number_from_image(session, test_image_path)
            print(f"Recognized number: {result}")
        else:
            print(f"\nNo test image found at: {test_image_path}")
            print("Model loaded successfully and ready for inference!")
