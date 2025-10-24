import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import PIL

# Define the character mapping (same as during training)
CHARACTERS = '0123456789'
char_to_idx = {char: idx for idx, char in enumerate(CHARACTERS)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS) + 1  # 10 characters + 1 for CTC blank = 11

class CTCLayer(layers.Layer):
    """CTC layer for model loading - simplified version"""
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def call(self, inputs):
        # During inference, just return the predictions
        if len(inputs) == 1:
            return inputs[0]
        # During training (not used in inference)
        return inputs[1]

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
def decode_predictions(pred):
    """Convert model output to text predictions using CTC decode"""
    batch_size = pred.shape[0]
    
    # Use tf.nn.ctc_greedy_decoder instead of tf.keras.backend.ctc_decode
    # tf.nn.ctc_greedy_decoder expects (time_major=True): (time_steps, batch_size, num_classes)
    inputs_tm = tf.transpose(pred, [1, 0, 2])  # (time_steps, batch_size, num_classes)
    sequence_length = tf.constant([pred.shape[1]] * batch_size)  # All sequences have same length
    
    # Use CTC greedy decoder with explicit blank index
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs_tm,
        sequence_length=sequence_length,
        blank_index=10  # CTC blank at index 10 (NUM_CLASSES - 1)
    )
    
    # Convert sparse tensor to dense
    dense_decoded = tf.sparse.to_dense(decoded[0]).numpy()
    
    # Convert indices to characters
    predicted_labels = []
    for i in range(batch_size):
        text = ''
        for idx in dense_decoded[i]:
            if idx >= 0 and idx < len(CHARACTERS):  # Valid character indices (0-9)
                text += CHARACTERS[idx]
        predicted_labels.append(text)
        
    return predicted_labels

def load_model_for_inference(model_path):
    """Load the CTC model for inference"""
    try:
        # Try loading as inference model first (should work with the new architecture)
        model = keras.models.load_model(model_path, compile=False, 
                                       custom_objects={'CTCLayer': CTCLayer})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to build model architecture manually...")
        
        # If loading fails, build the inference model manually
        return build_inference_model()

def build_inference_model():
    """Build inference model matching the training architecture"""
    input_shape = (64, 80, 1)
    
    # Input layer for images
    input_img = keras.Input(shape=input_shape, name='image')
    
    # CNN feature extraction (must match training exactly)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                           kernel_initializer='he_normal')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)  # Only downsample height
    
    # Reshape for RNN - create sequence from width dimension
    new_shape = (input_shape[1] // 4, (input_shape[0] // 8) * 128)  # (width_steps, features)
    x = keras.layers.Reshape(new_shape)(x)
    
    # LSTM layers
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, 
                                                     recurrent_dropout=0.1,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal'))(x)
    
    # Dense layer before CTC
    x = keras.layers.Dense(NUM_CLASSES, kernel_initializer='glorot_uniform', dtype='float32')(x)
    
    model = keras.Model(inputs=input_img, outputs=x)
    return model

def recognize_number_from_image(model, image):
    """Recognize number in a given image using CTC model"""
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
        # If it's a file path
        opencvImage = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        # Assume it's already a numpy array
        opencvImage = image
        if len(opencvImage.shape) == 3:
            if opencvImage.shape[2] == 3:  # BGR
                opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
            elif opencvImage.shape[2] == 4:  # BGRA
                opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGRA2GRAY)
            else:
                opencvImage = opencvImage[:, :, 0]  # Take first channel

    processed_img = preprocess_image(opencvImage)

    if processed_img is None:
        return "Error processing image"
    
    # Make prediction
    pred = model.predict(processed_img, verbose=False)
    
    # Decode the prediction
    result = decode_predictions(pred)
    
    return result[0] if result else ""

# Example usage
if __name__ == "__main__":
    
    # Load the saved model
    print("Loading CTC model...")
    model = load_model_for_inference('number_recognition_model.h5')
    print("Model loaded successfully!")
    
    test_dir = "dataset/croppedScores/"
    if os.path.exists(test_dir):
        for img_file in os.listdir(test_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, img_file)
                try:
                    img = PIL.Image.open(img_path)
                    number = recognize_number_from_image(model, img)
                    print(f"Image {img_file}: {number}")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    else:
        print(f"Test directory {test_dir} not found.")
        
        # Test with a synthetic image if no test directory
        print("Testing with synthetic image...")
        test_img = np.random.randint(0, 255, (64, 80), dtype=np.uint8)
        result = recognize_number_from_image(model, test_img)
        print(f"Synthetic image result: {result}")