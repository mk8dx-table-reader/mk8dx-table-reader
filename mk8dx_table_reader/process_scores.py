from typing import List, Tuple
import cv2
import numpy as np

class ScoreProcessor:

    """
    Class for processing LCD display images with cropping and segmentation.
    """
    
    def __init__(self, min_width_threshold: int = 10):
        """
        Initialize the ScoreProcessor.
        
        Args:
            min_width_threshold: Minimum width (in pixels) for a valid digit segment
        """
        self.min_width_threshold = min_width_threshold

    def preprocess_image(self, img):
        """Preprocess an image file for prediction - updated for CTC model"""
        
        img = np.array(img)
        
        avgColor = np.mean(img, axis=(0,1))

        if float(avgColor) > 175:
            # invert colors
            img = 255 - img
            lower_bound = np.array([0, 0, 0])     
            upper_bound = np.array([120, 120, 120])
            # img = cv2.addWeighted(img, 2, np.zeros(img.shape, img.dtype), 0,25)
            # img = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0,25)
        else:
            # lower bound and upper bound for White color
            lower_bound = np.array([0, 0, 0])     
            upper_bound = np.array([190, 190, 190])
            
        # cv2.imshow("Debug Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # Brightness normalization - normalize to target brightness level
        # target_brightness = 160.0  # Target average brightness (0-255 scale)
        # current_brightness = np.mean(img)
        
        # if current_brightness > 0:  # Avoid division by zero
        #     brightness_factor = target_brightness / current_brightness
        #     img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # mask = cv2.bitwise_not(mask)
        # mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # segmented_img = cv2.bitwise_and(img, img, mask=mask)
        # contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask =  cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)

        # cv2.imshow("Debug Image", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = mask

    

        
        return img
    def crop_image(self, img: np.ndarray) -> np.ndarray:
        # # Create binary image (threshold to ensure black=0, white=255)
        # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # # Calculate Y-profile: count black pixels (0) in each row
        # # Invert to count black pixels: 255 - pixel_value, then sum
        # y_profile = np.sum(255 - binary, axis=1)
        
        # Find rows with black pixels (non-zero values in profile)
        rows_with_content = np.where(img > 0)[0]
        
        if len(rows_with_content) == 0:
            # No content found, return original image
            return img
        
        # Get top and bottom bounds
        top = rows_with_content[0]
        bottom = rows_with_content[-1] + 1  # +1 for inclusive slicing
        
        # Crop the image vertically
        cropped = img[top:bottom, :]
        return cropped
    
    def segment_digits(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment the image into individual digits using X-profile analysis.
        
        The X-profile counts black pixels along each column. Each contiguous
        block of columns with black pixels represents a digit.
        
        Args:
            image: Binary image (grayscale or binary) where digits are black (0)
                   and background is white (255)
        
        Returns:
            List of segmented digit images, ordered left to right by x-coordinate
        """
        image = np.array(image)
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate X-profile: count black pixels (0) in each column
        x_profile = np.sum(255 - binary, axis=0)
        
        # Find local minima (valleys) in the profile to identify gaps between digits
        # A valley is where the profile drops significantly
        valleys = []
        threshold = np.mean(x_profile[x_profile > 0]) * 0.15  # 15% of average
        
        for x in range(1, len(x_profile) - 1):
            # Check if this is a local minimum and below threshold
            if x_profile[x] < threshold:
                if x_profile[x] <= x_profile[x-1] and x_profile[x] <= x_profile[x+1]:
                    valleys.append(x)
        
        # If valleys are too close together, keep only the deepest one
        filtered_valleys = []
        min_gap = 5  # Minimum pixels between valleys
        
        if valleys:
            current_group = [valleys[0]]
            for v in valleys[1:]:
                if v - current_group[-1] < min_gap:
                    current_group.append(v)
                else:
                    # Find deepest valley in current group
                    deepest = min(current_group, key=lambda x: x_profile[x])
                    filtered_valleys.append(deepest)
                    current_group = [v]
            # Add last group
            deepest = min(current_group, key=lambda x: x_profile[x])
            filtered_valleys.append(deepest)
        
        # Create segments based on valleys
        segments = []
        
        # Find first and last content columns
        content_cols = np.where(x_profile > 0)[0]
        if len(content_cols) == 0:
            return []
        
        start_col = content_cols[0]
        end_col = content_cols[-1] + 1
        
        if not filtered_valleys:
            # No valleys found, treat as single segment
            if end_col - start_col >= self.min_width_threshold:
                segments.append((start_col, end_col))
        else:
            # Create segments between valleys
            prev_split = start_col
            for valley in filtered_valleys:
                if valley - prev_split >= self.min_width_threshold:
                    segments.append((prev_split, valley))
                prev_split = valley
            
            # Add final segment
            if end_col - prev_split >= self.min_width_threshold:
                segments.append((prev_split, end_col))
        
        # Extract digit images from segments
        digit_images = []
        for start_x, end_x in segments:
            digit = image[:, start_x:end_x]
            digit_images.append(digit)
        
        return digit_images
    
    
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Complete processing pipeline: crop then segment.
        
        Args:
            image: Input image with LCD digits
        
        Returns:
            Tuple of (cropped_image, list_of_digit_segments)
        """
        processed_img = self.preprocess_image(image)
        # Step 6: Segment using X-profile
        digits = self.segment_digits(processed_img)
        # Step 5: Crop using Y-profile
        cropped = []
        for i, digit in enumerate(digits):
            cropped.append(self.crop_image(digit))
        
        # number = self.recognize_digits(digits)
        
        # visualize_results(image, processed_img, digits)

        return processed_img, cropped


def visualize_results(original: np.ndarray, cropped: np.ndarray, 
                     digits: List[np.ndarray], number: int = None, output_path: str = None):
    """
    Visualize the cropping and segmentation results.
    
    Args:
        original: Original input image
        cropped: Cropped image
        digits: List of segmented digits
        number: Recognized number as string
        output_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    num_plots = 2 + len(digits)
    fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 4))
    
    # Add main title with recognized number
    fig.suptitle(f'Recognized Number: {number}', fontsize=16, fontweight='bold')
    
    # Show original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show cropped
    axes[1].imshow(cropped, cmap='gray')
    axes[1].set_title('Cropped')
    axes[1].axis('off')
    
    # Show each digit
    for i, digit in enumerate(digits):
        axes[2 + i].imshow(digit, cmap='gray')
        axes[2 + i].set_title(f'Digit {i+1}')
        axes[2 + i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Specify the folder containing images
    input_folder = "/home/ujuj/gitFolders/shortcat.tips/dataset/croppedScores"
    output_folder = "/home/ujuj/gitFolders/shortcat.tips/dataset/segmentation_results"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Get all image files from the folder
    image_files = []
    if os.path.isdir(input_folder):
        for file in os.listdir(input_folder):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        print(f"No images found in {input_folder}")
    else:
        print(f"Found {len(image_files)} images to process\n")
        
        # Create processor with minimum width threshold
        processor = ScoreProcessor(min_width_threshold=5)
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"Processing [{idx}/{len(image_files)}]: {os.path.basename(image_path)}")
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"  Error: Could not load image")
                continue
            
            try:
                # Process the image
                cropped_image, digit_segments = processor.process_image(image)
                
                print(f"  Original shape: {image.shape}")
                print(f"  Cropped shape: {cropped_image.shape}")
                print(f"  Digits found: {len(digit_segments)}")
                
                # Create output filename
                base_name = Path(image_path).stem
                output_path = os.path.join(output_folder, f"{base_name}_segmentation.png")
                
                # Visualize results
                visualize_results(image, cropped_image, digit_segments, output_path)
                
            except Exception as e:
                print(f"  Error processing image: {str(e)}")
            
            print()  # Empty line for readability