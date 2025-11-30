"""
ONNX Runtime version of fullreader.py
This file uses ONNX Runtime instead of TensorFlow/Keras for production deployment.
No TensorFlow dependencies required!
Updated to use ONNX-based EasyOCR for lightweight inference.
"""
import cv2
import os
from pathlib import Path
import sys
relative_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(relative_dir)
import kerasUseModelScores as KS  # Now using ONNX version
from detectors import yoloUseModelTable as YTable
import detectors.yoloUseModelPlayers as YPlayers
# import detectors.base_table_detector as BTD
# from detectors import base_player_detector as BPD
import error_handling as er
import exceptions.LoadImgException as LoadImgException
import mk8dx_table_reader.exceptions.PlayersNotFoundException as PlayersNotFoundException
import mk8dx_table_reader.exceptions.TableNotFoundException as TableNotFoundException
try:
    import easyocr
except ImportError:
    import torchfree_ocr as easyocr
import warnings
import time
import PIL
import numpy as np
from ultralytics import YOLO

class Fullreader:
    """
    Fullreader class to handle the full OCR process for a given image.
    NOTE: As for now the program only works with grayscale images.
       
    Attributes:
        readerEasyOcr (easyocr.Reader or easyocr_onnx.Reader): EasyOCR reader instance for text recognition. 
            Uses ONNX-based implementation when available for lightweight inference without PyTorch.
            Used to read player and team names from cropped images of players.
            
        modelScore (onnxruntime.InferenceSession): ONNX Runtime session for recognizing scores.
            Used to recognize scores from cropped images of players.
        
        modelPosition: Currently NULL because easyocr worked just fine
        
        modelTablePath (str): Path to the custom YOLO model for table detection.
            Used to recognize the table area in the image, and the number of players in that table if there is one
        
        playerDetector (BasePlayerDetector): Instance of a player detector (ONNX or YOLO).
            Used to recognize player areas in the image, such as their positions, tags, names, and scores.
    """
    def __init__(self):
        # Initialize the EasyOCR reader for English language
        # Will use ONNX version if available, otherwise falls back to PyTorch version
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.readerEasyOcr = easyocr.Reader(['en'])
        # Load ONNX model for score recognition
        if relative_dir:
            onnx_model_path = relative_dir + '/models/number_recognition_model.onnx'
            self.modelScore = KS.load_model_for_inference(onnx_model_path)
            
            # Initialize Table Detector using YOLO with ONNX backend
            self.modelTable = YOLO(relative_dir + '/models/detectTable.pt')
            
            # Initialize Player Detector using YOLO with ONNX backend
            self.modelPlayers = YOLO(relative_dir + '/models/detectPlayers.pt')
        else:
            onnx_model_path = 'mk8dx_table_reader/models/number_recognition_model.onnx'
            self.modelScore = KS.load_model_for_inference(onnx_model_path)
            
            # Initialize Table Detector using YOLO with ONNX backend
            self.modelTable = YOLO('mk8dx_table_reader/models/detectTable.onnx')
            
            # Initialize Player Detector using YOLO with ONNX backend
            self.modelPlayers = YOLO('mk8dx_table_reader/models/detectPlayers.onnx')

    def fullOCR(self, img, score_error_exceptions=False, **kwargs):
        """_summary_

        :param img: PIL Image object to be processed.
        :type img: PIL.Image.Image
        :param score_error_exceptions: Whether to enable error handling for scores (for now), defaults to False. If True, will raise DetectionError if the detection went wrong.

        :raises ValueError: If the image cannot be processed, a ValueError is raised.
        """
        # Keep the original PIL Image for cropping operations
        img_pil = img
        
        # Convert PIL Image to RGB (in case it's RGBA or other format)
        try:
            img_rgb = img.convert('RGB')
        except Exception as e:
            raise ValueError(f"Image conversion failed: {e}")
        # Convert to numpy array (OpenCV format is BGR, so we need to convert)
        img_np = np.array(img_rgb)
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Convert the image to grayscale (this was the original failing line)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale back to PIL Image for YOLO processing
        img_gray_pil = PIL.Image.fromarray(img_gray)
                
        # Use grayscale image for YOLO table detection
        first_found, second_boxes = YTable.process_detections(img_gray_pil, self.modelTable)
        if first_found is None:
            raise TableNotFoundException.TableNotFoundException()
        if not second_boxes or len(second_boxes) == 0:
            raise PlayersNotFoundException.PlayersNotFoundException()
        box = first_found
        
        player_names = []
        player_scores = []
        # order second_boxes in the list from top to bottom according to their y coordinate
        second_boxes = sorted(second_boxes, key=lambda b: b[1])
        for i, box in enumerate(second_boxes):
            playerIMG = img_gray_pil.crop(box)
            name, score = YPlayers.process_detections(playerIMG, self.modelPlayers)
            
            name_results = self.readerEasyOcr.readtext(np.array(playerIMG.crop(name)))
            if name_results:
                player_names.append(name_results[0][1])
            else: # if easyOCR fails to recognize the name
                player_names.append("Error")  
            
            try:
                score = self.getScores(playerIMG.crop(score))
            except Exception:  
                None
            if score is not None: 
                player_scores.append(score)
            else:
                player_scores.append("Error")
                
        for i in range (len(player_scores)):
            player_scores[i] = player_scores[i]
        if score_error_exceptions:
            er.error_handling_scores(player_scores)
        return player_names, player_scores

    def getScores(self, tableIMG):
        """
        :param tableIMG: Cropped score image of one player (grayscale needed).
        :type tableIMG: PIL.Image.Image
        
        :return: The recognized single score from the cropped player image.
        :rtype: str
        """
        # Use the ONNX model to predict the scores

        scores = KS.recognize_number_from_image(self.modelScore, tableIMG)
        return scores



if __name__ == "__main__":
    # Example usage of Fullreader
    
    fullreader = Fullreader()
    folder_path = "dataset/endScreenData"  # Replace with your folder path
    # Get all image files from the folder
    image_extensions = {'.png'}
    image_files = [f for f in os.listdir(folder_path) 
                   if Path(f).suffix.lower() in image_extensions]
    num_img_files = len(image_files)
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_file}, Image {i+1}/{num_img_files}")
        # Load the image using PIL
        img = PIL.Image.open(image_path)
        try :
            extractedTableString= fullreader.fullOCR(img, score_error_exceptions=True)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        if extractedTableString is not None:
            names, scores = extractedTableString
            print(f"Extracted Names: {names}")
            print(f"Extracted Scores: {scores}")