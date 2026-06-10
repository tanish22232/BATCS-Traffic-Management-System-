
import cv2
import easyocr
import numpy as np
import re
import os
from collections import deque

class RedLightViolationSystem:
    def __init__(self):
        print("Initializing Red Light Violation Detection System...")
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
        
        # Store detected plates
        self.detected_plates = deque(maxlen=100)
        self.frame_count = 0
        
        # Output file
        self.output_file = "violation_plates.txt"
        
        # Create output file
        with open(self.output_file, 'w') as f:
            f.write("RED LIGHT VIOLATION - DETECTED LICENSE PLATES\n")
            f.write("=" * 50 + "\n\n")
    
    def detect_plates(self, frame):
        """Simple license plate detection using contour analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(blurred, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plates = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            if len(approx) == 4: # Rectangular shape
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate-like aspect ratio
                if 2.0 < aspect_ratio < 5.0 and w > 80 and h > 30:
                    plates.append((x, y, w, h))
        
        return plates
    
    def enhance_plate_image(self, plate_image):
        """Enhance plate image for better OCR"""
        if plate_image is None or plate_image.size == 0:
            return None
            
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Resize if too small
        height, width = gray.shape
        if width < 100 or height < 30:
            scale = max(200 / width, 60 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(denoised)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def read_license_plate_text(self, plate_image):
        """Read text from license plate"""
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        
        # Enhance the plate image
        enhanced_plate = self.enhance_plate_image(plate_image)
        
        if enhanced_plate is None:
            return "", 0.0
        
        # Use EasyOCR to read text
        best_text = ""
        best_confidence = 0.0
        
        try:
            results = self.reader.readtext(enhanced_plate, detail=1, paragraph=False, text_threshold=0.4)
            
            for bbox, text, confidence in results:
                if confidence > best_confidence and len(text) >= 4:
                    best_text = text.upper()
                    best_confidence = confidence
        except Exception as e:
            print(f"OCR Error: {e}")
        
        # Clean the text
        cleaned_text = re.sub(r'[^A-Z0-9]', '', best_text)
        
        # Common OCR error corrections
        corrections = {
            'I': '1', 'O': '0', 'S': '5', 'Z': '2', 
            'B': '8', 'G': '6', 'D': '0', 'T': '7',
            ' ': '', '-': '', '.': '', '_': ''
        }
        
        for wrong, right in corrections.items():
            cleaned_text = cleaned_text.replace(wrong, right)
        
        return cleaned_text, best_confidence
    
    def is_valid_plate(self, text):
        """Check if text could be a license plate"""
        if not text or len(text) < 5:
            return False
        
        # Should have both letters and numbers
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        
        return letter_count >= 2 and digit_count >= 2
    
    def save_to_file(self, plate_text, timestamp):
        """Save detected plate to text file"""
        with open(self.output_file, 'a') as f:
            f.write(f"{timestamp}: {plate_text}\n")
    
    def process_video(self, video_path):
        """Process video and detect license plates"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' does not exist")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        print("Processing video for red light violations...")
        print("Press 'q' to quit early")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
                
            self.frame_count += 1
            
            # Process every 5th frame for efficiency
            if self.frame_count % 5 != 0:
                continue
            
            # Detect license plates
            plates = self.detect_plates(frame)
            
            for x, y, w, h in plates:
                # Add padding
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                x, y = max(0, x - pad_x), max(0, y - pad_y)
                w, h = min(frame.shape[1] - x, w + 2 * pad_x), min(frame.shape[0] - y, h + 2 * pad_y)
                
                # Crop the license plate region
                cropped_plate = frame[y:y+h, x:x+w]
                
                if cropped_plate.size == 0:
                    continue
                
                # Read text from license plate
                plate_text, confidence = self.read_license_plate_text(cropped_plate)
                
                # Check if this is a valid plate
                if plate_text and self.is_valid_plate(plate_text) and confidence > 0.4:
                    timestamp = f"Frame {self.frame_count}"
                    
                    # Check if this plate is new
                    is_new = True
                    for existing_plate, _ in self.detected_plates:
                        if plate_text == existing_plate:
                            is_new = False
                            break
                    
                    if is_new:
                        self.detected_plates.append((plate_text, timestamp))
                        self.save_to_file(plate_text, timestamp)
                        print(f"Detected violation: {plate_text} (confidence: {confidence:.2f})")
            
            # Display progress
            if self.frame_count % 50 == 0:
                print(f"Processed {self.frame_count}/{total_frames} frames")
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Final report
        print(f"\nProcessing completed! Detected {len(self.detected_plates)} potential violations.")
        print(f"Results saved to: {self.output_file}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        return True

# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = RedLightViolationSystem()
    
    # Specify your video path
    video_path = r"C:\Users\sankalp.DESKTOP-1RPOQ63\Documents\traffic-violation-detection\input.mp4"
    
    # Process the video
    success = system.process_video(video_path)
    
    if success:
        print("\nRed light violation detection completed!")
        print("Check 'violation_plates.txt' for the results.")
    else:
        print("Failed to process video")
