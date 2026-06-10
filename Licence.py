import cv2
import easyocr
import numpy as np
import re
import time
import os
import urllib.request
from collections import deque

class HaarLicensePlateRecognizer:
    def __init__(self):
        print("Initializing License Plate Recognition System...")
        
        # Download or load Haar cascade
        haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
        self.haar_path = "haarcascade_russian_plate_number.xml"
        
        if not os.path.exists(self.haar_path):
            print("Downloading Haar cascade file...")
            try:
                urllib.request.urlretrieve(haar_url, self.haar_path)
                print("Download completed!")
            except:
                print("Failed to download Haar cascade. Using alternative method.")
                self.haar_path = None
        
        # Load Haar cascade
        if self.haar_path and os.path.exists(self.haar_path):
            self.plate_cascade = cv2.CascadeClassifier(self.haar_path)
        else:
            print("Using built-in plate detection without Haar cascade")
            self.plate_cascade = None
        
        # Initialize EasyOCR reader
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        
        # Tracking variables
        self.detected_plates = deque(maxlen=50)
        self.frame_count = 0
        self.last_plate_time = 0
        self.plate_cooldown = 2  # seconds between plate detections
        
        # Indian license plate patterns
        self.plate_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$',  # KA01AB1234
            r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$',  # Variants
            r'^\d{1,2}[A-Z]{1,3}\d{1,4}$',  # Numbers, letters, numbers
        ]
        
        # Video writer
        self.video_writer = None
    
    def detect_plates_haar(self, frame):
        """Detect license plates using Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.plate_cascade:
            # Use Haar cascade if available
            plates = self.plate_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(80, 30),
                maxSize=(400, 120)
            )
            return [(x, y, w, h, 0.8) for (x, y, w, h) in plates]  # Add confidence
        else:
            # Fallback to traditional method
            return self.detect_plates_traditional(gray)
    
    def detect_plates_traditional(self, gray):
        """Traditional license plate detection as fallback"""
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
            
            if len(approx) == 4:  # Rectangular shape
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate-like aspect ratio
                if 2.0 < aspect_ratio < 5.0 and w > 80 and h > 30:
                    plates.append((x, y, w, h, 0.7))  # Add confidence
        
        return plates
    
    def enhance_plate_image(self, plate_image):
        """Enhanced plate image preprocessing for better OCR"""
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
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.erode(cleaned, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        
        return cleaned
    
    def read_license_plate_text(self, plate_image):
        """Read text from license plate with improved OCR"""
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        
        # Enhance the plate image
        enhanced_plate = self.enhance_plate_image(plate_image)
        
        if enhanced_plate is None:
            return "", 0.0
        
        # Try multiple OCR configurations
        best_text = ""
        best_confidence = 0.0
        
        try:
            # Try with different parameter combinations
            results = self.reader.readtext(enhanced_plate, detail=1, paragraph=False, 
                                         batch_size=4, text_threshold=0.3)
            
            for bbox, text, confidence in results:
                if confidence > best_confidence and len(text) >= 3:
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
    
    def is_valid_indian_plate(self, text):
        """Check if text matches Indian license plate patterns"""
        if not text or len(text) < 5:
            return False
        
        # Check against known Indian plate patterns
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        
        # Additional checks for Indian plates
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        
        return letter_count >= 2 and digit_count >= 2 and len(text) >= 5
    
    def initialize_video_writer(self, frame, output_path):
        """Initialize video writer for output"""
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def process_frame(self, frame):
        """Process a single video frame"""
        self.frame_count += 1
        
        # Detect license plates
        plates = self.detect_plates_haar(frame)
        
        best_plate = None
        best_text = ""
        best_confidence = 0.0
        
        current_time = time.time()
        
        for x, y, w, h, confidence in plates:
            # Add padding
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            x, y = max(0, x - pad_x), max(0, y - pad_y)
            w, h = min(frame.shape[1] - x, w + 2 * pad_x), min(frame.shape[0] - y, h + 2 * pad_y)
            
            # Crop the license plate region
            cropped_plate = frame[y:y+h, x:x+w]
            
            if cropped_plate.size == 0:
                continue
            
            # Read text from license plate
            plate_text, text_confidence = self.read_license_plate_text(cropped_plate)
            
            # Check if this is a valid Indian plate with good confidence
            if (plate_text and self.is_valid_indian_plate(plate_text) and 
                text_confidence > best_confidence and
                current_time - self.last_plate_time > self.plate_cooldown):
                
                best_plate = (x, y, w, h)
                best_text = plate_text
                best_confidence = text_confidence
        
        if best_plate and best_text:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.detected_plates.append((best_text, timestamp, best_confidence))
            self.last_plate_time = current_time
            print(f"Frame {self.frame_count}: Detected plate '{best_text}' with confidence {best_confidence:.2f}")
            return frame, best_text, best_plate
        
        return frame, None, None
    
    def draw_results(self, frame, plate_text, bbox):
        """Draw detection results on the frame with plate text"""
        output_frame = frame.copy()
        
        if bbox is not None and plate_text is not None:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw text background
            text_bg_height = 40
            cv2.rectangle(output_frame, (x, y - text_bg_height), (x + w, y), (0, 255, 0), -1)
            
            # Draw plate text
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - (text_bg_height - text_size[1]) // 2
            
            cv2.putText(output_frame, plate_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Display frame counter and stats
        cv2.putText(output_frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(output_frame, f"Plates detected: {len(self.detected_plates)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_frame

    def process_video(self, video_source, output_video_path):
        """Process video from file and save output with bounding boxes"""
        if not os.path.exists(video_source):
            print(f"Error: Video file '{video_source}' does not exist")
            return False
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_source}'")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        paused = False
        start_time = time.time()
        processed_frames = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame, plate_text, bbox = self.process_frame(frame)
                processed_frames += 1
                
                # Draw results
                output_frame = self.draw_results(processed_frame, plate_text, bbox)
                
                # Write frame to output video
                self.video_writer.write(output_frame)
                
                # Display the frame
                cv2.imshow('License Plate Recognition', output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"frame_{timestamp}.jpg", frame)
                print(f"Frame saved as frame_{timestamp}.jpg")
        
        # Calculate performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nProcessing completed!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Processed frames: {processed_frames}")
        print(f"Total valid plates detected: {len(self.detected_plates)}")
        
        if self.detected_plates:
            print("\nDetected license plates:")
            for plate, timestamp, confidence in self.detected_plates:
                print(f"  {plate} (detected at {timestamp}, confidence: {confidence:.2f})")
        else:
            print("\nNo license plates detected. Trying alternative approach...")
            self.try_alternative_detection(video_source)
        
        # Clean up
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        return True
    
    def try_alternative_detection(self, video_path):
        """Try alternative detection methods"""
        print("Trying alternative detection method...")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 != 0:  # Process every 10th frame
                continue
                
            # Try simple color-based detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 2.0 < aspect_ratio < 5.0 and w > 80 and h > 30:
                    # Potential plate found
                    plate_img = frame[y:y+h, x:x+w]
                    plate_text, confidence = self.read_license_plate_text(plate_img)
                    
                    if plate_text and self.is_valid_indian_plate(plate_text):
                        print(f"Alternative method found plate: {plate_text}")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.imshow('Alternative Detection', frame)
                        cv2.waitKey(1000)  # Show for 1 second
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("HAAR CASCADE LICENSE PLATE RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Initialize the recognizer
    recognizer = HaarLicensePlateRecognizer()
    
    # Specify your video path
    video_path = r"C:\Users\sankalp.DESKTOP-1RPOQ63\Documents\traffic-violation-detection\input.mp4"
    output_path = r"C:\Users\sankalp.DESKTOP-1RPOQ63\Documents\traffic-violation-detection\output_detected.mp4"
    
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print("Starting processing...")
    
    success = recognizer.process_video(video_path, output_path)
    
    if success:
        print(f"\nOutput video saved successfully to: {output_path}")
    else:
        print("\nFailed to process video")