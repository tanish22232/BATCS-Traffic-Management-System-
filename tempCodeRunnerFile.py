import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO # type: ignore
from PIL import Image
import torchvision.transforms as transforms
import argparse
from cnn import EnhancedTrafficLightCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='best_enhanced_model.pth', help='Path to model weights')
    parser.add_argument('--conf-threshold', type=float, default=0.6, help='Confidence threshold')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load enhanced model
    model = EnhancedTrafficLightCNN(num_classes=7).to(device)
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            class_names = checkpoint.get('class_names', [])
        else:
            model.load_state_dict(checkpoint)
            class_names = ["go", "goforward", "goleft", "red", "stopleft", "warning", "warningleft"]
        model.eval()
        print(f"Enhanced model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Enhanced transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load YOLO models
    yolo_traffic = YOLO("yolov8n.pt")
    yolo_vehicles = YOLO("yolov8m.pt")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('enhanced_output.mp4', fourcc, fps, (frame_width, frame_height))

    print("Starting enhanced traffic violation detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect traffic lights with YOLO
        tl_results = yolo_traffic(frame, classes=[9], verbose=False, conf=args.conf_threshold)
        current_light_state = None
        light_confidence = 0.0

        for result in tl_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                
                if conf < args.conf_threshold:
                    continue
                    
                tl_roi = frame[y1:y2, x1:x2]
                if tl_roi.size == 0 or tl_roi.shape[0] < 20 or tl_roi.shape[1] < 20:
                    continue

                # Classify with enhanced CNN
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(tl_roi, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(tensor_img)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, pred_class = torch.max(probabilities, 1)
                    
                    current_light_state = class_names[pred_class.item()]
                    light_confidence = confidence.item()

                    # Determine color
                    if "go" in current_light_state:
                        color = (0, 255, 0)  # Green
                    elif "red" in current_light_state or "stop" in current_light_state:
                        color = (0, 0, 255)  # Red
                    elif "warning" in current_light_state:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White

                    # Draw enhanced bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{current_light_state} ({light_confidence:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                except Exception as e:
                    continue

        # Enhanced violation detection
        if current_light_state and ("red" in current_light_state or "stop" in current_light_state):
            # Detect vehicles with better YOLO model
            vehicle_results = yolo_vehicles(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.5)
            
            for result in vehicle_results:
                for box in result.boxes:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                    vehicle_conf = box.conf.item()
                    
                    if vehicle_conf < 0.5:
                        continue
                    
                    # Enhanced violation logic
                    if vy2 > frame_height * 0.7:  # Vehicle in intersection
                        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                        cv2.putText(frame, "RED LIGHT VIOLATION", (vx1, vy1-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write to output
        out.write(frame)
        cv2.imshow("Enhanced Traffic Violation Detection", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Enhanced processing complete. Output saved as 'enhanced_output.mp4'")

if __name__ == "__main__":
    main()