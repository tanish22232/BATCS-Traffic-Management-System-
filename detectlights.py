from ultralytics import YOLO # type: ignore
import cv2

# Load YOLO model (pretrained on COCO)
model = YOLO("yolov8n.pt")  # or yolov8s/m/l for better accuracy

# Open video
cap = cv2.VideoCapture(r"C:\Users\sankalp.DESKTOP-1RPOQ63\Documents\traffic-violation-detection\input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model.predict(frame, classes=[9])  # Class 9 = traffic light in COCO
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Traffic Light Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()