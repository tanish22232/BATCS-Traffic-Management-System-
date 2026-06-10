from os import stat
from pickle import FRAME
from tkinter import Frame

import cv2


if stat == "red":
    # Check if any vehicle is crossing the stop line
    vehicle_results = yolo.predict(FRAME, classes=[2, 3, 5, 7])  # type: ignore # Cars, trucks, buses
    for vehicle in vehicle_results[0].boxes:
        vx1, vy1, vx2, vy2 = map(int, vehicle.xyxy[0])
        if (vx2 > stop_line_x):  # type: ignore # If vehicle crosses stop line
            cv2.putText(Frame, "VIOLATION!", (vx1, vy1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)